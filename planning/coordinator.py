"""Single-coarse-trajectory MPR-MPC planning coordinator."""

from __future__ import annotations

import time

import numpy as np
import torch

from .config import mapping, section
from .evaluator import TrajectoryWorldModelEvaluator
from .local_mppi import LocalMPPI
from .residual_prior import ResidualTrajectoryPrior
from .residual_training import ResidualTargetGenerator
from .structured_proposal import StructuredProposalGenerator
from .trajectory_adapter import LatticeActionAdapter
from .types import MPRPlanContext, MPRPlanResult


class MPRMPCPlanner:
    """Global Lattice mode selection, one residual correction, local H+1 MPPI."""

    def __init__(self, cfg, tdmpc_agent, controller, *, action_space=None):
        self.cfg = cfg
        self.agent = tdmpc_agent
        self.device = tdmpc_agent.device
        self.horizon = int(cfg.horizon)
        self.action_dim = int(cfg.action_dim)
        self.mpr_cfg = section(cfg, "mpr_mpc")
        self.enabled = bool(self.mpr_cfg.get("enabled", True))
        simulator = mapping(section(cfg, "metadrive").get("simulator"))
        self.control_dt = float(simulator.get("physics_world_step_size", 0.02)) * float(
            simulator.get("decision_repeat", 5)
        )
        self.structured = StructuredProposalGenerator(controller)
        self.action_adapter = LatticeActionAdapter(controller, action_dim=self.action_dim)
        if action_space is not None:
            self.action_adapter.validate_action_space(action_space)
        self.evaluator = TrajectoryWorldModelEvaluator(tdmpc_agent, self.horizon)

        residual_cfg = mapping(self.mpr_cfg.get("residual_prior"))
        self.residual_cfg = residual_cfg
        self.delta_t_bound = float(residual_cfg.get("delta_t_bound", 0.25))
        if self.delta_t_bound <= 0:
            raise ValueError("residual_prior.delta_t_bound must be positive.")
        self.parameter_scales = torch.as_tensor(
            residual_cfg.get("parameter_scales", [4.0, 20.0, 3.0]),
            dtype=torch.float32,
            device=self.device,
        )
        if self.parameter_scales.shape != (3,) or torch.any(self.parameter_scales <= 0):
            raise ValueError("residual_prior.parameter_scales must contain three positive values.")
        wm_scales = residual_cfg.get("wm_feature_scales", {})
        wm_scales = mapping(wm_scales)
        self.wm_feature_scales = torch.tensor(
            [
                float(wm_scales.get("reward_sum", wm_scales.get("reward", 50.0))),
                float(wm_scales.get("terminal_q", 10.0)),
                float(wm_scales.get("total_value", 50.0)),
            ],
            device=self.device,
        )
        if torch.any(self.wm_feature_scales <= 0):
            raise ValueError("All residual_prior.wm_feature_scales must be positive.")
        wm_feature_dim = 3
        residual_input_dim = int(cfg.latent_dim) + 3 + wm_feature_dim
        self.residual_prior = ResidualTrajectoryPrior(
            residual_input_dim,
            hidden_dims=residual_cfg.get("hidden_dims", [256, 256]),
            log_std_min=float(residual_cfg.get("log_std_min", -5.0)),
            log_std_max=float(residual_cfg.get("log_std_max", 1.0)),
            initial_log_std=float(residual_cfg.get("initial_log_std", -3.0)),
        ).to(self.device)
        self.residual_input_dim = residual_input_dim

        refinement = mapping(self.mpr_cfg.get("refinement"))
        self.min_speed = float(refinement.get("min_speed", 0.5))
        self.max_speed = float(refinement.get("max_speed", 20.0))
        self.min_horizon = float(refinement.get("min_horizon", 1.5))
        self.max_horizon = float(refinement.get("max_horizon", 3.5))
        self.use_mppi = bool(self.mpr_cfg.get("use_mppi", True))
        self.local_mppi = LocalMPPI(
            self.evaluator,
            mapping(self.mpr_cfg.get("mppi")),
            horizon=self.horizon,
            action_dim=self.action_dim,
            device=self.device,
        )
        target_cfg = mapping(self.mpr_cfg.get("residual_training"))
        self.target_generator = ResidualTargetGenerator(
            self.structured,
            self.action_adapter,
            self.evaluator,
            horizon=self.horizon,
            control_dt=self.control_dt,
            sample_count=int(target_cfg.get("target_samples", 32)),
            elite_count=int(target_cfg.get("target_elites", 4)),
            target_mode=str(target_cfg.get("target_mode", "best")),
            temperature=float(target_cfg.get("target_temperature", 1.0)),
            parameter_clamper=self.clamp_parameters,
        )
        self.last_context: MPRPlanContext | None = None
        self.last_result: MPRPlanResult | None = None
        self.residual_invalid_count = 0

    def reset(self) -> None:
        self.structured.reset()
        self.local_mppi.reset()
        self.last_context = None
        self.last_result = None

    def clamp_parameters(self, parameters) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float32).copy()
        lateral_min, lateral_max = self.structured.planner.last_lateral_bounds
        values[0] = np.clip(values[0], lateral_min, lateral_max)
        values[1] = np.clip(values[1], self.min_speed, self.max_speed)
        values[2] = np.clip(values[2], self.min_horizon, self.max_horizon)
        return values

    def residual_bounds(self, coarse_path) -> torch.Tensor:
        """Per-plan physical bounds [lane_width, coarse_speed/2, delta_T_max]."""
        lane_width = float(self.structured.planner.last_lane_width)
        if not np.isfinite(lane_width) or lane_width <= 0:
            raise RuntimeError("Lattice planner did not provide a positive current lane width.")
        target_speed = max(0.0, float(coarse_path.target_speed))
        return torch.tensor(
            [[lane_width, 0.5 * target_speed, self.delta_t_bound]],
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def prepare(self, observation, vehicle) -> MPRPlanContext:
        observation = observation if torch.is_tensor(observation) else torch.as_tensor(
            observation, dtype=torch.float32
        )
        latent = self.evaluator.encode(observation)
        selection = self.structured.select_coarse(vehicle)
        coarse_path = selection.coarse_path
        coarse_actions = self.action_adapter.path_to_actions(
            coarse_path,
            vehicle,
            required_steps=self.horizon + 1,
            control_dt=self.control_dt,
            device=self.device,
        )
        consequence = self.evaluator.evaluate_trajectory_consequence(
            latent, coarse_actions.unsqueeze(0)
        )
        wm_features = consequence.features / self.wm_feature_scales
        parameters = torch.tensor(
            [[coarse_path.target_d, coarse_path.target_speed, coarse_path.horizon]],
            dtype=torch.float32,
            device=self.device,
        ) / self.parameter_scales
        residual_bounds = self.residual_bounds(coarse_path)
        residual_input = torch.cat([latent, parameters, wm_features], dim=-1)
        if residual_input.shape != (1, self.residual_input_dim):
            raise RuntimeError(
                f"Residual input shape {tuple(residual_input.shape)} != {(1, self.residual_input_dim)}."
            )
        context = MPRPlanContext(
            observation=observation,
            latent=latent,
            selection=selection,
            coarse_actions=coarse_actions,
            coarse_consequence=consequence,
            wm_features=wm_features,
            coarse_parameters=parameters,
            residual_bounds=residual_bounds,
            residual_input=residual_input,
        )
        self.last_context = context
        return context

    @torch.no_grad()
    def refine_and_plan(
        self,
        context: MPRPlanContext,
        vehicle,
        *,
        eval_mode: bool,
        force_zero_residual: bool = False,
    ) -> MPRPlanResult:
        started = time.perf_counter()
        coarse = context.selection.coarse_path
        if force_zero_residual or not bool(self.mpr_cfg.get("use_residual", True)):
            residual = torch.zeros((1, 3), device=self.device)
            mean = residual.clone()
            log_std = residual.clone()
        else:
            deterministic_residual = (
                bool(self.residual_cfg.get("deterministic_eval", True))
                if eval_mode
                else bool(self.residual_cfg.get("deterministic_train", False))
            )
            residual, prior_info = self.residual_prior(
                context.residual_input,
                delta_bounds=context.residual_bounds,
                deterministic=deterministic_residual,
            )
            mean = prior_info["bounded_mean"]
            log_std = prior_info["log_std"]

        coarse_parameters = np.asarray(
            [coarse.target_d, coarse.target_speed, coarse.horizon], dtype=np.float32
        )
        requested = residual.squeeze(0).detach().cpu().numpy()
        refined_parameters = self.clamp_parameters(coarse_parameters + requested)
        effective_residual = refined_parameters - coarse_parameters
        refined = coarse
        residual_valid = True
        fallback_reason = ""
        if np.any(np.abs(effective_residual) > 1e-8):
            generated, fallback_reason = self.structured.regenerate(refined_parameters)
            if generated is None:
                residual_valid = False
                self.residual_invalid_count += 1
                effective_residual[:] = 0.0
            else:
                refined = generated
        refined_actions = self.action_adapter.path_to_actions(
            refined,
            vehicle,
            required_steps=self.horizon + 1,
            control_dt=self.control_dt,
            device=self.device,
        )
        if self.use_mppi:
            mppi = self.local_mppi.plan(
                context.latent, refined_actions, eval_mode=eval_mode
            )
            action = mppi.action
            initial_mean = mppi.initial_mean
            final_mean = mppi.final_mean
            initial_std = mppi.initial_std
            final_std = mppi.final_std
            final_value = float(mppi.final_value.cpu())
        else:
            action = refined_actions[0]
            initial_mean = final_mean = refined_actions
            initial_std = final_std = torch.zeros_like(refined_actions)
            refined_consequence = self.evaluator.evaluate_trajectory_consequence(
                context.latent, refined_actions.unsqueeze(0)
            )
            final_value = float(refined_consequence.total_value[0].cpu())

        paths = list(context.selection.candidates)
        if not any(path is refined for path in paths):
            paths.append(refined)
        self.structured.planner.last_candidates = paths
        self.structured.planner.last_selected_index = next(
            index for index, path in enumerate(paths) if path is refined
        )
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        result = MPRPlanResult(
            action=action.detach().cpu(),
            coarse_path=coarse,
            refined_path=refined,
            coarse_actions=context.coarse_actions.detach(),
            refined_actions=refined_actions.detach(),
            residual=torch.as_tensor(effective_residual, device=self.device),
            residual_mean=mean.squeeze(0).detach(),
            residual_log_std=log_std.squeeze(0).detach(),
            residual_valid=residual_valid,
            fallback_reason=fallback_reason,
            coarse_consequence=context.coarse_consequence,
            metrics={
                "mpr/candidate_count": float(len(context.selection.candidates)),
                "mpr/feasible_count": float(len(context.selection.feasible)),
                "mpr/selected_index": float(context.selection.selected_index),
                "mpr/coarse_value": float(context.coarse_consequence.total_value[0].cpu()),
                "mpr/final_value": final_value,
                "mpr/residual_d": float(effective_residual[0]),
                "mpr/residual_v": float(effective_residual[1]),
                "mpr/residual_t": float(effective_residual[2]),
                "mpr/residual_bound_d": float(context.residual_bounds[0, 0].cpu()),
                "mpr/residual_bound_v": float(context.residual_bounds[0, 1].cpu()),
                "mpr/residual_bound_t": float(context.residual_bounds[0, 2].cpu()),
                "mpr/residual_valid": float(residual_valid),
                "mpr/residual_invalid_count": float(self.residual_invalid_count),
                "mpr/planning_ms": elapsed_ms,
            },
            debug={
                "initial_mppi_mean": initial_mean.detach().cpu(),
                "final_mppi_mean": final_mean.detach().cpu(),
                "initial_mppi_std": initial_std.detach().cpu(),
                "final_mppi_std": final_std.detach().cpu(),
            },
        )
        self.last_result = result
        return result

    def plan(self, observation, vehicle, *, t0=False, eval_mode=False, force_zero_residual=False):
        if t0:
            self.reset()
        context = self.prepare(observation, vehicle)
        return self.refine_and_plan(
            context,
            vehicle,
            eval_mode=eval_mode,
            force_zero_residual=force_zero_residual,
        )

    def generate_residual_target(self, context, vehicle):
        return self.target_generator.generate(context, vehicle)


__all__ = ["MPRMPCPlanner"]
