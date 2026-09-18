"""Single-coarse-trajectory MPR-MPC planning coordinator."""

from __future__ import annotations

import time

import numpy as np
import torch

from .config import mapping, section
from .evaluator import TrajectoryWorldModelEvaluator
from .kinematic_filter import CorridorContext
from .local_mppi import LocalMPPI
from .residual_prior import ResidualTrajectoryPrior
from .residual_training import ResidualTargetGenerator
from .structured_proposal import StructuredProposalGenerator
from .trajectory_adapter import LatticeActionAdapter
from .types import MPRPlanContext, MPRPlanResult, PlanningStage


class MPRMPCPlanner:
    """Global Lattice selection, optional 2-D residual, then bounded local MPPI."""

    def __init__(self, cfg, tdmpc_agent, controller, *, action_space=None):
        self.cfg = cfg
        self.agent = tdmpc_agent
        self.device = tdmpc_agent.device
        self.horizon = int(cfg.horizon)
        self.action_dim = int(cfg.action_dim)
        self.mpr_cfg = section(cfg, "mpr_mpc")
        self.enabled = bool(self.mpr_cfg.get("enabled", True))
        self.allow_residual = bool(self.mpr_cfg.get("use_residual", True))
        self.allow_mppi = bool(self.mpr_cfg.get("use_mppi", True))
        stages = mapping(self.mpr_cfg.get("stages"))
        self.mppi_start_step = int(stages.get("mppi_start_step", 20000))
        self.residual_start_step = int(stages.get("residual_start_step", 50000))
        if not 0 <= self.mppi_start_step <= self.residual_start_step:
            raise ValueError("Require 0 <= mppi_start_step <= residual_start_step.")

        simulator = mapping(section(cfg, "metadrive").get("simulator"))
        self.control_dt = float(simulator.get("physics_world_step_size", 0.02)) * float(
            simulator.get("decision_repeat", 5)
        )
        self.structured = StructuredProposalGenerator(controller)
        self.action_adapter = LatticeActionAdapter(controller, action_dim=self.action_dim)
        if action_space is not None:
            self.action_adapter.validate_action_space(action_space)
        self.evaluator = TrajectoryWorldModelEvaluator(tdmpc_agent, self.horizon)

        self.residual_cfg = mapping(self.mpr_cfg.get("residual_prior"))
        self.parameter_scales = torch.as_tensor(
            self.residual_cfg.get("parameter_scales", [4.0, 20.0, 3.0]),
            dtype=torch.float32,
            device=self.device,
        )
        if self.parameter_scales.shape != (3,) or torch.any(self.parameter_scales <= 0):
            raise ValueError("residual_prior.parameter_scales must contain three positive values.")
        wm_scales = mapping(self.residual_cfg.get("wm_feature_scales"))
        self.wm_feature_scales = torch.tensor(
            [
                float(wm_scales.get("reward_sum", 50.0)),
                float(wm_scales.get("terminal_q", 10.0)),
                float(wm_scales.get("total_value", 50.0)),
            ],
            device=self.device,
        )
        if torch.any(self.wm_feature_scales <= 0):
            raise ValueError("All residual_prior.wm_feature_scales must be positive.")
        self.residual_input_dim = int(cfg.latent_dim) + 3 + 3
        self.residual_prior = ResidualTrajectoryPrior(
            self.residual_input_dim,
            hidden_dims=self.residual_cfg.get("hidden_dims", [256, 256]),
            log_std_min=float(self.residual_cfg.get("log_std_min", -5.0)),
            log_std_max=float(self.residual_cfg.get("log_std_max", 1.0)),
            initial_log_std=float(self.residual_cfg.get("initial_log_std", -3.0)),
        ).to(self.device)

        refinement = mapping(self.mpr_cfg.get("refinement"))
        self.min_speed = float(refinement.get("min_speed", 0.5))
        self.max_speed = float(refinement.get("max_speed", 20.0))
        self.min_horizon = float(refinement.get("min_horizon", 1.5))
        self.max_horizon = float(refinement.get("max_horizon", 3.5))
        mppi_cfg = mapping(self.mpr_cfg.get("mppi"))
        self.mppi_cfg = mppi_cfg
        self.local_mppi = LocalMPPI(
            self.evaluator,
            mppi_cfg,
            horizon=self.horizon,
            action_dim=self.action_dim,
            device=self.device,
            control_dt=self.control_dt,
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
            target_mode=str(target_cfg.get("target_mode", "softmax_elite")),
            temperature=float(target_cfg.get("target_temperature", 0.5)),
            distribution=str(target_cfg.get("target_distribution", "truncated_gaussian")),
            std_scale=float(target_cfg.get("target_std_scale", 0.25)),
            min_improvement_abs=float(target_cfg.get("min_improvement_abs", 1.0)),
            min_improvement_ratio=float(target_cfg.get("min_improvement_ratio", 0.02)),
            parameter_clamper=self.clamp_parameters,
        )
        self.last_context: MPRPlanContext | None = None
        self.last_result: MPRPlanResult | None = None
        self.residual_invalid_count = 0
        self.residual_attempt_count = 0
        self.mppi_call_count = 0
        self.baseline_selected_count = 0

    def stage_for_step(self, global_step: int) -> PlanningStage:
        step = max(0, int(global_step))
        if step < self.mppi_start_step:
            return PlanningStage("A", False, False)
        if step < self.residual_start_step:
            return PlanningStage("B", False, self.allow_mppi)
        return PlanningStage("C", self.allow_residual, self.allow_mppi)

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
        lane_width = float(self.structured.planner.last_lane_width)
        if not np.isfinite(lane_width) or lane_width <= 0:
            raise RuntimeError("Lattice planner did not provide a positive current lane width.")
        target_speed = max(0.0, float(coarse_path.target_speed))
        return torch.tensor(
            [[lane_width, 0.5 * target_speed]], dtype=torch.float32, device=self.device
        )

    def apply_residual_parameters(self, coarse_parameters, residual) -> np.ndarray:
        coarse = np.asarray(coarse_parameters, dtype=np.float32)
        correction = np.asarray(residual, dtype=np.float32).reshape(2)
        requested = coarse.copy()
        requested[:2] += correction
        refined = self.clamp_parameters(requested)
        refined[2] = coarse[2]
        return refined

    @torch.no_grad()
    def prepare(
        self, observation, vehicle, *, need_world_model_features: bool = True
    ) -> MPRPlanContext:
        observation = observation if torch.is_tensor(observation) else torch.as_tensor(
            observation, dtype=torch.float32
        )
        selection = self.structured.select_coarse(vehicle)
        coarse_path = selection.coarse_path
        coarse_actions = self.action_adapter.path_to_actions(
            coarse_path,
            vehicle,
            required_steps=self.horizon + 1,
            control_dt=self.control_dt,
            device=self.device,
        )
        parameters = torch.tensor(
            [[coarse_path.target_d, coarse_path.target_speed, coarse_path.horizon]],
            dtype=torch.float32,
            device=self.device,
        ) / self.parameter_scales
        residual_bounds = self.residual_bounds(coarse_path)
        latent = consequence = wm_features = residual_input = None
        if need_world_model_features:
            latent = self.evaluator.encode(observation)
            consequence = self.evaluator.evaluate_trajectory_consequence(
                latent, coarse_actions.unsqueeze(0)
            )
            wm_features = consequence.features / self.wm_feature_scales
            residual_input = torch.cat([latent, parameters, wm_features], dim=-1)
            if residual_input.shape != (1, self.residual_input_dim):
                raise RuntimeError(
                    f"Residual input shape {tuple(residual_input.shape)} != "
                    f"{(1, self.residual_input_dim)}."
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

    def _vehicle_geometry(self, vehicle) -> tuple[float, float, float]:
        def positive(value, fallback):
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float("nan")
            return value if np.isfinite(value) and value > 0.0 else float(fallback)

        fallback_width = float(self.mppi_cfg.get("kinematic_vehicle_width", 1.852))
        fallback_wheelbase = float(self.mppi_cfg.get("kinematic_wheelbase", 2.46894))
        fallback_steering = float(
            self.mppi_cfg.get("kinematic_max_steering_angle", 0.6981317)
        )
        vehicle_width = positive(getattr(vehicle, "WIDTH", None), fallback_width)
        front = getattr(vehicle, "FRONT_WHEELBASE", None)
        rear = getattr(vehicle, "REAR_WHEELBASE", None)
        try:
            wheelbase = float(front) + float(rear)
        except (TypeError, ValueError):
            wheelbase = fallback_wheelbase
        wheelbase = positive(wheelbase, fallback_wheelbase)
        try:
            max_steering_angle = np.deg2rad(float(vehicle.max_steering))
        except (AttributeError, TypeError, ValueError):
            max_steering_angle = fallback_steering
        max_steering_angle = positive(max_steering_angle, fallback_steering)
        return vehicle_width, wheelbase, max_steering_angle

    def _corridor_context(self, refined, vehicle) -> CorridorContext:
        # H controls a_t...a_{t+H-1} produce H post-transition states. The
        # H+1-th control belongs only to terminal Q and is not integrated here.
        query = (np.arange(self.horizon, dtype=np.float64) + 1.0) * self.control_dt
        source_t = np.asarray(refined.t, dtype=np.float64)
        source_d = np.asarray(refined.d, dtype=np.float64)
        refined_d = np.interp(np.minimum(query, source_t[-1]), source_t, source_d)
        try:
            speed = float(np.linalg.norm(np.asarray(vehicle.velocity, dtype=float)[:2]))
        except Exception:
            speed = max(0.0, float(getattr(vehicle, "speed", 0.0)))
        planner = self.structured.planner
        vehicle_width, wheelbase, max_steering_angle = self._vehicle_geometry(vehicle)
        safe_min, safe_max = planner.last_lateral_bounds
        lattice_width = float(planner.last_vehicle_width or vehicle_width)
        lattice_margin = float(planner.config.get("frenet_lane_margin", 0.0))
        # last_lateral_bounds is a target/vehicle-center range, already inset by
        # half the Lattice vehicle width and frenet_lane_margin. Undo that inset
        # to recover raw road edges; the kinematic filter then applies the current
        # vehicle width and its own safety margin exactly once.
        road_min = float(safe_min) - 0.5 * lattice_width - lattice_margin
        road_max = float(safe_max) + 0.5 * lattice_width + lattice_margin
        return CorridorContext(
            refined_lateral=torch.as_tensor(refined_d, dtype=torch.float32, device=self.device),
            lane_width=float(self.structured.planner.last_lane_width),
            road_min=float(road_min),
            road_max=float(road_max),
            initial_speed=speed,
            vehicle_width=vehicle_width,
            wheelbase=wheelbase,
            max_steering_angle=max_steering_angle,
        )

    @torch.no_grad()
    def refine_and_plan(
        self,
        context: MPRPlanContext,
        vehicle,
        *,
        eval_mode: bool,
        global_step: int,
        force_zero_residual: bool = False,
    ) -> MPRPlanResult:
        started = time.perf_counter()
        stage = self.stage_for_step(global_step)
        coarse = context.selection.coarse_path
        if force_zero_residual or not stage.use_residual:
            residual = torch.zeros((1, 2), device=self.device)
            mean = residual.clone()
            log_std = residual.clone()
        else:
            if context.residual_input is None:
                raise RuntimeError("Stage C requires current world-model residual features.")
            deterministic = (
                bool(self.residual_cfg.get("deterministic_eval", True))
                if eval_mode
                else bool(self.residual_cfg.get("deterministic_train", True))
            )
            residual, prior_info = self.residual_prior(
                context.residual_input,
                delta_bounds=context.residual_bounds,
                deterministic=deterministic,
            )
            mean = prior_info["bounded_mean"]
            log_std = prior_info["log_std"]
            self.residual_attempt_count += 1

        coarse_parameters = np.asarray(
            [coarse.target_d, coarse.target_speed, coarse.horizon], dtype=np.float32
        )
        requested = residual.squeeze(0).detach().cpu().numpy()
        refined_parameters = self.apply_residual_parameters(coarse_parameters, requested)
        effective_residual = refined_parameters[:2] - coarse_parameters[:2]
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

        if stage.use_mppi:
            if context.latent is None:
                raise RuntimeError("Stage B/C MPPI requires an encoded current observation.")
            mppi = self.local_mppi.plan(
                context.latent,
                refined_actions,
                eval_mode=eval_mode,
                corridor_context=self._corridor_context(refined, vehicle),
            )
            self.mppi_call_count += 1
            self.baseline_selected_count += int(mppi.baseline_selected)
            action = mppi.action
            initial_mean, final_mean = mppi.initial_mean, mppi.final_mean
            initial_std, final_std = mppi.initial_std, mppi.final_std
            baseline_wm_value = float(mppi.baseline_wm_value.cpu())
            selected_wm_value = float(mppi.selected_wm_value.cpu())
            wm_value_gain = float(mppi.wm_value_gain.cpu())
            baseline_score = float(mppi.baseline_score.cpu())
            selected_score = float(mppi.selected_score.cpu())
            planner_score_gain = float(mppi.planner_score_gain.cpu())
            baseline_selected = float(mppi.baseline_selected)
            reject_count = float(mppi.corridor_reject_count)
            reject_rate = float(mppi.corridor_reject_rate)
            max_lateral_deviation = float(mppi.max_lateral_deviation)
            max_delta = mppi.max_action_delta
        else:
            action = refined_actions[0]
            initial_mean = final_mean = refined_actions
            initial_std = final_std = torch.zeros_like(refined_actions)
            if context.latent is None:
                # Stage A is Lattice-only: no planning-time encoder/WM rollout.
                baseline_wm_value = selected_wm_value = float("nan")
            else:
                refined_value = self.evaluator.evaluate_trajectory_consequence(
                    context.latent, refined_actions.unsqueeze(0)
                ).total_value[0]
                baseline_wm_value = selected_wm_value = float(refined_value.cpu())
            baseline_score = selected_score = baseline_wm_value
            wm_value_gain = planner_score_gain = 0.0
            baseline_selected = 1.0
            reject_count = reject_rate = max_lateral_deviation = 0.0
            max_delta = torch.zeros(self.action_dim, device=self.device)

        paths = list(context.selection.candidates)
        if not any(path is refined for path in paths):
            paths.append(refined)
        self.structured.planner.last_candidates = paths
        self.structured.planner.last_selected_index = next(
            index for index, path in enumerate(paths) if path is refined
        )
        invalid_rate = self.residual_invalid_count / max(1, self.residual_attempt_count)
        baseline_rate = self.baseline_selected_count / max(1, self.mppi_call_count)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        coarse_value = (
            float(context.coarse_consequence.total_value[0].cpu())
            if context.coarse_consequence is not None
            else float("nan")
        )
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
            stage=stage,
            metrics={
                "mpr/stage": float(ord(stage.name) - ord("A")),
                "mpr/candidate_count": float(len(context.selection.candidates)),
                "mpr/feasible_count": float(len(context.selection.feasible)),
                "mpr/selected_index": float(context.selection.selected_index),
                "mpr/coarse_value": coarse_value,
                "mpr/baseline_wm_value": baseline_wm_value,
                "mpr/selected_wm_value": selected_wm_value,
                "mpr/wm_value_gain": wm_value_gain,
                "mpr/baseline_score": baseline_score,
                "mpr/selected_score": selected_score,
                "mpr/planner_score_gain": planner_score_gain,
                # Compatibility aliases: all three are planner-score quantities.
                "mpr/baseline_value": baseline_score,
                "mpr/final_value": selected_score,
                "mpr/planner_gain": planner_score_gain,
                "mpr/residual_d": float(effective_residual[0]),
                "mpr/residual_v": float(effective_residual[1]),
                "mpr/residual_bound_d": float(context.residual_bounds[0, 0].cpu()),
                "mpr/residual_bound_v": float(context.residual_bounds[0, 1].cpu()),
                "mpr/residual_valid": float(residual_valid),
                "mpr/residual_invalid_count": float(self.residual_invalid_count),
                "mpr/residual_invalid_rate": float(invalid_rate),
                "mpr/initial_std_steer": float(initial_std[:, 0].mean().cpu()),
                "mpr/initial_std_throttle": float(initial_std[:, 1].mean().cpu()),
                "mpr/final_std_steer": float(final_std[:, 0].mean().cpu()),
                "mpr/final_std_throttle": float(final_std[:, 1].mean().cpu()),
                "mpr/max_delta_steer": float(max_delta[0].cpu()),
                "mpr/max_delta_throttle": float(max_delta[1].cpu()),
                "mpr/max_lateral_deviation": max_lateral_deviation,
                "mpr/corridor_reject_count": reject_count,
                "mpr/corridor_reject_rate": reject_rate,
                "mpr/baseline_selected": baseline_selected,
                "mpr/baseline_selected_rate": float(baseline_rate),
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

    def plan(
        self,
        observation,
        vehicle,
        *,
        global_step: int,
        t0=False,
        eval_mode=False,
        force_zero_residual=False,
    ):
        if t0:
            self.reset()
        stage = self.stage_for_step(global_step)
        context = self.prepare(
            observation,
            vehicle,
            need_world_model_features=stage.use_residual or stage.use_mppi,
        )
        return self.refine_and_plan(
            context,
            vehicle,
            eval_mode=eval_mode,
            global_step=global_step,
            force_zero_residual=force_zero_residual,
        )

    def generate_residual_target(self, context, vehicle):
        return self.target_generator.generate(context, vehicle)


__all__ = ["MPRMPCPlanner"]
