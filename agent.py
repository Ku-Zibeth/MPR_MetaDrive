"""Training-compatible facade combining unchanged TD-MPC2 and MPR-MPC planning."""

from __future__ import annotations

from pathlib import Path

import torch

from .planning.config import mapping, section
from .planning.residual_training import ResidualLearner


class MPRMPCAgent:
    """Expose the original trainer API while routing actions through MPR-MPC."""

    def __init__(self, cfg, tdmpc_agent, planner, env):
        self.cfg = cfg
        self.tdmpc_agent = tdmpc_agent
        self.planner = planner
        self.env = env
        self.model = tdmpc_agent.model
        self.device = tdmpc_agent.device
        training_cfg = mapping(section(cfg, "mpr_mpc").get("residual_training"))
        self.residual_training_enabled = bool(training_cfg.get("enabled", True))
        self.target_every = max(1, int(training_cfg.get("target_every", 1)))
        self.learning_starts = max(0, int(training_cfg.get("learning_starts", 0)))
        if self.learning_starts != int(planner.residual_start_step):
            raise ValueError(
                "residual_training.learning_starts must equal "
                "mpr_mpc.stages.residual_start_step so execution and training align."
            )
        self.updates_per_step = max(0, int(training_cfg.get("updates_per_step", 1)))
        debug_cfg = mapping(section(cfg, "mpr_mpc").get("debug"))
        self.debug_enabled = bool(debug_cfg.get("enabled", False))
        self.debug_every = max(1, int(debug_cfg.get("every", 100)))
        self.residual_learner = ResidualLearner(planner, training_cfg)
        self._planner_calls = 0
        self.global_env_step = 0
        self.best_eval_key = None
        self.last_plan = None
        self.last_target_metrics: dict[str, float] = {}

    def set_global_step(self, step: int) -> None:
        self.global_env_step = max(0, int(step))

    def act(self, obs, t0=False, eval_mode=False, task=None, global_step=None):
        if not self.planner.enabled:
            return self.tdmpc_agent.act(
                obs, t0=t0, eval_mode=eval_mode, task=task
            )
        if task is not None:
            raise ValueError("The current MetaDrive MPR-MPC integration is single-task only.")
        step = self.global_env_step if global_step is None else int(global_step)
        self.set_global_step(step)
        result = self.planner.plan(
            obs,
            self.env.unwrapped_metadrive.agent,
            global_step=step,
            t0=t0,
            eval_mode=eval_mode,
        )
        self.last_plan = result
        if not eval_mode:
            self._planner_calls += 1
            if self.debug_enabled and self._planner_calls % self.debug_every == 0:
                coarse = result.coarse_parameters
                refined = result.refined_parameters
                residual = result.residual.detach().cpu().numpy()
                print(
                    "[mpr] "
                    f"candidates={int(result.metrics['mpr/candidate_count'])} "
                    f"valid={int(result.metrics['mpr/feasible_count'])} "
                    f"selected={int(result.metrics['mpr/selected_index'])} "
                    f"coarse={coarse.tolist()} residual={residual.tolist()} "
                    f"refined={refined.tolist()} "
                    f"J={result.metrics['mpr/coarse_value']:.3f}->"
                    f"{result.metrics['mpr/final_value']:.3f} "
                    f"time_ms={result.metrics['mpr/planning_ms']:.2f}"
                )
            should_generate = (
                self.residual_training_enabled
                and result.stage.use_residual
                and step >= self.learning_starts
                and self._planner_calls % self.target_every == 0
            )
            if should_generate:
                target, metrics = self.planner.generate_residual_target(
                    self.planner.last_context, self.env.unwrapped_metadrive.agent
                )
                self.residual_learner.add(self.planner.last_context, target)
                self.last_target_metrics = metrics
        return result.action

    def update(self, buffer):
        info = dict(self.tdmpc_agent.update(buffer))
        for key in ("consistency_loss", "reward_loss", "value_loss", "termination_loss"):
            if key in info:
                info[f"wm/{key}"] = info[key]
        stage = self.planner.stage_for_step(self.global_env_step)
        if self.planner.enabled and self.residual_training_enabled and stage.use_residual:
            residual_info = {}
            for _ in range(self.updates_per_step):
                residual_info = self.residual_learner.update()
            info.update({f"mpr/{key}": value for key, value in residual_info.items()})
            info.update(
                {
                    f"mpr/{key}": torch.tensor(value, device=self.device)
                    for key, value in self.last_target_metrics.items()
                }
            )
        if self.last_plan is not None:
            info.update(
                {
                    key: torch.tensor(value, device=self.device)
                    for key, value in self.last_plan.metrics.items()
                }
            )
        env_metrics = {
            "env/metadrive_reward": "metadrive_reward",
            "env/tdmpc2_reward": "tdmpc2_reward",
            "env/cost": "cost",
            "env/risk_field_cost": "risk_field_cost",
            "env/risk_field_normalized_cost": "risk_field_normalized_cost",
            "env/risk_field_reward_penalty": "risk_field_reward_penalty",
            "env/risk_field_boundary_cost": "risk_field_boundary_cost",
            "env/risk_field_lane_cost": "risk_field_lane_cost",
            "env/risk_field_offroad_cost": "risk_field_offroad_cost",
            "env/risk_field_vehicle_cost": "risk_field_vehicle_cost",
            "env/risk_field_object_cost": "risk_field_object_cost",
            "env/safety_risk_cost": "safety_risk_cost",
            "env/safety_event_cost": "safety_event_cost",
            "env/route_completion": "route_completion",
        }
        last_env_info = self.env.last_info
        info.update(
            {
                metric: torch.tensor(
                    float(last_env_info.get(source, 0.0)), device=self.device
                )
                for metric, source in env_metrics.items()
            }
        )
        return info

    def save(self, filepath) -> None:
        state = {
            "format": "mpr_mpc_v1",
            "model": self.tdmpc_agent.model.state_dict(),
            "tdmpc_optim": self.tdmpc_agent.optim.state_dict(),
            "tdmpc_pi_optim": self.tdmpc_agent.pi_optim.state_dict(),
            "tdmpc_scale": self.tdmpc_agent.scale.state_dict(),
            "global_env_step": self.global_env_step,
            "best_eval_key": self.best_eval_key,
            "mpr_mpc": {
                "residual_prior": self.planner.residual_prior.state_dict(),
                "residual_optimizer": self.residual_learner.optimizer.state_dict(),
                "residual_update_steps": self.residual_learner.update_steps,
                "planner_calls": self._planner_calls,
                "residual_invalid_count": self.planner.residual_invalid_count,
                "residual_attempt_count": self.planner.residual_attempt_count,
                "mppi_call_count": self.planner.mppi_call_count,
                "baseline_selected_count": self.planner.baseline_selected_count,
            },
        }
        torch.save(state, Path(filepath))

    def load(self, filepath, *, load_optimizer: bool = False) -> None:
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        if not isinstance(state, dict) or state.get("format") != "mpr_mpc_v1":
            raise ValueError(
                "Refusing to resume a non-MPR-MPC checkpoint. Start from scratch with "
                "resume_checkpoint=null or provide an mpr_mpc_v1 checkpoint."
            )
        self.tdmpc_agent.load(state)
        mpr_state = state.get("mpr_mpc") if isinstance(state, dict) else None
        if not mpr_state:
            raise ValueError("MPR-MPC checkpoint is missing planner state.")
        prior_state = mpr_state.get("residual_prior")
        self.planner.residual_prior.load_state_dict(prior_state)
        if load_optimizer:
            self.tdmpc_agent.optim.load_state_dict(state["tdmpc_optim"])
            self.tdmpc_agent.pi_optim.load_state_dict(state["tdmpc_pi_optim"])
            self.tdmpc_agent.scale.load_state_dict(state["tdmpc_scale"])
            self.residual_learner.optimizer.load_state_dict(mpr_state["residual_optimizer"])
        self.residual_learner.update_steps = int(mpr_state.get("residual_update_steps", 0))
        self._planner_calls = int(mpr_state.get("planner_calls", 0))
        self.global_env_step = int(state.get("global_env_step", 0))
        best_key = state.get("best_eval_key")
        self.best_eval_key = tuple(best_key) if best_key is not None else None
        self.planner.residual_invalid_count = int(mpr_state.get("residual_invalid_count", 0))
        self.planner.residual_attempt_count = int(mpr_state.get("residual_attempt_count", 0))
        self.planner.mppi_call_count = int(mpr_state.get("mppi_call_count", 0))
        self.planner.baseline_selected_count = int(mpr_state.get("baseline_selected_count", 0))

    def eval(self):
        self.tdmpc_agent.eval()
        self.planner.residual_prior.eval()
        return self


__all__ = ["MPRMPCAgent"]
