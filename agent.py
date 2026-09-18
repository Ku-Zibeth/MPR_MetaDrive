"""Training-compatible facade combining unchanged TD-MPC2 and MPR-MPC planning."""

from __future__ import annotations

from pathlib import Path
import warnings

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
        self.updates_per_step = max(0, int(training_cfg.get("updates_per_step", 1)))
        debug_cfg = mapping(section(cfg, "mpr_mpc").get("debug"))
        self.debug_enabled = bool(debug_cfg.get("enabled", False))
        self.debug_every = max(1, int(debug_cfg.get("every", 100)))
        self.residual_learner = ResidualLearner(planner, training_cfg)
        self._planner_calls = 0
        self.last_plan = None
        self.last_target_metrics: dict[str, float] = {}

    def act(self, obs, t0=False, eval_mode=False, task=None):
        if not self.planner.enabled:
            return self.tdmpc_agent.act(
                obs, t0=t0, eval_mode=eval_mode, task=task
            )
        if task is not None:
            raise ValueError("The current MetaDrive MPR-MPC integration is single-task only.")
        result = self.planner.plan(
            obs,
            self.env.unwrapped_metadrive.agent,
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
                and self._planner_calls >= self.learning_starts
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
        if self.planner.enabled and self.residual_training_enabled:
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
            "model": self.tdmpc_agent.model.state_dict(),
            "mpr_mpc": {
                "residual_prior": self.planner.residual_prior.state_dict(),
                "residual_optimizer": self.residual_learner.optimizer.state_dict(),
                "residual_update_steps": self.residual_learner.update_steps,
                "planner_calls": self._planner_calls,
            },
        }
        torch.save(state, Path(filepath))

    def load(self, filepath, *, load_optimizer: bool = False) -> None:
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.tdmpc_agent.load(state)
        mpr_state = state.get("mpr_mpc") if isinstance(state, dict) else None
        if not mpr_state:
            return
        prior_state = mpr_state.get("residual_prior")
        current_state = self.planner.residual_prior.state_dict()
        compatible = prior_state is not None and all(
            key in current_state and current_state[key].shape == value.shape
            for key, value in prior_state.items()
        ) and set(prior_state) == set(current_state)
        if compatible:
            self.planner.residual_prior.load_state_dict(prior_state)
        else:
            warnings.warn(
                "Skipping incompatible legacy MPR residual head; the current 518-D "
                "head remains zero-mean initialized.",
                RuntimeWarning,
            )
        if compatible and load_optimizer and "residual_optimizer" in mpr_state:
            self.residual_learner.optimizer.load_state_dict(mpr_state["residual_optimizer"])
        self.residual_learner.update_steps = int(mpr_state.get("residual_update_steps", 0))
        self._planner_calls = int(mpr_state.get("planner_calls", 0))

    def eval(self):
        self.tdmpc_agent.eval()
        self.planner.residual_prior.eval()
        return self


__all__ = ["MPRMPCAgent"]
