"""Training facade combining TD-MPC2, MPR-MPC planning, and Residual SAC."""

from __future__ import annotations

from pathlib import Path

import torch

from .planning.config import mapping, section
from .residual_rl import ResidualReplayBuffer, ResidualSACAgent
from .residual_rl.types import ResidualPolicyOutput


class MPRMPCAgent:
    """Expose the original trainer API while routing actions through MPR-MPC."""

    def __init__(self, cfg, tdmpc_agent, planner, env):
        self.cfg = cfg
        self.tdmpc_agent = tdmpc_agent
        self.planner = planner
        self.env = env
        self.model = tdmpc_agent.model
        self.device = tdmpc_agent.device

        self.residual_cfg = mapping(section(cfg, "residual_rl"))
        self.residual_rl_enabled = bool(self.residual_cfg.get("enabled", True))
        self.planner.allow_residual = bool(self.planner.allow_residual and self.residual_rl_enabled)
        self.freeze_tdmpc2_on_start = bool(
            self.residual_cfg.get("freeze_tdmpc2_on_start", True)
        )
        self.seed_steps = max(0, int(self.residual_cfg.get("seed_steps", 5000)))
        self.warmup_action_scale = min(
            1.0, max(0.0, float(self.residual_cfg.get("warmup_action_scale", 0.5)))
        )
        self.residual_learning_starts = max(
            0, int(self.residual_cfg.get("learning_starts", 5000))
        )
        self.residual_update_per_step = max(
            0.0, float(self.residual_cfg.get("update_per_step", 0.2))
        )
        self.residual_batch_size = max(1, int(self.residual_cfg.get("batch_size", 256)))
        self.residual_gamma = float(self.residual_cfg.get("gamma", 0.99))
        self.invalid_residual_penalty = max(
            0.0, float(self.residual_cfg.get("invalid_residual_penalty", 0.2))
        )

        self.residual_sac = ResidualSACAgent(
            planner.residual_state_dim,
            self.residual_cfg,
            device=self.device,
        )
        self.residual_replay = ResidualReplayBuffer(
            int(self.residual_cfg.get("buffer_size", 200000)),
            planner.residual_state_dim,
            action_dim=2,
        )
        self.planner.set_residual_action_provider(self._select_residual_action)

        debug_cfg = mapping(section(cfg, "mpr_mpc").get("debug"))
        self.debug_enabled = bool(debug_cfg.get("enabled", False))
        self.debug_every = max(1, int(debug_cfg.get("every", 100)))
        self._planner_calls = 0
        self.global_env_step = 0
        self.best_eval_key = None
        self.last_plan = None
        self.last_residual_transition_metrics: dict[str, float] = {}
        self.residual_rl_step = 0
        self.residual_update_budget = 0.0
        self._pending_residual: dict | None = None
        self._residual_episode_cost = 0.0
        self._residual_episode_steps = 0
        self._tdmpc_frozen = False

    def set_global_step(self, step: int) -> None:
        self.global_env_step = max(0, int(step))

    def _maybe_freeze_tdmpc2(self, step: int) -> None:
        if self._tdmpc_frozen or not self.freeze_tdmpc2_on_start:
            return
        if not self.planner.stage_for_step(step).use_residual:
            return
        self.tdmpc_agent.model.eval()
        for parameter in self.tdmpc_agent.model.parameters():
            parameter.requires_grad_(False)
        self._tdmpc_frozen = True

    @torch.no_grad()
    def _select_residual_action(self, state: torch.Tensor, *, eval_mode: bool) -> ResidualPolicyOutput:
        if eval_mode:
            return self.residual_sac.select_action(state, deterministic=True)
        if self.residual_rl_step < self.seed_steps:
            action = (
                torch.rand((1, 2), device=self.device, dtype=torch.float32) * 2.0 - 1.0
            ) * self.warmup_action_scale
            zeros = torch.zeros_like(action)
            return ResidualPolicyOutput(action=action, mean=zeros, log_std=zeros, warmup=True)
        return self.residual_sac.select_action(state, deterministic=False)

    def _complete_pending_with_next_state(self, next_state: torch.Tensor) -> None:
        pending = self._pending_residual
        if not pending or "reward" not in pending:
            return
        self.residual_replay.add(
            pending["state"],
            pending["action"],
            pending["reward"],
            pending["cost"],
            next_state.detach().cpu().numpy(),
            pending["done"],
        )
        self._pending_residual = None

    def observe_transition(self, reward: float, cost: float, done: bool, info: dict | None = None) -> None:
        del info
        pending = self._pending_residual
        if not pending or "reward" in pending:
            return
        env_reward = float(reward)
        env_cost = float(cost)
        invalid_penalty = self.invalid_residual_penalty if not pending["valid"] else 0.0
        residual_reward = env_reward - invalid_penalty
        pending.update(
            reward=residual_reward,
            cost=env_cost,
            done=bool(done),
        )
        self._residual_episode_cost += env_cost
        self._residual_episode_steps += 1
        self.last_residual_transition_metrics = {
            "env/reward": env_reward,
            "residual_rl/reward": residual_reward,
            "residual_rl/invalid_penalty": invalid_penalty,
            "residual_rl/env_cost": env_cost,
        }
        if done:
            terminal_state = pending["state"]
            self.residual_replay.add(
                terminal_state,
                pending["action"],
                residual_reward,
                env_cost,
                terminal_state,
                True,
            )
            self._pending_residual = None
            if self._residual_episode_steps > 0:
                value = self.residual_sac.update_lagrangian(self._residual_episode_cost)
                self.last_residual_transition_metrics.update(
                    {
                        "residual_rl/episode_cost": self._residual_episode_cost,
                        "residual_rl/episode_steps": float(self._residual_episode_steps),
                        "residual_rl/lagrangian": value,
                    }
                )
            self._residual_episode_cost = 0.0
            self._residual_episode_steps = 0

    def act(self, obs, t0=False, eval_mode=False, task=None, global_step=None):
        if not self.planner.enabled:
            return self.tdmpc_agent.act(obs, t0=t0, eval_mode=eval_mode, task=task)
        if task is not None:
            raise ValueError("The current MetaDrive MPR-MPC integration is single-task only.")
        step = self.global_env_step if global_step is None else int(global_step)
        self.set_global_step(step)
        self._maybe_freeze_tdmpc2(step)
        result = self.planner.plan(
            obs,
            self.env.unwrapped_metadrive.agent,
            global_step=step,
            t0=t0,
            eval_mode=eval_mode,
        )
        self.last_plan = result
        context = self.planner.last_context
        if not eval_mode:
            if context is not None and context.residual_state is not None:
                self._complete_pending_with_next_state(context.residual_state)
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
                    f"WM={result.metrics['mpr/baseline_wm_value']:.3f}->"
                    f"{result.metrics['mpr/selected_wm_value']:.3f} "
                    f"score={result.metrics['mpr/baseline_score']:.3f}->"
                    f"{result.metrics['mpr/selected_score']:.3f} "
                    f"time_ms={result.metrics['mpr/planning_ms']:.2f}"
                )
            if (
                self.residual_rl_enabled
                and result.stage.use_residual
                and context is not None
                and context.residual_state is not None
            ):
                self._pending_residual = {
                    "state": context.residual_state.detach().cpu().numpy(),
                    "action": result.requested_normalized_residual.detach().cpu().numpy(),
                    "valid": bool(result.residual_valid),
                }
                self.residual_rl_step += 1
        return result.action

    def update(self, buffer):
        stage = self.planner.stage_for_step(self.global_env_step)
        info = {}
        if not (self._tdmpc_frozen and stage.use_residual):
            info.update(dict(self.tdmpc_agent.update(buffer)))
            for key in ("consistency_loss", "reward_loss", "value_loss", "termination_loss"):
                if key in info:
                    info[f"wm/{key}"] = info[key]
        else:
            info["wm/frozen"] = torch.tensor(1.0, device=self.device)

        if self.planner.enabled and self.residual_rl_enabled and stage.use_residual:
            residual_info = {
                "replay_size": torch.tensor(float(len(self.residual_replay)), device=self.device),
                "local_step": torch.tensor(float(self.residual_rl_step), device=self.device),
                "update_budget": torch.tensor(float(self.residual_update_budget), device=self.device),
                "lagrangian": torch.tensor(
                    float(self.residual_sac.lagrangian_multiplier), device=self.device
                ),
            }
            if (
                self.residual_rl_step >= self.residual_learning_starts
                and len(self.residual_replay) >= self.residual_batch_size
            ):
                self.residual_update_budget += self.residual_update_per_step
                updates = 0
                while self.residual_update_budget >= 1.0:
                    batch = self.residual_replay.sample(
                        self.residual_batch_size,
                        device=self.device,
                        gamma=self.residual_gamma,
                    )
                    residual_info.update(self.residual_sac.update(batch))
                    self.residual_update_budget -= 1.0
                    updates += 1
                residual_info["updates"] = torch.tensor(float(updates), device=self.device)
                residual_info["update_budget"] = torch.tensor(
                    float(self.residual_update_budget), device=self.device
                )
            info.update({f"residual_rl/{key}": value for key, value in residual_info.items()})
            info.update(
                {
                    key: torch.tensor(value, device=self.device)
                    for key, value in self.last_residual_transition_metrics.items()
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
                metric: torch.tensor(float(last_env_info.get(source, 0.0)), device=self.device)
                for metric, source in env_metrics.items()
            }
        )
        return info

    def save(self, filepath) -> None:
        state = {
            "format": "mpr_mpc_v2_residual_sac",
            "model": self.tdmpc_agent.model.state_dict(),
            "tdmpc_optim": self.tdmpc_agent.optim.state_dict(),
            "tdmpc_pi_optim": self.tdmpc_agent.pi_optim.state_dict(),
            "tdmpc_scale": self.tdmpc_agent.scale.state_dict(),
            "global_env_step": self.global_env_step,
            "best_eval_key": self.best_eval_key,
            "mpr_mpc": {
                "residual_sac": self.residual_sac.state_dict(),
                "residual_rl_step": self.residual_rl_step,
                "residual_update_budget": self.residual_update_budget,
                "planner_calls": self._planner_calls,
                "tdmpc_frozen": self._tdmpc_frozen,
                "residual_invalid_count": self.planner.residual_invalid_count,
                "residual_attempt_count": self.planner.residual_attempt_count,
                "mppi_call_count": self.planner.mppi_call_count,
                "baseline_selected_count": self.planner.baseline_selected_count,
            },
        }
        torch.save(state, Path(filepath))

    def load(self, filepath, *, load_optimizer: bool = False) -> None:
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        if not isinstance(state, dict) or state.get("format") != "mpr_mpc_v2_residual_sac":
            raise ValueError(
                "Refusing to resume an incompatible MPR-MPC checkpoint. Start from scratch with "
                "resume_checkpoint=null or provide an mpr_mpc_v2_residual_sac checkpoint."
            )
        self.tdmpc_agent.load(state)
        mpr_state = state.get("mpr_mpc") if isinstance(state, dict) else None
        if not mpr_state:
            raise ValueError("MPR-MPC checkpoint is missing planner state.")
        self.residual_sac.load_state_dict(
            mpr_state["residual_sac"], load_optimizer=load_optimizer
        )
        if load_optimizer:
            self.tdmpc_agent.optim.load_state_dict(state["tdmpc_optim"])
            self.tdmpc_agent.pi_optim.load_state_dict(state["tdmpc_pi_optim"])
            self.tdmpc_agent.scale.load_state_dict(state["tdmpc_scale"])
        self.residual_rl_step = int(mpr_state.get("residual_rl_step", 0))
        self.residual_update_budget = float(mpr_state.get("residual_update_budget", 0.0))
        self._planner_calls = int(mpr_state.get("planner_calls", 0))
        self.global_env_step = int(state.get("global_env_step", 0))
        best_key = state.get("best_eval_key")
        self.best_eval_key = tuple(best_key) if best_key is not None else None
        self._tdmpc_frozen = bool(mpr_state.get("tdmpc_frozen", False))
        if self._tdmpc_frozen:
            self.tdmpc_agent.model.eval()
            for parameter in self.tdmpc_agent.model.parameters():
                parameter.requires_grad_(False)
        self.planner.residual_invalid_count = int(mpr_state.get("residual_invalid_count", 0))
        self.planner.residual_attempt_count = int(mpr_state.get("residual_attempt_count", 0))
        self.planner.mppi_call_count = int(mpr_state.get("mppi_call_count", 0))
        self.planner.baseline_selected_count = int(mpr_state.get("baseline_selected_count", 0))

    def eval(self):
        self.tdmpc_agent.eval()
        self.residual_sac.eval()
        return self


__all__ = ["MPRMPCAgent"]
