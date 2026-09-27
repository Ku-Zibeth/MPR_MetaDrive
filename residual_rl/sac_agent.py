"""Residual SAC-Lagrangian agent for normalized ``[delta_d, delta_v]`` actions."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .lagrangian import ProjectedLagrangian
from .networks import Critic, GaussianActor
from .types import ResidualPolicyOutput


class ResidualSACAgent:
    action_dim = 2

    def __init__(
        self,
        state_dim: int,
        cfg: dict,
        *,
        device: str | torch.device,
    ):
        self.state_dim = int(state_dim)
        self.device = torch.device(device)
        hidden_dims: Sequence[int] = cfg.get("hidden_dims", [256, 256])
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.auto_alpha = bool(cfg.get("auto_alpha", True))
        self.target_entropy = float(cfg.get("target_entropy", -self.action_dim))
        self.cost_limit = float(cfg.get("cost_limit", 5.0))
        self.lagrangian_rescaling = bool(cfg.get("lagrangian_rescaling", True))

        self.actor = GaussianActor(
            self.state_dim,
            self.action_dim,
            hidden_dims,
            log_std_min=float(cfg.get("log_std_min", -20.0)),
            log_std_max=float(cfg.get("log_std_max", 2.0)),
        ).to(self.device)
        self.q1 = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.q2 = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_q1 = deepcopy(self.q1).to(self.device).requires_grad_(False)
        self.target_q2 = deepcopy(self.q2).to(self.device).requires_grad_(False)
        self.cost_q1 = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.cost_q2 = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_cost_q1 = deepcopy(self.cost_q1).to(self.device).requires_grad_(False)
        self.target_cost_q2 = deepcopy(self.cost_q2).to(self.device).requires_grad_(False)

        actor_lr = float(cfg.get("actor_lr", cfg.get("lr", 3e-4)))
        critic_lr = float(cfg.get("critic_lr", cfg.get("lr", 3e-4)))
        alpha_lr = float(cfg.get("alpha_lr", cfg.get("lr", 3e-4)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.cost_critic_optimizer = torch.optim.Adam(
            list(self.cost_q1.parameters()) + list(self.cost_q2.parameters()), lr=critic_lr
        )
        alpha = max(float(cfg.get("alpha", 0.2)), 1e-8)
        self.log_alpha = torch.tensor(
            np.log(alpha),
            device=self.device,
            dtype=torch.float32,
            requires_grad=self.auto_alpha,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=alpha_lr) if self.auto_alpha else None
        )
        self.lagrangian = ProjectedLagrangian(
            learning_rate=float(cfg.get("lagrangian_lr", 0.01)),
            initial_value=float(cfg.get("lagrangian_initial", 0.0)),
        )
        self.update_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lagrangian_multiplier(self) -> float:
        return self.lagrangian.value

    @torch.no_grad()
    def select_action(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> ResidualPolicyOutput:
        state = state.to(self.device, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Residual state dim {state.shape[-1]} != {self.state_dim}.")
        action, _, mean, log_std = self.actor(state, deterministic=deterministic)
        return ResidualPolicyOutput(
            action=action.clamp(-1.0, 1.0).detach(),
            mean=torch.tanh(mean).detach(),
            log_std=log_std.detach(),
            warmup=False,
        )

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        cost = batch["cost"]
        next_state = batch["next_state"]
        done = batch["done"]
        discount = batch["discount"]

        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor(next_state)
            next_q = torch.minimum(
                self.target_q1(next_state, next_action),
                self.target_q2(next_state, next_action),
            )
            reward_target = reward + discount * (1.0 - done) * (
                next_q - self.alpha.detach() * next_log_prob
            )
            next_cost_q = 0.5 * (
                self.target_cost_q1(next_state, next_action)
                + self.target_cost_q2(next_state, next_action)
            )
            cost_target = cost + discount * (1.0 - done) * next_cost_q

        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        critic_loss = (q1 - reward_target).square().mean() + (q2 - reward_target).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        cost_q1 = self.cost_q1(state, action)
        cost_q2 = self.cost_q2(state, action)
        cost_critic_loss = (
            F.smooth_l1_loss(cost_q1, cost_target)
            + F.smooth_l1_loss(cost_q2, cost_target)
        )
        self.cost_critic_optimizer.zero_grad(set_to_none=True)
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        critic_parameters = [
            parameter
            for critic in (self.q1, self.q2, self.cost_q1, self.cost_q2)
            for parameter in critic.parameters()
        ]
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            policy_action, log_prob, _, _ = self.actor(state)
            policy_q = torch.minimum(
                self.q1(state, policy_action),
                self.q2(state, policy_action),
            )
            actor_sac_loss = (self.alpha.detach() * log_prob - policy_q).mean()
            policy_cost = 0.5 * (
                self.cost_q1(state, policy_action)
                + self.cost_q2(state, policy_action)
            ).mean()
            lagrangian = self.lagrangian_multiplier
            actor_safety_loss = lagrangian * policy_cost
            rescaling = 1.0 / (1.0 + lagrangian) if self.lagrangian_rescaling else 1.0
            actor_loss = rescaling * (actor_sac_loss + actor_safety_loss)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)

        alpha_loss = torch.zeros((), device=self.device)
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            self._soft_update(self.target_q1, self.q1)
            self._soft_update(self.target_q2, self.q2)
            self._soft_update(self.target_cost_q1, self.cost_q1)
            self._soft_update(self.target_cost_q2, self.cost_q2)
        self.update_steps += 1
        return {
            "loss/critic": critic_loss.detach(),
            "loss/cost_critic": cost_critic_loss.detach(),
            "loss/actor": actor_loss.detach(),
            "loss/actor_sac": actor_sac_loss.detach(),
            "loss/actor_safety": actor_safety_loss.detach(),
            "loss/alpha": alpha_loss.detach(),
            "train/alpha": self.alpha.detach(),
            "train/entropy": (-log_prob).detach().mean(),
            "train/q1": q1.detach().mean(),
            "train/q2": q2.detach().mean(),
            "train/target_q": reward_target.detach().mean(),
            "train/cost_q": (0.5 * (cost_q1 + cost_q2)).detach().mean(),
            "train/cost_target_q": cost_target.detach().mean(),
            "lagrangian/value": torch.tensor(float(lagrangian), device=self.device),
            "lagrangian/cost_limit": torch.tensor(self.cost_limit, device=self.device),
            "lagrangian/lr": torch.tensor(self.lagrangian.learning_rate, device=self.device),
            "lagrangian/rescaling": torch.tensor(float(rescaling), device=self.device),
        }

    def update_lagrangian(self, episode_cost: float) -> float:
        return self.lagrangian.step(episode_cost, self.cost_limit)

    def _soft_update(self, target, source) -> None:
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.lerp_(parameter, self.tau)

    def train(self) -> None:
        self.actor.train()
        self.q1.train()
        self.q2.train()
        self.cost_q1.train()
        self.cost_q2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.cost_q1.eval()
        self.cost_q2.eval()

    def state_dict(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "cost_q1": self.cost_q1.state_dict(),
            "cost_q2": self.cost_q2.state_dict(),
            "target_cost_q1": self.target_cost_q1.state_dict(),
            "target_cost_q2": self.target_cost_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "cost_critic_optimizer": self.cost_critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "lagrangian": self.lagrangian.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state: dict, *, load_optimizer: bool = False) -> None:
        if int(state["state_dim"]) != self.state_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("Residual SAC checkpoint dimensions do not match this run.")
        for name in (
            "actor",
            "q1",
            "q2",
            "target_q1",
            "target_q2",
            "cost_q1",
            "cost_q2",
            "target_cost_q1",
            "target_cost_q2",
        ):
            getattr(self, name).load_state_dict(state[name])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.lagrangian.load_state_dict(state["lagrangian"])
        self.update_steps = int(state.get("update_steps", 0))
        if load_optimizer:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.cost_critic_optimizer.load_state_dict(state["cost_critic_optimizer"])
            if self.alpha_optimizer is not None and state.get("alpha_optimizer") is not None:
                self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
