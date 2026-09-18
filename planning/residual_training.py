"""World-model-guided targets and a replay buffer for residual supervision."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ResidualExample:
    latent: torch.Tensor
    coarse_parameters: torch.Tensor
    wm_features: torch.Tensor
    residual_bounds: torch.Tensor
    target_residual: torch.Tensor


class ResidualSupervisionBuffer:
    """Separate from TD-MPC2 replay; stores no environment transitions."""

    def __init__(self, capacity: int):
        self._items: deque[ResidualExample] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._items)

    def add(self, context, target_residual: torch.Tensor) -> None:
        self._items.append(
            ResidualExample(
                latent=context.latent.squeeze(0).detach().cpu(),
                coarse_parameters=context.coarse_parameters.squeeze(0).detach().cpu(),
                wm_features=context.wm_features.squeeze(0).detach().cpu(),
                residual_bounds=context.residual_bounds.squeeze(0).detach().cpu(),
                target_residual=target_residual.reshape(3).detach().cpu(),
            )
        )

    def sample(self, batch_size: int, device) -> ResidualExample:
        if len(self) < int(batch_size):
            raise RuntimeError("Residual supervision buffer does not contain a full batch.")
        indices = np.random.choice(len(self), size=int(batch_size), replace=False)
        examples = [self._items[int(index)] for index in indices]
        stack = lambda name: torch.stack([getattr(item, name) for item in examples]).to(device)
        return ResidualExample(
            latent=stack("latent"),
            coarse_parameters=stack("coarse_parameters"),
            wm_features=stack("wm_features"),
            residual_bounds=stack("residual_bounds"),
            target_residual=stack("target_residual"),
        )


class ResidualTargetGenerator:
    """Search bounded corrections around the one planner-selected coarse path."""

    def __init__(
        self,
        structured,
        action_adapter,
        evaluator,
        *,
        horizon: int,
        control_dt: float,
        sample_count: int = 32,
        elite_count: int = 4,
        target_mode: str = "best",
        temperature: float = 1.0,
        parameter_clamper=None,
    ):
        self.structured = structured
        self.action_adapter = action_adapter
        self.evaluator = evaluator
        self.horizon = int(horizon)
        self.control_dt = float(control_dt)
        self.sample_count = int(sample_count)
        self.elite_count = int(elite_count)
        self.target_mode = str(target_mode)
        self.temperature = float(temperature)
        self.parameter_clamper = parameter_clamper
        if self.sample_count < 1 or not 1 <= self.elite_count <= self.sample_count:
            raise ValueError("Require 1 <= elite_count <= sample_count.")
        if self.target_mode not in {"best", "softmax_elite"}:
            raise ValueError("target_mode must be best or softmax_elite.")

    @torch.no_grad()
    def generate(self, context, vehicle) -> tuple[torch.Tensor, dict[str, float]]:
        coarse = context.selection.coarse_path
        coarse_parameters = np.asarray(
            [coarse.target_d, coarse.target_speed, coarse.horizon], dtype=np.float32
        )
        delta_bounds = context.residual_bounds.squeeze(0).detach().cpu().numpy()
        samples = np.random.uniform(
            -delta_bounds,
            delta_bounds,
            size=(self.sample_count, 3),
        ).astype(np.float32)
        samples[0] = 0.0
        paths = []
        residuals = []
        for index, residual in enumerate(samples):
            parameters = coarse_parameters + residual
            if self.parameter_clamper is not None:
                parameters = self.parameter_clamper(parameters)
            actual_residual = np.asarray(parameters, dtype=np.float32) - coarse_parameters
            if index == 0 or np.all(np.abs(actual_residual) < 1e-8):
                path = coarse
            else:
                path, _ = self.structured.regenerate(parameters)
            if path is not None:
                paths.append(path)
                residuals.append(actual_residual)
        if not paths:
            return torch.zeros(3, device=context.latent.device), {
                "target_candidates": 0.0,
                "target_best_value": float(context.coarse_consequence.total_value[0]),
            }
        actions = self.action_adapter.paths_to_actions(
            paths,
            vehicle,
            required_steps=self.horizon + 1,
            control_dt=self.control_dt,
            device=context.latent.device,
        )
        consequence = self.evaluator.evaluate_trajectory_consequence(context.latent, actions)
        values = consequence.total_value
        residual_tensor = torch.as_tensor(
            np.asarray(residuals), device=values.device, dtype=values.dtype
        )
        elite_count = min(self.elite_count, len(paths))
        elite_indices = torch.topk(values, elite_count).indices
        if self.target_mode == "best":
            target = residual_tensor[elite_indices[0]]
        else:
            elite_values = values[elite_indices]
            weights = torch.softmax(self.temperature * (elite_values - elite_values.max()), dim=0)
            target = (weights.unsqueeze(-1) * residual_tensor[elite_indices]).sum(dim=0)
        return target.detach(), {
            "target_candidates": float(len(paths)),
            "target_best_value": float(values.max().cpu()),
            "target_zero_value": float(values[0].cpu()),
            "target_improvement": float((values.max() - values[0]).cpu()),
        }


class ResidualLearner:
    def __init__(self, planner, cfg):
        self.planner = planner
        self.batch_size = int(cfg.get("batch_size", 64))
        self.reg_coef = float(cfg.get("residual_reg_coef", 0.01))
        self.grad_clip = float(cfg.get("grad_clip_norm", 10.0))
        self.buffer = ResidualSupervisionBuffer(int(cfg.get("buffer_size", 100000)))
        parameters = list(planner.residual_prior.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=float(cfg.get("lr", 3e-4)))
        self.update_steps = 0

    def add(self, context, target_residual: torch.Tensor) -> None:
        self.buffer.add(context, target_residual)

    def update(self) -> dict[str, torch.Tensor]:
        if len(self.buffer) < self.batch_size:
            return {"residual_buffer_size": torch.tensor(float(len(self.buffer)))}
        batch = self.buffer.sample(self.batch_size, self.planner.device)
        residual_input = torch.cat(
            [batch.latent, batch.coarse_parameters, batch.wm_features], dim=-1
        )
        loss, info = self.planner.residual_prior.supervised_loss(
            residual_input,
            batch.target_residual,
            delta_bounds=batch.residual_bounds,
            residual_reg_coef=self.reg_coef,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.planner.residual_prior.parameters(),
            self.grad_clip,
        )
        self.optimizer.step()
        self.update_steps += 1
        return {
            "residual_loss": loss.detach(),
            "residual_grad_norm": torch.as_tensor(grad_norm).detach(),
            "residual_nll": info["residual_nll"],
            "residual_reg": info["residual_reg"],
            "residual_buffer_size": torch.tensor(float(len(self.buffer))),
        }


__all__ = [
    "ResidualExample",
    "ResidualLearner",
    "ResidualSupervisionBuffer",
    "ResidualTargetGenerator",
]
