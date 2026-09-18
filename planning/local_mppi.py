"""H+1 local MPPI centered on one refined structured action sequence."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalMPPIResult:
    action: torch.Tensor
    selected_sequence: torch.Tensor
    initial_mean: torch.Tensor
    final_mean: torch.Tensor
    initial_std: torch.Tensor
    final_std: torch.Tensor
    final_value: torch.Tensor
    elite_values: torch.Tensor


class LocalMPPI:
    def __init__(self, evaluator, cfg, *, horizon: int, action_dim: int, device):
        self.evaluator = evaluator
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.num_samples = int(cfg.get("num_samples", 512))
        self.num_elites = int(cfg.get("num_elites", 64))
        self.iterations = int(cfg.get("iterations", 6))
        self.min_std = float(cfg.get("min_std", 0.05))
        self.max_std = float(cfg.get("max_std", 0.5))
        self.initial_std_value = float(cfg.get("initial_std", self.max_std))
        self.temperature = float(cfg.get("temperature", 0.5))
        self.proposal_init = str(cfg.get("proposal_init", "structured_baseline"))
        self.blend_alpha = float(cfg.get("blend_alpha", 0.8))
        if self.num_samples <= 0 or not 0 < self.num_elites <= self.num_samples:
            raise ValueError("Require 0 < num_elites <= num_samples.")
        if self.iterations <= 0:
            raise ValueError("MPPI iterations must be positive.")
        if self.proposal_init not in {"structured_baseline", "previous_mean", "blend"}:
            raise ValueError("Unknown proposal_init mode.")
        self._prev_mean: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_mean = None

    def _shift_previous(self, baseline: torch.Tensor) -> torch.Tensor:
        if self._prev_mean is None:
            return baseline.clone()
        shifted = baseline.clone()
        shifted[:-1] = self._prev_mean[1:]
        shifted[-1] = baseline[-1]
        return shifted

    def initial_mean(self, baseline: torch.Tensor) -> torch.Tensor:
        expected = (self.horizon + 1, self.action_dim)
        if baseline.shape != expected:
            raise ValueError(f"Baseline shape {tuple(baseline.shape)} != {expected}.")
        baseline = baseline.to(self.device).clamp(-1.0, 1.0)
        if self.proposal_init == "structured_baseline":
            return baseline.clone()
        shifted = self._shift_previous(baseline)
        if self.proposal_init == "previous_mean":
            return shifted
        return self.blend_alpha * baseline + (1.0 - self.blend_alpha) * shifted

    @torch.no_grad()
    def plan(self, z_t: torch.Tensor, baseline: torch.Tensor, *, eval_mode: bool, task=None):
        mean = self.initial_mean(baseline)
        initial_mean = mean.clone()
        std = torch.full_like(mean, self.initial_std_value).clamp(self.min_std, self.max_std)
        initial_std = std.clone()
        elite_actions = None
        elite_values = None
        weights = None
        for _ in range(self.iterations):
            noise = torch.randn(
                self.horizon + 1,
                self.num_samples,
                self.action_dim,
                device=self.device,
            )
            actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1.0, 1.0)
            actions[:, 0] = mean
            consequence = self.evaluator.evaluate_mppi_candidates(
                z_t, actions.permute(1, 0, 2).contiguous(), task=task
            )
            values = consequence.total_value.nan_to_num(0.0)
            elite_indices = torch.topk(values, self.num_elites, dim=0).indices
            elite_values = values[elite_indices]
            elite_actions = actions[:, elite_indices]
            maximum = elite_values.max()
            weights = torch.exp(self.temperature * (elite_values - maximum))
            weights = weights / weights.sum().clamp_min(1e-9)
            mean = (weights.view(1, -1, 1) * elite_actions).sum(dim=1)
            variance = (
                weights.view(1, -1, 1) * (elite_actions - mean.unsqueeze(1)).square()
            ).sum(dim=1)
            std = variance.sqrt().clamp(self.min_std, self.max_std)

        selected = int(torch.multinomial(weights, 1).item())
        sequence = elite_actions[:, selected]
        action = sequence[0]
        if not eval_mode:
            action = action + std[0] * torch.randn_like(action)
        self._prev_mean = mean.detach().clone()
        return LocalMPPIResult(
            action=action.clamp(-1.0, 1.0),
            selected_sequence=sequence,
            initial_mean=initial_mean,
            final_mean=mean,
            initial_std=initial_std,
            final_std=std,
            final_value=elite_values[selected],
            elite_values=elite_values,
        )


__all__ = ["LocalMPPI", "LocalMPPIResult"]
