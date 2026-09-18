"""Safety-bounded H+1 local MPPI around one refined Lattice action sequence."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematic_filter import BatchedKinematicCorridor, CorridorContext


def _action_vector(value, action_dim: int, name: str, device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(action_dim)
    if tensor.shape != (action_dim,):
        raise ValueError(f"{name} must be scalar or [{action_dim}], got {tuple(tensor.shape)}.")
    if torch.any(tensor < 0):
        raise ValueError(f"{name} values must be non-negative.")
    return tensor


@dataclass(frozen=True)
class LocalMPPIResult:
    action: torch.Tensor
    selected_sequence: torch.Tensor
    initial_mean: torch.Tensor
    final_mean: torch.Tensor
    initial_std: torch.Tensor
    final_std: torch.Tensor
    baseline_value: torch.Tensor
    final_value: torch.Tensor
    planner_gain: torch.Tensor
    elite_values: torch.Tensor
    baseline_selected: bool
    corridor_reject_count: int
    corridor_reject_rate: float
    max_lateral_deviation: float
    max_action_delta: torch.Tensor


class LocalMPPI:
    def __init__(self, evaluator, cfg, *, horizon: int, action_dim: int, device, control_dt=0.1):
        self.evaluator = evaluator
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.num_samples = int(cfg.get("num_samples", 512))
        self.num_elites = int(cfg.get("num_elites", 64))
        self.iterations = int(cfg.get("iterations", 6))
        self.min_std_vector = _action_vector(cfg.get("min_std", [0.005, 0.02]), self.action_dim, "min_std", self.device)
        self.max_std_vector = _action_vector(cfg.get("max_std", [0.05, 0.20]), self.action_dim, "max_std", self.device)
        self.initial_std_vector = _action_vector(cfg.get("initial_std", [0.025, 0.10]), self.action_dim, "initial_std", self.device)
        self.max_delta = _action_vector(cfg.get("max_delta", [0.05, 0.20]), self.action_dim, "max_delta", self.device)
        self.temperature = float(cfg.get("temperature", 0.5))
        self.proposal_init = str(cfg.get("proposal_init", "structured_baseline"))
        self.blend_alpha = float(cfg.get("blend_alpha", 0.8))
        self.deviation_coef = float(cfg.get("deviation_coef", 5.0))
        self.smoothness_coef = float(cfg.get("smoothness_coef", 1.0))
        self.throttle_deviation_weight = float(cfg.get("throttle_deviation_weight", 0.25))
        self.min_improvement_abs = float(cfg.get("min_improvement_abs", 1.0))
        self.min_improvement_ratio = float(cfg.get("min_improvement_ratio", 0.01))
        self.uncertainty_coef = float(cfg.get("uncertainty_coef", 0.0))
        self.eval_seed = int(cfg.get("eval_seed", 0))
        if self.num_samples < 2 or not 0 < self.num_elites <= self.num_samples:
            raise ValueError("Require 2 <= num_samples and 0 < num_elites <= num_samples.")
        if self.iterations <= 0:
            raise ValueError("MPPI iterations must be positive.")
        if torch.any(self.min_std_vector > self.max_std_vector):
            raise ValueError("min_std cannot exceed max_std.")
        if torch.any(self.initial_std_vector < self.min_std_vector) or torch.any(self.initial_std_vector > self.max_std_vector):
            raise ValueError("initial_std must be within [min_std,max_std].")
        if torch.any(self.max_delta <= 0):
            raise ValueError("max_delta must be positive.")
        if self.proposal_init not in {"structured_baseline", "previous_mean", "blend"}:
            raise ValueError("Unknown proposal_init mode.")
        self.corridor = BatchedKinematicCorridor(cfg, control_dt=control_dt, device=device)
        self._prev_mean: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_mean = None

    def _shift_previous(self, baseline: torch.Tensor) -> torch.Tensor:
        if self._prev_mean is None:
            return baseline.clone()
        shifted = baseline.clone()
        shifted[:-1] = self._prev_mean[1:]
        shifted[-1] = baseline[-1]
        return self._project(shifted, baseline)

    def _project(self, actions: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        delta = actions - baseline.reshape(*([1] * (actions.ndim - 2)), *baseline.shape)
        max_delta = self.max_delta.reshape(*([1] * (actions.ndim - 1)), self.action_dim)
        delta = torch.maximum(torch.minimum(delta, max_delta), -max_delta)
        return (baseline.reshape(*([1] * (actions.ndim - 2)), *baseline.shape) + delta).clamp(-1.0, 1.0)

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
        return self._project(
            self.blend_alpha * baseline + (1.0 - self.blend_alpha) * shifted,
            baseline,
        )

    def _penalties(self, actions: torch.Tensor, baseline: torch.Tensor):
        delta = actions - baseline.unsqueeze(0)
        normalized = delta / self.max_delta.view(1, 1, -1)
        deviation = normalized[..., 0].square()
        if self.action_dim > 1:
            deviation = deviation + self.throttle_deviation_weight * normalized[..., 1].square()
        deviation = deviation.mean(dim=1)
        if delta.shape[1] > 1:
            smoothness = (delta[:, 1:] - delta[:, :-1]).square().sum(dim=-1).mean(dim=1)
        else:
            smoothness = delta.new_zeros(delta.shape[0])
        return deviation, smoothness

    def _candidate_batch(self, mean, std, baseline, generator=None):
        noise = torch.randn(
            self.num_samples,
            self.horizon + 1,
            self.action_dim,
            device=self.device,
            generator=generator,
        )
        actions = self._project(mean.unsqueeze(0) + std.unsqueeze(0) * noise, baseline)
        actions[0] = baseline
        actions[1] = self._project(mean, baseline)
        return actions

    @torch.no_grad()
    def plan(
        self,
        z_t: torch.Tensor,
        baseline: torch.Tensor,
        *,
        eval_mode: bool,
        corridor_context: CorridorContext | None = None,
        task=None,
    ):
        baseline = baseline.to(self.device).clamp(-1.0, 1.0)
        mean = self.initial_mean(baseline)
        initial_mean = mean.clone()
        std = self.initial_std_vector.view(1, -1).expand_as(mean).clone()
        initial_std = std.clone()
        generator = None
        if eval_mode:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.eval_seed)

        total_rejected = 0
        total_checked = 0
        actions = elite_actions = elite_values = weights = None
        baseline_value = None
        for _ in range(self.iterations):
            actions = self._candidate_batch(mean, std, baseline, generator)
            if corridor_context is None:
                valid = torch.ones(self.num_samples, dtype=torch.bool, device=self.device)
            else:
                corridor = self.corridor.check(actions, baseline, corridor_context)
                valid = corridor.valid.clone()
            # The Lattice baseline is the certified fallback and must never disappear.
            valid[0] = True
            total_rejected += int((~valid).sum().item())
            total_checked += self.num_samples
            consequence = self.evaluator.evaluate_mppi_candidates(z_t, actions, task=task)
            wm_values = (
                consequence.total_value
                - self.uncertainty_coef * consequence.terminal_q_std
            ).nan_to_num(0.0)
            deviation, smoothness = self._penalties(actions, baseline)
            scores = wm_values - self.deviation_coef * deviation - self.smoothness_coef * smoothness
            scores = scores.masked_fill(~valid, -torch.inf)
            baseline_value = wm_values[0]
            elite_indices = torch.topk(scores, self.num_elites, dim=0).indices
            elite_values = scores[elite_indices]
            elite_actions = actions[elite_indices]
            finite = torch.isfinite(elite_values)
            logits = self.temperature * (elite_values - elite_values[finite].max())
            weights = torch.where(finite, torch.exp(logits), torch.zeros_like(logits))
            weights = weights / weights.sum().clamp_min(1e-9)
            mean = (weights.view(-1, 1, 1) * elite_actions).sum(dim=0)
            mean = self._project(mean, baseline)
            variance = (
                weights.view(-1, 1, 1) * (elite_actions - mean.unsqueeze(0)).square()
            ).sum(dim=0)
            std = torch.maximum(
                torch.minimum(variance.sqrt(), self.max_std_vector.view(1, -1)),
                self.min_std_vector.view(1, -1),
            )
        if eval_mode:
            selected_index = int(torch.argmax(elite_values).item())
        else:
            selected_index = int(torch.multinomial(weights, 1).item())
        sequence = elite_actions[selected_index]
        selected_value = elite_values[selected_index]
        planner_gain = selected_value - baseline_value
        threshold = max(
            self.min_improvement_abs,
            self.min_improvement_ratio * abs(float(baseline_value.cpu())),
        )
        baseline_selected = bool(float(planner_gain.cpu()) <= threshold)
        if baseline_selected:
            sequence = baseline
            selected_value = baseline_value
            planner_gain = selected_value - baseline_value

        self._prev_mean = mean.detach().clone()
        action_delta = (sequence - baseline).abs().amax(dim=0)
        selected_lateral = 0.0
        if corridor_context is not None:
            selected_lateral = float(
                self.corridor.check(sequence.unsqueeze(0), baseline, corridor_context)
                .max_lateral_deviation[0]
                .cpu()
            )
        return LocalMPPIResult(
            action=sequence[0].clamp(-1.0, 1.0),
            selected_sequence=sequence,
            initial_mean=initial_mean,
            final_mean=mean,
            initial_std=initial_std,
            final_std=std,
            baseline_value=baseline_value,
            final_value=selected_value,
            planner_gain=planner_gain,
            elite_values=elite_values,
            baseline_selected=baseline_selected,
            corridor_reject_count=total_rejected,
            corridor_reject_rate=float(total_rejected / max(1, total_checked)),
            max_lateral_deviation=selected_lateral,
            max_action_delta=action_delta,
        )


__all__ = ["LocalMPPI", "LocalMPPIResult", "_action_vector"]
