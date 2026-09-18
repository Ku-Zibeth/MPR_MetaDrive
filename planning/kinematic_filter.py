"""Batched short-horizon geometry filter for local MPPI candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CorridorContext:
    refined_lateral: torch.Tensor
    lane_width: float
    road_min: float
    road_max: float
    initial_speed: float


@dataclass(frozen=True)
class CorridorResult:
    valid: torch.Tensor
    max_lateral_deviation: torch.Tensor
    lateral: torch.Tensor


class BatchedKinematicCorridor:
    """Use a bicycle approximation only to reject geometrically unsafe actions."""

    def __init__(self, cfg, *, control_dt: float, device):
        self.device = torch.device(device)
        self.dt = float(control_dt)
        self.ratio = float(cfg.get("lateral_corridor_ratio", 0.5))
        self.safety_margin = float(cfg.get("lateral_safety_margin", 0.15))
        self.wheelbase = float(cfg.get("kinematic_wheelbase", 2.8))
        self.max_steering_angle = float(cfg.get("kinematic_max_steering_angle", 0.6))
        self.max_acceleration = float(cfg.get("kinematic_max_acceleration", 4.0))
        self.max_deceleration = float(cfg.get("kinematic_max_deceleration", 6.0))
        if min(self.dt, self.ratio, self.wheelbase, self.max_steering_angle) <= 0:
            raise ValueError("Kinematic corridor positive parameters must be > 0.")

    def rollout(self, actions: torch.Tensor, initial_speed: float) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError("actions must have shape [N,H+1,2].")
        count, steps, _ = actions.shape
        speed = actions.new_full((count,), max(0.0, float(initial_speed)))
        yaw = actions.new_zeros(count)
        lateral_position = actions.new_zeros(count)
        traces = []
        for step in range(steps):
            steering = actions[:, step, 0] * self.max_steering_angle
            throttle = actions[:, step, 1]
            acceleration = torch.where(
                throttle >= 0.0,
                throttle * self.max_acceleration,
                throttle * self.max_deceleration,
            )
            yaw = yaw + self.dt * speed / self.wheelbase * torch.tan(steering)
            lateral_position = lateral_position + self.dt * speed * torch.sin(yaw)
            speed = (speed + self.dt * acceleration).clamp_min(0.0)
            traces.append(lateral_position)
        return torch.stack(traces, dim=1)

    def check(
        self,
        actions: torch.Tensor,
        baseline: torch.Tensor,
        context: CorridorContext,
    ) -> CorridorResult:
        actions = actions.to(self.device)
        baseline = baseline.to(self.device)
        refined_d = context.refined_lateral.to(self.device, dtype=actions.dtype)
        if refined_d.shape != (actions.shape[1],):
            raise ValueError("refined_lateral must contain one value per action step.")
        candidate_y = self.rollout(actions, context.initial_speed)
        baseline_y = self.rollout(baseline.unsqueeze(0), context.initial_speed)[0]
        lateral_deviation = candidate_y - baseline_y.unsqueeze(0)
        predicted_d = refined_d.unsqueeze(0) + lateral_deviation

        corridor_radius = self.ratio * float(context.lane_width)
        lower = torch.maximum(
            refined_d - corridor_radius,
            refined_d.new_full(refined_d.shape, float(context.road_min) + self.safety_margin),
        )
        upper = torch.minimum(
            refined_d + corridor_radius,
            refined_d.new_full(refined_d.shape, float(context.road_max) - self.safety_margin),
        )
        valid = ((predicted_d >= lower) & (predicted_d <= upper)).all(dim=1)
        return CorridorResult(
            valid=valid,
            max_lateral_deviation=lateral_deviation.abs().amax(dim=1),
            lateral=predicted_d,
        )


__all__ = ["BatchedKinematicCorridor", "CorridorContext", "CorridorResult"]
