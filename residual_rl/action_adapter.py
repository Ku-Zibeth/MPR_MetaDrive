"""Map normalized SAC actions to MPR-MPC physical residuals."""

from __future__ import annotations

import torch


class ResidualActionAdapter:
    """Linear ``[-1, 1]^2`` to ``[delta_d, delta_v]`` adapter."""

    action_dim = 2

    @staticmethod
    def to_physical(normalized_action: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(
            normalized_action,
            dtype=bounds.dtype,
            device=bounds.device,
        )
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != 2:
            raise ValueError(f"Residual SAC action must end in dim 2, got {tuple(action.shape)}.")
        if bounds.ndim == 1:
            bounds = bounds.unsqueeze(0)
        if bounds.shape[-1] != 2 or bounds.shape[0] not in {1, action.shape[0]}:
            raise ValueError("Residual bounds must be [2] or [B,2].")
        if not torch.isfinite(bounds).all() or torch.any(bounds < 0):
            raise ValueError("Residual bounds must be finite and non-negative.")
        return action.clamp(-1.0, 1.0) * bounds.expand(action.shape[0], -1)
