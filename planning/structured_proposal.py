"""Lattice-owned multimodal generation, filtering, and single-mode selection."""

from __future__ import annotations

from .types import StructuredSelection


class StructuredProposalGenerator:
    def __init__(self, controller):
        self.controller = controller
        self.planner = controller.planner

    def reset(self) -> None:
        self.controller.reset()

    def select_coarse(self, vehicle) -> StructuredSelection:
        candidates = list(self.planner.generate_candidates(vehicle))
        if not candidates:
            raise RuntimeError("Structured planner returned no candidates.")
        feasible = list(self.planner.filter_feasible_paths(candidates))
        pool = feasible if feasible else candidates
        pool_index = int(self.planner.select_nominal_path(pool))
        coarse = pool[pool_index]
        selected_index = next(
            index for index, candidate in enumerate(candidates) if candidate is coarse
        )
        self.planner.last_selected_index = selected_index
        return StructuredSelection(
            candidates=tuple(candidates),
            feasible=tuple(feasible),
            coarse_path=coarse,
            selected_index=selected_index,
            used_fallback=not bool(feasible),
        )

    def regenerate(self, parameters) -> tuple[object | None, str]:
        target_d, target_speed, horizon = (float(value) for value in parameters)
        try:
            path = self.planner.generate_parameterized_paths(
                [(target_d, target_speed)], horizon=horizon
            )[0]
        except (RuntimeError, ValueError, IndexError, FloatingPointError) as exc:
            return None, f"regeneration_error:{type(exc).__name__}"
        if not self.planner.filter_feasible_paths([path]):
            return None, "infeasible_refined_trajectory"
        return path, ""


__all__ = ["StructuredProposalGenerator"]
