"""Lattice, TD-MPC2 and continuous residual SAC integration."""

import sys
from pathlib import Path


_UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "tdmpc2"
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

from .action_adapter import LatticeActionAdapter
from .evaluator import WorldModelEvaluator
from .planner import ResidualPlanResult, ResidualSACLatticePlanner
from .residual_action import ResidualActionAdapter
from .state_builder import ResidualStateBuilder

__all__ = [
    "LatticeActionAdapter",
    "ResidualPlanResult",
    "ResidualActionAdapter",
    "ResidualSACLatticePlanner",
    "ResidualStateBuilder",
    "WorldModelEvaluator",
]
