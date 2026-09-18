"""Thin TD-MPC2 extension point used by the independent MPR-MPC package."""

from __future__ import annotations

from mpr_mpc._bootstrap import bootstrap


bootstrap()

from tdmpc2 import TDMPC2  # noqa: E402


class MPRTDMPC2(TDMPC2):
    """Keep the original learner intact and expose its model to MPR-MPC.

    Structured proposal construction and H+1 local MPPI live outside this class.
    Consequently the original action policy, TD targets, replay format, and model
    update remain byte-for-byte identical to the existing TD-MPC2 implementation.
    """


__all__ = ["MPRTDMPC2"]
