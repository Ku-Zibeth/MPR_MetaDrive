"""Import helpers for the repository's upstream-style TD-MPC2 layout."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
TDMPC2_ROOT = REPO_ROOT / "tdmpc2"


def bootstrap() -> None:
    """Expose the repository and upstream TD-MPC2 source roots."""
    for path in (REPO_ROOT, TDMPC2_ROOT):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


__all__ = ["PACKAGE_ROOT", "REPO_ROOT", "TDMPC2_ROOT", "bootstrap"]
