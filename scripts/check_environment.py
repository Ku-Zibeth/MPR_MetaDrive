#!/usr/bin/env python
"""Check the runtime environment required by MPR-MPC."""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata
from pathlib import Path


SCRIPT = Path(__file__).resolve()
MPR_ROOT = SCRIPT.parents[1]
REPO_ROOT = MPR_ROOT.parent
TDMPC2_ROOT = REPO_ROOT / "tdmpc2"

for path in (REPO_ROOT, TDMPC2_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def ok(name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[OK] {name}{suffix}")


def fail(name: str, detail: str) -> None:
    print(f"[FAIL] {name} - {detail}")
    raise SystemExit(1)


def dist_version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError as exc:
        fail(dist, "package is not installed")
        raise AssertionError from exc


def import_module(module: str, dist: str | None = None) -> object:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report exact import failure to user
        fail(module, repr(exc))
    version = dist_version(dist) if dist else getattr(mod, "__version__", "imported")
    ok(module, str(version))
    return mod


def main() -> None:
    version = sys.version_info
    if version.major != 3 or version.minor != 11:
        fail("Python", f"expected 3.11.x, got {platform.python_version()}")
    ok("Python", platform.python_version())

    torch = import_module("torch", "torch")
    ok("PyTorch CUDA runtime", str(torch.version.cuda))
    if not torch.cuda.is_available():
        fail("CUDA", "torch.cuda.is_available() is False")
    ok("CUDA", torch.cuda.get_device_name(0))
    ok("CUDA device count", str(torch.cuda.device_count()))

    import_module("numpy", "numpy")
    import_module("hydra", "hydra-core")
    import_module("omegaconf", "omegaconf")
    import_module("wandb", "wandb")
    import_module("gymnasium", "gymnasium")
    import_module("metadrive", "metadrive-simulator")
    import_module("tensordict", "tensordict")
    import_module("torchrl", "torchrl")
    import_module("pandas", "pandas")
    import_module("scipy", "scipy")

    import_module("env")
    import_module("common.buffer")
    import_module("mpr_mpc")
    import_module("mpr_mpc.agent")
    import_module("mpr_mpc.planning.coordinator")
    import_module("mpr_mpc.residual_rl")

    config_path = MPR_ROOT / "config.yaml"
    if not config_path.is_file():
        fail("MPR-MPC config", f"missing {config_path}")
    text = config_path.read_text(encoding="utf-8")
    expected = "algorithm_version: mpr_mpc_residual_sac_mppi_v2"
    if expected not in text:
        fail("algorithm_version", f"expected line not found in {config_path}")
    ok("algorithm_version", "mpr_mpc_residual_sac_mppi_v2")

    print("Environment check passed.")


if __name__ == "__main__":
    main()
