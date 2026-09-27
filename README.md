# MPR-MPC

MPR-MPC is a MetaDrive planner/training stack built on the TD-MPC2 codebase. It combines a structured Lattice proposal, Residual SAC-Lagrangian refinement, and local MPPI selection for the `metadrive-risk` task.

This README is written for deploying the current working project on a fresh Ubuntu machine with an NVIDIA CUDA GPU. The commands were checked against the local training machine on 2026-09-27.

## Quick Start

The current public `MPR_MetaDrive` repository contains the `mpr_mpc/` package, but not the whole outer TD-MPC2 workspace required by `mpr_mpc/train.py`. Start from a full TD-MPC2 workspace that also contains `env/`, `tdmpc2/`, `lattice_tdmpc2/`, and `mpr_mpc/`:

```bash
mkdir -p "$HOME/workspace"
cd "$HOME/workspace"

git clone https://github.com/nicklashansen/tdmpc2.git tdmpc2
cd tdmpc2
git checkout e9f59321933cbc8e11a002b842adc7d4ffae8ff1

git clone https://github.com/Ku-Zibeth/MPR_MetaDrive.git mpr_mpc
git clone https://github.com/Ku-Zibeth/lattice_tdmpc2.git lattice_tdmpc2
git -C lattice_tdmpc2 checkout 638fa40cffadf547d025f5aecbad5ffe97a6aad7
```

The inspected local workspace also has project-specific top-level `env/`, `lattice/`, `tdmpc2/common`, `tdmpc2/trainer`, and `tdmpc2/tdmpc2.py` changes. A plain upstream TD-MPC2 clone plus `mpr_mpc/` is therefore not guaranteed to run until those workspace files are included in the release/branch you deploy from.

Once the source tree is complete:

```bash
conda create -n tdmpc2 python=3.11 -y
conda activate tdmpc2
python -m pip install --upgrade pip

python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r mpr_mpc/requirements.txt

python mpr_mpc/scripts/check_environment.py
python -m compileall mpr_mpc

PYTHONPATH="$PWD:$PWD/tdmpc2" python -m pytest mpr_mpc/tests -q
```

Run the short smoke training before launching a 1M-step experiment:

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
python mpr_mpc/train.py \
checkpoint=null \
resume_checkpoint=null \
steps=1500 \
mpr_mpc.stages.residual_start_step=300 \
mpr_mpc.stages.mppi_start_step=700 \
residual_rl.seed_steps=100 \
residual_rl.learning_starts=100 \
milestone_checkpoints.world_model_step=300 \
milestone_checkpoints.world_model_residual_step=700 \
milestone_checkpoints.full_model_step=1500 \
enable_wandb=false \
eval_episodes=1 \
metadrive.simulator.horizon=150
```

## Project Overview

Expected working layout:

```text
tdmpc2/
├── env/                         # MetaDrive TD-MPC2 adapter used by mpr_mpc/train.py
├── lattice_tdmpc2/              # safety-cost implementation reused by MetaDrive env
├── mpr_mpc/                     # this project
│   ├── train.py
│   ├── evaluate.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── environment.yml
│   ├── environment-lock.txt
│   ├── scripts/check_environment.py
│   ├── residual_rl/
│   ├── planning/
│   ├── lattice/
│   ├── tdmpc2/                  # shim modules, not a full TD-MPC2 copy
│   └── tests/
└── tdmpc2/                      # upstream-style TD-MPC2 source package
```

`mpr_mpc/_bootstrap.py` inserts both the outer repository root and `tdmpc2/tdmpc2` package root into `sys.path`. That is why commands should be run from the outer `tdmpc2/` directory.

## Training Architecture

Current `algorithm_version`:

```yaml
algorithm_version: mpr_mpc_residual_sac_mppi_v2
```

The MetaDrive safety-cost adapter maps this MPR-MPC name to the existing v3 safety-cost implementation. This changes checkpoint metadata naming without changing the safety-cost formula.

Core components:

```text
Lattice proposal
Residual SAC-Lagrangian
Local MPPI refinement
TD-MPC2 world model and value evaluation
MetaDrive risk-field environment
```

Residual SAC uses a 527-D state by default:

```text
512 TD-MPC2 latent dims
12 coarse-path feature dims
1 world-model return dim
2 feasibility/fallback dims
```

Residual action dim is 2:

```text
delta_d = u_d * lane_width
delta_v = u_v * 0.5 * coarse_target_speed
```

Local MPPI defaults:

```yaml
mpr_mpc:
  mppi:
    num_samples: 64
    num_elites: 8
    iterations: 4
```

## Training Stages

Default total training steps: `1,000,000`.

```text
Stage A [0, 20K):
  Lattice + TD-MPC2 high-frequency training
  Residual OFF
  MPPI OFF

Stage B [20K, 50K):
  Lattice + Residual SAC-Lagrangian + TD-MPC2 high-frequency training
  MPPI OFF

Stage C [50K, 1M]:
  Lattice + Residual SAC-Lagrangian + Local MPPI
  TD-MPC2 slow online adaptation
```

TD-MPC2 update schedule:

```yaml
tdmpc_stage_training:
  stage_a_update_ratio: 1.0
  stage_b_update_ratio: 1.0
  stage_c_update_ratio: 0.25
```

Residual starts at `20K`. With the default:

```yaml
residual_rl:
  seed_steps: 5000
  learning_starts: 5000
```

the approximate behavior is:

```text
20K-25K: residual warmup / exploration data collection
after ~25K: SAC policy and critics update
```

## Tested Platform

Actual inspected machine:

```text
Ubuntu: 24.04.5 LTS (noble)
Kernel: Linux 7.0.0-34-generic x86_64
CPU: Intel Core i7-8700K, 6 cores / 12 threads
RAM: 31 GiB
Disk checked: 916G filesystem, 643G available at inspection time
Conda: 24.5.0
Training env: tdmpc2
Python in training env: 3.11.16
pip in training env: 24.0
GPU: NVIDIA GeForce RTX 3070
GPU memory: 8192 MiB
NVIDIA driver: 595.91.07
nvidia-smi reported CUDA compatibility: 13.2
PyTorch: 2.7.1+cu126
PyTorch CUDA runtime: 12.6
MetaDrive package: metadrive-simulator==0.4.3
```

This is a tested configuration, not a measured minimum hardware requirement.

## Hardware Requirements

MPR-MPC training currently requires an NVIDIA CUDA GPU. `mpr_mpc/train.py` raises an error if:

```python
torch.cuda.is_available()
```

is false.

Current training is single-GPU. Select the GPU with:

```bash
CUDA_VISIBLE_DEVICES=0
```

Use `CUDA_VISIBLE_DEVICES=1` for another GPU. Distributed multi-GPU training is not implemented here.

## NVIDIA Driver / CUDA / PyTorch

Keep these concepts separate:

```text
NVIDIA Driver:
  System driver that lets Linux talk to the GPU. Verified here: 595.91.07.

nvidia-smi CUDA Version:
  Maximum CUDA API compatibility reported by the driver. Verified here: 13.2.

PyTorch bundled CUDA runtime:
  CUDA runtime shipped with the PyTorch wheel. Verified here: torch 2.7.1+cu126, torch.version.cuda = 12.6.

System CUDA Toolkit:
  nvcc/development toolkit. It is not required just because torch.version.cuda is 12.6.
```

For this project, you need a working NVIDIA driver and a CUDA-enabled PyTorch build. A full system CUDA Toolkit is not required for the inspected training path.

Verify:

```bash
python - <<'PY'
import torch
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("device count =", torch.cuda.device_count())
PY
```

Expected:

```text
cuda available = True
```

## Conda Environment

Use the verified Python minor version:

```bash
conda create -n tdmpc2 python=3.11 -y
conda activate tdmpc2
python -m pip install --upgrade pip
```

Install PyTorch separately:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Then install the direct project dependencies:

```bash
python -m pip install -r mpr_mpc/requirements.txt
```

You can also create the environment from:

```bash
conda env create -f mpr_mpc/environment.yml
conda activate tdmpc2
```

## MetaDrive

Current MetaDrive is installed from PyPI as:

```text
metadrive-simulator==0.4.3
```

The import module is `metadrive`; there is no separate installed package named `metadrive`, and `metadrive.__version__` was not exposed in this environment.

Sanity test verified on the inspected machine:

```bash
python - <<'PY'
from metadrive import MetaDriveEnv

env = MetaDriveEnv({
    "use_render": False,
    "log_level": 50,
    "num_scenarios": 1,
    "start_seed": 0,
})
try:
    reset_result = env.reset(seed=0)
    print("reset type:", type(reset_result))
    print("reset len:", len(reset_result) if isinstance(reset_result, tuple) else "not tuple")
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    print("obs type:", type(obs))
    step_result = env.step([0.0, 0.0])
    print("step type:", type(step_result))
    print("step len:", len(step_result) if isinstance(step_result, tuple) else "not tuple")
finally:
    env.close()
print("MetaDrive sanity test passed.")
PY
```

Verified output shape:

```text
reset returns a 2-tuple
step returns a 5-tuple
```

## Linux System Packages

The inspected conda environment contains OpenGL-related conda packages such as `libgl`, `libglvnd`, `libglx`, and `libopengl`. On a fresh Ubuntu server, install these system packages if MetaDrive/Panda3D/OpenCV reports missing shared libraries:

```bash
sudo apt update
sudo apt install -y \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libglvnd0 \
    ffmpeg
```

Headless training normally uses `use_render=false`, so it should not require an interactive desktop window. Rendering and video export are more likely to need the OpenGL/ffmpeg pieces above.

## Headless / EGL

`mpr_mpc/train.py` sets:

```python
os.environ.setdefault("MUJOCO_GL", "egl")
```

For SSH/server training, it is still fine to export it explicitly:

```bash
export MUJOCO_GL=egl
```

Default training config:

```yaml
metadrive:
  simulator:
    use_render: false
```

## Verify Environment

Run:

```bash
cd "$PROJECT_ROOT"
python mpr_mpc/scripts/check_environment.py
```

The script checks Python, PyTorch, CUDA, MetaDrive, Hydra, OmegaConf, NumPy, W&B, and local MPR-MPC imports. It does not install anything.

## Compile And Unit Tests

Verified compile command:

```bash
cd "$PROJECT_ROOT"
python -m compileall mpr_mpc
```

Verified pytest command after installing `pytest==9.1.1`:

```bash
PYTHONPATH="$PWD:$PWD/tdmpc2" python -m pytest mpr_mpc/tests -q
```

The residual and planner unit tests run on CPU. They import project modules and do not start a full MetaDrive training episode.

## Smoke Training

Use this before full training:

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
python mpr_mpc/train.py \
checkpoint=null \
resume_checkpoint=null \
steps=1500 \
mpr_mpc.stages.residual_start_step=300 \
mpr_mpc.stages.mppi_start_step=700 \
residual_rl.seed_steps=100 \
residual_rl.learning_starts=100 \
milestone_checkpoints.world_model_step=300 \
milestone_checkpoints.world_model_residual_step=700 \
milestone_checkpoints.full_model_step=1500 \
enable_wandb=false \
eval_episodes=1 \
metadrive.simulator.horizon=150
```

This verifies:

```text
Stage A starts
Stage B Residual SAC starts
Stage C MPPI starts
CUDA is used by TD-MPC2
MetaDrive runs headlessly
milestone checkpoints are saved
```

The same command completed on this machine. In a fresh experiment directory it produces:

```text
logs/metadrive-risk/1/mpr_mpc/models/milestone_300_world_model.pt
logs/metadrive-risk/1/mpr_mpc/models/milestone_700_world_model_residual_rl.pt
logs/metadrive-risk/1/mpr_mpc/models/milestone_1500_full_mpr_mpc.pt
```

If a milestone file with the same name already exists, the trainer skips overwriting it. A separate `exp_name=mpr_mpc_version_check` run verified that new milestone files record:

```text
algorithm_version = mpr_mpc_residual_sac_mppi_v2
```

Peak VRAM for the current smoke run was not instrumented in-code. Use this during a smoke run if you need a measured value:

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

## Full Training

Fresh 1M-step training:

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
python mpr_mpc/train.py \
checkpoint=null \
resume_checkpoint=null \
exp_name=mpr_mpc_residual_sac_v1 \
wandb_run_name=mpr_mpc_residual_sac_v1 \
wandb_project=mpr_mpc_metadrive
```

Meaning:

```text
CUDA_VISIBLE_DEVICES=0 selects the physical GPU visible as cuda:0.
checkpoint=null is required for fresh MPR-MPC training.
resume_checkpoint=null means do not resume a previous MPR-MPC checkpoint.
```

## W&B

Default config:

```yaml
wandb_project: mpr_mpc_metadrive
wandb_entity: null
enable_wandb: true
```

`wandb_entity: null` uses the account configured by:

```bash
wandb login
```

For non-interactive servers:

```bash
export WANDB_API_KEY=...
```

Disable W&B:

```bash
enable_wandb=false
```

## Checkpoints

Regular resumable checkpoints use:

```text
format = mpr_mpc_v2_residual_sac
```

Resume:

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
python mpr_mpc/train.py \
checkpoint=null \
resume_checkpoint=/absolute/path/to/latest.pt
```

Replay buffers are not serialized:

```text
TD-MPC2 replay is rebuilt after resume.
Residual replay is rebuilt after resume.
resume is not a bit-exact continuation.
```

## Milestone Checkpoints

Milestone format:

```text
mpr_mpc_milestone_v1
```

Default milestones:

```text
20K: milestone_20k_world_model.pt
  world_model only

50K: milestone_50k_world_model_residual_rl.pt
  world_model + residual_rl
  MPPI has not participated in Stage B

1M: milestone_1m_full_mpr_mpc.pt
  world_model + residual_rl + planner/MPPI state
```

Programmatic loading:

```python
agent.load_milestone("/path/to/milestone_20k_world_model.pt")
agent.load_milestone("/path/to/milestone_50k_world_model_residual_rl.pt")
agent.load_milestone("/path/to/milestone_1m_full_mpr_mpc.pt")
```

You may request specific components:

```python
agent.load_milestone(path, components=["world_model"])
agent.load_milestone(path, components=["world_model", "residual_rl"])
agent.load_milestone(path, components=["world_model", "residual_rl", "planner"])
```

`evaluate.py` calls `agent.load(checkpoint)`, and `agent.load()` supports `mpr_mpc_milestone_v1`.

## Evaluation

Evaluate a full milestone:

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
python mpr_mpc/evaluate.py \
checkpoint=/absolute/path/to/milestone_1m_full_mpr_mpc.pt \
eval_episodes=20 \
enable_wandb=false
```

Rendered local desktop evaluation:

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
python mpr_mpc/evaluate.py \
checkpoint=/absolute/path/to/milestone_1m_full_mpr_mpc.pt \
eval_episodes=3 \
enable_wandb=false \
metadrive.simulator.use_render=true
```

Interactive rendering is intended for a machine with a display session. SSH/headless servers may need EGL/virtual display setup, or use `use_render=false`.

## Moving To Another Computer

Checkpoints should not depend on the original absolute path. Move them with `scp` or `rsync`:

```bash
scp milestone_50k_world_model_residual_rl.pt \
    user@server:/path/to/project/checkpoints/

rsync -av logs/metadrive-risk/1/mpr_mpc/models/ \
    user@server:/path/to/project/checkpoints/
```

Load using the new absolute path on the target machine:

```bash
python mpr_mpc/evaluate.py \
checkpoint=/path/to/project/checkpoints/milestone_1m_full_mpr_mpc.pt \
enable_wandb=false
```

## Hydra Overrides

Command-line overrides take priority over `mpr_mpc/config.yaml`.

Examples:

```bash
python mpr_mpc/train.py seed=2 steps=100000 enable_wandb=false

python mpr_mpc/train.py \
mpr_mpc.stages.residual_start_step=20000 \
mpr_mpc.stages.mppi_start_step=50000

python mpr_mpc/train.py mpr_mpc.mppi.num_samples=128

python mpr_mpc/train.py residual_rl.update_per_step=0.2
```

## Reproducibility

Default seed:

```yaml
seed: 1
```

The project seeds Python/NumPy/PyTorch through TD-MPC2 helpers and passes scenario seeds into MetaDrive. GPU training is not promised to be bitwise deterministic. Resumed training is also not bit-exact because replay buffers are not saved.

## Dependency Files

```text
requirements.txt:
  Direct Python dependencies used by the current MPR-MPC + MetaDrive training path.
  PyTorch is installed separately so the CUDA runtime choice is explicit.

environment.yml:
  Recommended conda environment using Python 3.11 and pip dependencies.

environment-lock.txt:
  Full package snapshot from the inspected tdmpc2 environment, including indirect deps.
  It is for forensic reproduction, not as the preferred install recipe.
```

## Troubleshooting

### `torch.cuda.is_available() == False`

Run:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Usually this means the NVIDIA driver is missing/broken, or the installed PyTorch build is CPU-only or incompatible with the driver.

### `ModuleNotFoundError: common`

Run commands from the outer `tdmpc2/` root:

```bash
cd "$PROJECT_ROOT"
PYTHONPATH="$PWD:$PWD/tdmpc2" python -m pytest mpr_mpc/tests -q
```

Also confirm `tdmpc2/common` exists. `mpr_mpc/tdmpc2/common` contains shim modules and depends on the outer TD-MPC2 source.

### `ModuleNotFoundError: env`

The top-level `env/` package is required by `mpr_mpc/train.py`. The public `MPR_MetaDrive` clone alone did not contain it during inspection.

### `ModuleNotFoundError: metadrive`

Install:

```bash
python -m pip install metadrive-simulator==0.4.3
```

### Panda3D / OpenGL / EGL Errors

For headless training, keep:

```yaml
metadrive.simulator.use_render=false
```

and:

```bash
export MUJOCO_GL=egl
```

For missing `libGL.so`/EGL libraries, install the Ubuntu packages listed above.

### W&B Login Problems

Either login:

```bash
wandb login
```

or disable logging:

```bash
enable_wandb=false
```

### CUDA OOM

First reduce the main memory drivers:

```bash
mpr_mpc.mppi.num_samples=32
batch_size=128
```

Changing these can affect experiment results.

### Checkpoint Incompatible

Supported current formats:

```text
mpr_mpc_v2_residual_sac
mpr_mpc_milestone_v1
```

Older TD-MPC2 or Lattice-only checkpoints may need explicit migration or partial component loading.
