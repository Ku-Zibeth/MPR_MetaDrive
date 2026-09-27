# MPR-MPC

Self-contained MetaDrive training code for MPR-MPC: Lattice proposal + Residual SAC-Lagrangian + Local MPPI, using a vendored TD-MPC2 runtime.

The repository vendors the runtime components required by the tested MPR-MPC training path. No external TD-MPC2 checkout and no external `lattice_tdmpc2` checkout are required for the default MetaDrive workflow.

## Quick Start

```bash
git clone https://github.com/Ku-Zibeth/MPR_MetaDrive.git
cd MPR_MetaDrive

conda env create -f environment.yml
conda activate tdmpc2

python -m pip install --upgrade pip

python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install -r requirements.txt

python scripts/check_environment.py
python scripts/check_environment.py --full
python -m pytest tests -q
```

Start a short three-stage smoke run before a full experiment:

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
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

## Repository Layout

```text
MPR_MetaDrive/
├── train.py
├── evaluate.py
├── config.yaml
├── agent.py
├── trainer.py
├── training_schedule.py
├── _bootstrap.py
├── env/
├── lattice/
├── lattice_tdmpc2/
├── planning/
├── residual_rl/
├── tdmpc2/
├── vendor/
│   └── tdmpc2_runtime/
├── scripts/
│   └── check_environment.py
├── tests/
├── requirements.txt
├── environment.yml
├── environment-lock.txt
├── THIRD_PARTY.md
└── README.md
```

Runtime imports are repository-local:

```text
env/                         MetaDrive TD-MPC2 environment adapter
lattice/                     vendored Frenet-Lattice runtime
lattice_tdmpc2/              vendored safety-cost/runtime components
vendor/tdmpc2_runtime/       vendored TD-MPC2 runtime
tdmpc2/                      thin MPR-MPC compatibility shim
```

`scripts/check_environment.py` verifies these modules resolve inside the clone.

## Tested Platform

Verified local machine on 2026-09-27:

```text
Ubuntu: 24.04.5 LTS
Python: 3.11.16
Conda: 24.5.0
GPU: NVIDIA GeForce RTX 3070, 8192 MiB
NVIDIA driver: 595.91.07
nvidia-smi CUDA compatibility: 13.2
PyTorch: 2.7.1+cu126
PyTorch CUDA runtime: 12.6
MetaDrive: metadrive-simulator==0.4.3
```

MPR-MPC training currently requires an NVIDIA CUDA GPU. `train.py` fails fast if `torch.cuda.is_available()` is false.

## PyTorch And CUDA

Install PyTorch separately before `requirements.txt` so the CUDA build is explicit:

```bash
python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

`requirements.txt` intentionally does not pin `torch`. Run:

```bash
python -m pip check
```

after installation to confirm there are no dependency conflicts.

The system CUDA Toolkit is not required for the inspected training path. You need a working NVIDIA driver and the CUDA-enabled PyTorch wheel.

## Output Directory

Default output is always repository-relative, independent of the shell current working directory:

```text
MPR_MetaDrive/logs/<task>/<seed>/<exp_name>/
```

Default example:

```text
MPR_MetaDrive/logs/metadrive-risk/1/mpr_mpc/
├── models/
│   ├── latest.pt
│   ├── best.pt
│   ├── final.pt
│   ├── milestone_20k_world_model.pt
│   ├── milestone_50k_world_model_residual_rl.pt
│   └── milestone_1m_full_mpr_mpc.pt
├── eval.csv
├── wandb/
└── ...
```

The rule is:

```text
default log_root = <repository root>/logs
default work_dir = <log_root>/<task>/<seed>/<exp_name>
default model_dir = <work_dir>/models
```

Override the log root for a large disk:

```bash
python train.py log_root=/mnt/ssd/mpr_runs
```

Then outputs go to:

```text
/mnt/ssd/mpr_runs/metadrive-risk/1/<exp_name>/
```

Override one exact run directory:

```bash
python train.py work_dir=/mnt/ssd/mpr_runs/metadrive-risk/1/my_run
```

W&B local files use `cfg.work_dir`, so they also stay under the run directory.

## Training Architecture

Current algorithm name:

```yaml
algorithm_version: mpr_mpc_residual_sac_mppi_v2
```

Stages:

```text
Stage A [0, 20K):
  Lattice + TD-MPC2
  Residual OFF
  MPPI OFF

Stage B [20K, 50K):
  Lattice + Residual SAC + TD-MPC2
  MPPI OFF

Stage C [50K, 1M]:
  Lattice + Residual SAC + Local MPPI
  TD-MPC2 slow adaptation
```

TD-MPC2 update ratios:

```yaml
tdmpc_stage_training:
  stage_a_update_ratio: 1.0
  stage_b_update_ratio: 1.0
  stage_c_update_ratio: 0.25
```

Residual action mapping:

```text
delta_d = u_d * lane_width
delta_v = u_v * 0.5 * coarse_target_speed
```

Local MPPI defaults:

```yaml
num_samples: 64
num_elites: 8
iterations: 4
```

## Full Training

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
checkpoint=null \
resume_checkpoint=null \
exp_name=mpr_mpc_residual_sac_v1 \
wandb_run_name=mpr_mpc_residual_sac_v1 \
wandb_project=mpr_mpc_metadrive
```

Disable W&B:

```bash
python train.py enable_wandb=false
```

For headless MetaDrive training, keep:

```yaml
metadrive:
  simulator:
    use_render: false
```

`MUJOCO_GL=egl` is set by the entrypoints for vendored TD-MPC2 / dm-control / MuJoCo compatibility; MetaDrive headless training primarily depends on `use_render=false`.

## Checkpoints

Regular resumable checkpoints:

```text
logs/metadrive-risk/1/<exp_name>/models/latest.pt
logs/metadrive-risk/1/<exp_name>/models/best.pt
logs/metadrive-risk/1/<exp_name>/models/final.pt
```

Milestones:

```text
20K:
  logs/metadrive-risk/1/<exp_name>/models/milestone_20k_world_model.pt
  Evaluation stack: Lattice ON, Residual SAC OFF, Local MPPI OFF

50K:
  logs/metadrive-risk/1/<exp_name>/models/milestone_50k_world_model_residual_rl.pt
  Evaluation stack: Lattice ON, Residual SAC ON, Local MPPI OFF

1M:
  logs/metadrive-risk/1/<exp_name>/models/milestone_1m_full_mpr_mpc.pt
  Evaluation stack: Lattice ON, Residual SAC ON, Local MPPI ON
```

Resume:

```bash
python train.py \
checkpoint=null \
resume_checkpoint="$PWD/logs/metadrive-risk/1/<exp_name>/models/latest.pt"
```

Replay buffers are not serialized, so resume is not bit-exact continuation.

## Evaluation

By default `evaluation_stack=checkpoint` infers the runtime stack from the loaded milestone:

```text
world_model               -> Lattice ON, Residual SAC OFF, Local MPPI OFF
world_model_residual_rl   -> Lattice ON, Residual SAC ON,  Local MPPI OFF
full_mpr_mpc              -> Lattice ON, Residual SAC ON,  Local MPPI ON
regular latest/best/final -> normal stage-based behavior
```

You can override the stack explicitly with `evaluation_stack=world_model`,
`evaluation_stack=world_model_residual_rl`, or `evaluation_stack=full_mpr_mpc`.

```bash
python evaluate.py \
checkpoint="$PWD/logs/metadrive-risk/1/<exp_name>/models/milestone_1m_full_mpr_mpc.pt" \
eval_episodes=20 \
enable_wandb=false
```

Rendered local evaluation:

```bash
python evaluate.py \
checkpoint="$PWD/logs/metadrive-risk/1/<exp_name>/models/milestone_1m_full_mpr_mpc.pt" \
eval_episodes=3 \
enable_wandb=false \
metadrive.simulator.use_render=true
```

Interactive rendering expects a desktop/display-capable environment.

## Environment Check

```bash
python scripts/check_environment.py
python scripts/check_environment.py --full
```

The script checks:

```text
Python 3.11
PyTorch 2.7.1 / CUDA runtime 12.6
CUDA availability and GPU name
metadrive-simulator 0.4.3
gymnasium 0.29.1
hydra-core 1.3.2
omegaconf 2.3.0
tensordict 0.8.3
torchrl 0.8.1
repository-local runtime sources
default log root
algorithm_version
```

`--full` additionally creates the configured MetaDrive environment, runs a reset plus a few random steps, and verifies `reward`, `cost`, and `risk_field_cost`.

Version drift is reported as `[WARN]`; missing CUDA or external runtime source resolution is `[FAIL]`.

## Tests

```bash
python -m compileall .
python -m pytest tests -q
python -m pip check
```

Project MetaDrive sanity check:

```bash
python - <<'PY'
from omegaconf import OmegaConf
from _bootstrap import LOG_ROOT, bootstrap

bootstrap()

from common.parser import parse_cfg
from env import make_env

cfg = OmegaConf.load("config.yaml")
cfg.log_root = str(LOG_ROOT)
cfg.enable_wandb = False
cfg.metadrive.simulator.horizon = 5

env = make_env(parse_cfg(cfg))
try:
    obs = env.reset()
    for _ in range(3):
        obs, reward, done, info = env.step(env.rand_act())
    print("cost:", info.get("cost"))
    print("risk_field_cost:", info.get("risk_field_cost"))
finally:
    env.close()
PY
```

## Vendored Runtime

See [THIRD_PARTY.md](THIRD_PARTY.md).

Summary:

```text
vendor/tdmpc2_runtime/
  Source: https://github.com/nicklashansen/tdmpc2
  Base commit: e9f59321933cbc8e11a002b842adc7d4ffae8ff1
  Includes local runtime modifications required by MPR-MPC.

lattice_tdmpc2/
  Source: https://github.com/Ku-Zibeth/lattice_tdmpc2
  Commit: 638fa40cffadf547d025f5aecbad5ffe97a6aad7

lattice/
  Project-specific Frenet-Lattice runtime snapshot used by MPR-MPC.
```

Vendoring is used to make the research code reproducible and to avoid upstream API drift or missing local runtime changes.

## Troubleshooting

`torch.cuda.is_available() == False`:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`ModuleNotFoundError: common`, `env`, or `lattice`:

```bash
python scripts/check_environment.py
```

All runtime modules should resolve inside the `MPR_MetaDrive` clone.

Panda3D/OpenGL issues:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 libegl1 libglvnd0 ffmpeg
```

W&B login issues:

```bash
wandb login
```

or:

```bash
python train.py enable_wandb=false
```
