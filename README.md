# MPR-MPC

The online controller is deliberately single-mode:

```text
MetaDrive state
  -> Lattice candidates / hard feasibility / select one coarse path
  -> residual policy [delta_d, delta_v] (Stage C only)
  -> regenerate and recheck the refined Frenet path; T stays unchanged
  -> H+1 refined baseline controls
  -> bounded local MPPI (Stages B/C)
  -> execute selected_sequence[0]
```

TD-MPC2 is initialized from scratch. `checkpoint` must stay `null`; only an
`mpr_mpc_v1` checkpoint produced by this package may be supplied through
`resume_checkpoint`.

## Staged training

- Stage A, steps `[0, 20000)`: Lattice action, Residual OFF, MPPI OFF.
- Stage B, steps `[20000, 50000)`: Lattice + bounded Local MPPI, Residual OFF.
- Stage C, steps `[50000, ...)`: Lattice + deterministic Residual + Local MPPI.

Collection is Lattice-driven from step zero. There is no random action seed and
no concentrated seed-data pretraining. Once replay contains enough transitions
for a complete batch, TD-MPC2 receives one online update per environment step.
Stage A also skips the planning-time encoder/WM rollout; replay training of the
world model continues normally.

## Tensor flow

```text
observation                                      [B, obs_dim]
current encoder latent                           [B, 512]
normalized coarse [target_d,target_speed,T]       [B, 3]
current WM [sum(r_0..r_9), Q_H, J]                [B, 3]
residual input                                    [B, 518]
raw mean/log_std                                  [B, 2] + [B, 2]
bounded deterministic residual [delta_d,delta_v]  [B, 2]
coarse/refined controls                           [B, 11, 2]
```

Residual supervision stores raw observation, coarse parameters, coarse H+1
controls, bounds and target. It recomputes latent and WM consequence features at
update time, so replay does not retain stale encoder or world-model outputs.

Local MPPI uses vector standard deviations, a hard action trust region relative
to the original refined baseline, a batched bicycle-model corridor check, and
deviation/smoothness penalties. Candidate zero is always the baseline and
candidate one is the current mean. Evaluation samples reproducibly and selects
the highest-scoring real elite. No noise is added after selection.
The corridor integrates only the first H controls into states `t+1...t+H`; the
H+1-th control is reserved for terminal Q. Vehicle width, wheelbase and maximum
steering are read from the live MetaDrive vehicle, with config values as fallback.

## Fresh training

```bash
cd /home/kzb/kzb_code/tdmpc2
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
  python mpr_mpc/train.py \
  checkpoint=null \
  resume_checkpoint=null \
  exp_name=mpr_mpc_v1 \
  wandb_run_name=mpr_mpc_v1 \
  wandb_project=mpr_mpc_metadrive
```

Resume only from a checkpoint created by the command above:

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
  python mpr_mpc/train.py \
  checkpoint=null \
  resume_checkpoint=/absolute/path/to/latest.pt \
  exp_name=mpr_mpc_v1_resume \
  wandb_run_name=mpr_mpc_v1_resume
```

`latest.pt` is overwritten at every evaluation and every 50k steps. `best.pt`
uses the lexicographic key `(success higher, offroad lower, reward higher)`;
`final.pt` is written on normal completion. Checkpoints include the TD-MPC2
model and both optimizers, running value scale, residual model/optimizer,
global environment step and planner counters.

Replay contents and an unfinished episode are intentionally not serialized.
After resume, model/optimizer/planner state is restored, while TD replay and the
Residual supervision buffer rebuild from new Stage-C planner data. Updates stay
disabled until `train/replay_ready=1`; progress is visible in
`train/replay_steps`.

MPPI logging separates raw WM return (`baseline_wm_value`, `selected_wm_value`,
`wm_value_gain`) from penalized planner score (`baseline_score`,
`selected_score`, `planner_score_gain`). Legacy `baseline_value`, `final_value`
and `planner_gain` are retained as planner-score aliases.

## Evaluation and visualization

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
  python mpr_mpc/evaluate.py \
  checkpoint=/absolute/path/to/best.pt \
  eval_episodes=20 \
  enable_wandb=false
```

Add `metadrive.simulator.use_render=true` for the interactive main/top-down
views. Evaluation uses the stage encoded by the checkpoint's global step.

## Tests and short smoke run

```bash
conda run -n tdmpc2 env PYTHONPATH="$PWD:$PWD/tdmpc2" \
  python -m unittest mpr_mpc.tests.test_mpr_mpc -v

conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
  python mpr_mpc/train.py checkpoint=null resume_checkpoint=null \
  steps=2 eval_episodes=1 enable_wandb=false save_video=false \
  metadrive.simulator.horizon=20 exp_name=mpr_mpc_smoke
```

The environment remains the existing `MetaDriveTDMPC2Env` with the configured
risk-field reward. No multimodal residual family, Gaussian-mixture planner, or
new cost critic is introduced.
