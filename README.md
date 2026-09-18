# MPR-MPC

MPR-MPC combines the existing MetaDrive Frenet Lattice planner with the existing
TD-MPC2 world model without changing either package in place. The online path is:

```text
observation -> TD-MPC2 encoder -> z_t
MetaDrive state -> Lattice candidates -> feasibility filter -> planner-cost selection
selected coarse path -> H+1 actions -> H-step WM consequence features
[z_t, normalized coarse (d,v,T), WM (sum_r,Q_H,J)]
    -> bounded Gaussian residual [delta_d, delta_v, delta_T]
    -> Frenet trajectory regeneration and safety check
    -> refined H+1 action baseline
    -> local Gaussian MPPI
    -> H dynamics steps + Q(z_H, trajectory_action_H)
    -> execute action[0]
```

There is no per-Lattice-candidate residual inference. Lattice owns multimodal
generation and mode selection. The residual prior only corrects the one selected
coarse trajectory, and MPPI only performs continuous local refinement around it.

## Module responsibilities

- `lattice/`: namespace for the existing multimodal Frenet generator.
- `tdmpc2/`: namespace and extension point for the unchanged TD-MPC2 learner.
- `planning/structured_proposal.py`: feasibility filtering and planner-cost selection.
- `planning/evaluator.py`: policy-terminal and trajectory-terminal WM evaluation.
- `planning/residual_prior.py`: bounded 3D Gaussian trajectory residual.
- `planning/trajectory_adapter.py`: Frenet path to normalized H+1 MetaDrive actions.
- `planning/local_mppi.py`: local H+1 MPPI with H explicit dynamics transitions.
- `planning/residual_training.py`: separate residual supervision buffer and targets.
- `planning/coordinator.py`: inference data flow and fallbacks.
- `agent.py`: facade compatible with the original `OnlineTrainer`.

The original action policy `WorldModel._pi` is retained for TD targets,
`update_pi()`, and original TD-MPC2 inference. In MPR-MPC mode it is not used for
MPPI proposal trajectories or planning-time terminal actions.

## Tensor flow

```text
observation                         [B,259]
latent                              [B,512]
normalized [d,v,T]                  [B,3]
WM [sum(r_0...r_9),Q_H,J]           [B,3]
residual input                      [B,518]
residual raw mean/log-std           [B,6]
bounded residual                    [B,3]
coarse/refined action sequence      [B,11,2]
MPPI samples                        [11,N,2]
explicit WM rollout                 first 10 actions
terminal Q                          Q(z_10, action_10)
executed action                     [2]
```

The World Model still predicts each one-step reward internally because `J` needs
the time-ordered discounted return. The residual policy receives only their
undiscounted sum, not the ten individual values. Its bounds are computed for every
selected coarse path: `|delta_d| <= current_lane_width`,
`|delta_v| <= coarse_target_speed/2`, and `|delta_T| <= delta_t_bound`.

At current defaults `control_dt=0.02*5=0.1 s`, the World Model explicitly covers
1.0 s, while the full structured trajectory remains 2.0 or 3.0 s.

## Training

Warm-start from the existing pure TD-MPC2 checkpoint:

```bash
cd /home/kzb/kzb_code/tdmpc2
conda activate tdmpc2
CUDA_VISIBLE_DEVICES=0 python mpr_mpc/train.py \
  checkpoint=logs/metadrive-risk/1/tdmpc2_metadrive_risk/models/final.pt \
  exp_name=mpr_mpc_seed1
```

Training uses `MetaDriveTDMPC2Env`, the same `metadrive-risk` environment and
risk-field parameters as `tdmpc2_fm`. The scalar reward learned by TD-MPC2 is:

```text
clip(base_reward_weight * MetaDriveReward
     - risk_field_reward_scale * clip(raw_risk / risk_field_raw_clip, 0, 1),
     reward_min, reward_max)
```

With the default configuration this is
`clip(MetaDriveReward - 25 * clip(raw_risk / 10, 0, 1), -10, 10)`.
The risk field contains road-boundary, lane-line, off-road, surrounding-vehicle,
and static-object terms. Headway and TTC terms are present but default to zero
weight, matching the existing TD-MPC2 risk-field run.

Weights & Biases is enabled by default under project `mpr_mpc_metadrive`. Run
`wandb login` once before training. `wandb_entity: null` uses the account selected
by that login, and `wandb_run_name: null` produces `mpr_mpc-seed1`. Override them
normally with Hydra:

```bash
CUDA_VISIBLE_DEVICES=0 python mpr_mpc/train.py \
  checkpoint=logs/metadrive-risk/1/tdmpc2_metadrive_risk/models/final.pt \
  exp_name=mpr_mpc_seed1 \
  wandb_project=mpr_mpc_metadrive \
  wandb_run_name=mpr_mpc_seed1
```

W&B receives TD-MPC2 losses, residual-prior loss/buffer/target metrics, MPR
candidate/value/residual/timing metrics, per-step risk-field components at train
logging points, and evaluation episode return, cost, base MetaDrive reward,
risk-field cost, risk penalty, collision, and off-road rates. For an offline
debug run only, pass `enable_wandb=false`.

World-model/Q/action-policy learning still uses the original TD-MPC2 replay and
updates. Residual targets use a separate buffer. For each selected coarse path,
bounded parameter residuals (including zero) are regenerated, filtered, converted
to H+1 controls, and scored with trajectory-consistent terminal Q. The best or a
softmax-weighted elite residual becomes the supervised Gaussian target.

To freeze TD-MPC2 and train only the residual prior, use the evaluation/data
components directly; the default online command continues TD-MPC2 learning. A
frozen-WM trainer is not yet exposed as a separate CLI mode.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python mpr_mpc/evaluate.py \
  checkpoint=logs/metadrive-risk/1/mpr_mpc_seed1/models/final.pt \
  eval_episodes=20
```

Evaluation also logs per-scenario return, safety cost, base reward, risk-field
cost/penalty, success, collision, off-road status, and episode length to W&B.
Pass `enable_wandb=false` when a local-only evaluation is desired.

Enable main-camera and top-down visualization with:

```bash
CUDA_VISIBLE_DEVICES=0 python mpr_mpc/evaluate.py \
  checkpoint=/path/to/checkpoint.pt \
  eval_episodes=20 \
  metadrive.simulator.use_render=true
```

An old pure TD-MPC2 checkpoint is accepted. The residual head is then safely
zero-mean initialized, so deterministic evaluation initially uses zero correction.

## Original TD-MPC2 compatibility

The independent entry point can reproduce the original planning branch:

```bash
CUDA_VISIBLE_DEVICES=0 python mpr_mpc/evaluate.py \
  checkpoint=logs/metadrive-risk/1/tdmpc2_metadrive_risk/models/final.pt \
  mpr_mpc.enabled=false
```

This delegates to the original `TDMPC2.act()`, including policy proposals,
previous-mean warm start, H-step action tensors, and policy terminal action.

## Ablations

```bash
# A: original TD-MPC2
python mpr_mpc/evaluate.py checkpoint=... mpr_mpc.enabled=false

# B: Lattice baseline + local MPPI, zero residual
python mpr_mpc/evaluate.py checkpoint=... \
  mpr_mpc.use_residual=false mpr_mpc.use_mppi=true

# C: Lattice + residual, no MPPI
python mpr_mpc/evaluate.py checkpoint=... \
  mpr_mpc.use_residual=true mpr_mpc.use_mppi=false

# D: complete MPR-MPC
python mpr_mpc/evaluate.py checkpoint=... \
  mpr_mpc.use_residual=true mpr_mpc.use_mppi=true

# E: structured center / previous mean / blend
python mpr_mpc/evaluate.py checkpoint=... \
  mpr_mpc.mppi.proposal_init=structured_baseline
python mpr_mpc/evaluate.py checkpoint=... \
  mpr_mpc.mppi.proposal_init=previous_mean
python mpr_mpc/evaluate.py checkpoint=... \
  mpr_mpc.mppi.proposal_init=blend mpr_mpc.mppi.blend_alpha=0.8

# F: deterministic / stochastic residual at evaluation
python mpr_mpc/evaluate.py checkpoint=... \
  mpr_mpc.residual_prior.deterministic_eval=true
python mpr_mpc/evaluate.py checkpoint=... \
  mpr_mpc.residual_prior.deterministic_eval=false
```

World-Model candidate selection is intentionally not an ablation in the revised
architecture: Lattice planner cost always owns global mode selection.

## Important configuration

- `mpr_mpc.enabled`: switch between MPR-MPC and untouched TD-MPC2 planning.
- `use_residual`, `use_mppi`: component ablations.
- `residual_prior.delta_t_bound`: fixed maximum `|delta_T|`; lateral and speed
  bounds are dynamic.
- `residual_prior.wm_feature_scales`: summed-reward, terminal-Q, and total-value normalization.
- `refinement`: absolute speed and horizon safety bounds.
- `mppi.initial_std/min_std/max_std`: local action-space search scale.
- `mppi.proposal_init`: structured baseline, shifted previous mean, or blend.
- `residual_training.target_samples`: low-dimensional residual targets per step.
- `debug.enabled/every`: compact proposal, residual, value, and timing logs.

## Tests

```bash
/home/kzb/anaconda3/envs/tdmpc2/bin/python -m unittest discover \
  -s mpr_mpc/tests -v
```

## Known limitations

- Frenet generation, collision filtering, and path tracking are NumPy/Python and
  non-differentiable; residual learning therefore uses value-guided supervision.
- H+1 controls are produced with virtual future tracking states, not a full inverse
  vehicle dynamics model.
- Residual target generation adds multiple Frenet regenerations and WM evaluations
  per selected coarse path and is the main training-time bottleneck.
- The residual supervision target inherits bias from the learned World Model/Q.
- The current integration is single-task MetaDrive and assumes normalized two-axis
  `[steering, throttle_brake]` control.
- A refined trajectory that fails geometry/collision checks falls back to the
  selected coarse path; this branch is non-differentiable by design.
- Planning always uses Lattice planner-cost mode selection. This is deliberate and
  prevents the removed 54-candidate residual/WM-selection architecture from
  reappearing through configuration.
