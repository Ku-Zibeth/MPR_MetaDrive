# MPR-MPC Residual SAC-Lagrangian

The controller is a staged single-coarse-trajectory MPR-MPC stack:

```text
MetaDrive state
  -> Lattice coarse trajectory
  -> Residual SAC-Lagrangian
  -> refined Frenet trajectory
  -> Local MPPI
  -> execute first control
```

The residual module learns online from real MetaDrive reward and real environment
safety cost. The world model is used for latent state, prediction, and MPPI
scoring, not as a residual teacher.

## Training Stages

- Stage A `[0, 20K)`: Lattice + TD-MPC2 training. Residual SAC OFF. Local MPPI OFF.
- Stage B `[20K, 50K)`: Lattice + Residual SAC + TD-MPC2 training. Local MPPI OFF.
- Stage C `[50K, 1M]`: Lattice + Residual SAC + Local MPPI. TD-MPC2 continues slow online adaptation.

Stage A learns a basic predictive world model from structured Lattice behavior.
Stage B exposes the Residual SAC policy directly to environment consequences,
without Local MPPI correction, encouraging distinct trajectory-level corrections.
Stage C introduces Local MPPI only after the residual policy is established, so
MPPI performs local control-sequence refinement around the residual trajectory.

Stage C TD-MPC2 adaptation uses update-frequency reduction rather than optimizer
recreation:

```yaml
tdmpc_stage_training:
  stage_a_update_ratio: 1.0
  stage_b_update_ratio: 1.0
  stage_c_update_ratio: 0.25
```

Residual SAC starts at `mpr_mpc.stages.residual_start_step` and its local
`residual_rl_step` starts from zero there. With defaults, steps `20K..25K` use
`Uniform(-0.5, +0.5)` residual warmup actions, then stochastic SAC actions and
updates begin once residual replay is ready.

## Residual Action

Residual action remains two-dimensional:

```text
[u_d, u_v] in [-1, 1]^2
delta_d = u_d * current_lane_width
delta_v = u_v * 0.5 * coarse_target_speed
T_refined = T_lattice
```

If Frenet regeneration or feasibility checking rejects a residual, execution
falls back to the coarse Lattice trajectory with effective residual `[0, 0]`.
Residual replay still stores the actor's requested normalized action.

## Residual SAC

The default residual state is 527-D:

```text
TD-MPC2 latent        512
Lattice path features 12
coarse WM return       1
feasible/fallback      2
WM cost                0
```

`include_wm_cost_in_state=true` is unsupported until `TrajectoryConsequence` is
extended with a trained TD-MPC2 cost head.

The SAC-Lagrangian implementation contains a Gaussian actor, twin reward critics
and targets, twin cost critics and targets, automatic entropy alpha, and a
projected dual-gradient Lagrangian multiplier. With `use_lagrangian=false`,
reward SAC still trains but the cost branch, actor safety loss, target cost
updates, and lambda updates are disabled.

Cost critics use real MetaDrive environment safety cost only. Invalid residuals
only affect residual reward:

```text
residual_reward = env_reward - invalid_residual_penalty * I_invalid
```

## Local MPPI

Local MPPI is OFF in Stage A and Stage B. It turns ON only in Stage C.

```yaml
mpr_mpc:
  mppi:
    num_samples: 64
    num_elites: 8
    iterations: 4
```

MPPI has no learned model weights. Full checkpoints save MPPI configuration and
planner counters, not an MPPI `state_dict`.

## Milestone Checkpoints

Milestones are permanent and coexist with `latest.pt`, `best.pt`, and `final.pt`.
They do not serialize TD-MPC2 replay or Residual SAC replay, so resume is not a
bit-exact continuation.

- `milestone_20k_world_model.pt`: basic TD-MPC2 world model trained on Lattice data.
- `milestone_50k_world_model_residual_rl.pt`: TD-MPC2 + trained Residual SAC, with Stage B MPPI OFF.
- `milestone_1m_full_mpr_mpc.pt`: full MPR-MPC stack, including planner counters and MPPI config.

Milestone format:

```text
format = mpr_mpc_milestone_v1
components.world_model = true
components.residual_rl = milestone-dependent
components.planner = full milestone only
components.mppi = full milestone only
```

Component-aware loading is available through:

```python
agent.load_milestone(path, components=["world_model"], load_optimizer=True)
agent.load_milestone(path, components=["world_model", "residual_rl"], load_optimizer=True)
agent.load_milestone(path, components=["world_model", "residual_rl", "planner"])
```

Examples:

```python
# Start a new Residual SAC experiment from the 20K world model.
agent.load_milestone("milestone_20k_world_model.pt", components=["world_model"])

# Evaluate or continue Lattice + Residual SAC without MPPI from 50K.
agent.load_milestone(
    "milestone_50k_world_model_residual_rl.pt",
    components=["world_model", "residual_rl"],
)

# Evaluate the full final stack.
agent.load_milestone(
    "milestone_1m_full_mpr_mpc.pt",
    components=["world_model", "residual_rl", "planner"],
)
```

When resuming beyond a historical milestone, the trainer will not fabricate that
old milestone from the later model. Existing milestone files are never
overwritten. The final milestone is checked once more after the training loop so
the 1M full checkpoint is not missed at loop exit.

## Metrics

Important metrics include:

```text
train/stage
train/tdmpc_update_ratio
train/tdmpc_updates
train/residual_updates
train/residual_active
train/mppi_active

checkpoint/milestone_20k_saved
checkpoint/milestone_50k_saved
checkpoint/milestone_1m_saved

residual_rl/action_d
residual_rl/action_v
residual_rl/reward
residual_rl/invalid_penalty
residual_rl/env_cost
residual_rl/replay_size
residual_rl/local_step
residual_rl/update_steps

sac/loss_actor
sac/loss_reward_critic
sac/loss_cost_critic
sac/alpha
sac/log_alpha
sac/entropy
sac/q_mean
sac/target_q_mean
sac/cost_q_mean
sac/cost_target_mean

lagrangian/value
lagrangian/cost_limit
lagrangian/episode_cost
lagrangian/episode_cost_minus_limit
lagrangian/update_delta

residual_stats/mean_abs_delta_d
residual_stats/mean_abs_delta_v
residual_stats/zero_like_ratio

mpr/residual_valid
mpr/residual_invalid_rate
mpr/used_lattice_fallback
mpr/feasible_ratio
mpr/baseline_wm_value
mpr/selected_wm_value
mpr/planner_score_gain
mpr/corridor_reject_rate
mpr/planning_ms
```

## Running

```bash
cd /home/kzb/kzb_code/tdmpc2
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
  python mpr_mpc/train.py \
  checkpoint=null \
  resume_checkpoint=null \
  exp_name=mpr_mpc_residual_sac_v1 \
  wandb_run_name=mpr_mpc_residual_sac_v1 \
  wandb_project=mpr_mpc_metadrive
```

Fast stage-switch smoke:

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

Expected smoke milestones:

```text
milestone_300_world_model.pt
milestone_700_world_model_residual_rl.pt
milestone_1500_full_mpr_mpc.pt
```

Tests:

```bash
python -m compileall mpr_mpc
PYTHONPATH="$PWD:$PWD/tdmpc2" python -m pytest mpr_mpc/tests -q
```

The broader MPR-MPC tests require the MetaDrive package because they instantiate
the Frenet/MetaDrive controller.
