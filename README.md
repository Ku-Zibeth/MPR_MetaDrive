# MPR-MPC Residual SAC-Lagrangian

The current controller is a single-coarse-trajectory MPR-MPC stack:

```text
MetaDrive state
  -> Lattice coarse trajectory
  -> Residual SAC-Lagrangian
  -> refined Frenet trajectory
  -> Local MPPI
  -> execute first control
```

The residual module is no longer supervised by world-model residual targets. It
learns online from real MetaDrive reward and real environment safety cost.

## Training Stages

- Stage A, steps `[0, 20000)`: Lattice only, TD-MPC2 online.
- Stage B, steps `[20000, 50000)`: Lattice + Local MPPI, TD-MPC2 online.
- Stage C, steps `[50000, ...)`: Lattice + Residual SAC + Local MPPI. TD-MPC2 is frozen by default and Residual SAC trains online.

Stage C keeps MPR-MPC's dynamic physical residual bounds:

```text
[u_d, u_v] in [-1, 1]^2
delta_d = u_d * current_lane_width
delta_v = u_v * 0.5 * coarse_target_speed
T_refined = T_lattice
```

If Frenet regeneration or feasibility checking rejects the residual, execution
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

`include_wm_cost_in_state=true` is intentionally unsupported until
`TrajectoryConsequence` is extended with a trained TD-MPC2 cost head.

The SAC-Lagrangian implementation contains:

- Gaussian actor for normalized `[u_d, u_v]`.
- Twin reward critics and target reward critics.
- Twin cost critics and target cost critics.
- Automatic entropy temperature alpha.
- Projected dual-gradient Lagrangian lambda.
- Reward critic MSE targets.
- Cost critic Huber targets using real MetaDrive environment safety cost.

With `use_lagrangian=false`, reward SAC still trains, but cost targets, cost
critic optimizer steps, actor safety loss, cost target-network updates, and
lambda updates are disabled.

Invalid residuals only affect residual reward:

```text
residual_reward = env_reward - invalid_residual_penalty * I_invalid
```

They do not affect TD-MPC2 reward, environment cost, or cost critic targets.

## Local MPPI

Local MPPI remains a small refinement around the residual-refined Frenet
baseline:

```yaml
mpr_mpc:
  mppi:
    num_samples: 64
    num_elites: 8
    iterations: 4
```

The top-level TD-MPC2 `num_samples`, `num_elites`, and `iterations` are separate
and are not used by the MPR Local MPPI controller.

## Checkpoints

Current checkpoints use:

```text
mpr_mpc_v2_residual_sac
```

They include:

- TD-MPC2 model.
- TD-MPC2 optimizer states and running value scale.
- Residual SAC actor.
- Reward critics and target reward critics.
- Cost critics and target cost critics.
- Actor, critic, cost critic, and alpha optimizer states.
- Alpha and Lagrangian lambda state.
- Global environment step.
- Residual local step and update budget.
- Planner counters.

TD-MPC2 replay and Residual SAC replay are not serialized, so resume is not a
bit-exact continuation. After resume, replay buffers rebuild from fresh
environment interaction.

## Metrics

Important training metrics include:

```text
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
sac/target_entropy
sac/q1
sac/q2
sac/q_mean
sac/target_q_mean
sac/target_q_max
sac/cost_q1
sac/cost_q2
sac/cost_q_mean
sac/cost_target_mean
sac/cost_target_max

lagrangian/value
lagrangian/cost_limit
lagrangian/episode_cost
lagrangian/episode_cost_minus_limit
lagrangian/episode_cost_per_step
lagrangian/update_delta

residual_stats/mean_abs_delta_d
residual_stats/mean_abs_delta_v
residual_stats/max_abs_delta_d
residual_stats/max_abs_delta_v
residual_stats/mean_abs_action_d
residual_stats/mean_abs_action_v
residual_stats/zero_like_ratio
residual_stats/fallback_mean_abs_delta
residual_stats/normal_mean_abs_delta

mpr/requested_residual_d
mpr/requested_residual_v
mpr/residual_d
mpr/residual_v
mpr/residual_valid
mpr/residual_invalid_rate
mpr/used_lattice_fallback
mpr/feasible_ratio
mpr/baseline_wm_value
mpr/selected_wm_value
mpr/wm_value_gain
mpr/baseline_score
mpr/selected_score
mpr/planner_score_gain
mpr/baseline_selected_rate
mpr/corridor_reject_rate
mpr/planning_ms
```

Legacy aliases may appear for old planner score names, but the metrics above are
the canonical keys for the Residual SAC-Lagrangian path.

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

Evaluation:

```bash
conda run -n tdmpc2 env CUDA_VISIBLE_DEVICES=0 \
  python mpr_mpc/evaluate.py \
  checkpoint=/absolute/path/to/best.pt \
  eval_episodes=20 \
  enable_wandb=false
```

Residual-only tests do not require MetaDrive:

```bash
python -m pytest mpr_mpc/tests/test_residual_rl.py -q
```

The broader MPR-MPC tests require the MetaDrive package because they instantiate
the Frenet/MetaDrive controller.
