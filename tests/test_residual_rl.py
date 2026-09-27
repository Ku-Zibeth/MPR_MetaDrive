from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mpr_mpc.agent import MPRMPCAgent
from mpr_mpc.planning.types import PlanningStage
from mpr_mpc.residual_rl import ResidualReplayBuffer, ResidualSACAgent, ResidualStateBuilder


def _batch(batch_size=8, state_dim=6, *, done=0.0):
    return {
        "state": torch.randn((batch_size, state_dim)),
        "action": torch.randn((batch_size, 2)).clamp(-1.0, 1.0),
        "reward": torch.randn((batch_size, 1)),
        "cost": torch.rand((batch_size, 1)),
        "next_state": torch.randn((batch_size, state_dim)),
        "done": torch.full((batch_size, 1), float(done)),
        "discount": torch.full((batch_size, 1), 0.99),
    }


def _bare_agent(*, state_dim=3, batch_size=2):
    agent = MPRMPCAgent.__new__(MPRMPCAgent)
    agent.device = torch.device("cpu")
    agent.planner = SimpleNamespace(
        enabled=True,
        stage_for_step=lambda _step: PlanningStage("C", True, True),
    )
    agent.residual_rl_enabled = True
    agent.residual_learning_starts = 0
    agent.residual_batch_size = batch_size
    agent.residual_update_per_step = 1.0
    agent.residual_update_budget = 0.0
    agent.residual_gamma = 0.99
    agent.residual_rl_step = batch_size
    agent.residual_sac = ResidualSACAgent(
        state_dim,
        {"hidden_dims": [16], "lr": 1e-3, "use_lagrangian": True},
        device="cpu",
    )
    agent.residual_replay = ResidualReplayBuffer(16, state_dim=state_dim)
    for _ in range(batch_size):
        agent.residual_replay.add(
            np.ones(state_dim, dtype=np.float32),
            np.asarray([0.25, -0.5], dtype=np.float32),
            1.0,
            0.25,
            np.full(state_dim, 2.0, dtype=np.float32),
            False,
        )
    agent.last_residual_transition_metrics = {}
    agent._residual_stats = []
    agent.last_plan = None
    agent.global_env_step = 60000
    agent.env = SimpleNamespace(last_info={})
    agent.tdmpc_agent = SimpleNamespace(
        update=lambda _buffer: (_ for _ in ()).throw(AssertionError("TD-MPC2 update called"))
    )
    return agent


def test_use_lagrangian_false_disables_cost_training():
    agent = ResidualSACAgent(
        6,
        {"hidden_dims": [16], "lr": 1e-3, "use_lagrangian": False},
        device="cpu",
    )
    before = [parameter.detach().clone() for parameter in agent.cost_q1.parameters()]
    info = agent.update(_batch(state_dim=6))
    after = list(agent.cost_q1.parameters())
    assert float(info["sac/loss_actor_safety"]) == 0.0
    assert float(info["lagrangian/value"]) == 0.0
    assert float(info["sac/loss_cost_critic"]) == 0.0
    for lhs, rhs in zip(before, after):
        torch.testing.assert_close(lhs, rhs)
    metrics = agent.update_lagrangian(episode_cost=10.0, episode_steps=5)
    assert metrics["lagrangian/value"] == 0.0
    assert metrics["lagrangian/update_delta"] == 0.0


def test_lagrangian_increases_above_cost_limit():
    agent = ResidualSACAgent(
        4,
        {"hidden_dims": [16], "lagrangian_lr": 0.01, "cost_limit": 5.0},
        device="cpu",
    )
    metrics = agent.update_lagrangian(episode_cost=10.0, episode_steps=5)
    assert abs(metrics["lagrangian/value"] - 0.05) < 1e-8
    assert abs(metrics["lagrangian/update_delta"] - 0.05) < 1e-8
    assert metrics["lagrangian/episode_cost_minus_limit"] == 5.0
    assert metrics["lagrangian/episode_cost_per_step"] == 2.0


def test_lagrangian_never_negative():
    agent = ResidualSACAgent(
        4,
        {"hidden_dims": [16], "lagrangian_lr": 0.01, "cost_limit": 5.0},
        device="cpu",
    )
    metrics = agent.update_lagrangian(episode_cost=-100.0, episode_steps=4)
    assert metrics["lagrangian/value"] == 0.0
    assert metrics["lagrangian/new_value"] == 0.0


def test_pending_transition_completed_on_next_state():
    agent = MPRMPCAgent.__new__(MPRMPCAgent)
    agent.residual_replay = ResidualReplayBuffer(8, state_dim=3)
    agent.invalid_residual_penalty = 0.2
    agent._pending_residual = {
        "state": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "action": np.asarray([0.4, -0.3], dtype=np.float32),
        "valid": True,
    }
    agent._residual_episode_cost = 0.0
    agent._residual_episode_steps = 0
    agent.last_residual_transition_metrics = {}
    agent.observe_transition(reward=2.0, cost=0.5, done=False)
    assert len(agent.residual_replay) == 0
    agent._complete_pending_with_next_state(torch.tensor([4.0, 5.0, 6.0]))
    assert len(agent.residual_replay) == 1
    np.testing.assert_allclose(agent.residual_replay.states[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(agent.residual_replay.actions[0], [0.4, -0.3])
    np.testing.assert_allclose(agent.residual_replay.rewards[0], [2.0])
    np.testing.assert_allclose(agent.residual_replay.costs[0], [0.5])
    np.testing.assert_allclose(agent.residual_replay.next_states[0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(agent.residual_replay.dones[0], [0.0])
    assert agent._pending_residual is None
    agent._complete_pending_with_next_state(torch.tensor([7.0, 8.0, 9.0]))
    assert len(agent.residual_replay) == 1


def test_terminal_transition_targets_do_not_bootstrap():
    agent = ResidualSACAgent(6, {"hidden_dims": [16], "lr": 1e-3}, device="cpu")
    batch = _batch(batch_size=5, state_dim=6, done=1.0)
    batch["reward"].fill_(3.0)
    batch["cost"].fill_(0.75)
    info = agent.update(batch)
    torch.testing.assert_close(info["sac/target_q_mean"], torch.tensor(3.0))
    torch.testing.assert_close(info["sac/cost_target_mean"], torch.tensor(0.75))


def test_invalid_residual_replay_keeps_requested_action():
    agent = MPRMPCAgent.__new__(MPRMPCAgent)
    agent.residual_replay = ResidualReplayBuffer(8, state_dim=2)
    agent.invalid_residual_penalty = 0.2
    agent._pending_residual = {
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
        "action": np.asarray([0.8, -0.6], dtype=np.float32),
        "valid": False,
    }
    agent._residual_episode_cost = 0.0
    agent._residual_episode_steps = 0
    agent.last_residual_transition_metrics = {}
    agent.residual_sac = SimpleNamespace(update_lagrangian=lambda *_args: {})
    agent.observe_transition(reward=1.5, cost=0.7, done=True)
    assert len(agent.residual_replay) == 1
    np.testing.assert_allclose(agent.residual_replay.actions[0], [0.8, -0.6])
    np.testing.assert_allclose(agent.residual_replay.rewards[0], [1.3])
    np.testing.assert_allclose(agent.residual_replay.costs[0], [0.7])
    np.testing.assert_allclose(agent.residual_replay.dones[0], [1.0])


def test_include_wm_cost_in_state_is_explicitly_unsupported():
    try:
        ResidualStateBuilder({"include_wm_cost_in_state": True}, latent_dim=4, device="cpu")
    except ValueError as exc:
        assert "currently unsupported" in str(exc)
    else:
        raise AssertionError("include_wm_cost_in_state=true should fail loudly")


def test_tdmpc_freezes_once_on_stage_c():
    agent = MPRMPCAgent.__new__(MPRMPCAgent)
    agent.freeze_tdmpc2_on_start = True
    agent._tdmpc_frozen = False
    agent.planner = SimpleNamespace(stage_for_step=lambda _step: PlanningStage("C", True, True))
    model = torch.nn.Linear(3, 2)
    agent.tdmpc_agent = SimpleNamespace(model=model)
    agent.residual_sac = ResidualSACAgent(4, {"hidden_dims": [16]}, device="cpu")
    agent._maybe_freeze_tdmpc2(60000)
    assert agent._tdmpc_frozen
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in agent.residual_sac.actor.parameters())


def test_residual_update_not_blocked_by_tdmpc_replay():
    agent = _bare_agent(state_dim=3, batch_size=2)
    info = agent.update(None, tdmpc_update_enabled=False, residual_update_enabled=True)
    assert float(info["wm/replay_ready"]) == 0.0
    assert agent.residual_sac.update_steps == 1
    assert float(info["residual_rl/updates"]) == 1.0
