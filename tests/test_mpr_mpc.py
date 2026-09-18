from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG, FrenetPath, MetaDriveFrenetController
from mpr_mpc.planning.coordinator import MPRMPCPlanner
from mpr_mpc.planning.evaluator import TrajectoryWorldModelEvaluator
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.residual_prior import ResidualTrajectoryPrior
from mpr_mpc.planning.time_alignment import align_frenet_path
from mpr_mpc.planning.trajectory_adapter import LatticeActionAdapter


def make_path(horizon=2.0, dt=0.1, target_d=0.2, target_speed=10.0):
    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    s = target_speed * times
    d = target_d * (times / horizon) ** 2
    d_d = np.gradient(d, dt)
    d_dd = np.gradient(d_d, dt)
    d_ddd = np.gradient(d_dd, dt)
    zeros = np.zeros_like(times)
    return FrenetPath(
        t=times.tolist(), d=d.tolist(), d_d=d_d.tolist(), d_dd=d_dd.tolist(),
        d_ddd=d_ddd.tolist(), s=s.tolist(), s_d=np.full_like(times, target_speed).tolist(),
        s_dd=zeros.tolist(), s_ddd=zeros.tolist(), x=s.tolist(), y=d.tolist(),
        yaw=np.arctan2(d_d, np.full_like(d_d, target_speed)).tolist(),
        target_d=target_d, target_speed=target_speed, horizon=horizon,
    )


class FakeVehicle:
    position = np.asarray([0.0, 0.0])
    heading_theta = 0.0
    speed = 5.0


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._termination = None

    def reward(self, z, action, task):
        return action[:, :1]

    def next(self, z, action, task):
        return z + action[:, :2]

    def pi(self, z, task):
        return torch.zeros((len(z), 2), device=z.device), {}

    def Q(self, z, action, task, return_type="avg"):
        return 10.0 * action[:, :1]


def tiny_agent(horizon=3):
    del horizon
    cfg = SimpleNamespace(
        action_dim=2, latent_dim=2, multitask=False, episodic=False, num_bins=1,
        vmin=-10.0, vmax=10.0, bin_size=1.0,
    )
    return SimpleNamespace(
        model=TinyModel(), cfg=cfg, device=torch.device("cpu"), discount=0.9,
    )


class MPRMPCUnitTests(unittest.TestCase):
    def test_h_plus_one_action_adapter(self):
        controller = MetaDriveFrenetController(dict(FRENET_DEFAULT_CONFIG))
        adapter = LatticeActionAdapter(controller)
        actions = adapter.path_to_actions(
            make_path(), FakeVehicle(), required_steps=11, control_dt=0.1, device="cpu"
        )
        self.assertEqual(actions.shape, (11, 2))
        self.assertTrue(torch.isfinite(actions).all())

    def test_time_alignment_with_different_dt(self):
        path = make_path(dt=0.1)
        aligned = align_frenet_path(path, required_steps=6, control_dt=0.2)
        np.testing.assert_allclose(aligned.times, np.arange(6) * 0.2)
        np.testing.assert_allclose(aligned.positions[:, 0], np.arange(6) * 2.0, atol=1e-5)

    def test_residual_shape_dynamic_bounds_and_zero_initial_mean(self):
        prior = ResidualTrajectoryPrior(518)
        bounds = torch.tensor(
            [[3.5, 5.0, 0.25], [4.0, 2.0, 0.25], [3.0, 0.0, 0.25], [3.5, 6.0, 0.25]]
        )
        residual, info = prior(
            torch.zeros((4, 518)), delta_bounds=bounds, deterministic=True
        )
        self.assertEqual(residual.shape, (4, 3))
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.all(residual.abs() <= bounds + 1e-6))
        self.assertTrue(torch.equal(info["delta_bounds"], bounds))
        self.assertEqual(info["log_std"].shape, (4, 3))

    def test_dynamic_bounds_use_lane_width_and_half_coarse_speed(self):
        planner_like = SimpleNamespace(
            structured=SimpleNamespace(planner=SimpleNamespace(last_lane_width=3.6)),
            delta_t_bound=0.25,
            device=torch.device("cpu"),
        )
        bounds = MPRMPCPlanner.residual_bounds(
            planner_like, make_path(target_speed=10.0)
        )
        torch.testing.assert_close(bounds, torch.tensor([[3.6, 5.0, 0.25]]))

    def test_residual_supervised_loss_backpropagates(self):
        prior = ResidualTrajectoryPrior(32, hidden_dims=(16, 16))
        inputs = torch.randn((8, 32))
        targets = torch.zeros((8, 3))
        bounds = torch.tensor([3.5, 5.0, 0.25]).expand(8, -1)
        loss, _ = prior.supervised_loss(
            inputs, targets, delta_bounds=bounds, residual_reg_coef=0.01
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in prior.parameters()))

    def test_trajectory_terminal_action_replaces_policy_action(self):
        evaluator = TrajectoryWorldModelEvaluator(tiny_agent(), horizon=3)
        latent = torch.zeros((1, 2))
        trajectory_actions = torch.zeros((1, 4, 2))
        trajectory_actions[:, 3, 0] = 0.7
        trajectory = evaluator.evaluate_trajectory_consequence(latent, trajectory_actions)
        policy = evaluator.evaluate_policy_terminal(latent, trajectory_actions[:, :3])
        self.assertAlmostEqual(float(trajectory.terminal_q[0]), 7.0, places=5)
        self.assertAlmostEqual(float(policy.terminal_q[0]), 0.0, places=5)
        self.assertEqual(trajectory.reward_sequence.shape, (1, 3))
        self.assertEqual(trajectory.features.shape, (1, 3))
        self.assertTrue(torch.equal(trajectory.features[:, 0], trajectory.reward_sum))

    def test_local_mppi_proposal_equals_structured_baseline(self):
        cfg = {
            "num_samples": 8, "num_elites": 2, "iterations": 1,
            "initial_std": 0.2, "min_std": 0.05, "max_std": 0.5,
            "temperature": 0.5, "proposal_init": "structured_baseline",
        }
        evaluator = TrajectoryWorldModelEvaluator(tiny_agent(), horizon=3)
        planner = LocalMPPI(evaluator, cfg, horizon=3, action_dim=2, device="cpu")
        baseline = torch.linspace(-0.5, 0.5, 8).reshape(4, 2)
        self.assertTrue(torch.equal(planner.initial_mean(baseline), baseline))
        result = planner.plan(torch.zeros((1, 2)), baseline, eval_mode=True)
        self.assertEqual(result.selected_sequence.shape, (4, 2))
        self.assertTrue(torch.equal(result.initial_mean, baseline))


if __name__ == "__main__":
    unittest.main()
