"""Lattice-driven online trainer with staged MPR-MPC activation."""

from __future__ import annotations

from time import time

import numpy as np
import torch

from mpr_mpc._bootstrap import bootstrap


bootstrap()
from trainer.online_trainer import OnlineTrainer  # noqa: E402


class MPROnlineTrainer(OnlineTrainer):
    """Collect from Lattice at step zero and perform one smooth update per step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = int(self.agent.global_env_step)
        self._replay_steps = 0
        self._best_eval_key = self.agent.best_eval_key
        self._checkpoint_freq = int(self.cfg.get("checkpoint_freq", 50000))
        self._start_time = time()

    def _save_checkpoint(self, identifier: str) -> None:
        self.agent.set_global_step(self._step)
        self.logger.save_agent(self.agent, identifier)

    def _handle_eval(self) -> dict[str, float]:
        metrics = self.eval()
        metrics.update(self.common_metrics())
        self.logger.log(metrics, "eval")
        key = (
            float(metrics["success"]),
            -float(metrics["offroad"]),
            float(metrics["episode_reward"]),
        )
        if self._best_eval_key is None or key > self._best_eval_key:
            self._best_eval_key = key
            self.agent.best_eval_key = key
            self._save_checkpoint("best")
        self._save_checkpoint("latest")
        return metrics

    @torch.no_grad()
    def eval(self):
        rewards, costs, successes, lengths = [], [], [], []
        collisions, offroads, route_completions = [], [], []
        base_rewards, risk_costs, normalized_risks, risk_penalties = [], [], [], []
        for episode in range(int(self.cfg.eval_episodes)):
            observation = self.env.reset()
            done = False
            episode_reward = episode_cost = episode_base_reward = 0.0
            episode_risk_cost = episode_normalized_risk = episode_risk_penalty = 0.0
            step, info = 0, {}
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=episode == 0)
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(
                    observation,
                    t0=step == 0,
                    eval_mode=True,
                    global_step=self._step,
                )
                observation, reward, done, info = self.env.step(action)
                episode_reward += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                episode_base_reward += float(info.get("metadrive_reward", 0.0))
                episode_risk_cost += float(info.get("risk_field_cost", 0.0))
                episode_normalized_risk += float(info.get("risk_field_normalized_cost", 0.0))
                episode_risk_penalty += float(info.get("risk_field_reward_penalty", 0.0))
                step += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            rewards.append(episode_reward)
            costs.append(episode_cost)
            successes.append(float(info.get("success", 0.0)))
            lengths.append(step)
            collisions.append(float(info.get("crash", 0.0)))
            offroads.append(float(info.get("out_of_road", 0.0)))
            route_completions.append(float(info.get("route_completion", 0.0)))
            base_rewards.append(episode_base_reward)
            risk_costs.append(episode_risk_cost)
            normalized_risks.append(episode_normalized_risk)
            risk_penalties.append(episode_risk_penalty)
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        mean = lambda values: float(np.nanmean(values))
        return {
            "episode_reward": mean(rewards),
            "episode_cost": mean(costs),
            "success": mean(successes),
            "offroad": mean(offroads),
            "collision": mean(collisions),
            "route_completion": mean(route_completions),
            "episode_length": mean(lengths),
            "episode_metadrive_reward": mean(base_rewards),
            "episode_risk_field_cost": mean(risk_costs),
            "episode_normalized_risk": mean(normalized_risks),
            "episode_risk_penalty": mean(risk_penalties),
        }

    def train(self):
        train_metrics, done, info = {}, True, {}
        eval_next = self._step == 0
        eval_freq = max(1, int(self.cfg.eval_freq))
        next_eval = ((self._step // eval_freq) + 1) * eval_freq
        next_checkpoint = ((self._step // self._checkpoint_freq) + 1) * self._checkpoint_freq
        minimum_replay_steps = int(self.cfg.batch_size) * (int(self.cfg.horizon) + 1)

        while self._step < int(self.cfg.steps):
            if done:
                if eval_next:
                    self._handle_eval()
                    eval_next = False
                if self._step > 0:
                    if info["terminated"] and not self.cfg.episodic:
                        raise ValueError("Termination detected but episodic=false.")
                    train_metrics.update(
                        episode_reward=torch.tensor([td["reward"] for td in self._tds[1:]]).sum(),
                        episode_cost=torch.tensor([td["cost"] for td in self._tds[1:]]).sum(),
                        episode_success=info["success"],
                        episode_length=len(self._tds),
                        episode_terminated=info["terminated"],
                        Reason=info.get("end_reason", "unknown"),
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    episode = torch.cat(self._tds)
                    self._ep_idx = self.buffer.add(episode)
                    self._replay_steps += max(0, len(episode) - 1)
                observation = self.env.reset()
                self._tds = [self.to_td(observation)]

            # No random seed policy: Stage A is Lattice-controlled from the first step.
            self.agent.set_global_step(self._step)
            action = self.agent.act(
                observation,
                t0=len(self._tds) == 1,
                global_step=self._step,
            )
            observation, reward, done, info = self.env.step(action)
            cost = torch.tensor(
                float(info.get("cost", info.get("risk_field_cost", 0.0))),
                dtype=torch.float32,
            )
            self._tds.append(self.to_td(observation, action, reward, info["terminated"], cost))

            # Once a complete batch can be formed, exactly one online update per env step.
            if self._replay_steps >= minimum_replay_steps:
                train_metrics.update(self.agent.update(self.buffer))

            self._step += 1
            self.agent.set_global_step(self._step)
            if self._step >= next_eval:
                eval_next = True
                while next_eval <= self._step:
                    next_eval += eval_freq
            if self._step >= next_checkpoint:
                self._save_checkpoint("latest")
                while next_checkpoint <= self._step:
                    next_checkpoint += self._checkpoint_freq

        self.agent.set_global_step(self._step)
        self.logger.finish(self.agent)


__all__ = ["MPROnlineTrainer"]
