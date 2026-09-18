"""Online trainer additions for MetaDrive risk-field evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch

from mpr_mpc._bootstrap import bootstrap


bootstrap()
from trainer.online_trainer import OnlineTrainer  # noqa: E402


class MPROnlineTrainer(OnlineTrainer):
    """Keep TD-MPC2 training unchanged and expose safety metrics to W&B."""

    @torch.no_grad()
    def eval(self):
        rewards = []
        costs = []
        successes = []
        lengths = []
        base_rewards = []
        risk_costs = []
        normalized_risks = []
        risk_penalties = []
        collisions = []
        offroads = []

        for episode in range(int(self.cfg.eval_episodes)):
            observation = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_cost = 0.0
            episode_base_reward = 0.0
            episode_risk_cost = 0.0
            episode_normalized_risk = 0.0
            episode_risk_penalty = 0.0
            step = 0
            info = {}
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=episode == 0)

            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(
                    observation, t0=step == 0, eval_mode=True
                )
                observation, reward, done, info = self.env.step(action)
                episode_reward += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                episode_base_reward += float(info.get("metadrive_reward", 0.0))
                episode_risk_cost += float(info.get("risk_field_cost", 0.0))
                episode_normalized_risk += float(
                    info.get("risk_field_normalized_cost", 0.0)
                )
                episode_risk_penalty += float(
                    info.get("risk_field_reward_penalty", 0.0)
                )
                step += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)

            rewards.append(episode_reward)
            costs.append(episode_cost)
            successes.append(float(info.get("success", 0.0)))
            lengths.append(step)
            base_rewards.append(episode_base_reward)
            risk_costs.append(episode_risk_cost)
            normalized_risks.append(episode_normalized_risk)
            risk_penalties.append(episode_risk_penalty)
            collisions.append(float(info.get("crash", 0.0)))
            offroads.append(float(info.get("out_of_road", 0.0)))
            if self.cfg.save_video:
                self.logger.video.save(self._step)

        return {
            "episode_reward": float(np.mean(rewards)),
            "episode_cost": float(np.mean(costs)),
            "episode_success": float(np.mean(successes)),
            "episode_length": float(np.mean(lengths)),
            "episode_metadrive_reward": float(np.mean(base_rewards)),
            "episode_risk_field_cost": float(np.mean(risk_costs)),
            "episode_normalized_risk": float(np.mean(normalized_risks)),
            "episode_risk_penalty": float(np.mean(risk_penalties)),
            "episode_collision": float(np.mean(collisions)),
            "episode_offroad": float(np.mean(offroads)),
        }


__all__ = ["MPROnlineTrainer"]
