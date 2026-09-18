"""Evaluate original TD-MPC2 or MPR-MPC on reproducible MetaDrive scenes."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ROOT = Path(__file__).resolve().parent
while str(SCRIPT_ROOT) in sys.path:
    sys.path.remove(str(SCRIPT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpr_mpc._bootstrap import bootstrap

bootstrap()

from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from env import make_env  # noqa: E402

from mpr_mpc.agent import MPRMPCAgent  # noqa: E402
from mpr_mpc.lattice.frenet_metadrive import (  # noqa: E402
    FRENET_DEFAULT_CONFIG,
    MetaDriveFrenetController,
)
from mpr_mpc.planning.coordinator import MPRMPCPlanner  # noqa: E402
from mpr_mpc.planning.visualization import draw_refinement_overlay  # noqa: E402
from mpr_mpc.tdmpc2.core import MPRTDMPC2  # noqa: E402


def _wandb_run(raw_cfg):
    if not bool(raw_cfg.enable_wandb):
        return None
    import wandb

    name = raw_cfg.wandb_run_name or f"eval_mpr_mpc_seed{raw_cfg.seed}"
    return wandb.init(
        project=str(raw_cfg.wandb_project),
        entity=raw_cfg.wandb_entity,
        name=str(name),
        config=OmegaConf.to_container(raw_cfg, resolve=True),
        job_type="evaluation",
    )


def _scenarios(cfg) -> np.ndarray:
    simulator = cfg.metadrive["simulator"]
    start = int(simulator["start_seed"])
    count = int(simulator["num_scenarios"])
    episodes = int(cfg.eval_episodes)
    rng = np.random.default_rng(int(cfg.seed))
    return rng.choice(
        np.arange(start, start + count), size=episodes, replace=episodes > count
    )


def _render(env, result, episode: int, scenario: int, step: int) -> None:
    text = {"episode": episode, "scenario": scenario, "step": step}
    if result is not None:
        text.update(
            {
                "coarse d/v/T": (
                    f"{result.coarse_path.target_d:+.2f}/"
                    f"{result.coarse_path.target_speed:.2f}/"
                    f"{result.coarse_path.horizon:.2f}"
                ),
                "residual d/v": "/".join(
                    f"{float(value):+.2f}" for value in result.residual.cpu()
                ),
                "coarse/final J": (
                    f"{result.metrics['mpr/coarse_value']:.2f}/"
                    f"{result.metrics['mpr/final_value']:.2f}"
                ),
            }
        )
    env.unwrapped_metadrive.render(text=text)
    env.unwrapped_metadrive.render(
        text=text,
        mode="topdown",
        window=True,
        screen_record=False,
        screen_size=(720, 720),
        film_size=(1200, 1200),
        scaling=5,
        target_agent_heading_up=True,
        draw_target_vehicle_trajectory=True,
    )
    if result is not None:
        draw_refinement_overlay(env.unwrapped_metadrive, result)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA for TD-MPC2.")
    raw_cfg.compile = False
    raw_cfg.save_agent = False
    raw_cfg.save_csv = False
    cfg = parse_cfg(raw_cfg)
    set_seed(int(cfg.seed))
    checkpoint = Path(str(cfg.checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    env = make_env(cfg)
    run = None
    summaries = []
    try:
        run = _wandb_run(raw_cfg)
        tdmpc_agent = MPRTDMPC2(cfg)
        lattice_config = dict(FRENET_DEFAULT_CONFIG)
        lattice_config.update(dict(cfg.lattice))
        planner = MPRMPCPlanner(
            cfg,
            tdmpc_agent,
            MetaDriveFrenetController(lattice_config),
            action_space=env.action_space,
        )
        agent = MPRMPCAgent(cfg, tdmpc_agent, planner, env)
        agent.load(checkpoint)
        agent.eval()
        scenes = _scenarios(cfg)
        print(f"MPR-MPC enabled: {planner.enabled}; scenarios: {scenes.tolist()}")
        for episode, scenario in enumerate(scenes, start=1):
            obs = env.reset(seed=int(scenario))
            done = False
            step = 0
            episode_return = 0.0
            episode_cost = 0.0
            episode_base_reward = 0.0
            episode_risk_cost = 0.0
            episode_risk_penalty = 0.0
            info = {}
            while not done:
                action = agent.act(obs, t0=step == 0, eval_mode=True)
                obs, reward, done, info = env.step(action)
                step += 1
                episode_return += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                episode_base_reward += float(info.get("metadrive_reward", 0.0))
                episode_risk_cost += float(info.get("risk_field_cost", 0.0))
                episode_risk_penalty += float(
                    info.get("risk_field_reward_penalty", 0.0)
                )
                if bool(cfg.metadrive["simulator"].get("use_render", False)):
                    _render(env, agent.last_plan, episode, int(scenario), step)
            row = {
                "return": episode_return,
                "cost": episode_cost,
                "success": float(info.get("success", 0.0)),
                "length": step,
                "metadrive_reward": episode_base_reward,
                "risk_field_cost": episode_risk_cost,
                "risk_penalty": episode_risk_penalty,
                "collision": float(info.get("crash", 0.0)),
                "offroad": float(info.get("out_of_road", 0.0)),
                "route_completion": float(info.get("route_completion", 0.0)),
            }
            summaries.append(row)
            if run is not None:
                run.log(
                    {f"eval/{key}": value for key, value in row.items()},
                    step=episode,
                )
            print(
                f"[{episode:02d}/{len(scenes):02d}] seed={int(scenario)} "
                f"R={episode_return:.2f} C={episode_cost:.2f} "
                f"S={row['success']:.0f} L={step}"
            )
    finally:
        env.close()
        if run is not None:
            if summaries:
                for key in summaries[0]:
                    run.summary[f"eval/mean_{key}"] = float(
                        np.mean([row[key] for row in summaries])
                    )
            run.finish()
    if summaries:
        print(
            f"Summary: return={np.mean([x['return'] for x in summaries]):.3f} "
            f"cost={np.mean([x['cost'] for x in summaries]):.3f} "
            f"success={np.mean([x['success'] for x in summaries]):.3f}"
        )


if __name__ == "__main__":
    main()
