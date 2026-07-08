"""Evaluator: load a checkpoint, run N episodes across the four tasks, report metrics.

Section 5.3's generalization protocol is directly supported: pass a list of team sizes
via `--team-sizes` and each is evaluated independently.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dh_marl.comm.diffusion_channel import ChannelConfig, DiffusionChannel
from dh_marl.env.tasks import TASK_TYPES
from dh_marl.env.vascular_env import EnvConfig, VascularSwarmEnv
from dh_marl.env.vessel_geometry import VesselScene
from dh_marl.eval.metrics import team_success_rate, collision_rate, coordination_efficiency
from dh_marl.models.critic import CriticConfig, GraphAttentionCritic
from dh_marl.models.layers import AttentionPool
from dh_marl.models.message_encoder import MessageEncoder, MessageEncoderConfig
from dh_marl.models.policy import PerAgentPolicy, PolicyConfig
from dh_marl.train.normalizer import RunningMeanStd
from dh_marl.train.rollout import collect_rollout


@dataclass
class EvalResult:
    task: str
    team_size: int
    n_episodes: int
    success_rate: float
    collision_rate: float
    coord_efficiency: float


class Evaluator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ):
        self.device = device
        # weights_only=False because our checkpoints legitimately contain numpy arrays
        # for the running-mean-std normalizer state. Only load checkpoints you produced.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.trainer_config = ckpt.get("trainer_config", {})

        # build modules from the saved config
        obs_dim_placeholder = self._infer_obs_dim()
        self.policy = PerAgentPolicy(PolicyConfig(obs_dim=obs_dim_placeholder)).to(device)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic = GraphAttentionCritic(CriticConfig()).to(device)
        try:
            self.critic.load_state_dict(ckpt["critic"])
        except RuntimeError:
            pass    # baseline critics have a different key set; skip
        self.message = MessageEncoder(
            MessageEncoderConfig(obs_dim=obs_dim_placeholder),
        ).to(device)
        self.message.load_state_dict(ckpt["message"])

        self.recv_pool = AttentionPool(d_query=16, d_msg=16).to(device)
        if "recv_pool" in ckpt:
            self.recv_pool.load_state_dict(ckpt["recv_pool"])

        self.obs_norm = RunningMeanStd(shape=(obs_dim_placeholder,))
        norm = ckpt.get("obs_norm", {})
        if "mean" in norm:
            self.obs_norm.mean = np.asarray(norm["mean"])
            self.obs_norm.var = np.asarray(norm["var"])
            self.obs_norm.count = float(norm.get("count", 1.0))

        # env prototype
        self.env_cfg = EnvConfig()

    def _infer_obs_dim(self) -> int:
        # 7 + 3 + 10 + 20 + 16 + 1 = 57 (default channel d_msg=16, n_ray=20)
        return 57

    def evaluate(
        self,
        task: str,
        team_size: int,
        n_episodes: int = 20,
        steps_per_episode: int = 1000,
    ) -> EvalResult:
        self.env_cfg.n_agents = team_size
        self.env_cfg.task_name = task
        if task == "BBT":
            self.env_cfg.scene = VesselScene.with_bottleneck()
        elif task == "DSM":
            self.env_cfg.scene = VesselScene.with_stenoses()
        else:
            self.env_cfg.scene = VesselScene()
        env = VascularSwarmEnv(self.env_cfg)
        channel = DiffusionChannel(ChannelConfig())
        channel.attach_receiver_pool(self.recv_pool)

        successes: list[bool] = []
        collisions: list[int] = []
        lengths: list[int] = []
        team_returns: list[float] = []

        for _ in range(n_episodes):
            r = collect_rollout(
                env, self.policy, self.message, channel,
                n_steps=steps_per_episode,
                device=self.device,
                obs_normalizer=self.obs_norm,
            )
            successes.append(r.task_success)
            collisions.append(r.collision_events)
            lengths.append(r.length)
            team_returns.append(float(r.rewards_team.sum()))

        return EvalResult(
            task=task,
            team_size=team_size,
            n_episodes=n_episodes,
            success_rate=team_success_rate(successes),
            collision_rate=collision_rate(collisions, lengths),
            coord_efficiency=coordination_efficiency(team_returns, ideal_return=50.0),
        )


def evaluate_checkpoint(
    checkpoint: str | Path,
    tasks: list[str] = list(TASK_TYPES),
    team_sizes: list[int] = [4, 8, 16, 32],
    n_episodes: int = 20,
    device: str = "cpu",
) -> list[EvalResult]:
    evaluator = Evaluator(checkpoint, device=device)
    out = []
    for task in tasks:
        for n in team_sizes:
            res = evaluator.evaluate(task, n, n_episodes=n_episodes)
            out.append(res)
            print(
                f"[{task}] N={n:3d}  success={res.success_rate:.3f}  "
                f"coll={res.collision_rate:.3f}  eff={res.coord_efficiency:.3f}"
            )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_TYPES))
    parser.add_argument("--team-sizes", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = evaluate_checkpoint(
        args.checkpoint, tasks=args.tasks, team_sizes=args.team_sizes,
        n_episodes=args.n_episodes, device=args.device,
    )
    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)


if __name__ == "__main__":
    main()
