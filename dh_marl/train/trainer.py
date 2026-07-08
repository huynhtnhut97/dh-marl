"""Full training loop for DH-MARL (Algorithm 1).

Wires the environment, policy, critic, message encoder, diffusion channel, PPO update
with FCB advantage, running normalizers, and checkpointing. Supports task randomization
across the four benchmarks and swarm-size randomization at training time.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dh_marl.comm.diffusion_channel import ChannelConfig, DiffusionChannel
from dh_marl.env.hemodynamics import HIMConfig
from dh_marl.env.tasks import TASK_TYPES, RewardCoeffs
from dh_marl.env.vascular_env import EnvConfig, VascularSwarmEnv
from dh_marl.env.vessel_geometry import VesselScene
from dh_marl.models.critic import CriticConfig, GraphAttentionCritic
from dh_marl.models.layers import AttentionPool
from dh_marl.models.message_encoder import MessageEncoder, MessageEncoderConfig
from dh_marl.models.policy import PerAgentPolicy, PolicyConfig
from dh_marl.train.normalizer import RunningMeanStd
from dh_marl.train.ppo import PPOConfig, ppo_update
from dh_marl.train.rollout import collect_rollout
from dh_marl.utils.config import load_config
from dh_marl.utils.logging import get_logger, TensorboardLogger
from dh_marl.utils.seeding import seed_everything


@dataclass
class TrainerConfig:
    # top-level knobs
    total_episodes: int = 20_000
    steps_per_rollout: int = 256
    workers: int = 4                 # workers per iteration; each yields a rollout
    n_train: int = 8                 # N_train from Table 1
    swarm_size_jitter: bool = False   # if True, sample N uniformly in {6, 8, 10}
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # component-level configs
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig | None = None      # obs_dim filled in at init
    critic: CriticConfig = field(default_factory=CriticConfig)
    message: MessageEncoderConfig | None = None
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    # optimizer learning rates (Table 1)
    lr_policy: float = 5e-4
    lr_critic: float = 1e-3
    lr_message: float = 5e-4

    # infra
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/dhmarl"
    eval_every: int = 200            # iterations
    checkpoint_every: int = 500
    task_schedule: str = "random"    # 'random' or 'round_robin'

    # ablation switches
    ablation: dict[str, bool] = field(default_factory=dict)


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("trainer")
        self.tb = TensorboardLogger(self.out_dir / "tb")

        # env
        self.env = VascularSwarmEnv(cfg.env)
        obs_dim = self.env.obs_dim

        # models
        self.policy = PerAgentPolicy(
            cfg.policy or PolicyConfig(obs_dim=obs_dim),
        ).to(cfg.device)
        self.critic = GraphAttentionCritic(cfg.critic).to(cfg.device)
        self.message = MessageEncoder(
            cfg.message or MessageEncoderConfig(obs_dim=obs_dim, d_msg=cfg.channel.d_msg),
        ).to(cfg.device)

        # receiver attention pool for the diffusion channel
        self.recv_pool = AttentionPool(
            d_query=cfg.channel.d_msg, d_msg=cfg.channel.d_msg,
        ).to(cfg.device)
        self.channel = DiffusionChannel(cfg.channel, seed=cfg.seed)
        self.channel.attach_receiver_pool(self.recv_pool)

        # optimizer with per-module learning rates
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": cfg.lr_policy},
                {"params": self.critic.parameters(), "lr": cfg.lr_critic},
                {"params": self.message.parameters(), "lr": cfg.lr_message},
                {"params": self.recv_pool.parameters(), "lr": cfg.lr_message},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # normalizers
        self.obs_norm = RunningMeanStd(shape=(obs_dim,))
        self.ret_norm = RunningMeanStd(shape=())

        # bookkeeping
        self.iter_idx = 0
        self.best_success_rate = 0.0
        self.task_rr_idx = 0

    # ---- iteration loop --------------------------------------------------

    def _sample_task(self) -> str:
        if self.cfg.task_schedule == "round_robin":
            name = TASK_TYPES[self.task_rr_idx % len(TASK_TYPES)]
            self.task_rr_idx += 1
            return name
        return str(np.random.choice(TASK_TYPES))

    def _sample_n(self) -> int:
        if not self.cfg.swarm_size_jitter:
            return self.cfg.n_train
        return int(np.random.choice([6, 8, 10]))

    def train(self):
        cfg = self.cfg
        total_iters = max(1, cfg.total_episodes // (cfg.workers))
        self.log.info(f"starting training for {total_iters} iterations")

        for it in range(1, total_iters + 1):
            self.iter_idx = it

            rollouts = []
            successes = []
            for _ in range(cfg.workers):
                task_name = self._sample_task()
                n_agents = self._sample_n()
                # rebuild task without rebuilding the whole env (scene stays)
                if task_name == "BBT" and self.env.scene.bottleneck_x is None:
                    # BBT needs a bottleneck; hot-swap scene if not present
                    self.env.scene = VesselScene.with_bottleneck()
                else:
                    self.env.scene = cfg.env.scene
                self.env.cfg.scene = self.env.scene
                self.env.cfg.task_name = task_name
                self.env.cfg.n_agents = n_agents
                self.env.n = n_agents

                r = collect_rollout(
                    self.env,
                    self.policy,
                    self.message,
                    self.channel,
                    cfg.steps_per_rollout,
                    device=cfg.device,
                    obs_normalizer=self.obs_norm,
                )
                # update obs stats from the collected data
                self.obs_norm.update(r.obs.reshape(-1, r.obs.shape[-1]))
                rollouts.append(r)
                successes.append(int(r.task_success))

            # PPO update
            t0 = time.time()
            stats = ppo_update(
                self.policy,
                self.critic,
                self.message,
                self.optimizer,
                rollouts,
                self.obs_norm,
                self.ret_norm,
                cfg.ppo,
                device=cfg.device,
            )
            dt = time.time() - t0

            success_rate = float(np.mean(successes))
            stats.update(success_rate=success_rate, update_time_s=dt)
            self.log.info(
                f"iter {it}/{total_iters} success={success_rate:.3f} "
                f"pol={stats['policy_loss']:.3f} val={stats['value_loss']:.3f} "
                f"ent={stats['entropy']:.3f} kl={stats['kl']:.4f} dt={dt:.1f}s"
            )
            for k, v in stats.items():
                self.tb.scalar(k, v, it)

            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                self.save_checkpoint(self.out_dir / "best.pt")

            if it % cfg.checkpoint_every == 0:
                self.save_checkpoint(self.out_dir / f"ckpt_iter{it}.pt")

        self.save_checkpoint(self.out_dir / "final.pt")
        self.tb.close()
        self.log.info("training finished.")

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "message": self.message.state_dict(),
                "recv_pool": self.recv_pool.state_dict(),
                "obs_norm": {
                    "mean": self.obs_norm.mean,
                    "var": self.obs_norm.var,
                    "count": self.obs_norm.count,
                },
                "ret_norm": {
                    "mean": self.ret_norm.mean,
                    "var": self.ret_norm.var,
                    "count": self.ret_norm.count,
                },
                "iter": self.iter_idx,
                "best_success_rate": self.best_success_rate,
                "trainer_config": _dataclass_to_dict(self.cfg),
            },
            path,
        )
        self.log.info(f"checkpoint saved to {path}")


# ---- helpers -------------------------------------------------------------

def _dataclass_to_dict(x: Any) -> Any:
    """Recursively convert a nested dataclass tree to plain dicts (for YAML/JSON dump).

    numpy arrays are converted to lists so YAML round-trips cleanly.
    """
    if hasattr(x, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(getattr(x, k)) for k in x.__dataclass_fields__}
    if isinstance(x, dict):
        return {k: _dataclass_to_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_dataclass_to_dict(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ---- CLI entry -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--algo", type=str, default="dhmarl",
                        choices=["dhmarl", "ind_sa", "sa_la", "mappo", "coma_gnn"])
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = _dict_to_trainer_config(cfg_dict)
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    if args.algo == "dhmarl":
        Trainer(cfg).train()
        return

    from dh_marl.baselines import get_baseline_trainer
    baseline = get_baseline_trainer(args.algo, cfg)
    baseline.train()


def _dict_to_trainer_config(d: dict) -> TrainerConfig:
    """Manual coercion of nested-dict YAML into the TrainerConfig dataclass tree.

    Deliberately explicit (rather than `dacite` or Pydantic) so users can see exactly
    which knobs map where without diving into a schema library.
    """
    cfg = TrainerConfig()
    for k, v in d.items():
        if k in ("ppo", "critic", "channel", "env", "policy", "message", "ablation"):
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if "ppo" in d:
        cfg.ppo = PPOConfig(**{**asdict(cfg.ppo), **d["ppo"]})
    if "critic" in d:
        cfg.critic = CriticConfig(**{**asdict(cfg.critic), **d["critic"]})
    if "channel" in d:
        cfg.channel = ChannelConfig(**{**asdict(cfg.channel), **d["channel"]})
    if "env" in d:
        env_kwargs = {**asdict(cfg.env), **d["env"]}
        # rebuild HIM / RewardCoeffs / scene from sub-dicts if present
        if "him" in d["env"]:
            env_kwargs["him"] = HIMConfig(**{**asdict(cfg.env.him), **d["env"]["him"]})
        if "coeffs" in d["env"]:
            env_kwargs["coeffs"] = RewardCoeffs(**{**asdict(cfg.env.coeffs), **d["env"]["coeffs"]})
        # scene is not a dataclass with plain fields (has method factories); allow a preset name
        scene_spec = d["env"].get("scene", "default")
        if scene_spec == "with_bottleneck":
            env_kwargs["scene"] = VesselScene.with_bottleneck()
        elif scene_spec == "with_stenoses":
            env_kwargs["scene"] = VesselScene.with_stenoses()
        else:
            env_kwargs["scene"] = VesselScene()
        cfg.env = EnvConfig(**env_kwargs)
    if "ablation" in d:
        cfg.ablation = dict(d["ablation"])
        cfg.ppo.use_fcb = cfg.ablation.get("use_fcb", True)
        cfg.ppo.use_dc = cfg.ablation.get("use_dc", True)
        cfg.ppo.use_him_edges = cfg.ablation.get("use_him_edges", True)
    return cfg


if __name__ == "__main__":
    main()
