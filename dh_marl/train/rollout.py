"""Rollout data container and single-worker collector.

A `Rollout` is the T-step trajectory of a single environment for a single team of N
agents, holding the per-step observations, actions, log-probs, rewards, and the graph
information (adjacency, edge-prior, task tag) needed by the critic.

Parallelism: the paper uses four parallel workers per training iteration. In this
reference implementation, we stack rollouts from independent workers in `collect_rollout`.
The default is one worker; increase with `--workers` in the training script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from dh_marl.env.vascular_env import VascularSwarmEnv


@dataclass
class Rollout:
    obs: np.ndarray                           # [T, N, obs_dim]
    actions: np.ndarray                        # [T, N] long
    log_probs: np.ndarray                      # [T, N] float
    rewards_per_agent: np.ndarray              # [T, N] float
    rewards_team: np.ndarray                   # [T] float
    dones: np.ndarray                          # [T] bool
    adjacencies: np.ndarray                    # [T, N, N] bool
    edge_priors: np.ndarray                    # [T, N, N] float
    embeddings: np.ndarray                     # [T, N, embed_dim] float
    messages_sent: np.ndarray                  # [T, N, d_msg] float
    task_success: bool = False
    collision_events: int = 0
    n_agents: int = 0

    @property
    def length(self) -> int:
        return self.obs.shape[0]


@torch.no_grad()
def collect_rollout(
    env: VascularSwarmEnv,
    policy,                          # PerAgentPolicy
    message_encoder,                 # MessageEncoder
    diffusion_channel,               # DiffusionChannel
    n_steps: int,
    device: str = "cpu",
    obs_normalizer=None,              # RunningMeanStd | None
) -> Rollout:
    """Collect one rollout of exactly n_steps or until the episode ends."""
    diffusion_channel.reset()
    state = env.reset()
    obs_np = state["obs"]                                     # [N, obs_dim]

    obs_buf, act_buf, lp_buf = [], [], []
    ra_buf, rt_buf, done_buf = [], [], []
    adj_buf, prior_buf, emb_buf, msg_buf = [], [], [], []

    collisions = 0
    success = False

    for _ in range(n_steps):
        # normalize obs for the network
        obs_norm = obs_normalizer.normalize(obs_np) if obs_normalizer is not None else obs_np
        obs_t = torch.from_numpy(obs_norm).to(device)
        step = policy.sample_action(obs_t)
        action = step["action"].cpu().numpy()
        log_prob = step["log_prob"].cpu().numpy()
        embed = step["embed"].cpu().numpy()

        # send messages through the channel
        msg = message_encoder(obs_t, hard=True).detach().cpu().numpy()
        recv = diffusion_channel.transmit(msg, env.positions, env.scene.body_diameter)
        env.set_incoming_messages(recv)

        # step env
        result = env.step(action.astype(np.int64))
        info = result["info"]

        obs_buf.append(obs_norm.copy())
        act_buf.append(action.astype(np.int64))
        lp_buf.append(log_prob)
        ra_buf.append(result["reward_per_agent"])
        rt_buf.append(float(result["reward_team"]))
        done_buf.append(result["done"])
        adj_buf.append(info["adjacency"].copy())
        prior_buf.append(info["edge_prior"].copy())
        emb_buf.append(embed)
        msg_buf.append(msg)

        if result["reward_per_agent"].min() < -0.5:
            collisions += 1

        obs_np = result["obs"]
        if result["done"]:
            success = bool(info.get("success", False))
            break

    return Rollout(
        obs=np.stack(obs_buf, axis=0),
        actions=np.stack(act_buf, axis=0),
        log_probs=np.stack(lp_buf, axis=0),
        rewards_per_agent=np.stack(ra_buf, axis=0),
        rewards_team=np.array(rt_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=bool),
        adjacencies=np.stack(adj_buf, axis=0),
        edge_priors=np.stack(prior_buf, axis=0),
        embeddings=np.stack(emb_buf, axis=0),
        messages_sent=np.stack(msg_buf, axis=0),
        task_success=success,
        collision_events=collisions,
        n_agents=env.n_agents,
    )


def gae_advantages(
    rewards: np.ndarray,               # [T]
    values: np.ndarray,                # [T + 1]
    dones: np.ndarray,                 # [T]
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation. Returns (advantages [T], returns [T])."""
    t = rewards.shape[0]
    adv = np.zeros(t, dtype=np.float32)
    last = 0.0
    for k in reversed(range(t)):
        mask = 1.0 - float(dones[k])
        delta = rewards[k] + gamma * values[k + 1] * mask - values[k]
        last = delta + gamma * lam * mask * last
        adv[k] = last
    ret = adv + values[:-1]
    return adv, ret
