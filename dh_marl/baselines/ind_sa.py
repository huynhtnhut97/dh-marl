"""Independent Single-Agent baseline (IND-SA).

N independent single-agent policies with no communication and no inter-robot awareness
in the policy inputs. Concretely:
- messages_in is forced to zero at every step (no diffusion channel),
- the policy trunk is still shared across agents (weight-tied), matching the paper's
  "N independent instances" phrasing, which refers to the environment coupling: the
  agents do not exchange messages and the critic is an independent per-agent V head
  rather than a graph attention critic.

Implemented as a slim subclass of the main Trainer that overrides `_build_networks`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dh_marl.train.trainer import Trainer, TrainerConfig
from dh_marl.models.policy import PerAgentPolicy, PolicyConfig


class _IndependentValueCritic(nn.Module):
    """Per-agent value head; no graph, no cross-agent attention."""

    def __init__(self, embed_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, obs_embed, action, adj, edge_prior):
        # signature-compatible with GraphAttentionCritic; ignores graph inputs.
        v_i = self.v(obs_embed).squeeze(-1)
        q_joint = v_i.mean()
        return q_joint, v_i


class IndependentSingleAgentTrainer(Trainer):
    """Trainer for IND-SA: zero incoming messages and per-agent value critic."""

    def __init__(self, cfg: TrainerConfig):
        # force channel off before base init
        cfg.ppo.use_dc = False
        cfg.ppo.use_him_edges = False
        cfg.ppo.use_fcb = False
        super().__init__(cfg)
        # replace critic
        self.critic = _IndependentValueCritic(embed_dim=self.policy.cfg.embed_dim).to(cfg.device)
        # rebuild optimizer with the new critic
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": cfg.lr_policy},
                {"params": self.critic.parameters(), "lr": cfg.lr_critic},
                {"params": self.message.parameters(), "lr": cfg.lr_message},
                {"params": self.recv_pool.parameters(), "lr": cfg.lr_message},
            ],
            betas=(0.9, 0.999), eps=1e-8,
        )

    def _sample_task(self):
        # IND-SA sees no team reward and no messages; task sampling proceeds normally.
        return super()._sample_task()
