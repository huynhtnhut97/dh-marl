"""Centralized MAPPO baseline.

Standard MAPPO controller (Yu et al., 2022): decentralized policy at execution, but a
centralized value function that consumes all agents' observations concatenated together.
No graph attention, no counterfactual credit assignment. Communication is enabled
(consistent with the paper's phrasing that MAPPO gets "the same swarm-aware state at
training time"), but the flow-following counterfactual is disabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dh_marl.train.trainer import Trainer, TrainerConfig
from dh_marl.models.layers import MLP


class _CentralizedValueCritic(nn.Module):
    """Concatenated-observation MLP value function producing per-agent V and joint Q."""

    def __init__(self, embed_dim: int = 128, hidden: int = 128, max_n: int = 32):
        super().__init__()
        # per-agent V head on the local embedding
        self.v_head = MLP(embed_dim, hidden, 1, n_layers=2)
        # joint Q head on the mean-pooled team embedding
        self.q_head = MLP(embed_dim, hidden, 1, n_layers=2)

    def forward(self, obs_embed, action, adj, edge_prior):
        v_i = self.v_head(obs_embed).squeeze(-1)
        team = obs_embed.mean(dim=0, keepdim=True)
        q_joint = self.q_head(team).squeeze(-1).squeeze(-1)
        return q_joint, v_i


class MAPPOTrainer(Trainer):
    def __init__(self, cfg: TrainerConfig):
        cfg.ppo.use_fcb = False
        cfg.ppo.use_him_edges = False
        cfg.ppo.use_dc = True
        super().__init__(cfg)
        self.critic = _CentralizedValueCritic(embed_dim=self.policy.cfg.embed_dim).to(cfg.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": cfg.lr_policy},
                {"params": self.critic.parameters(), "lr": cfg.lr_critic},
                {"params": self.message.parameters(), "lr": cfg.lr_message},
                {"params": self.recv_pool.parameters(), "lr": cfg.lr_message},
            ],
            betas=(0.9, 0.999), eps=1e-8,
        )
