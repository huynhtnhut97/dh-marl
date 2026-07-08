"""Graph attention critic with hemodynamic-aware edge priors.

Implements the two-layer GAT critic from Section 4.4 and Table 4:
- Node features are the trunk embedding phi(o_i) concatenated with a 128-dim one-hot
  action encoding.
- Edge weights come from the interaction kernel of Equation (5). The critic reads them
  as a prior on attention logits and continues to refine them via standard GAT attention.
- Two value heads: a joint Q(s, a) and per-agent V^i.

Design note: to keep this repo pure-PyTorch (no torch_geometric hard dependency for the
critic path — PyG is still a project requirement, but the critic can be swapped for a
PyG `GATv2Conv` if desired), the GAT layers here operate on dense [N, N] adjacencies.
Micro-robot swarm sizes stay well below the point where sparse propagation becomes
necessary (paper's largest deployment is N=32; N^2 = 1024, trivial for dense ops).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dh_marl.models.layers import MLP, GATLayer


@dataclass
class CriticConfig:
    obs_embed_dim: int = 128
    action_dim: int = 10
    action_embed_dim: int = 128
    node_dim: int = 128
    gat_hidden: int = 128
    gat_heads: int = 4
    value_head_hidden: int = 128


class GraphAttentionCritic(nn.Module):
    """Two-layer GAT producing joint Q(s,a) and per-agent V^i(s).

    Layer 1: concat aggregation over heads (output width heads * gat_hidden).
    Layer 2: mean aggregation (output width gat_hidden).
    Two-layer MLP value heads on top.
    """

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.action_encoder = nn.Linear(cfg.action_dim, cfg.action_embed_dim)
        node_in = cfg.obs_embed_dim + cfg.action_embed_dim
        self.node_encoder = nn.Linear(node_in, cfg.node_dim)

        self.gat1 = GATLayer(
            in_dim=cfg.node_dim,
            out_dim=cfg.gat_hidden,
            heads=cfg.gat_heads,
            aggregate="concat",
        )
        self.gat2 = GATLayer(
            in_dim=cfg.gat_hidden * cfg.gat_heads,
            out_dim=cfg.gat_hidden,
            heads=cfg.gat_heads,
            aggregate="mean",
        )
        self.norm1 = nn.LayerNorm(cfg.gat_hidden * cfg.gat_heads)
        self.norm2 = nn.LayerNorm(cfg.gat_hidden)

        self.q_head = MLP(cfg.gat_hidden, cfg.value_head_hidden, 1, n_layers=2)
        self.v_head = MLP(cfg.gat_hidden, cfg.value_head_hidden, 1, n_layers=2)

    def _forward_graph(
        self,
        obs_embed: torch.Tensor,       # [N, obs_embed_dim]
        action: torch.Tensor,           # [N] long
        adj: torch.Tensor,              # [N, N] bool
        edge_prior: torch.Tensor,       # [N, N] non-negative
    ) -> torch.Tensor:
        """Return per-agent node embeddings after two GAT layers."""
        a_onehot = F.one_hot(action, num_classes=self.cfg.action_dim).to(obs_embed.dtype)
        a_emb = self.action_encoder(a_onehot)                    # [N, action_embed_dim]
        node_in = torch.cat([obs_embed, a_emb], dim=-1)          # [N, .]
        h = self.node_encoder(node_in)                            # [N, node_dim]
        h = F.gelu(self.norm1(self.gat1(h, adj, edge_prior)))
        h = F.gelu(self.norm2(self.gat2(h, adj, edge_prior)))
        return h                                                  # [N, gat_hidden]

    def forward(
        self,
        obs_embed: torch.Tensor,
        action: torch.Tensor,
        adj: torch.Tensor,
        edge_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return joint Q(s,a) as the mean per-node Q, and per-agent V^i.

        Returns:
            q_joint: [] scalar tensor.
            v_per_agent: [N] tensor.
        """
        h = self._forward_graph(obs_embed, action, adj, edge_prior)
        q_i = self.q_head(h).squeeze(-1)                          # [N]
        v_i = self.v_head(h).squeeze(-1)                          # [N]
        q_joint = q_i.mean()
        return q_joint, v_i

    def q_value_batched(
        self,
        obs_embed: torch.Tensor,       # [T, N, obs_embed_dim]
        action: torch.Tensor,           # [T, N] long
        adj: torch.Tensor,              # [T, N, N] bool
        edge_prior: torch.Tensor,       # [T, N, N]
    ) -> torch.Tensor:
        """Sequential over T (dense-adjacency Q; T is typically small in a minibatch).

        Returns q_joint [T].
        """
        t = obs_embed.size(0)
        out = []
        for k in range(t):
            q, _ = self.forward(obs_embed[k], action[k], adj[k], edge_prior[k])
            out.append(q)
        return torch.stack(out, dim=0)
