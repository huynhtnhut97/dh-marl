"""Shared building blocks: MLP trunk, GAT layer, attention pooling, Gumbel-softmax quantizer.

All modules use GELU activations and LayerNorm (not BatchNorm), matching the on-policy
setting described in Section 4.6 of the paper (correlated transitions in a rollout batch
make BatchNorm statistics unreliable).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Layer-normed MLP with GELU activations. Final layer is linear (no norm, no activation).

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden width shared across intermediate layers.
        out_dim: Output feature dimension.
        n_layers: Total number of Linear layers (must be >= 1). n_layers=1 collapses to a
            single linear projection with no hidden units.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 3):
        super().__init__()
        assert n_layers >= 1
        if n_layers == 1:
            self.net = nn.Linear(in_dim, out_dim)
            return
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class GATLayer(nn.Module):
    """Multi-head graph attention layer (Velickovic et al., 2018).

    Supports optional edge-weight priors: when `edge_weight_prior` is provided, its
    (log-)value is added to the attention logits, which lets the interaction-kernel
    values of Equation (5) initialize and continue to influence the critic's attention
    over training.

    Args:
        in_dim: Node feature dimension.
        out_dim: Output feature dimension per head (not multiplied by heads).
        heads: Number of attention heads.
        aggregate: 'concat' (layer 1) or 'mean' (layer 2), matching Table 4.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        aggregate: str = "concat",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert aggregate in {"concat", "mean"}
        self.heads = heads
        self.out_dim = out_dim
        self.aggregate = aggregate
        self.dropout = dropout
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,           # [N, in_dim]
        adj: torch.Tensor,          # [N, N] bool or {0,1}
        edge_weight_prior: Optional[torch.Tensor] = None,   # [N, N] non-negative
    ) -> torch.Tensor:
        n = x.size(0)
        h = self.W(x).view(n, self.heads, self.out_dim)          # [N, H, D]
        e_src = (h * self.a_src).sum(dim=-1)                     # [N, H]
        e_dst = (h * self.a_dst).sum(dim=-1)                     # [N, H]
        # broadcast to pairwise logits [N, N, H]
        e = e_src.unsqueeze(1) + e_dst.unsqueeze(0)
        e = F.leaky_relu(e, negative_slope=0.2)

        if edge_weight_prior is not None:
            # log(1 + w) is a numerically stable prior injection: zero prior contributes
            # zero logit, and larger interaction kernel values bias attention up smoothly.
            prior = torch.log1p(edge_weight_prior.clamp(min=0.0))    # [N, N]
            e = e + prior.unsqueeze(-1)

        # mask non-edges
        mask = adj.to(dtype=torch.bool)
        # every node attends to itself
        eye = torch.eye(n, dtype=torch.bool, device=x.device)
        mask = mask | eye
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = F.softmax(e, dim=1)                              # softmax over source
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # aggregate: sum_j alpha_{ij} h_j
        out = torch.einsum("ijh,jhd->ihd", alpha, h)             # [N, H, D]
        if self.aggregate == "concat":
            return out.reshape(n, self.heads * self.out_dim)
        return out.mean(dim=1)


class AttentionPool(nn.Module):
    """Single-head attention pooling for message aggregation at the receiver.

    Given per-neighbour message vectors and a receiver query vector, returns a weighted
    sum of the messages using scaled dot-product attention.
    """

    def __init__(self, d_query: int, d_msg: int):
        super().__init__()
        self.d = d_msg
        self.q_proj = nn.Linear(d_query, d_msg, bias=False)
        self.k_proj = nn.Linear(d_msg, d_msg, bias=False)
        self.scale = 1.0 / math.sqrt(d_msg)

    def forward(
        self,
        query: torch.Tensor,          # [B, d_query]
        messages: torch.Tensor,       # [B, K, d_msg]
        mask: Optional[torch.Tensor] = None,   # [B, K] bool, True = valid
    ) -> torch.Tensor:
        q = self.q_proj(query).unsqueeze(1)         # [B, 1, d]
        k = self.k_proj(messages)                    # [B, K, d]
        logits = (q * k).sum(dim=-1) * self.scale    # [B, K]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        # handle the empty-neighbourhood case: all-masked rows produce NaN under softmax;
        # substitute a zero pooled vector rather than propagating NaN.
        valid_row = mask.any(dim=1, keepdim=True) if mask is not None else None
        alpha = F.softmax(logits, dim=1)             # [B, K]
        pooled = (alpha.unsqueeze(-1) * messages).sum(dim=1)   # [B, d_msg]
        if valid_row is not None:
            pooled = pooled * valid_row.to(pooled.dtype)
        return pooled


class GumbelQuantizer(nn.Module):
    """B-bit scalar quantizer with a Gumbel-softmax relaxation for training.

    During training, each of `d_msg` channels is mapped to a soft distribution over
    2**B levels via Gumbel-softmax; gradients flow through the resulting expectation.
    At evaluation, the hard arg-max is returned (Section 4.6).

    Args:
        d_msg: Number of message channels (each quantized independently).
        bits: Bits per channel.
        levels_range: Tuple (lo, hi) defining the value range that levels are placed on.
        tau: Softmax temperature during training.
    """

    def __init__(
        self,
        d_msg: int,
        bits: int = 4,
        levels_range: tuple[float, float] = (-1.0, 1.0),
        tau: float = 1.0,
    ):
        super().__init__()
        self.d_msg = d_msg
        self.bits = bits
        self.n_levels = 2 ** bits
        self.tau = tau
        lo, hi = levels_range
        levels = torch.linspace(lo, hi, self.n_levels)   # [L]
        self.register_buffer("levels", levels)
        # per-channel projection into logits over the L levels
        self.proj = nn.Linear(1, self.n_levels)

    def forward(self, x: torch.Tensor, hard: Optional[bool] = None) -> torch.Tensor:
        """Quantize each channel of x independently.

        Args:
            x: [B, d_msg] pre-quantization continuous message.
            hard: If None, uses self.training. If True, arg-max; if False, soft.
        Returns:
            Quantized tensor of shape [B, d_msg] whose values lie on `self.levels`.
        """
        if hard is None:
            hard = not self.training
        b, d = x.shape
        assert d == self.d_msg
        logits = self.proj(x.view(b * d, 1))                   # [B*d, L]
        if hard:
            idx = logits.argmax(dim=-1)                         # [B*d]
            out = self.levels[idx]                              # [B*d]
        else:
            y = F.gumbel_softmax(logits, tau=self.tau, hard=False)   # [B*d, L]
            out = (y * self.levels.unsqueeze(0)).sum(dim=-1)         # [B*d]
        return out.view(b, d)
