"""Per-agent policy with a flow-disturbance risk head.

Implements Equations (2) and (3): a shared MLP trunk phi produces an embedding used by
- a linear policy head over 10 discrete primitives (six translational, four rotational),
- a scalar risk head passed through sigmoid to yield r_t^i in [0, 1].

The risk score modulates the epsilon-greedy exploration rate via Equation (4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from dh_marl.models.layers import MLP


@dataclass
class PolicyConfig:
    obs_dim: int              # dimension of Equation (1) observation
    action_dim: int = 10
    hidden_dim: int = 256
    embed_dim: int = 128
    n_trunk_layers: int = 3
    eps_min: float = 0.01
    eps_max: float = 1.0
    lambda_eps: float = 2.0


class PerAgentPolicy(nn.Module):
    """Shared-parameter policy trunk with policy and risk output heads.

    Parameter sharing across agents follows Lowe et al. (2017) and Yu et al. (2022); a
    single copy of this module is applied to every agent's observation.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = MLP(cfg.obs_dim, cfg.hidden_dim, cfg.embed_dim, n_layers=cfg.n_trunk_layers)
        self.policy_head = nn.Linear(cfg.embed_dim, cfg.action_dim)
        self.risk_head = nn.Linear(cfg.embed_dim, 1)

    # -- forward primitives ------------------------------------------------

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the trunk embedding phi(o)."""
        return self.trunk(obs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (embedding, action logits, risk logit).

        Args:
            obs: [B, obs_dim] tensor.
        """
        z = self.encode(obs)
        logits = self.policy_head(z)
        risk_logit = self.risk_head(z).squeeze(-1)
        return z, logits, risk_logit

    # -- action selection --------------------------------------------------

    def risk_score(self, risk_logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(risk_logit)

    def epsilon(self, risk: torch.Tensor) -> torch.Tensor:
        """Equation (4): risk-modulated exploration rate.

        High risk -> risk score close to 1 -> exp(-lambda * risk) small -> epsilon
        close to eps_min (exploit); low risk -> epsilon close to eps_max (explore).
        """
        c = self.cfg
        return c.eps_min + (c.eps_max - c.eps_min) * torch.exp(-c.lambda_eps * risk)

    def sample_action(
        self,
        obs: torch.Tensor,
        eps_override: Optional[float] = None,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample a discrete action with the risk-modulated epsilon-greedy schedule.

        Returns dict with keys:
            action:  [B] long
            log_prob:[B] float, log pi(a|o) under the *base* Categorical policy
                     (not the epsilon-greedy mixture); this is what PPO's importance
                     ratio uses.
            entropy: [B] float, base policy entropy.
            risk:    [B] float
            eps:     [B] float
            logits:  [B, A] base policy logits (needed by the critic for one-hot inputs)
            embed:   [B, embed_dim]
        """
        z, logits, risk_logit = self.forward(obs)
        dist = Categorical(logits=logits)
        risk = self.risk_score(risk_logit)
        eps = self.epsilon(risk) if eps_override is None else torch.full_like(risk, eps_override)

        if greedy:
            action = logits.argmax(dim=-1)
        else:
            # epsilon-greedy mixture: with prob eps, sample uniform; else, sample from pi.
            uniform_choice = torch.randint(0, self.cfg.action_dim, size=risk.shape, device=obs.device)
            policy_choice = dist.sample()
            take_uniform = (torch.rand_like(risk) < eps)
            action = torch.where(take_uniform, uniform_choice, policy_choice)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "risk": risk,
            "eps": eps,
            "logits": logits,
            "embed": z,
        }

    # -- auxiliary losses --------------------------------------------------

    def risk_bce_loss(
        self,
        risk_logit: torch.Tensor,
        turbulence_label: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy auxiliary loss against simulator-provided labels.

        The label is a scalar in [0, 1] summarising local flow disturbance (e.g. the
        mean of the 10-directional tau vector); the head is trained to predict it.
        """
        return F.binary_cross_entropy_with_logits(risk_logit, turbulence_label)
