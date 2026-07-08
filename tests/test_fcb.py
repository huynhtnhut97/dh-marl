"""Tests for the flow-following counterfactual advantage."""

from __future__ import annotations

import torch

from dh_marl.models.critic import CriticConfig, GraphAttentionCritic
from dh_marl.train.fcb import compute_fcb_advantage


def test_fcb_advantage_shape():
    critic = GraphAttentionCritic(CriticConfig())
    n, embed_dim = 6, 128
    obs_embed = torch.randn(n, embed_dim)
    action = torch.randint(0, 10, size=(n,))
    adj = torch.rand(n, n) < 0.5
    prior = torch.rand(n, n) * adj.float()
    adv = compute_fcb_advantage(critic, obs_embed, action, adj, prior)
    assert adv.shape == (n,)


def test_fcb_advantage_zero_when_i_has_no_edges():
    """If agent i has no neighbours in the interaction graph, its FCB advantage should
    depend only on the action-encoding contribution to its own Q term."""
    critic = GraphAttentionCritic(CriticConfig())
    critic.eval()
    n = 4
    obs_embed = torch.randn(n, 128)
    action = torch.zeros(n, dtype=torch.long)
    adj = torch.eye(n, dtype=torch.bool)
    prior = torch.zeros(n, n)
    with torch.no_grad():
        adv = compute_fcb_advantage(critic, obs_embed, action, adj, prior)
    # When the realized action already equals the passive action index and the graph is
    # trivial, the advantage should be numerically zero for every agent.
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5)
