"""Unit tests for GraphAttentionCritic: forward, backward, and shape invariance."""

from __future__ import annotations

import torch

from dh_marl.models.critic import CriticConfig, GraphAttentionCritic


def _mk_inputs(n: int, embed_dim: int = 128):
    obs_embed = torch.randn(n, embed_dim)
    action = torch.randint(0, 10, size=(n,))
    adj = torch.rand(n, n) < 0.5
    edge_prior = torch.rand(n, n) * adj.to(torch.float32)
    return obs_embed, action, adj, edge_prior


def test_critic_forward_shapes():
    critic = GraphAttentionCritic(CriticConfig())
    for n in (2, 4, 8, 16):
        obs_embed, action, adj, edge_prior = _mk_inputs(n)
        q_joint, v_i = critic(obs_embed, action, adj, edge_prior)
        assert q_joint.shape == ()
        assert v_i.shape == (n,)


def test_critic_backward_flows_to_all_inputs():
    critic = GraphAttentionCritic(CriticConfig())
    obs_embed, action, adj, edge_prior = _mk_inputs(6)
    obs_embed.requires_grad_(True)
    edge_prior.requires_grad_(True)
    q_joint, _ = critic(obs_embed, action, adj, edge_prior)
    q_joint.backward()
    assert obs_embed.grad is not None
    assert edge_prior.grad is not None


def test_critic_is_permutation_covariant():
    """Permuting the input node order permutes the output V_i in the same way."""
    critic = GraphAttentionCritic(CriticConfig())
    n = 5
    obs_embed, action, adj, edge_prior = _mk_inputs(n)

    perm = torch.randperm(n)
    obs_embed_p = obs_embed[perm]
    action_p = action[perm]
    adj_p = adj[perm][:, perm]
    edge_prior_p = edge_prior[perm][:, perm]

    critic.eval()
    with torch.no_grad():
        _, v = critic(obs_embed, action, adj, edge_prior)
        _, v_p = critic(obs_embed_p, action_p, adj_p, edge_prior_p)

    assert torch.allclose(v[perm], v_p, atol=1e-5)
