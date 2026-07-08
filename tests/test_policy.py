"""Unit tests for PerAgentPolicy: shapes, risk-modulated epsilon schedule, sampling."""

from __future__ import annotations

import torch

from dh_marl.models.policy import PerAgentPolicy, PolicyConfig


def test_policy_forward_shapes():
    cfg = PolicyConfig(obs_dim=57)
    policy = PerAgentPolicy(cfg)
    obs = torch.randn(8, 57)
    z, logits, risk_logit = policy(obs)
    assert z.shape == (8, cfg.embed_dim)
    assert logits.shape == (8, 10)
    assert risk_logit.shape == (8,)


def test_epsilon_schedule_monotonic_in_risk():
    cfg = PolicyConfig(obs_dim=57, eps_min=0.01, eps_max=1.0, lambda_eps=2.0)
    policy = PerAgentPolicy(cfg)
    risk = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    eps = policy.epsilon(risk)
    # eps decreases as risk increases (Equation 4)
    assert all(eps[i] > eps[i + 1] for i in range(len(risk) - 1))
    assert torch.isclose(eps[0], torch.tensor(cfg.eps_max), atol=1e-4)


def test_sample_action_shapes_and_types():
    cfg = PolicyConfig(obs_dim=57)
    policy = PerAgentPolicy(cfg)
    obs = torch.randn(4, 57)
    out = policy.sample_action(obs)
    assert out["action"].shape == (4,)
    assert out["action"].dtype == torch.long
    assert out["log_prob"].shape == (4,)
    assert out["entropy"].shape == (4,)
    assert out["risk"].shape == (4,)
    assert out["eps"].shape == (4,)
    assert out["logits"].shape == (4, 10)
    assert out["embed"].shape == (4, cfg.embed_dim)


def test_greedy_matches_argmax():
    cfg = PolicyConfig(obs_dim=57)
    policy = PerAgentPolicy(cfg)
    obs = torch.randn(4, 57)
    z, logits, _ = policy(obs)
    out = policy.sample_action(obs, greedy=True)
    assert torch.equal(out["action"], logits.argmax(dim=-1))
