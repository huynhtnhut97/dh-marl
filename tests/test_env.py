"""Integration-style tests for the vascular environment and per-task rewards."""

from __future__ import annotations

import numpy as np

from dh_marl.env.tasks import TASK_TYPES
from dh_marl.env.vascular_env import ACTION_DELTAS, EnvConfig, VascularSwarmEnv
from dh_marl.env.vessel_geometry import VesselScene


def _make(task: str, n: int = 8) -> VascularSwarmEnv:
    cfg = EnvConfig()
    cfg.task_name = task
    cfg.n_agents = n
    if task == "BBT":
        cfg.scene = VesselScene.with_bottleneck()
    elif task == "DSM":
        cfg.scene = VesselScene.with_stenoses()
    return VascularSwarmEnv(cfg)


def test_env_reset_and_obs_shape_for_all_tasks():
    for task in TASK_TYPES:
        env = _make(task, n=6)
        state = env.reset()
        assert state["obs"].shape == (6, env.obs_dim)


def test_env_step_returns_expected_keys():
    env = _make("MSDD", n=4)
    env.reset()
    actions = np.zeros(4, dtype=np.int64)
    out = env.step(actions)
    for k in ("obs", "reward_per_agent", "reward_team", "done", "info"):
        assert k in out
    assert out["obs"].shape == (4, env.obs_dim)
    assert out["reward_per_agent"].shape == (4,)


def test_env_positions_stay_finite():
    env = _make("MSDD", n=6)
    env.reset()
    for _ in range(20):
        actions = np.random.randint(0, 10, size=6)
        out = env.step(actions.astype(np.int64))
        assert np.all(np.isfinite(env.positions))


def test_action_deltas_are_unit_or_zero():
    for a in range(10):
        norm = float(np.linalg.norm(ACTION_DELTAS[a]))
        assert norm in (0.0, 1.0), (a, norm)


def test_bbt_task_success_when_all_traverse():
    env = _make("BBT", n=4)
    env.reset()
    # teleport all agents past the bottleneck and step once
    x_neck = env.scene.bottleneck_x + 1e-4
    env.positions[:, 0] = x_neck + 1e-4
    env.positions[:, 1] = 0.0
    env.positions[:, 2] = 0.0
    out = env.step(np.zeros(4, dtype=np.int64))
    assert out["info"]["success"] is True
