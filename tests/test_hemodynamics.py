"""Unit tests for the HIM kernels and the combined interaction force."""

from __future__ import annotations

import numpy as np
import pytest

from dh_marl.env.hemodynamics import (
    HIM, HIMConfig,
    wake_kernel, turbulence_kernel, lubrication_kernel,
)


def test_wake_kernel_is_zero_when_sender_is_downstream():
    d = np.array([[0.0, 2.0], [2.0, 0.0]])
    cos_theta = np.array([[0.0, -1.0], [-1.0, 0.0]])   # sender opposite direction
    w = wake_kernel(d, cos_theta, xi_w=5.0)
    assert np.all(w == 0.0)


def test_wake_kernel_positive_when_upstream():
    d = np.array([[0.0, 2.0]])
    cos_theta = np.array([[0.0, 1.0]])
    w = wake_kernel(d, cos_theta, xi_w=5.0)
    assert w[0, 1] > 0.0


def test_turbulence_kernel_decays_with_distance():
    d = np.array([[0.0, 1.0, 2.0, 5.0]])
    t = turbulence_kernel(d, xi_t=3.0)
    assert t[0, 1] > t[0, 2] > t[0, 3]
    assert t[0, 0] == pytest.approx(1.0)


def test_lubrication_kernel_diverges_at_contact_and_is_zero_far_away():
    d = np.array([[0.001, 0.5, 1.0, 5.0]])
    l = lubrication_kernel(d, xi_l=1.2, cutoff=3.0)
    assert l[0, 0] > l[0, 1] > l[0, 2] > 0
    assert l[0, 3] == 0.0


def test_him_force_reduces_to_zero_for_single_agent():
    him = HIM(HIMConfig())
    positions = np.zeros((1, 3), dtype=np.float32)
    velocities = np.zeros_like(positions)
    tau = np.zeros((1, 10), dtype=np.float32)
    f, k, _ = him.compute(positions, velocities, tau)
    assert f.shape == (1, 3)
    assert np.all(f == 0.0)
    assert k.shape == (1, 1)
    assert np.all(k == 0.0)


def test_him_lubrication_force_pushes_close_pair_apart():
    him = HIM(HIMConfig())
    bd = him.cfg.body_diameter
    positions = np.array([[0.0, 0.0, 0.0], [0.5 * bd, 0.0, 0.0]], dtype=np.float32)  # 0.5 bd apart
    velocities = np.zeros_like(positions)
    tau = np.zeros((2, 10), dtype=np.float32)
    f, _, _ = him.compute(positions, velocities, tau)
    # agent 0 should be pushed in -x, agent 1 in +x
    assert f[0, 0] < 0.0
    assert f[1, 0] > 0.0


def test_him_edge_graph_threshold():
    him = HIM(HIMConfig(edge_threshold=0.05))
    bd = him.cfg.body_diameter
    positions = np.array([[0.0, 0, 0], [1e6 * bd, 0, 0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    tau = np.zeros((2, 10), dtype=np.float32)
    _, kernel_sum, _ = him.compute(positions, velocities, tau)
    adj, _ = him.build_graph(kernel_sum)
    assert adj.sum() == 0     # very far apart -> no edges
