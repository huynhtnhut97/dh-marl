"""Tests for the message encoder, Gumbel quantizer, and diffusion channel."""

from __future__ import annotations

import numpy as np
import torch

from dh_marl.comm.diffusion_channel import ChannelConfig, DiffusionChannel
from dh_marl.models.layers import AttentionPool, GumbelQuantizer
from dh_marl.models.message_encoder import MessageEncoder, MessageEncoderConfig


def test_quantizer_hard_mode_outputs_on_levels():
    q = GumbelQuantizer(d_msg=8, bits=4)
    q.eval()
    x = torch.randn(3, 8)
    out = q(x)
    # every element must be one of the quantizer levels
    for value in out.flatten().tolist():
        assert min(abs(value - lvl.item()) for lvl in q.levels) < 1e-5


def test_quantizer_soft_mode_backpropagates():
    q = GumbelQuantizer(d_msg=4, bits=3)
    q.train()
    x = torch.randn(2, 4, requires_grad=True)
    out = q(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_message_encoder_forward_shape():
    cfg = MessageEncoderConfig(obs_dim=57, d_msg=16, bits=4)
    m = MessageEncoder(cfg)
    obs = torch.randn(5, 57)
    out = m(obs, hard=True)
    assert out.shape == (5, 16)


def test_channel_attenuation_decreases_with_distance():
    ch = DiffusionChannel(ChannelConfig(xi_d=4.0))
    dist_bd = np.array([[0.0, 1.0, 2.0, 10.0]])
    a = ch.attenuation_matrix(dist_bd)
    # column 0 corresponds to distance 0 which is the diagonal fixup -> zeroed
    assert a[0, 0] == 0.0
    assert a[0, 1] > a[0, 2] > a[0, 3]


def test_channel_transmit_shape():
    n, d = 6, 16
    ch = ChannelConfig(d_msg=d, xi_d=4.0)
    channel = DiffusionChannel(ch)
    pool = AttentionPool(d_query=d, d_msg=d)
    channel.attach_receiver_pool(pool)

    messages = np.random.randn(n, d).astype(np.float32)
    positions = np.random.uniform(-5e-4, 5e-4, size=(n, 3)).astype(np.float32)
    bd = 50e-6
    out = channel.transmit(messages, positions, bd)
    assert out.shape == (n, d)
