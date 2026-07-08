"""Message encoder g_psi: shared MLP producing a d_c=16 message vector, then quantized.

The quantized encoding is what a sender transmits into the diffusion channel of
Equation (7). The channel model itself (distance-dependent attenuation, propagation
delay, additive noise) lives in `dh_marl.comm.diffusion_channel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dh_marl.models.layers import MLP, GumbelQuantizer


@dataclass
class MessageEncoderConfig:
    obs_dim: int
    d_msg: int = 16
    hidden_dim: int = 128
    n_layers: int = 2
    bits: int = 4
    tau: float = 1.0


class MessageEncoder(nn.Module):
    """g_psi: local observation -> continuous message -> per-channel B-bit quantization."""

    def __init__(self, cfg: MessageEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MLP(cfg.obs_dim, cfg.hidden_dim, cfg.d_msg, n_layers=cfg.n_layers)
        self.quantizer = GumbelQuantizer(d_msg=cfg.d_msg, bits=cfg.bits, tau=cfg.tau)

    def forward(self, obs: torch.Tensor, hard: bool | None = None) -> torch.Tensor:
        """Return the transmitted quantized message vector for each agent in the batch.

        Args:
            obs: [B, obs_dim] pre-normalized local observation.
            hard: If None, hard=True at eval and hard=False at train.
        """
        continuous = self.encoder(obs)
        quantized = self.quantizer(continuous, hard=hard)
        return quantized

    def bit_budget_regularizer(self, obs: torch.Tensor) -> torch.Tensor:
        """A soft channel-bottleneck penalty on encoder activations.

        Encourages the pre-quantization values to stay within the quantizer level range,
        which helps the arg-max encoding at evaluation match the soft encoding at
        training. Adds negligible cost in practice; hyperparameter defaults are
        set to a low weight in the trainer.
        """
        continuous = self.encoder(obs)
        return continuous.pow(2).mean()
