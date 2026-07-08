"""Diffusion-based constrained communication channel (Equation 7).

Given per-agent transmitted messages g_psi(o_i), the channel:
1. attenuates each pairwise transmission by D(d_ij, delta_ij) (distance-dependent),
2. applies a propagation delay proportional to d_ij (implemented as an integer step delay
   in the reduced-order model; messages older than the buffer horizon are dropped),
3. adds Gaussian noise with variance sigma^2,
4. aggregates at the receiver via attention pooling (see `dh_marl.models.layers.AttentionPool`).

The chemical interpretation of Table 3 governs default parameter values:
- attenuation length (in body diameters): xi_d = 4.0
- delay per body diameter: 1 step (proxy for advection time)
- noise sigma: 0.05
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from dh_marl.models.layers import AttentionPool


@dataclass
class ChannelConfig:
    d_msg: int = 16
    xi_d: float = 4.0                  # attenuation length in body diameters
    delay_per_bd: float = 1.0            # steps of delay per body diameter of separation
    delay_max: int = 8                   # buffer horizon
    noise_sigma: float = 0.05


class DiffusionChannel:
    """Pairwise-attenuated, delayed, noisy communication with attention-pool aggregation.

    The channel does not own an encoder: it takes already-encoded quantized message
    vectors and returns per-receiver aggregated message vectors ready for the next
    observation.
    """

    def __init__(self, cfg: ChannelConfig | None = None, seed: int = 0):
        self.cfg = cfg or ChannelConfig()
        self.rng = np.random.default_rng(seed)
        self._buffer: deque[np.ndarray] = deque(maxlen=self.cfg.delay_max + 1)
        # attention pool at the receiver (learned; owned externally in the trainer)
        self.pool_net: AttentionPool | None = None

    def attach_receiver_pool(self, pool: AttentionPool) -> None:
        self.pool_net = pool

    def reset(self) -> None:
        self._buffer.clear()

    # ---- physical channel transform --------------------------------------

    def attenuation_matrix(self, dist_bd: np.ndarray) -> np.ndarray:
        """D(d_ij) = exp(-d_ij / xi_d) in body-diameter units. Diagonal set to zero."""
        d = np.exp(-dist_bd / self.cfg.xi_d).astype(np.float32)
        np.fill_diagonal(d, 0.0)
        return d

    def delay_matrix(self, dist_bd: np.ndarray) -> np.ndarray:
        """Integer step-delay per pair, capped at delay_max."""
        d = np.round(dist_bd * self.cfg.delay_per_bd).astype(np.int32)
        return np.clip(d, 0, self.cfg.delay_max)

    def transmit(self, messages: np.ndarray, positions: np.ndarray, body_diameter: float) -> np.ndarray:
        """Return per-receiver aggregated messages given per-agent transmitted messages.

        Args:
            messages: [N, d_msg] transmitted (already-quantized) messages.
            positions: [N, 3] agent positions (metres).
            body_diameter: physical body diameter scale (metres).

        Returns:
            [N, d_msg] aggregated per-receiver message vectors.
        """
        n = messages.shape[0]
        dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) / body_diameter
        atten = self.attenuation_matrix(dist)                        # [N, N]
        delay = self.delay_matrix(dist)                              # [N, N]

        # push the current message into the buffer; index 0 is now, higher indices older
        self._buffer.appendleft(messages.astype(np.float32))
        # for each pair (i, j), fetch the message from j at time (now - delay[i, j])
        received = np.zeros((n, n, self.cfg.d_msg), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = int(delay[i, j])
                if d < len(self._buffer):
                    received[i, j] = self._buffer[d][j]
                else:
                    received[i, j] = 0.0

        received *= atten[..., None]
        received += self.rng.normal(0.0, self.cfg.noise_sigma, size=received.shape).astype(np.float32)

        # aggregate at each receiver
        if self.pool_net is None:
            # fallback: attenuation-weighted mean
            weights = atten / (atten.sum(axis=1, keepdims=True) + 1e-8)
            return (weights[..., None] * received).sum(axis=1).astype(np.float32)

        # pool via the learned attention head; query = mean-message as a stub embedding
        # (in the full trainer, `query` is the receiver's trunk embedding; see PPO step)
        with torch.no_grad():
            msgs_t = torch.from_numpy(received)                              # [N, N, d]
            query_t = msgs_t.mean(dim=1)                                     # [N, d]
            # mask out i==j
            mask = torch.from_numpy(~np.eye(n, dtype=bool))                  # [N, N]
            out = self.pool_net(query_t, msgs_t, mask)
        return out.detach().cpu().numpy().astype(np.float32)

    def transmit_with_query(
        self,
        messages: torch.Tensor,             # [N, d]
        positions: np.ndarray,
        body_diameter: float,
        query_embed: torch.Tensor,          # [N, d_query] receiver embedding
    ) -> torch.Tensor:
        """Torch-side transmit that keeps gradients wired through the attention pool.

        Called at training time by the PPO step so the receiver's attention pool
        gradients flow to the encoder that produced `messages`.
        """
        assert self.pool_net is not None, "receiver pool not attached"
        n = messages.shape[0]
        device = messages.device
        dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) / body_diameter
        atten = torch.from_numpy(self.attenuation_matrix(dist)).to(device)
        # Approximation: assume all messages arrive in this step (no delay) to keep the
        # gradient graph flat. Delay only affects the runtime-quantized transmission that
        # feeds the *next* obs; it's not needed in the current-step attention loss.
        expanded = messages.unsqueeze(0).expand(n, n, -1) * atten.unsqueeze(-1)
        noise = torch.randn_like(expanded) * self.cfg.noise_sigma
        expanded = expanded + noise
        mask = torch.from_numpy(~np.eye(n, dtype=bool)).to(device)
        return self.pool_net(query_embed, expanded, mask)
