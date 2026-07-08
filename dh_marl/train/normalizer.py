"""Welford-style running mean and variance for observation and return normalization.

Used as `RunningMeanStd(shape=(obs_dim,))` for observations and `RunningMeanStd(shape=())`
for scalar returns. Matches the standard PPO recipe used in the reference implementation
(Section 4.6 mentions "Observations and returns are normalized via running statistics").
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        new_var = M2 / tot
        self.mean = new_mean
        self.var = new_var
        self.count = tot

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        z = (x - self.mean) / std
        return np.clip(z, -clip, clip).astype(np.float32)
