"""Role emergence analyses from Section 5.4.

For each rollout, we extract per-agent behavioral descriptors, cluster them into K roles,
and report:
- adjusted Rand index (ARI) of role labels across seeds,
- mutual information between message tokens and role labels,
- temporal consistency of role labels within a rollout,
- causal-importance drops when each role's messages are replaced with noise.

The four behavioral roles the paper identifies (scout, carrier, relayer, returner) are
recovered by K=4 clustering on per-agent statistics. This module returns numerical
summaries only; interpretation is left to the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mutual_info_score


def _behavioral_descriptors(rollout, scene_length: float) -> np.ndarray:
    """Return [N, 5] descriptors: mean axial progress, radial dispersion, msg entropy,
    action entropy, and self-role slot."""
    n = rollout.n_agents
    obs = rollout.obs                     # [T, N, obs_dim]
    actions = rollout.actions             # [T, N]
    messages = rollout.messages_sent      # [T, N, d_msg]

    axial = obs[:, :, 0]                  # first pose component
    progress = (axial[-1] - axial[0]) / max(1e-6, scene_length)

    radial = np.linalg.norm(obs[:, :, 1:3], axis=-1)
    dispersion = radial.std(axis=0)

    # action entropy per agent
    a_ent = np.zeros(n, dtype=np.float32)
    for i in range(n):
        counts = np.bincount(actions[:, i], minlength=10) / max(1, actions.shape[0])
        counts = counts[counts > 0]
        a_ent[i] = -(counts * np.log(counts)).sum()

    # message entropy proxy: variance of the message across time
    msg_var = messages.var(axis=0).mean(axis=1)

    # last-role slot (as reported by env)
    role_slot = obs[-1, :, -1]

    return np.stack([progress, dispersion, a_ent, msg_var, role_slot], axis=1)


def role_cluster_labels(rollout, scene_length: float, k: int = 4, seed: int = 0) -> np.ndarray:
    x = _behavioral_descriptors(rollout, scene_length)
    if x.shape[0] < k:
        return np.zeros(x.shape[0], dtype=np.int32)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x)
    return km.labels_.astype(np.int32)


def adjusted_rand_across_seeds(labelings: list[np.ndarray]) -> float:
    """Mean pairwise ARI across a list of per-seed label arrays for the same agents."""
    if len(labelings) < 2:
        return 1.0
    scores = []
    for i in range(len(labelings)):
        for j in range(i + 1, len(labelings)):
            scores.append(adjusted_rand_score(labelings[i], labelings[j]))
    return float(np.mean(scores))


def message_role_mutual_information(messages: np.ndarray, role_labels: np.ndarray) -> float:
    """Mutual information between message tokens and role labels (in nats).

    Messages are quantized per-channel; we discretize each channel by rounding to the
    nearest integer level index and sum per-channel MI, following the paper's phrasing.
    """
    if messages.ndim == 3:
        # [T, N, d] -> [T*N, d]
        m = messages.reshape(-1, messages.shape[-1])
        roles = np.tile(role_labels, messages.shape[0])
    else:
        m = messages
        roles = role_labels
    # discretize each channel
    m_disc = np.round((m - m.min(axis=0, keepdims=True)) * 8).astype(np.int32)
    mi = 0.0
    for c in range(m_disc.shape[1]):
        mi += mutual_info_score(roles, m_disc[:, c])
    return float(mi)


def temporal_consistency(role_labels_over_time: np.ndarray, interval: int = 50) -> float:
    """Fraction of intervals in which each agent's role label is unchanged.

    role_labels_over_time: [T_windows, N]. Returns mean over agents and windows.
    """
    if role_labels_over_time.shape[0] < 2:
        return 1.0
    unchanged = role_labels_over_time[1:] == role_labels_over_time[:-1]
    return float(unchanged.mean())


def role_intervention_drops(
    baseline_success_rate: float,
    per_role_success_rates: dict[int, float],
) -> dict[int, float]:
    """Signed drop in team success when each role's messages are replaced with noise."""
    return {r: float(sr - baseline_success_rate) for r, sr in per_role_success_rates.items()}
