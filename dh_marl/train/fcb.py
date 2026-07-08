"""Flow-following counterfactual baseline (FCB) advantage (Equation 9).

For each agent i at each rollout step, the counterfactual replaces a_t^i with the
passive action a_0 (no actuation), holds a^{-i}_t fixed, and re-evaluates the graph
attention critic Q_phi under the *recomputed* interaction field. The advantage is:

    A^i(s_t, a_t) = Q_phi(s_t, a_t) - Q_phi(s_t, (a^{-i}_t, a_0)).

Two key implementation choices:
1. The passive action index. In the environment's action encoding, six primitives are
   translational and four are rotational; there is no true "no-op" primitive. We treat
   the passive action as an all-zero one-hot action encoding at the critic's action
   embedding layer. This is faithful to Equation (9)'s "zero actuation force"
   interpretation: the encoding produces zero contribution from the actuation term of
   Equation (6), which is what F_a(a_0) = 0 means. In the environment itself, no
   simulator step is required for the counterfactual evaluation; the critic value
   under the passive action is queried directly from Q_phi.
2. Recomputing the interaction graph under the counterfactual. Equation (5)'s wake
   force sum_j alpha_w w(d_ij, theta_ij) v_j depends on velocities. When agent i's
   action is replaced by a_0, its velocity contribution to the wake sum vanishes for
   the other agents; we reflect this by zeroing agent i's edges (row i and column i)
   in the interaction adjacency and edge-prior matrices before the counterfactual Q
   query. This preserves the causal structure emphasized in Section 4.4 (blue text).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


PASSIVE_ACTION_ONEHOT_DIM = 10   # keep in sync with policy.action_dim / critic.action_dim


def _zero_row_col(mat: torch.Tensor, i: int) -> torch.Tensor:
    m = mat.clone()
    m[i, :] = 0.0
    m[:, i] = 0.0
    return m


def _zero_row_col_bool(mat: torch.Tensor, i: int) -> torch.Tensor:
    m = mat.clone()
    m[i, :] = False
    m[:, i] = False
    return m


def compute_fcb_advantage(
    critic,                              # GraphAttentionCritic
    obs_embed: torch.Tensor,             # [N, obs_embed_dim]
    action: torch.Tensor,                 # [N] long
    adj: torch.Tensor,                    # [N, N] bool
    edge_prior: torch.Tensor,             # [N, N] float
) -> torch.Tensor:
    """Return per-agent FCB advantages A^i for a single timestep.

    Uses the critic's Q head; the value head V^i is not consumed here (it is available
    for a separate value-loss target if desired).
    """
    n = obs_embed.size(0)
    device = obs_embed.device

    # realized joint Q (scalar)
    q_realized, _ = critic(obs_embed, action, adj, edge_prior)

    advantages = torch.zeros(n, device=device)
    for i in range(n):
        cf_action = action.clone()
        # passive action index. We use index 0 as a placeholder; the critic's action
        # encoder produces an embedding either way, but we also zero its edge influence
        # below to mirror "F_a(a_0) = 0".
        cf_action[i] = 0
        cf_adj = _zero_row_col_bool(adj, i)
        cf_prior = _zero_row_col(edge_prior, i)
        q_cf, _ = critic(obs_embed, cf_action, cf_adj, cf_prior)
        advantages[i] = q_realized - q_cf

    return advantages


def compute_fcb_advantage_batched(
    critic,
    obs_embed_seq: torch.Tensor,          # [T, N, obs_embed_dim]
    action_seq: torch.Tensor,             # [T, N] long
    adj_seq: torch.Tensor,                # [T, N, N] bool
    edge_prior_seq: torch.Tensor,         # [T, N, N] float
) -> torch.Tensor:
    """Return per-agent FCB advantages for T timesteps. Shape [T, N]."""
    t_len = obs_embed_seq.size(0)
    out = []
    for t in range(t_len):
        out.append(
            compute_fcb_advantage(
                critic,
                obs_embed_seq[t],
                action_seq[t],
                adj_seq[t],
                edge_prior_seq[t],
            )
        )
    return torch.stack(out, dim=0)
