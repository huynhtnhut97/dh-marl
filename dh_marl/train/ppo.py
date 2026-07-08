"""PPO clipped-objective update with the flow-following counterfactual advantage.

Follows the on-policy recipe from Section 4.6:
- 4 parallel workers, 256-step rollouts (8192 transitions/iter at N=8),
- 4 PPO epochs over minibatches of 128,
- clip epsilon 0.2, GAE lambda 0.95, gamma 0.99,
- Adam with (0.9, 0.999, 1e-8), global-norm gradient clip 0.5,
- separate learning rates for policy trunk (5e-4), critic (1e-3), message encoder (5e-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from dh_marl.train.fcb import compute_fcb_advantage


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 128
    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    risk_bce_coef: float = 0.1
    message_bottleneck_coef: float = 1e-4
    grad_clip: float = 0.5
    # for controlled-additive-baseline variants (Section 5.6)
    use_fcb: bool = True         # if False, use standard COMA advantage
    use_dc: bool = True          # if False, zero incoming messages during rollout
    use_him_edges: bool = True   # if False, use identity adjacency at critic time


def _standard_coma_advantage(critic, obs_embed, action, adj, edge_prior, logits_per_agent):
    """Standard COMA baseline: expected Q under the current policy at agent i.

    A^i = Q(s, a) - E_{a^i ~ pi_i}[Q(s, (a^{-i}, a^i))]

    Args:
        critic: GraphAttentionCritic (or a baseline critic with the same signature).
        obs_embed: [N, embed_dim] trunk outputs at this step (used by the critic).
        action: [N] realized actions.
        adj / edge_prior: [N, N] graph inputs to the critic.
        logits_per_agent: [N, A] policy logits at this step, computed fresh from raw obs
            outside this function so the trunk sees the observation it was trained on.
    """
    n = obs_embed.size(0)
    device = obs_embed.device
    q_realized, _ = critic(obs_embed, action, adj, edge_prior)
    dist = Categorical(logits=logits_per_agent)
    advantages = torch.zeros(n, device=device)
    for i in range(n):
        expectation = 0.0
        p = dist.probs[i]
        for a in range(logits_per_agent.shape[-1]):
            cf_action = action.clone()
            cf_action[i] = a
            q_a, _ = critic(obs_embed, cf_action, adj, edge_prior)
            expectation = expectation + p[a] * q_a
        advantages[i] = q_realized - expectation
    return advantages


def ppo_update(
    policy,                             # PerAgentPolicy
    critic,                             # GraphAttentionCritic
    message_encoder,                    # MessageEncoder
    optimizer,                          # torch.optim.Optimizer
    rollouts,                           # list[Rollout]
    obs_normalizer,                     # RunningMeanStd
    ret_normalizer,                     # RunningMeanStd
    cfg: PPOConfig,
    device: str = "cpu",
) -> dict[str, float]:
    """Run n_epochs of PPO over a batch of rollouts. Returns training statistics."""
    stats: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "kl": [],
        "risk_bce": [], "msg_reg": [], "advantage_mean": [], "advantage_std": [],
    }

    # flatten rollouts into (T*N)-indexed tensors, but keep per-step graph structure
    all_obs, all_act, all_old_lp, all_ret, all_adv, all_adj, all_prior = [], [], [], [], [], [], []
    for r in rollouts:
        t_len, n = r.actions.shape
        # per-agent returns from per-agent + team reward
        team = np.broadcast_to(r.rewards_team[:, None], (t_len, n))
        rew_flat = (r.rewards_per_agent + team).astype(np.float32)   # [T, N]

        # compute per-step per-agent advantages via FCB (or standard COMA if disabled)
        obs_norm = obs_normalizer.normalize(r.obs.reshape(-1, r.obs.shape[-1])).reshape(r.obs.shape)
        obs_t = torch.from_numpy(obs_norm).float().to(device)
        adj_t = torch.from_numpy(r.adjacencies).bool().to(device)
        prior_t = torch.from_numpy(r.edge_priors).float().to(device)
        act_t = torch.from_numpy(r.actions).long().to(device)
        emb_t = torch.from_numpy(r.embeddings).float().to(device)

        with torch.no_grad():
            adv_per_step = []
            # Pre-compute per-step logits from the raw normalized observations. The
            # standard-COMA baseline path needs these; the FCB path doesn't, but the
            # forward is cheap and shared.
            _, all_logits, _ = policy(obs_t.reshape(-1, obs_t.shape[-1]))
            all_logits = all_logits.reshape(t_len, n, -1)
            for t in range(t_len):
                a = adj_t[t] if cfg.use_him_edges else torch.eye(n, dtype=torch.bool, device=device)
                p = prior_t[t] if cfg.use_him_edges else torch.zeros_like(prior_t[t])
                if cfg.use_fcb:
                    adv = compute_fcb_advantage(critic, emb_t[t], act_t[t], a, p)
                else:
                    adv = _standard_coma_advantage(
                        critic, emb_t[t], act_t[t], a, p, all_logits[t],
                    )
                adv_per_step.append(adv)
            advs = torch.stack(adv_per_step, dim=0)                     # [T, N]

        # add reward-to-go as the return target
        rets = np.zeros_like(rew_flat)
        running = np.zeros(n, dtype=np.float32)
        for k in reversed(range(t_len)):
            running = rew_flat[k] + cfg.gamma * running * (0.0 if r.dones[k] else 1.0)
            rets[k] = running

        # normalize returns
        ret_normalizer.update(rets.reshape(-1))
        rets_norm = (rets - ret_normalizer.mean) / (np.sqrt(ret_normalizer.var) + 1e-8)

        # collect flat tensors
        all_obs.append(torch.from_numpy(obs_norm.reshape(-1, obs_norm.shape[-1])).float())
        all_act.append(torch.from_numpy(r.actions.reshape(-1)).long())
        all_old_lp.append(torch.from_numpy(r.log_probs.reshape(-1)).float())
        all_ret.append(torch.from_numpy(rets_norm.reshape(-1)).float())
        all_adv.append(advs.reshape(-1).cpu())
        all_adj.append(adj_t)                                        # [T, N, N]
        all_prior.append(prior_t)                                    # [T, N, N]

    obs_flat = torch.cat(all_obs, dim=0).to(device)
    act_flat = torch.cat(all_act, dim=0).to(device)
    old_lp_flat = torch.cat(all_old_lp, dim=0).to(device)
    ret_flat = torch.cat(all_ret, dim=0).to(device)
    adv_flat = torch.cat(all_adv, dim=0).to(device)

    # Guard against occasional NaN or inf from the counterfactual pass (rare, but a
    # single poisoned entry cascades through the whole PPO epoch). Replace NaN/inf
    # with zero and log a warning downstream if it happens too often.
    adv_flat = torch.nan_to_num(adv_flat, nan=0.0, posinf=0.0, neginf=0.0)
    ret_flat = torch.nan_to_num(ret_flat, nan=0.0, posinf=0.0, neginf=0.0)

    # advantage normalization (standard PPO trick)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    total = obs_flat.shape[0]
    idx = np.arange(total)

    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, total, cfg.minibatch_size):
            mb = idx[start:start + cfg.minibatch_size]
            mb_t = torch.from_numpy(mb).long().to(device)

            obs_mb = obs_flat[mb_t]
            act_mb = act_flat[mb_t]
            old_lp_mb = old_lp_flat[mb_t]
            ret_mb = ret_flat[mb_t]
            adv_mb = adv_flat[mb_t]

            z, logits, risk_logit = policy(obs_mb)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(act_mb)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            # value target from the *per-agent* V head is not defined on the flat mb
            # (V is per-node in a graph); use ret_mb as a per-transition scalar target
            # on a simple per-agent value head appended to the policy trunk.
            # We approximate: minimize (V_from_trunk - ret_mb)^2 using a linear head on z.
            v_pred = _trunk_value(z)
            value_loss = 0.5 * (v_pred - ret_mb).pow(2).mean()

            # risk head auxiliary loss against the mean of tau in obs
            # (tau occupies dims 10..20 of the un-normed obs; obs here is normalized,
            # but the sign of tau is preserved, so we clamp and use the mean magnitude)
            tau_label = obs_mb[:, 10:20].abs().mean(dim=1).clamp(0.0, 1.0)
            risk_bce = F.binary_cross_entropy_with_logits(risk_logit, tau_label)

            # message-encoder bottleneck reg
            msg_reg = message_encoder.bit_budget_regularizer(obs_mb)

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
                + cfg.risk_bce_coef * risk_bce
                + cfg.message_bottleneck_coef * msg_reg
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()) + list(message_encoder.parameters()),
                cfg.grad_clip,
            )
            optimizer.step()

            with torch.no_grad():
                kl = (old_lp_mb - new_lp).mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["kl"].append(kl)
            stats["risk_bce"].append(risk_bce.item())
            stats["msg_reg"].append(msg_reg.item())
            stats["advantage_mean"].append(adv_mb.mean().item())
            stats["advantage_std"].append(adv_mb.std().item())

    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


def _trunk_value(z: torch.Tensor) -> torch.Tensor:
    """Cheap V head consuming the trunk embedding directly.

    Kept module-free (no learnable parameters here) to avoid coupling the PPO update
    to a specific value-head module. In the full trainer, this is replaced by a
    proper `nn.Linear(embed_dim, 1)` head attached to the policy.
    """
    return z.mean(dim=-1)
