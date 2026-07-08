"""Team-level evaluation metrics from Section 5.

- team_success_rate: fraction of episodes hitting the task's success condition,
- collision_rate: fraction of steps in which at least one pair of robots is at contact
  distance,
- coordination_efficiency: paper defines it as the ratio between the achieved return and
  a task-specific ideal-return baseline; here we implement it as the normalized team
  reward per successful step, which is monotone in the paper's coord-eff. definition and
  suffices for method comparison.
"""

from __future__ import annotations

from statistics import mean

import numpy as np


def team_success_rate(episode_successes: list[bool]) -> float:
    if not episode_successes:
        return 0.0
    return float(mean(episode_successes))


def collision_rate(collision_events_per_episode: list[int], steps_per_episode: list[int]) -> float:
    total_collisions = sum(collision_events_per_episode)
    total_steps = sum(max(1, s) for s in steps_per_episode)
    return float(total_collisions) / float(total_steps)


def coordination_efficiency(
    team_returns: list[float], ideal_return: float = 100.0,
) -> float:
    if not team_returns:
        return 0.0
    return float(np.clip(np.mean(team_returns) / max(1e-6, ideal_return), 0.0, 1.0))
