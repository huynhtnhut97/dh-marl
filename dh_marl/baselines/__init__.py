"""Baselines from Section 5.2."""

from dh_marl.baselines.ind_sa import IndependentSingleAgentTrainer
from dh_marl.baselines.sa_la import SingleAgentLocalAvoidanceTrainer
from dh_marl.baselines.mappo import MAPPOTrainer
from dh_marl.baselines.coma_gnn import COMAGNNTrainer


BASELINE_REGISTRY = {
    "ind_sa": IndependentSingleAgentTrainer,
    "sa_la": SingleAgentLocalAvoidanceTrainer,
    "mappo": MAPPOTrainer,
    "coma_gnn": COMAGNNTrainer,
}


def get_baseline_trainer(name: str, cfg):
    cls = BASELINE_REGISTRY[name]
    return cls(cfg)


__all__ = [
    "IndependentSingleAgentTrainer",
    "SingleAgentLocalAvoidanceTrainer",
    "MAPPOTrainer",
    "COMAGNNTrainer",
    "get_baseline_trainer",
    "BASELINE_REGISTRY",
]
