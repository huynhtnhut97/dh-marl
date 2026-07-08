"""COMA-GNN baseline.

Same graph attention critic as DH-MARL, same interaction-kernel edge priors, but the
counterfactual baseline is the standard COMA baseline (expected Q under the current
policy) rather than the flow-following one. Isolating this switch is the whole point of
the ablation-vs-baseline distinction in Section 5.5 and Section 5.2.
"""

from __future__ import annotations

from dh_marl.train.trainer import Trainer, TrainerConfig


class COMAGNNTrainer(Trainer):
    def __init__(self, cfg: TrainerConfig):
        cfg.ppo.use_fcb = False       # <- the only change vs. DH-MARL
        cfg.ppo.use_dc = True
        cfg.ppo.use_him_edges = True
        super().__init__(cfg)
