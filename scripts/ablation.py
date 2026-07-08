#!/usr/bin/env python
"""Ablation study runner (Section 5.5).

Runs the full DH-MARL model and its ablated variants:
    fcb   - flow-following counterfactual removed (falls back to standard COMA baseline)
    dc    - diffusion communication removed (zero messages)
    him   - inter-robot hydrodynamic interaction removed at critic time (identity edges)
    all   - all three removed simultaneously

Each variant trains for `--episodes` episodes and its final team success rate is dumped
to the report JSON. Also emits a plain-text summary comparable to the Section 5.5 table.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dh_marl.train.trainer import Trainer, _dict_to_trainer_config
from dh_marl.utils.config import load_config


def _apply_ablation(cfg, key: str):
    """Apply one ablation switch by mutating the trainer config in place."""
    if key == "full":
        cfg.ppo.use_fcb = True
        cfg.ppo.use_dc = True
        cfg.ppo.use_him_edges = True
    elif key == "fcb":
        cfg.ppo.use_fcb = False
    elif key == "dc":
        cfg.ppo.use_dc = False
    elif key == "him":
        cfg.ppo.use_him_edges = False
    elif key == "all":
        cfg.ppo.use_fcb = False
        cfg.ppo.use_dc = False
        cfg.ppo.use_him_edges = False
    else:
        raise ValueError(f"unknown ablation component: {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--components", nargs="+", default=["full", "fcb", "dc", "him", "all"])
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override total_episodes for a quick ablation.")
    parser.add_argument("--out", default="ablation_report.json")
    args = parser.parse_args()

    base_dict = load_config(args.config)
    report = {}

    for comp in args.components:
        cfg = _dict_to_trainer_config(copy.deepcopy(base_dict))
        _apply_ablation(cfg, comp)
        if args.episodes is not None:
            cfg.total_episodes = args.episodes
        cfg.out_dir = f"runs/ablation_{comp}"
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        trainer = Trainer(cfg)
        trainer.train()
        report[comp] = {
            "best_success_rate": trainer.best_success_rate,
            "out_dir": cfg.out_dir,
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
