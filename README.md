# DH-MARL: Decentralized Hemodynamic-Aware Multi-Agent Reinforcement Learning

Reference implementation of the framework described in:

> **Huynh, T. N., and Nguyen, K.-D.** *Decentralized Learning and Control of Multi-Microrobots in Complex Hemodynamic Environments.* Electronics (MDPI), 2026. Manuscript ID electronics-4377489.

This repository contains the algorithmic core of DH-MARL: a per-agent flow-disturbance-aware policy, a reduced-order inter-robot hydrodynamic interaction model (HIM), a bandwidth-constrained diffusion-based communication channel (DC), and a graph attention critic with a flow-following counterfactual baseline (FCB), all trained under a PPO-style CTDE loop.

The production simulator described in the paper is a Unity build. To keep the repository language-portable and CI-testable, this codebase ships a lightweight NumPy/PyTorch vascular environment that reproduces the essential dynamics (Equation 4 with the three interaction kernels, pulsatile flow proxy, and the four task success criteria of Table 2). The environment implements a `gymnasium`-style multi-agent API; a Unity `mlagents` wrapper is a drop-in replacement.

---

## What's here

| Component | File | Paper reference |
|---|---|---|
| Per-agent policy + risk head | `dh_marl/models/policy.py` | Eq. (2)-(3) |
| Graph attention critic | `dh_marl/models/critic.py` | Sec. 4.4, Fig. 1 |
| Message encoder + Gumbel quantizer | `dh_marl/models/message_encoder.py` | Eq. (7) |
| GAT layer, attention pool | `dh_marl/models/layers.py` | Sec. 4.4 |
| HIM interaction force | `dh_marl/env/hemodynamics.py` | Eq. (5), Fig. 2 |
| Diffusion channel | `dh_marl/comm/diffusion_channel.py` | Eq. (7), Table 3 |
| Four cooperative tasks | `dh_marl/env/tasks.py` | Table 2, Sec. 5.1 |
| Vascular environment | `dh_marl/env/vascular_env.py` | Sec. 3, Sec. 5.1 |
| Flow-following counterfactual | `dh_marl/train/fcb.py` | Eq. (9) |
| PPO update | `dh_marl/train/ppo.py` | Sec. 4.6 |
| Training loop | `dh_marl/train/trainer.py` | Algorithm 1 |
| IND-SA, SA-LA, MAPPO, COMA-GNN | `dh_marl/baselines/` | Sec. 5.2 |
| Role analysis (ARI, MI, temporal) | `dh_marl/eval/role_analysis.py` | Sec. 5.4 |

---

## Install

```bash
git clone https://github.com/huynhtnhut97/dh-marl.git
cd dh-marl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+, PyTorch 2.1+, PyTorch Geometric 2.5+. Training is GPU-friendly but not required for the Python environment; the paper's 20,000-episode run used two NVIDIA A100 GPUs.

---

## Quick start

Train DH-MARL on all four tasks with the default configuration ($N=8$, 20k episodes):

```bash
python scripts/train.py --config configs/default.yaml
```

Evaluate a trained checkpoint at unseen team sizes ($N \in \{4, 8, 16, 32\}$, matching Section 5.3):

```bash
python scripts/evaluate.py --checkpoint runs/dhmarl/best.pt --team-sizes 4 8 16 32
```

Run the ablation study of Section 5.5 (remove FCB / DC / HIM independently and jointly):

```bash
python scripts/ablation.py --config configs/default.yaml --components fcb dc him all
```

Baselines:

```bash
python scripts/train.py --config configs/default.yaml --algo ind_sa
python scripts/train.py --config configs/default.yaml --algo sa_la
python scripts/train.py --config configs/default.yaml --algo mappo
python scripts/train.py --config configs/default.yaml --algo coma_gnn
```

---

## HPC (SLURM)

Two SLURM scripts targeted at the AI-Panther cluster are in `scripts/slurm/`:

```bash
sbatch scripts/slurm/train_dhmarl.slurm
sbatch scripts/slurm/ablation.slurm
```

Adjust `--account`, `--partition`, and module loads to match your site.

---

## Tests

```bash
pytest tests/ -v
```

Unit tests cover HIM kernel shapes and symmetry, GAT critic forward/backward, Gumbel-softmax quantizer, per-task reward correctness, and FCB advantage sign under a controlled two-agent setup.

---

## Reproducing the paper's headline numbers

The paper reports 88.7% team success at $N=16$ averaged across the four tasks against 65.3% (IND-SA), 71.0% (SA-LA), 74.6% (MAPPO), and 80.4% (COMA-GNN). The bundled Python environment is a reduced-order proxy for the Unity simulator; numerical values from this repo will not exactly match the paper's, but the qualitative ordering (DH-MARL > COMA-GNN > MAPPO > SA-LA > IND-SA) and the ablation ordering (FCB removal produces the largest single drop) are reproduced. For exact numerical parity, use the Unity build referenced in the paper.

---

## Citation

```bibtex
@article{huynh2026dhmarl,
  title   = {Decentralized Learning and Control of Multi-Microrobots in Complex Hemodynamic Environments},
  author  = {Huynh, Truong Nhut and Nguyen, Kim-Doang},
  journal = {Electronics},
  year    = {2026},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

---

## License

MIT. See `LICENSE`.

## Acknowledgments

Compute was provided by the AI-Panther HPC cluster at Florida Institute of Technology.
