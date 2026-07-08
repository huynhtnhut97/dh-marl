#!/bin/bash
#SBATCH --job-name=dhmarl-ablation
#SBATCH --output=logs/dhmarl_ablation_%j.out
#SBATCH --error=logs/dhmarl_ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --partition=gpu

set -euo pipefail

module load anaconda cuda/12.1 || true

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source .venv/bin/activate || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CONFIG=${CONFIG:-configs/default.yaml}

python scripts/ablation.py \
    --config "${CONFIG}" \
    --components full fcb dc him all \
    --out "runs/ablation_${SLURM_JOB_ID}.json"
