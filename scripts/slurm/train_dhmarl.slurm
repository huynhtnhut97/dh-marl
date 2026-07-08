#!/bin/bash
#SBATCH --job-name=dhmarl-train
#SBATCH --output=logs/dhmarl_%j.out
#SBATCH --error=logs/dhmarl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --partition=gpu

# Adjust --account, --partition, and module loads to match the AI-Panther cluster or
# your institution's scheduler.

set -euo pipefail

module load anaconda cuda/12.1 || true

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source .venv/bin/activate || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIG=${CONFIG:-configs/default.yaml}
OUT_DIR=${OUT_DIR:-runs/dhmarl_${SLURM_JOB_ID}}

python scripts/train.py \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --seed 0
