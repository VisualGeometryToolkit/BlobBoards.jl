#!/bin/bash
# sbatch wrapper for descriptor training.
# Single task, single GPU (training is single-device — no Distributed).
# Per-GPU share of an 8-GPU node: 28 cpus + 230 GB.
#
# Submitted by submit_train.sh which exports BB_* env vars.
#SBATCH --job-name=bb-train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bb-train-%j.log
#SBATCH --error=logs/bb-train-%j.log

set -euo pipefail

: "${BB_PROJECT:?must be set}"
: "${BB_TRAIN_PATH:?must be set}"
: "${BB_VAL_PATH:?must be set}"
: "${BB_OUTPUT_DIR:?must be set}"
: "${BB_DRIVER:?must be set (absolute path to train_driver.jl)}"

echo "=== sbatch job $SLURM_JOB_ID on $(hostname) — single GPU training ==="
echo "    train=$BB_TRAIN_PATH"
echo "    val  =$BB_VAL_PATH"
echo "    out  =$BB_OUTPUT_DIR"

exec julia -O0 --startup-file=no --project="$BB_PROJECT" "$BB_DRIVER"
