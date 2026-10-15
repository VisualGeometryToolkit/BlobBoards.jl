#!/bin/bash
# sbatch wrapper for the SlurmClusterManager driver.
# Allocation: --ntasks=N+1 (driver as task 0 + N workers), each task
# pinned to its own GPU via --gpus-per-task=1 (cgroup isolation —
# each task sees its GPU as CUDA_VISIBLE_DEVICES=0).
#
# Submitted by submit_blobboard_training_scm.sh which exports BB_* env vars
# and supplies --ntasks / --gpus-per-task at submit time.
#SBATCH --job-name=bb-annot-scm
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bb-annot-scm-%j.log
#SBATCH --error=logs/bb-annot-scm-%j.log

set -euo pipefail

: "${BB_PROJECT:?must be set}"
: "${BB_OUTPUT_DIR:?must be set}"
: "${BB_BOARDS_DIR:?must be set}"
: "${BB_MODEL_PATH:?must be set}"
: "${BB_GALLERY_PATH:?must be set}"
: "${BB_SOURCES_FILE:?must be set}"
: "${BB_DRIVER:?must be set (absolute path to driver_scm.jl)}"

DRIVER="$BB_DRIVER"

echo "=== sbatch job $SLURM_JOB_ID on $(hostname) — NTASKS=$SLURM_NTASKS (one GPU per task via cgroup) ==="

exec julia --startup-file=no --project="$BB_PROJECT" "$DRIVER"
