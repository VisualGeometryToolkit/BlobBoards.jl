#!/bin/bash
# sbatch wrapper for the multi-GPU track-building driver (driver_build_tracks.jl).
#
# Allocation: --ntasks=N (driver as task 0 + N-1 workers), each task pinned to
# its own GPU via --gpus-per-task=1 (cgroup isolation — each task sees its GPU as
# CUDA_VISIBLE_DEVICES=0, which is required so VulkanSift detection also lands on
# that GPU, not just CUDA/ONNX).
#
# GPU BUDGET: at most 6 of the node's 8 B200 in use at once. Because cgroup
# isolation forces --gpus-per-task=1 uniformly, task 0 (the coordinator) also
# holds a GPU. So --ntasks=6 → 6 GPUs = 1 coordinator + 5 detection workers.
# Raising to 6 detection workers would need a heterogeneous job (CPU-only
# coordinator); not done here — 5 workers keeps us safely within the cap.
#
# Submit with the BB_* env vars exported, e.g.:
#   BB_PROJECT=/path/to/bb-slurm-env \
#   BB_DRIVER=$PWD/examples/training/slurm/driver_build_tracks.jl \
#   BB_VIDEO_DIR=/data/footage BB_BOARDS_DIR=/data/boards \
#   BB_MODEL_PATH=/data/model.onnx BB_OUTPUT_DIR=/data/out \
#   BB_MAX_DETECTIONS=15000 \
#   mkdir -p logs && sbatch examples/training/slurm/sbatch_build_tracks_scm.sh   # log -> logs/
#
#SBATCH --job-name=bb-tracks-scm
#SBATCH --partition=gpu
#SBATCH --ntasks=6
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bb-tracks-scm-%j.log
#SBATCH --error=logs/bb-tracks-scm-%j.log

set -euo pipefail

: "${BB_PROJECT:?must be set (Julia env with SlurmClusterManager)}"
: "${BB_DRIVER:?must be set (absolute path to driver_build_tracks.jl)}"
: "${BB_VIDEO_DIR:?must be set}"
: "${BB_BOARDS_DIR:?must be set}"
: "${BB_MODEL_PATH:?must be set}"
: "${BB_OUTPUT_DIR:?must be set}"

echo "=== sbatch job $SLURM_JOB_ID on $(hostname) — NTASKS=$SLURM_NTASKS (one GPU per task via cgroup; ≤6 of 8 B200) ==="

# -t auto gives each worker threads for the parallel symmetric_iou_match step.
exec julia --startup-file=no -t auto --project="$BB_PROJECT" "$BB_DRIVER"
