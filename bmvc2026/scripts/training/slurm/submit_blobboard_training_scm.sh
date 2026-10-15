#!/bin/bash
# Submit annotation as ONE sbatch allocation using SlurmClusterManager.jl.
# Total tasks = N_WORKERS + 1 (driver = task 0; workers = tasks 1..N).
# Each task gets its own GPU via --gpus-per-task=1 (cgroup isolation,
# CUDA_VISIBLE_DEVICES=0 inside each task).

set -euo pipefail

# Config — via env vars; the paths are required.
# - BB_DATA:    blobboard_training root (boards/, images/, videos/, models/)
# - BB_OUT:     output dir (per-submission timestamped subdir by default)
# - BB_PROJECT: Julia project with BlobBoards + training/SLURM deps
# - N_WORKERS:  number of GPU workers (driver gets +1 task w/ 1 GPU)
# - CPUS_PER_TASK: override --cpus-per-task (default 28 = 1/8 of an 8-GPU, 224-cpu node)
#
# OUT must be on LOCAL disk: network (SMB) shares go I/O-error-on-stat under
# load and hang JLD2 writes.
DATA="${BB_DATA:?set BB_DATA to the blobboard_training root}"
OUT="${BB_OUT:?set BB_OUT to an output directory on local disk}/$(date +%Y-%m-%dT%H-%M-%S)"
PROJECT="${BB_PROJECT:?set BB_PROJECT to a Julia env with BlobBoards and the training/SLURM deps}"

N_WORKERS="${N_WORKERS:-2}"             # number of GPU workers (driver gets +1 task w/ 1 GPU)
N_TASKS=$((N_WORKERS + 1))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCES_FILE="$OUT/_sources.txt"

mkdir -p "$OUT"

echo "=== Building source list ==="
julia --project="$PROJECT" "$SCRIPT_DIR/build_source_list.jl" "$SOURCES_FILE" \
    --group "$DATA/images/dataset_sonyfx3_tamron20mm_f8|$DATA/images/dataset_sonyfx3_tamron20mm_f8.json" \
    --group "$DATA/images/dataset_sonyfx3_tamron_20mm_f11|$DATA/images/dataset_sonyfx3_tamron_20mm_f11.json" \
    --group "$DATA/images/old_dataset_35mm|" \
    --group "$DATA/videos/iphone_till|$DATA/videos/iphone_till.json" \
    --group "$DATA/videos/outdoor_videos|"

echo
echo "=== Submitting sbatch (N_TASKS=$N_TASKS, --gpus-per-task=1) ==="

export BB_PROJECT="$PROJECT"
export BB_OUTPUT_DIR="$OUT"
export BB_BOARDS_DIR="$DATA/boards"
export BB_MODEL_PATH="$DATA/models/blob_matcher_dynamic_batch_size_fix.onnx"  # keep in sync with BlobBoards.DEFAULT_DESCRIPTOR_MODEL
export BB_GALLERY_PATH="$DATA/models/board_gallery.jld2"
export BB_SOURCES_FILE="$SOURCES_FILE"
export BB_N_FRAMES=30
export BB_DRIVER="$SCRIPT_DIR/driver_scm.jl"

mkdir -p logs   # SLURM opens --output before the job runs; it will not create the dir
sbatch --export=ALL --ntasks="$N_TASKS" --gpus-per-task=1 \
    --cpus-per-task="${CPUS_PER_TASK:-28}" "$SCRIPT_DIR/sbatch_scm.sh"

echo
echo "Output dir: $OUT"
echo "Monitor with: squeue -u \$USER  +  tail -f logs/bb-annot-scm-*.log"
