#!/bin/bash
# Annotate ALL blobboard_training data via ClusterManagers.SlurmManager.
# Run from a login-node shell (use `tmux` so the driver survives ssh exits).
# The driver allocates SLURM workers, dispatches per-source work, and
# tears workers down on exit.

set -euo pipefail

DATA="${BB_DATA:?set BB_DATA to the blobboard_training root (boards/, images/, videos/, models/)}"
OUT="${BB_OUT:?set BB_OUT to the output directory (local disk)}"
PROJECT="${BB_PROJECT:?set BB_PROJECT to a Julia env with BlobBoards + CUDA + AcceleratedKernels + NNlib + ClusterManagers as direct deps}"

SCRIPT_DIR="$(dirname "$0")"
SOURCES_FILE="$OUT/_sources.txt"

mkdir -p "$OUT"

# 1. Build flat sources file from the 5 (source, camera) groups
echo "=== Building source list ==="
julia --project="$PROJECT" "$SCRIPT_DIR/build_source_list.jl" "$SOURCES_FILE" \
    --group "$DATA/images/dataset_sonyfx3_tamron20mm_f8|$DATA/images/sony-fx3_tamron-20mm-f-2.8-f-8.json" \
    --group "$DATA/images/dataset_sonyfx3_tamron_20mm_f11|$DATA/images/sony-fx3_tamron-20mm-f-2.8-f-11.json" \
    --group "$DATA/images/old_dataset_35mm|" \
    --group "$DATA/videos/iphone_till|$DATA/videos/apple-iphone-13-mini-till_f1.6-1x-video-fullhd.json" \
    --group "$DATA/videos/outdoor_videos|"

# 2. Run the driver — it allocates SLURM workers + dispatches
echo
echo "=== Driver starting ==="
export BB_OUTPUT_DIR="$OUT"
export BB_BOARDS_DIR="$DATA/boards"
export BB_MODEL_PATH="$DATA/models/blob_matcher_dynamic_batch_size_fix.onnx"  # keep in sync with BlobBoards.DEFAULT_DESCRIPTOR_MODEL
export BB_GALLERY_PATH="$DATA/models/board_gallery.jld2"
export BB_N_FRAMES=30
export BB_N_WORKERS=5
export BB_PARTITION=gpu
export BB_GRES=gpu:1
export BB_TIME=01:00:00
export BB_CPUS_PER_TASK=4
export BB_MEM_PER_CPU=16G

exec julia --startup-file=no --project="$PROJECT" "$SCRIPT_DIR/driver.jl" "$SOURCES_FILE"
