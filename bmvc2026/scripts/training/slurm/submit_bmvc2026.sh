#!/bin/bash
# BMVC 2026 end-to-end orchestrator: annotate -> split -> train as one
# SLURM dependency chain. Each step is its own sbatch with afterok wiring.
#
# See examples/training/BMVC2026.md for the full runbook (paths, config,
# troubleshooting). The paths are required env vars; the rest have defaults.

set -euo pipefail

# ── Config — override at submit time via env vars ─────────────────────
DATA="${BB_DATA:?set BB_DATA to the blobboard_training root}"
OUT="${BB_OUT:?set BB_OUT to a results directory on local disk}/$(date +%Y-%m-%dT%H-%M-%S)"
PROJECT="${BB_PROJECT:?set BB_PROJECT to a Julia env with BlobBoards and the training/SLURM deps}"

N_WORKERS="${N_WORKERS:-2}"          # GPU workers for annotation
CPUS_PER_TASK="${CPUS_PER_TASK:-28}" # per-task share of an 8-GPU, 224-cpu node
N_TASKS=$((N_WORKERS + 1))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANNOT_DIR="$OUT/truth_images"
SPLITS_DIR="$OUT/splits"
CHECKPOINTS_DIR="$OUT/checkpoints"
SOURCES_FILE="$OUT/_sources.txt"

mkdir -p "$ANNOT_DIR" "$SPLITS_DIR" "$CHECKPOINTS_DIR"

# ── Sanity ─────────────────────────────────────────────────────────────
[ -d "$DATA/boards" ]  || { echo "missing $DATA/boards"  >&2; exit 1; }
[ -d "$DATA/images" ]  || { echo "missing $DATA/images"  >&2; exit 1; }
[ -d "$DATA/videos" ]  || { echo "missing $DATA/videos"  >&2; exit 1; }
[ -d "$DATA/models" ]  || { echo "missing $DATA/models"  >&2; exit 1; }

# ── Step 1: build sources manifest (login-node, ~seconds) ─────────────
echo "=== Building source list ==="
julia --project="$PROJECT" "$SCRIPT_DIR/build_source_list.jl" "$SOURCES_FILE" \
    --group "$DATA/images/dataset_sonyfx3_tamron20mm_f8|$DATA/images/dataset_sonyfx3_tamron20mm_f8.json"   \
    --group "$DATA/images/dataset_sonyfx3_tamron_20mm_f11|$DATA/images/dataset_sonyfx3_tamron_20mm_f11.json" \
    --group "$DATA/images/iphone_till|$DATA/images/iphone_till.json"        \
    --group "$DATA/images/old_dataset_35mm|$DATA/images/old_dataset_35mm.json" \
    --group "$DATA/videos/iphone_till|$DATA/videos/iphone_till.json"        \
    --group "$DATA/videos/outdoor_videos|"                                   \
    --group "$DATA/videos/20260301_sonyfx3_multiplelenses_office_and_hallway|"

# ── Step 2: submit annotate sbatch ────────────────────────────────────
echo
echo "=== Submitting annotate (N_TASKS=$N_TASKS, --gpus-per-task=1) ==="
export BB_PROJECT="$PROJECT"
export BB_OUTPUT_DIR="$ANNOT_DIR"
export BB_BOARDS_DIR="$DATA/boards"
export BB_MODEL_PATH="$DATA/models/blob_matcher_dynamic_batch_size_fix.onnx"
export BB_GALLERY_PATH="$DATA/models/board_gallery.jld2"
export BB_SOURCES_FILE="$SOURCES_FILE"
export BB_N_FRAMES="${BB_N_FRAMES:-30}"
export BB_DRIVER="$SCRIPT_DIR/driver_scm.jl"

mkdir -p logs   # every sbatch below writes --output=logs/…; SLURM will not create the dir
ANNOTATE_JOB=$(sbatch --parsable --export=ALL \
    --ntasks="$N_TASKS" --gpus-per-task=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    "$SCRIPT_DIR/sbatch_scm.sh")
echo "  annotate job: $ANNOTATE_JOB"

# ── Step 3: submit split sbatch (depends on annotate) ─────────────────
echo
echo "=== Submitting split (afterok:$ANNOTATE_JOB) ==="
export BB_ANNOTATION_DIR="$ANNOT_DIR"
export BB_SPLITS_DIR="$SPLITS_DIR"

SPLIT_JOB=$(sbatch --parsable --export=ALL \
    --dependency=afterok:"$ANNOTATE_JOB" \
    "$SCRIPT_DIR/sbatch_split.sh")
echo "  split job:    $SPLIT_JOB"

# ── Step 4: submit train sbatch (depends on split) ────────────────────
echo
echo "=== Submitting train (afterok:$SPLIT_JOB) ==="
export BB_TRAIN_PATH="$SPLITS_DIR/train.jld2"
export BB_VAL_PATH="$SPLITS_DIR/val.jld2"
export BB_OUTPUT_DIR="$CHECKPOINTS_DIR"
export BB_DRIVER="$( cd "$SCRIPT_DIR/.." && pwd )/train_bmvc2026.jl"

TRAIN_JOB=$(sbatch --parsable --export=ALL \
    --dependency=afterok:"$SPLIT_JOB" \
    "$SCRIPT_DIR/sbatch_train.sh")
echo "  train job:    $TRAIN_JOB"

# ── Summary ───────────────────────────────────────────────────────────
echo
echo "=== Chain submitted ==="
echo "  $ANNOTATE_JOB → $SPLIT_JOB → $TRAIN_JOB"
echo "  output root: $OUT"
echo
echo "Monitor:  squeue -u \$USER  +  tail -F bb-*-${ANNOTATE_JOB}.log bb-split-${SPLIT_JOB}.log bb-train-${TRAIN_JOB}.log"
