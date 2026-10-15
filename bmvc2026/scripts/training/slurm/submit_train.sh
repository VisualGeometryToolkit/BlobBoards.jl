#!/bin/bash
# Submit single-GPU descriptor training to SLURM.
set -euo pipefail

SPLITS="${SPLITS:?set SPLITS to the directory holding train.jld2 and val.jld2}"
OUT="${OUT:?set OUT to the checkpoint directory}"
PROJECT="${PROJECT:?set PROJECT to a Julia env with BlobBoards and the training deps}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

[ -f "$SPLITS/train.jld2" ] || { echo "missing $SPLITS/train.jld2" >&2; exit 1; }
[ -f "$SPLITS/val.jld2" ]   || { echo "missing $SPLITS/val.jld2" >&2; exit 1; }
mkdir -p "$OUT"

export BB_PROJECT="$PROJECT"
export BB_TRAIN_PATH="$SPLITS/train.jld2"
export BB_VAL_PATH="$SPLITS/val.jld2"
export BB_OUTPUT_DIR="$OUT"
export BB_DRIVER="$SCRIPT_DIR/train_driver.jl"

echo "=== Submitting training sbatch ==="
echo "    splits: $SPLITS"
echo "    output: $OUT"
mkdir -p logs   # SLURM opens --output before the job runs; it will not create the dir
sbatch --export=ALL "$SCRIPT_DIR/sbatch_train.sh"

echo
echo "Monitor: squeue -u \$USER  +  tail -f logs/bb-train-*.log"
