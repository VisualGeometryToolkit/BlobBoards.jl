#!/bin/bash
# sbatch wrapper: run split_training_data on a high-memory node.
# Concatenates per-frame patches + confusers (≈240 GB for the BMVC 2026
# corpus) into one tensor, partitions by board UID, writes train/val/test.
#
# Submitted by submit_bmvc2026.sh as an afterok dependency of the annotate
# job, with BB_ANNOTATION_DIR / BB_SPLITS_DIR / BB_PROJECT exported.
#SBATCH --job-name=bb-split
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bb-split-%j.log
#SBATCH --error=logs/bb-split-%j.log

set -euo pipefail

: "${BB_PROJECT:?must be set}"
: "${BB_ANNOTATION_DIR:?must be set (path to truth_images/)}"
: "${BB_SPLITS_DIR:?must be set (output dir for train/val/test .jld2)}"

VALIDATION_FRACTION="${BB_VALIDATION_FRACTION:-0.1}"
TEST_FRACTION="${BB_TEST_FRACTION:-0.1}"
SPLIT_SEED="${BB_SPLIT_SEED:-42}"

echo "=== sbatch job $SLURM_JOB_ID on $(hostname) — split ==="
echo "    annot       = $BB_ANNOTATION_DIR"
echo "    splits      = $BB_SPLITS_DIR"
echo "    validation_fraction = $VALIDATION_FRACTION"
echo "    test_fraction       = $TEST_FRACTION"
echo "    seed                = $SPLIT_SEED"

mkdir -p "$BB_SPLITS_DIR"

exec julia --startup-file=no --project="$BB_PROJECT" -e "
    using BlobBoards
    BlobBoards.split_training_data(\"$BB_ANNOTATION_DIR\";
        output_dir   = \"$BB_SPLITS_DIR\",
        validation_fraction = $VALIDATION_FRACTION,
        test_fraction = $TEST_FRACTION,
        seed = $SPLIT_SEED)
"
