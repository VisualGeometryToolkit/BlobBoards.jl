#=
BMVC 2026 descriptor training driver.

Reads BB_TRAIN_PATH / BB_VAL_PATH / BB_OUTPUT_DIR from env (set by the
sbatch wrapper). Single-GPU, single-process — no Distributed. Used by
examples/training/slurm/submit_bmvc2026.sh as the BB_DRIVER for
sbatch_train.sh.

Config matches B200 (180 GB HBM3e). Drop blobs_per_batch / confusers_per_batch
4× and halve AdamW LR (sqrt rule) for A100 (40 GB).
=#

using BlobBoards
using Logging, Dates

# Training pipeline lives in the BlobBoardsTrainingExt extension. Loading
# any trigger package activates it; CUDA + cuDNN intentionally NOT loaded:
# Reactant bundles its own cuDNN 9.14 via XLA. Loading CUDA.jl's cuDNN
# alongside causes libReactantExtra.so symbol mismatches (see
# slurm/train_driver.jl comment).
using Lux, Optimisers, Enzyme, Reactant, NNlib, TensorBoardLogger, Zygote
const Training = Base.get_extension(BlobBoards, :BlobBoardsTrainingExt)
@assert Training !== nothing "BlobBoardsTrainingExt failed to load"
const TrainingConfig = Training.TrainingConfig
const train_descriptor = Training.train_descriptor

const TRAIN_PATH = ENV["BB_TRAIN_PATH"]
const VAL_PATH = ENV["BB_VAL_PATH"]
const OUTPUT_DIR = ENV["BB_OUTPUT_DIR"]
mkpath(OUTPUT_DIR)

@info "BMVC 2026 training paths" TRAIN_PATH VAL_PATH OUTPUT_DIR

cfg = TrainingConfig(
    blobs_per_batch = 384,
    views_per_blob = 10,
    confusers_per_batch = 2048,
    batches_per_epoch = 0,
    patch_size = 64,
)

@info "TrainingConfig" cfg.blobs_per_batch cfg.views_per_blob cfg.confusers_per_batch cfg.patch_size
flush(stderr)

result = train_descriptor(TRAIN_PATH, VAL_PATH; cfg = cfg, output_dir = OUTPUT_DIR)

@info "Training done" epochs = result.epoch best_fpr = result.best_fpr
flush(stderr)
