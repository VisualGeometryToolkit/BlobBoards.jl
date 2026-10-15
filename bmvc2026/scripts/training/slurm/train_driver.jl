#=
SLURM-side training driver. Single GPU, single Julia process — no Distributed.
The sbatch wrapper allocates one task with --gpus-per-task=1 and exports BB_*
paths so this script can run unchanged on different splits.

Required env vars (from sbatch wrapper):
  BB_TRAIN_PATH    train split jld2 (e.g. <splits>/train.jld2)
  BB_VAL_PATH      val split jld2
  BB_OUTPUT_DIR    where checkpoints land
=#

using BlobBoards
using Logging

# Training pipeline lives in the BlobBoardsTrainingExt extension; loading
# any of these trigger packages activates it, then we grab the extension
# module to access TrainingConfig + train_descriptor.
using Lux, Optimisers, Enzyme, Reactant, NNlib, TensorBoardLogger, Zygote
# CUDA + cuDNN intentionally NOT loaded: Reactant brings its own bundled
# cuDNN 9.14 via XLA. Loading CUDA.jl's cuDNN 9.20 alongside causes
# libReactantExtra.so symbol mismatches.
const Training = Base.get_extension(BlobBoards, :BlobBoardsTrainingExt)
const TrainingConfig = Training.TrainingConfig
const train_descriptor = Training.train_descriptor

const TRAIN_PATH = ENV["BB_TRAIN_PATH"]
const VAL_PATH = ENV["BB_VAL_PATH"]
const OUTPUT_DIR = ENV["BB_OUTPUT_DIR"]
mkpath(OUTPUT_DIR)

@info "training paths" TRAIN_PATH VAL_PATH OUTPUT_DIR

# B200-tuned batch size (mirrors examples/training/train_descriptor.jl).
# Total batch = blobs * views + confusers = 384*10 + 2048 = 5888 patches (~30 GB).
train_cfg = TrainingConfig(
    blobs_per_batch = 384,
    views_per_blob = 10,
    confusers_per_batch = 2048,
    batches_per_epoch = 0,
    patch_size = 64,
)

@info "TrainingConfig" blobs_per_batch = train_cfg.blobs_per_batch views_per_blob = train_cfg.views_per_blob confusers_per_batch = train_cfg.confusers_per_batch patch_size = train_cfg.patch_size
flush(stderr)

result = train_descriptor(TRAIN_PATH, VAL_PATH; cfg = train_cfg, output_dir = OUTPUT_DIR)

@info "Training done" epochs = result.epoch best_fpr = result.best_fpr
flush(stderr)
