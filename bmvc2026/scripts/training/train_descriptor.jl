using BlobBoards
using Logging, Dates

# Training pipeline lives in the BlobBoardsTrainingExt extension; loading
# any of these trigger packages activates it, then we grab the extension
# module to access TrainingConfig + train_descriptor.
using Lux, Optimisers, Enzyme, Reactant, NNlib, TensorBoardLogger, Zygote
const Training = Base.get_extension(BlobBoards, :BlobBoardsTrainingExt)
@assert Training !== nothing "BlobBoardsTrainingExt failed to load — check that all triggers (Lux/Optimisers/Enzyme/Reactant/NNlib/TensorBoardLogger/Zygote) are in the active project."
const TrainingConfig = Training.TrainingConfig
const train_descriptor = Training.train_descriptor


# Direct all @info/@warn/@error to a log file (tail -f to monitor)
# _build_log_io = open(joinpath(@__DIR__, "train_descriptor.log"), "w")
# global_logger(SimpleLogger(_build_log_io, Logging.Info))

# @info "=== train descriptor started ==="
# flush(_build_log_io)

root = data_dir()

# =============================
# Parameters
# =============================
train_path = joinpath(root, "blobboard_training/training/splits/train.jld2")
val_path = joinpath(root, "blobboard_training/training/splits/val.jld2")
out_path = joinpath(root, "checkpoints")
latest_model = joinpath(".", "checkpoints", "best_model.jld2")

train_cfg = TrainingConfig()

# ---- Batch size (tuned for NVIDIA B200, 180 GB HBM3e) ----
# Total batch = blobs_per_batch * views_per_blob + confusers_per_batch
#             = 384 * 10 + 2048 = 5888 patches per step (≈30 GB GPU memory).
#
# B200 has ~6× the memory headroom of A100; the previous A100-tuned config
# (96 / 10 / 512 = 1472 patches, ~10 GB) was leaving most of the GPU idle.
# SupCon literature shows monotonic gains up to ~4–8K batch size; 5888 is
# well within the sweet spot.
#
# If you change these, also rescale:
#   * Learning rate: linear-scaling rule for SGD, sqrt rule for AdamW.
#     Going from old 1472 → new 5888 = 4× larger; multiply AdamW LR by 2.
#   * Warmup: bigger batches benefit from longer LR warmup (5–10% of training).
#
# Monitor `nvidia-smi -l 5` during the first run; if HBM usage stays well
# below 60 GB, push blobs_per_batch / confusers_per_batch up. Above ~80 GB
# back off to leave room for activation-memory peaks.
train_cfg.blobs_per_batch = 384    # was 96 (A100 default 64)
train_cfg.views_per_blob = 10     # mean of training dataset is ~10
train_cfg.confusers_per_batch = 2048   # was 512
train_cfg.batches_per_epoch = 0      # auto: ceil(n_labels / blobs_per_batch)

train_cfg.patch_size = 64


# =============================
# Train
# =============================
@info "Start training..."

result = train_descriptor(
    train_path,
    val_path;
    cfg = train_cfg,
    output_dir = out_path
)

@info "\n\nTraining done."
@info "Results:"
@info "  Epochs: $(result.epoch)"
@info "  Best FPR@95: $(result.best_fpr)"
