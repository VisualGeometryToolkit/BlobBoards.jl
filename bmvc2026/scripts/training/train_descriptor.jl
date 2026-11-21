#=
train_descriptor.jl — train a blob descriptor on the corpus of
configs/training/annotations.toml (BlobBoardsTrainingExt `train_descriptor`,
single GPU, single process).

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/training/train_descriptor.jl [--config=<toml>]

Its ONE config of record is configs/training/train_descriptor.toml (or
--config): the corpus (`[runs] annotations`, whose run key is folded into this
run's) and every TrainingConfig field (`[train]`). The checkpoints (best_model.jld2
whenever val FPR@95 improves, the TensorBoard events) land in the run's keyed
product (VisualGeometryToolkitEval `Product`: …/bmvc2026/descriptor/<key>/).

ENVIRONMENT: BlobBoardsTrainingExt triggers on Lux + Optimisers + Enzyme +
Reactant + NNlib + TensorBoardLogger + Zygote, which the paper project does not
declare (README.md beside this file); they must be on the load path. CUDA is
NOT loaded here: Reactant brings its own cuDNN.
=#
# reproduce: reads configs/training/train_descriptor.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

using BlobBoards
using Lux, Optimisers, Enzyme, Reactant, NNlib, TensorBoardLogger, Zygote
using VisualGeometryToolkitEval.Experiments: produce, product_dir, Produced

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: figure_config, load_config, stage_product

const Training = Base.get_extension(BlobBoards, :BlobBoardsTrainingExt)
Training === nothing && error("BlobBoardsTrainingExt is not loaded: its triggers (Lux, Optimisers, Enzyme, Reactant, NNlib, " *
    "TensorBoardLogger, Zygote) must be on the load path (scripts/training/README.md)")

const CFG, CFG_PATH = figure_config("train_descriptor"; default = projectdir("configs", "training", "train_descriptor.toml"))
const ANNOTATIONS = load_config(CFG["runs"]["annotations"])
# The run is keyed on this config with the corpus' product folded in (`upstream`).
const CORPUS = stage_product(ANNOTATIONS, "annotations"; output = "splits")
const PRODUCT = stage_product(CFG, "descriptor"; output = "best_model.jld2", upstream = Dict("annotations" => CORPUS))
const KEY = PRODUCT.key
const RUN_DIR = product_dir(PRODUCT)
# the corpus, where its producer wrote it (scripts/training/build_annotations.jl)
const SPLITS = joinpath(product_dir(CORPUS), "splits")

function main()
    T = CFG["train"]
    cfg = Training.TrainingConfig(;
        blobs_per_batch = T["blobs_per_batch"], views_per_blob = T["views_per_blob"],
        confusers_per_batch = T["confusers_per_batch"], min_track_length = T["min_track_length"],
        temperature = Float32(T["temperature"]), optimizer = Symbol(T["optimizer"]), lr = T["lr"],
        weight_decay = T["weight_decay"], epochs = T["epochs"], batches_per_epoch = T["batches_per_epoch"],
        eval_interval = T["eval_interval"], patch_size = T["patch_size"], descriptor_dim = T["descriptor_dim"],
        dropout = Float32(T["dropout"]),
    )
    train, val = joinpath(SPLITS, "train.jld2"), joinpath(SPLITS, "val.jld2")
    isfile(train) || error("no corpus at $(SPLITS): run scripts/training/build_annotations.jl on $(CFG["runs"]["annotations"]) first")
    @info "training" run = KEY train val RUN_DIR
    result = Training.train_descriptor(train, val; cfg, output_dir = RUN_DIR)
    @info "training done" epochs = result.epoch best_fpr = result.best_fpr
    return Produced(nothing; inputs = [train, val])
end

produce(_ -> main(), PRODUCT; driver = "bmvc2026/scripts/training/train_descriptor.jl", config_source = CFG_PATH)
println("train_descriptor: run $(KEY) at $(RUN_DIR)")
