#=
train_descriptor.jl — train a blob descriptor on the corpus of
configs/training/annotations.toml (BlobBoardsTrainingExt `train_descriptor`,
single GPU, single process).

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/training/train_descriptor.jl [--config=<toml>]

Its ONE config of record is configs/training/train_descriptor.toml (or
--config): the corpus (`[runs] annotations`, whose run key is folded into this
run's) and every TrainingConfig field (`[train]`). The checkpoints (best_model.jld2
whenever val FPR@95 improves, the TensorBoard events) land in the run's keyed
directory (VisualGeometryToolkitEval `keyed_cell_dir`: …/bmvc2026/descriptor-<run_key>/).

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
using VisualGeometryToolkitEval.Experiments: run_key, keyed_cell_dir, keyed_data_dir, write_run_records, run_complete

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: figure_config, load_config, EXPERIMENT

const Training = Base.get_extension(BlobBoards, :BlobBoardsTrainingExt)
Training === nothing && error("BlobBoardsTrainingExt is not loaded: its triggers (Lux, Optimisers, Enzyme, Reactant, NNlib, " *
    "TensorBoardLogger, Zygote) must be on the load path (scripts/training/README.md)")

const CFG, CFG_PATH = figure_config("train_descriptor"; default = projectdir("configs", "training", "train_descriptor.toml"))
const ANNOTATIONS = load_config(CFG["runs"]["annotations"])
# The run is keyed on this config with the corpus' run key folded in (`upstream`).
const UPSTREAM = Dict{String, String}("annotations" => run_key(ANNOTATIONS))
const KEY = run_key(CFG; upstream = UPSTREAM)
const RUN_DIR = keyed_cell_dir(CFG, EXPERIMENT, "descriptor"; upstream = UPSTREAM)
# the corpus: data of its tier, where its producer wrote it (Eval keyed_data_dir)
const SPLITS = joinpath(keyed_data_dir(ANNOTATIONS, ANNOTATIONS["output"]["tier"], "annotations"), "splits")

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
    write_run_records(RUN_DIR, CFG; driver = "bmvc2026/scripts/training/train_descriptor.jl", config_source = CFG_PATH,
        key = KEY, inputs = [train, val])
    return RUN_DIR
end

run_complete(RUN_DIR; sidecar = "best_model.jld2") ? println("train_descriptor: run $(KEY) complete at $(RUN_DIR)") : main()
