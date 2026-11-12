#=
build_annotations.jl — the descriptor training corpus: annotate every source of
the training footage with its boards (per-frame canonical patches), then split
by board identity into train/val/test.

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/training/build_annotations.jl [--config=<toml>]

Its ONE config of record is configs/training/annotations.toml (or --config):
the inputs (`[inputs.<id>]`: the boards, the photo and video sets, the network
that finds the boards), the detector (`[annotate]`) and the split (`[split]`).
The corpus is data of its own tier (`[output] tier`): it lands in
VisualGeometryToolkitEval's `keyed_data_dir` (`data_root()/<tier>/annotations-<key>`),
written by this one producer and never into another tier's tree:

  <run>/truth_images/   per-frame annotations (+ refs/, _log.jsonl)
  <run>/splits/         train.jld2, val.jld2, test.jld2
  <run>/build.log       the timestamped log of every launch on this run

The same config is the same directory: an interrupted run resumes there (each
source's _log.jsonl marker skips what is done); a changed value is a new run.
A source directory is calibrated when a sibling `<dir>.json` camera exists
(PoseGeometry), else homography-only. CUDA is required (the adaptation of
elliptic frames).
=#
# reproduce: reads configs/training/annotations.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

using BlobBoards
using BlobBoards: build_annotations   # advanced API (unexported)
using CUDA, cuDNN
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: BoundedCamera   # dispatch token for VisualGeometryToolkitIO.load (pixel operations take a BoundedCamera)
using VisualGeometryToolkitEval.Experiments: run_key, keyed_data_dir, write_run_records, run_complete
using Logging, Dates

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: figure_config, input_path, network_path, gallery_path

const CFG, CFG_PATH = figure_config("annotations"; default = projectdir("configs", "training", "annotations.toml"))
const KEY = run_key(CFG)
const RUN_DIR = keyed_data_dir(CFG, CFG["output"]["tier"], "annotations")
const ANN, SPLIT = CFG["annotate"], CFG["split"]

# Timestamped lines, tee'd to stdout and the run's build.log.
struct TimestampLogger <: Logging.AbstractLogger
    streams::Vector{IO}
    min_level::Logging.LogLevel
end
Logging.shouldlog(::TimestampLogger, _, _, _, _) = true
Logging.min_enabled_level(logger::TimestampLogger) = logger.min_level
Logging.catch_exceptions(::TimestampLogger) = false
function Logging.handle_message(logger::TimestampLogger, level, message, _module, group, id, filepath, line; kwargs...)
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")   # a log line's time, not an output path
    for io in logger.streams
        println(io, "[$timestamp] [$level] $message")
        flush(io)
    end
    return
end

const _IMG_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp")
const _VID_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")
_has_ext(p, exts) = lowercase(splitext(p)[2]) in exts

# Every immediate subdirectory D of `kind_dir`: its files of `exts` (sorted,
# the first `max_sources` when that is positive) and its camera, the sibling
# D.json when it exists.
function _discover(kind_dir::AbstractString, exts, max_sources::Integer)
    out = Tuple{Vector{String}, Union{String, Nothing}}[]
    isdir(kind_dir) || return out
    for entry in sort(readdir(kind_dir; join = true))
        isdir(entry) || continue
        files = sort!(filter(f -> _has_ext(f, exts), readdir(entry; join = true)))
        max_sources > 0 && (files = first(files, max_sources))
        isempty(files) && continue
        push!(out, (files, isfile(entry * ".json") ? entry * ".json" : nothing))
    end
    return out
end

# The detector of record, per source: PoseGeometry with the source's camera, or
# the homography path for an uncalibrated one.
function _config_for(cam)
    detect = DetectConfig(;
        polarity = Symbol(ANN["polarity"]), peak_threshold = ANN["peak_threshold"],
        edge_threshold = ANN["edge_threshold"], nms = EllipseNms(cutoff = Float32(ANN["nms_cutoff"])),
    )
    n = Int(ANN["min_inliers"])
    geometry = cam === nothing ? HomographyGeometry(min_inliers = n) : PoseGeometry(cam; min_inliers = n)
    return FindBoardsConfig(; geometry, detect)
end

function main()
    mkpath(RUN_DIR)
    log_io = open(joinpath(RUN_DIR, "build.log"), "a")
    global_logger(TimestampLogger(IO[stdout, log_io], Logging.Info))
    @info "=== build_annotations: run $(KEY) -> $(RUN_DIR) ==="

    boards_dir = input_path(CFG, "boards")
    network = DescriptorNetwork(network_path(CFG))
    gallery = BoardGallery(gallery_path(CFG), network)
    board_paths = filter(endswith(".json"), readdir(boards_dir; join = true))
    board_feature_dict = Dict(uid => f for (uid, f) in [insert!(gallery, bp, network) for bp in board_paths])
    board_bundles = Dict(let (b, _, ch) = BlobBoards.load_bundle(bp); uid_from_hash(ch) => b end for bp in board_paths)
    @info "$(length(board_paths)) boards loaded"

    max_sources = Int(ANN["max_sources"])
    videos = _discover(input_path(CFG, "videos"), _VID_EXTS, max_sources)
    images = _discover(input_path(CFG, "images"), _IMG_EXTS, max_sources)
    @info "$(sum(length(f) for (f, _) in videos; init = 0)) videos in $(length(videos)) dirs, " *
        "$(sum(length(f) for (f, _) in images; init = 0)) images in $(length(images)) dirs"

    output_dir = joinpath(RUN_DIR, "truth_images")
    common = (; board_bundles, board_paths, output_dir, device = CUDABackend(),
        cutoff = ANN["cutoff"], outer_extent = Float64(ANN["outer_extent"]), min_iou = Float64(ANN["min_iou"]))
    for (files, cam) in videos
        build_annotations(board_feature_dict, network, files; common...,
            n_frames = Int(ANN["video_frames"]), config = _config_for(cam === nothing ? nothing : VgtkIO.load(BoundedCamera, cam)))
    end
    for (files, cam) in images
        build_annotations(board_feature_dict, network, files; common...,
            config = _config_for(cam === nothing ? nothing : VgtkIO.load(BoundedCamera, cam)))
    end

    result = split_training_data(output_dir; output_dir = joinpath(RUN_DIR, "splits"),
        validation_fraction = SPLIT["validation_fraction"], test_fraction = SPLIT["test_fraction"], seed = SPLIT["seed"])
    @info "train $(result.stats.train_patches) patches ($(result.stats.train_boards) boards), " *
        "val $(result.stats.val_patches) ($(result.stats.val_boards_n)), test $(result.stats.test_patches) ($(result.stats.test_boards_n))"
    write_run_records(RUN_DIR, CFG; driver = "bmvc2026/scripts/training/build_annotations.jl", config_source = CFG_PATH, key = KEY,
        inputs = [boards_dir, input_path(CFG, "images"), input_path(CFG, "videos"), network_path(CFG)], producer = true)
    close(log_io)
    return RUN_DIR
end

run_complete(RUN_DIR; sidecar = joinpath("splits", "train.jld2")) ?
    println("build_annotations: run $(KEY) complete at $(RUN_DIR)") : main()
