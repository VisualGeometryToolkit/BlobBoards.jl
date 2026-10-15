#=
Build Annotation Database — annotate all data sources into a per-frame
JLD2 directory. See summarize_annotations(output_dir) for after-run stats.

Processes videos and images from the NFS data directory, detecting boards
and extracting canonical patches. Calibrated sources use camera intrinsics;
uncalibrated sources use homography-only matching.

Usage:
  QUICK_TEST=true  include("examples/training/build_annotations.jl")  # 2 videos, 3 images per source
  include("examples/training/build_annotations.jl")                    # full dataset
=#

using BlobBoards
using BlobBoards: build_annotations   # advanced API (unexported)
using CUDA, cuDNN

import VisualGeometryIo   # canonical vgio/cameras/v3 camera stores
using VisualGeometryCore: CameraModel   # dispatch token for VisualGeometryIo.load
using Logging, Dates
import VisualGeometryEval
using Infiltrator


root = joinpath(data_dir(), "blobboard_training")

# Custom logger: timestamped lines, tee'd to stdout + a per-run log file.
struct TimestampLogger <: Logging.AbstractLogger
    streams::Vector{IO}
    min_level::Logging.LogLevel
end

Logging.shouldlog(::TimestampLogger, _, _, _, _) = true
Logging.min_enabled_level(logger::TimestampLogger) = logger.min_level
Logging.catch_exceptions(::TimestampLogger) = false

function Logging.handle_message(logger::TimestampLogger, level, message, _module, group, id, filepath, line; kwargs...)
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")   # a log line's time, not an output path
    msg = "[$timestamp] [$level] $message"
    for io in logger.streams
        println(io, msg)
        flush(io)
    end
    return
end


QUICK_TEST = get(ENV, "QUICK_TEST", "false") == "true" || @isdefined(QUICK_TEST) && QUICK_TEST == true
if QUICK_TEST
    @info "QUICK_TEST mode — small subset only"
end
_take(v, n) = v[1:min(n, length(v))]

# The run directory is KEYED by the annotation settings (VisualGeometryEval
# Experiments `run_key`: SHA-256 of their sorted TOML), not by the wall clock:
# the same settings land in the same directory, which `_is_source_processed`
# then resumes; changed settings are a new directory. RESUME_DIR still points
# at an explicit directory.
const ANNOTATION_SETTINGS = Dict{String, Any}(
    "experiment" => Dict{String, Any}(
        "driver" => "build_annotations", "root" => basename(root), "quick_test" => QUICK_TEST,
        "network" => basename(descriptor_network_path()),
    ),
)
run_dir = let env_dir = get(ENV, "RESUME_DIR", "")
    isempty(env_dir) ?
        joinpath(root, "training", "annotations-" * VisualGeometryEval.Experiments.run_key(ANNOTATION_SETTINGS)) :
        env_dir
end
mkpath(run_dir)
_build_log_io = open(joinpath(run_dir, "build.log"), "a")
global_logger(TimestampLogger(IO[stdout, _build_log_io], Logging.Info))

@info "=== build_annotations started ==="
@info "run_dir = $run_dir"

# =============================================================================
# Paths
# =============================================================================

boards_dir = joinpath(root, "boards")
# Network from configs/default.toml — the one place a default entry point
# learns which descriptor to load. Same log-polar _cuda net as the paper.
model_path = descriptor_network_path()
gallery_path = joinpath(root, "models/board_gallery.jld2")
output_dir = joinpath(run_dir, "truth_images")
output_splits_path = joinpath(run_dir, "splits")
nfs_dir = nothing # joinpath(root, "training")

# =============================================================================
# Setup — boards, descriptor model, gallery
# =============================================================================

@info "Building gallery $gallery_path ..."
descriptor_network = DescriptorNetwork(model_path)
gallery = BoardGallery(gallery_path, descriptor_network)
board_paths = filter(endswith(".json"), readdir(boards_dir; join = true))
board_feature_dict = Dict(
    uid => features for (uid, features) in
        [insert!(gallery, bp, descriptor_network) for bp in board_paths]
)
board_bundles = Dict(
    let (b, _, ch) = BlobBoards.load_bundle(bp)
            uid_from_hash(ch) => b
    end
        for bp in board_paths
)
@info "$(length(board_paths)) boards loaded"

# =============================================================================
# Data sources — auto-discover from <root>/{images,videos}/<subdir>/
#
# For each immediate subdir D of images/ (or videos/):
#   • calibration = sibling D.json if it exists, else nothing
#   • files       = matching extensions inside D (non-recursive)
# =============================================================================

@isdefined(_IMG_EXTS) || (const _IMG_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"))
@isdefined(_VID_EXTS) || (const _VID_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"))

_has_ext(p, exts) = lowercase(splitext(p)[2]) in exts

function _discover(kind_dir::AbstractString, exts)
    out = Tuple{Vector{String}, Union{String, Nothing}}[]
    isdir(kind_dir) || return out
    for entry in sort(readdir(kind_dir; join = true))
        isdir(entry) || continue
        cal_path = entry * ".json"
        calib = isfile(cal_path) ? cal_path : nothing
        files = sort!(filter(f -> _has_ext(f, exts), readdir(entry; join = true)))
        isempty(files) && continue
        push!(out, (files, calib))
    end
    return out
end

videos_calibrated = _discover(joinpath(root, "videos"), _VID_EXTS)
images_calibrated = _discover(joinpath(root, "images"), _IMG_EXTS)

@info "Discovered $(sum(length(f) for (f, _) in videos_calibrated; init = 0)) videos across $(length(videos_calibrated)) dirs"
@info "Discovered $(sum(length(f) for (f, _) in images_calibrated; init = 0)) images across $(length(images_calibrated)) dirs"

# --- Quick test: subset everything ---
if QUICK_TEST
    videos_calibrated = [(images[1:min(2, end)], calib) for (images, calib) in videos_calibrated]
    images_calibrated = [(images[1:min(3, end)], calib) for (images, calib) in images_calibrated]
    output_dir = joinpath(run_dir, "truth_quicktest")
end

# =============================================================================
# Common kwargs
# =============================================================================

# Detector settings matched to examples/pipeline/find_boards.jl —
# unified detection across the find_boards pipeline and annotation pipeline.
CUTOFF = BlobBoards.DEFAULT_CUTOFF
DETECT_CONFIG = DetectConfig(
    polarity = :dark,
    peak_threshold = 0.02,
    edge_threshold = 5.0,
    nms = EllipseNms(cutoff = Float32(CUTOFF))
)

# Per-source camera lives in config.geometry: PoseGeometry(cam) for calibrated
# sources, HomographyGeometry() otherwise. min_inliers=10 either way.
_config_for(cam) = FindBoardsConfig(
    geometry = isnothing(cam) ? HomographyGeometry(min_inliers = 10) :
        PoseGeometry(cam; min_inliers = 10),
    detect = DETECT_CONFIG
)

common_kw = (;
    board_bundles,
    board_paths,
    output_dir,
    device = CUDABackend(),
)

# =============================================================================
# Annotate — uncalibrated sources (homography)
# =============================================================================

n_vid_frames = QUICK_TEST ? 5 : 30
# TODO: wait until homography patch extraction bug is fixed

# =============================================================================
# Annotate — calibrated sources
# =============================================================================

@info "=== Annotation ==="
@info "Annotate videos..."
for (videos, calib_file) in videos_calibrated
    video_camera = isnothing(calib_file) ? nothing : VisualGeometryIo.load(CameraModel, calib_file)
    build_annotations(
        board_feature_dict, descriptor_network, videos;
        common_kw..., n_frames = n_vid_frames, config = _config_for(video_camera)
    )
end

@info "Annotate images..."
for (images, calib_file) in images_calibrated
    image_camera = isnothing(calib_file) ? nothing : VisualGeometryIo.load(CameraModel, calib_file)
    build_annotations(
        board_feature_dict, descriptor_network, images;
        common_kw..., config = _config_for(image_camera)
    )
end


# =============================================================================
# Split into train / val / test
# =============================================================================

@info "=== Splitting ==="
# By board identity; the corpus split is chosen HERE (no library default).
result = split_training_data(
    output_dir;
    output_dir = output_splits_path,
    validation_fraction = 0.1,
    test_fraction = 0.1,
    seed = 42,
)

@info "Done!"
@info "  Train: $(result.stats.train_patches) patches ($(result.stats.train_boards) boards)"
@info "  Val:   $(result.stats.val_patches) patches ($(result.stats.val_boards_n) boards)"
@info "  Test:  $(result.stats.test_patches) patches ($(result.stats.test_boards_n) boards)"
@info "  Files: $(result.train_path), $(result.val_path), $(result.test_path)"

# =============================================================================
# Copy to NFS for preservation
# =============================================================================

if !isnothing(nfs_dir)
    mkpath(nfs_dir)
    # Output dir (recursive) + per-split JLD2 files
    for src in [output_dir, result.train_path, result.val_path, result.test_path]
        ispath(src) || continue
        dst = joinpath(nfs_dir, basename(src))
        sz = isfile(src) ? filesize(src) : sum(filesize.(filter(isfile, readdir(src; join = true))); init = 0)
        @info "Copying $(basename(src)) → $nfs_dir ($(round(sz / 1.0e9; digits = 2)) GB)"
        cp(src, dst; force = true)
    end
    @info "All files copied to NFS: $nfs_dir"
end
