#=
Distributed annotation driver — runs INSIDE an sbatch allocation,
spawns SLURM_NTASKS-1 workers via SlurmClusterManager.

The sbatch wrapper requests --ntasks=N+1 --gpus-per-task=1 (driver + N workers,
each in its own cgroup → CUDA_VISIBLE_DEVICES=0 inside each task → no
manual pinning needed; CUDA picks device 0 by default).

Required env vars (from sbatch wrapper):
  BB_OUTPUT_DIR    annotation output directory
  BB_BOARDS_DIR    directory of board JSONs
  BB_MODEL_PATH    descriptor model (.onnx)
  BB_GALLERY_PATH  board gallery (.jld2)
  BB_SOURCES_FILE  flat sources file (one "src|cam" per line)
  BB_N_FRAMES      stride videos to ~this many frames each (default 30)
=#

using Distributed
using SlurmClusterManager
# Master also needs CUDA + cuDNN loaded: line ~107 calls
# BlobBoards.DescriptorNetwork(MODEL_PATH) on the master to compute the
# model hash; ONNXRunTime's :cuda execution provider checks at call time
# that both packages are loaded in the current process. Workers get them
# via the @everywhere block below.
using CUDA, cuDNN

import VisualGeometryIo   # canonical vgio/cameras/v3 camera stores
using VisualGeometryCore: CameraModel   # dispatch token for VisualGeometryIo.load

# ── Master-side config ────────────────────────────────────────────────
const SOURCES_FILE = ENV["BB_SOURCES_FILE"]
const OUTPUT_DIR = ENV["BB_OUTPUT_DIR"]
const BOARDS_DIR = ENV["BB_BOARDS_DIR"]
const MODEL_PATH = ENV["BB_MODEL_PATH"]
const GALLERY_PATH = ENV["BB_GALLERY_PATH"]
const N_FRAMES = parse(Int, get(ENV, "BB_N_FRAMES", "30"))

source_camera_pairs = Tuple{String, String}[]
for line in eachline(SOURCES_FILE)
    isempty(strip(line)) && continue
    parts = split(line, '|'; limit = 2)
    push!(
        source_camera_pairs,
        (String(parts[1]), length(parts) == 2 ? String(parts[2]) : "")
    )
end
@info "Loaded $(length(source_camera_pairs)) source-camera pairs from $SOURCES_FILE"

mkpath(OUTPUT_DIR)

# ── Workers — SlurmClusterManager reads $SLURM_NTASKS, spawns N-1 ─────
@info "Calling addprocs(SlurmManager())..."
addprocs(
    SlurmManager(); exeflags = [
        "--project=$(dirname(Base.active_project()))",
        "--startup-file=no",
    ]
)
@info "Got $(nworkers()) workers: $(workers())"

# ── Worker imports + state ────────────────────────────────────────────
@everywhere begin
    using BlobBoards
    using BlobBoards: annotate_blobs   # advanced API (unexported)
    using CUDA, cuDNN, AcceleratedKernels, NNlib
    using FileIO
    using Images: Gray
end

@everywhere const _WORKER = Ref{NamedTuple}()

@everywhere function _init_worker!(model_path, gallery_path, boards_dir)
    descriptor_network = DescriptorNetwork(model_path)
    gallery = BoardGallery(gallery_path, descriptor_network)
    board_paths = filter(endswith(".json"), readdir(boards_dir; join = true))
    bfd = Dict(
        uid => f for (uid, f) in
            [insert!(gallery, p, descriptor_network) for p in board_paths]
    )
    bb = Dict(
        let (b, _, ch) = BlobBoards.load_bundle(p)
                uid_from_hash(ch) => b
        end
            for p in board_paths
    )
    uid_to_path = Dict(
        let (_, _, ch) = BlobBoards.load_bundle(p)
                uid_from_hash(ch) => p
        end
            for p in board_paths
    )
    _WORKER[] = (
        descriptor_network = descriptor_network,
        board_feature_dict = bfd,
        board_bundles = bb,
        uid_to_path = uid_to_path,
        board_paths = board_paths,
        camera_cache = Dict{String, Any}(),
        find_boards_config = FindBoardsConfig(geometry = HomographyGeometry(min_inliers = 10)),
    )
    "ready on worker $(myid()) — GPU=$(CUDA.device()) (uuid=$(CUDA.uuid(CUDA.device()))) host=$(gethostname())"
end

@everywhere function _process_source(
        source::String, camera_path::String,
        output_dir::String, n_frames::Int
    )
    # Self-init guard: the master's @sync init loop has been observed to
    # silently leave workers uninitialized (likely an @async/@sync swallow
    # of an init exception). Worker env vars are inherited from srun, so
    # we can recover by initializing on first source.
    if !isassigned(_WORKER)
        _init_worker!(ENV["BB_MODEL_PATH"], ENV["BB_GALLERY_PATH"], ENV["BB_BOARDS_DIR"])
    end
    s = _WORKER[]
    BlobBoards.is_source_processed(output_dir, source) && return :skipped
    camera = if isempty(camera_path)
        nothing
    else
        get!(s.camera_cache, camera_path) do;
            VisualGeometryIo.load(CameraModel, camera_path)
        end
    end
    # Camera lives in config.geometry — PoseGeometry(camera) for calibrated
    # sources, HomographyGeometry otherwise. Reuse the worker's detect config.
    base = s.find_boards_config
    geom = isnothing(camera) ? HomographyGeometry(min_inliers = 10) :
        PoseGeometry(camera; min_inliers = 10)
    config = FindBoardsConfig(geometry = geom, detect = base.detect)
    if BlobBoards.is_video(source)
        annotate_blobs(
            s.board_feature_dict, s.descriptor_network, source;
            output_dir, board_bundles = s.board_bundles, board_paths = s.board_paths,
            config, device = CUDABackend(), n_frames
        )
        return :video_done
    else
        img = Gray{Float32}.(FileIO.load(source))
        fp = BlobBoards.image_path(output_dir, basename(source))
        meta = BlobBoards.annotate_frame!(
            BlobBoards.existing_ref_uids(output_dir), output_dir,
            s.board_feature_dict, s.descriptor_network, s.uid_to_path,
            fp, basename(source), img;
            board_bundles = s.board_bundles,
            config, cutoff = BlobBoards.DEFAULT_CUTOFF,
            patch_size = 64, outer_extent = 96.0, min_iou = 0.5,
            device = CUDABackend()
        )
        BlobBoards.append_log(output_dir, meta)
        return meta
    end
end

# ── Master: hash check + per-worker init with explicit GPU index ──────
@info "Master-side hash check + _config.jld2 init..."
let dm = BlobBoards.DescriptorNetwork(MODEL_PATH),
        cfg = BlobBoards.FindBoardsConfig(geometry = BlobBoards.HomographyGeometry(min_inliers = 10))
    BlobBoards.check_annotation_hashes(
        OUTPUT_DIR,
        BlobBoards.network_hash(dm),
        BlobBoards.config_hash(;
            config = cfg, cutoff = BlobBoards.DEFAULT_CUTOFF,
            patch_size = 64, outer_extent = 96.0, min_iou = 0.5
        )
    )
end

# Each worker is in its own SLURM task cgroup → sees its single allocated
# GPU as device 0. Just call _init_worker! on each — no pinning math.
@info "Initializing $(nworkers()) workers (one GPU each via cgroup)..."
@sync for w in workers()
    @async @info remotecall_fetch(
        _init_worker!, w,
        MODEL_PATH, GALLERY_PATH, BOARDS_DIR
    )
end

# ── Drive ─────────────────────────────────────────────────────────────
@info "Annotating $(length(source_camera_pairs)) sources..."
t_start = time()
results = pmap(WorkerPool(workers()), source_camera_pairs) do (src, cam)
    try
        _process_source(src, cam, OUTPUT_DIR, N_FRAMES)
    catch e
        @warn "Source failed" source = basename(src) exception = (e, catch_backtrace())
        :failed
    end
end
elapsed = time() - t_start

n_skip = count(==(:skipped), results)
n_vid = count(==(:video_done), results)
n_fail = count(==(:failed), results)
n_img = length(results) - n_skip - n_vid - n_fail
@info "Done in $(round(elapsed; digits = 1))s — $n_img images, $n_vid videos, $n_skip skipped, $n_fail failed"
