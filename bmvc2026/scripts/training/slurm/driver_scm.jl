#=
import VisualGeometryToolkitIO as VgtkIO
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

import VisualGeometryToolkitIO   # canonical vgio/cameras/v3 camera stores
using VisualGeometryToolkitCore: BoundedCamera   # dispatch token for VisualGeometryToolkitIO.load

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

@everywhere include(joinpath(@__DIR__, "annotate_worker.jl"))   # _WORKER, _init_worker!, _process_source

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
