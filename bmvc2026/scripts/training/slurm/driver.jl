#=
import VisualGeometryToolkitIO as VgtkIO
Distributed annotation driver. Run from a login-node Julia (or `tmux`); it
allocates SLURM workers via ClusterManagers.SlurmManager, dispatches per-
source annotation work, and tears the workers down on exit.

Usage:
  julia --project=<env-with-CUDA-AcceleratedKernels-NNlib> \
        examples/training/slurm/driver.jl <sources_file>

Required env vars:
  BB_OUTPUT_DIR    annotation output directory (where per-image JLD2s go)
  BB_BOARDS_DIR    directory of board JSONs
  BB_MODEL_PATH    descriptor model (.onnx)
  BB_GALLERY_PATH  board gallery (.jld2)

Optional env vars:
  BB_N_FRAMES      stride videos to ~this many frames each (default 30)
  BB_N_WORKERS     number of SLURM-allocated workers (default 5)
  BB_PARTITION     SLURM partition (default "gpu")
  BB_GRES          per-worker SLURM --gres (default "gpu:1")
  BB_TIME          per-worker SLURM --time (default "01:00:00")
  BB_CPUS_PER_TASK SLURM --cpus-per-task per worker (default 4)
  BB_MEM_PER_CPU   SLURM --mem-per-cpu per worker (default "16G")

The sources file supplied as ARGS[1] has one entry per line:
  /abs/path/to/source.jpg|/abs/path/to/camera.json
  /abs/path/to/uncalibrated.png|
=#

using Distributed
using ClusterManagers

import VisualGeometryToolkitIO   # canonical vgio/cameras/v3 camera stores
using VisualGeometryToolkitCore: BoundedCamera   # dispatch token for VisualGeometryToolkitIO.load

# ── Master-side config ────────────────────────────────────────────────
const SOURCES_FILE = ARGS[1]
const OUTPUT_DIR = ENV["BB_OUTPUT_DIR"]
const BOARDS_DIR = ENV["BB_BOARDS_DIR"]
const MODEL_PATH = ENV["BB_MODEL_PATH"]
const GALLERY_PATH = ENV["BB_GALLERY_PATH"]
const N_FRAMES = parse(Int, get(ENV, "BB_N_FRAMES", "30"))
const N_WORKERS = parse(Int, get(ENV, "BB_N_WORKERS", "5"))

# Parse source-camera pairs
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

# ── Allocate SLURM workers ────────────────────────────────────────────
# Forward the driver's active project to each worker so `using BlobBoards`
# (and the ML weakdeps) finds them — fresh Julia processes don't inherit
# the driver's project automatically.
#
# GPU isolation strategy: ask SLURM for N GPUs at the allocation level
# (--gres=gpu:N). All N tasks in the allocation see all N GPUs via
# CUDA_VISIBLE_DEVICES. Then in _init_worker! we manually pin each Julia
# worker to a unique GPU via CUDA.device!(myid() - 2). This avoids
# --gpus-per-task which this cluster's accounting plugin rejects with
# InvalidAccount.
const _PROJECT = dirname(Base.active_project())
@info "Allocating $N_WORKERS SLURM workers (project=$_PROJECT)..."
addprocs(
    SlurmManager(N_WORKERS);
    partition = get(ENV, "BB_PARTITION", "gpu"),
    gres = "gpu:$N_WORKERS",     # whole allocation gets N_WORKERS GPUs
    t = get(ENV, "BB_TIME", "01:00:00"),
    cpus_per_task = parse(Int, get(ENV, "BB_CPUS_PER_TASK", "4")),
    mem_per_cpu = get(ENV, "BB_MEM_PER_CPU", "16G"),
    job_file_loc = mktempdir(prefix = "bb_slurm_"),
    exeflags = ["--project=$_PROJECT", "--startup-file=no"],
)
@info "Got $(nworkers()) workers: $(workers())"

# ── Worker-side imports + state ───────────────────────────────────────
@everywhere begin
    using BlobBoards
    using BlobBoards: annotate_blobs   # advanced API (unexported)
    using CUDA, AcceleratedKernels, NNlib
    using FileIO
    using Images: Gray
end

@everywhere include(joinpath(@__DIR__, "annotate_worker.jl"))   # _WORKER, _init_worker!, _process_source

# ── Master: hash check + fan out worker init ──────────────────────────
@info "Master-side hash check + _config.jld2 init..."
let dm = (@eval Main using BlobBoards; Main.BlobBoards.DescriptorNetwork(MODEL_PATH)),
        cfg = (@eval Main using BlobBoards; Main.BlobBoards.FindBoardsConfig(geometry = Main.BlobBoards.HomographyGeometry(min_inliers = 10)))
    BlobBoards.check_annotation_hashes(
        OUTPUT_DIR,
        BlobBoards.network_hash(dm),
        BlobBoards.config_hash(;
            config = cfg, cutoff = BlobBoards.DEFAULT_CUTOFF,
            patch_size = 64, outer_extent = 96.0, min_iou = 0.5
        )
    )
end

@info "Initializing workers..."
@sync for w in workers()
    @async @info remotecall_fetch(_init_worker!, w, MODEL_PATH, GALLERY_PATH, BOARDS_DIR)
end

# ── Drive: pmap with dynamic work-stealing ────────────────────────────
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

# Workers shut down on driver exit (Distributed default).
