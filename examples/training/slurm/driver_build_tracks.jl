#=
Distributed multi-GPU track-building driver — mirrors driver_scm.jl.

Runs INSIDE an sbatch allocation and spawns SLURM_NTASKS-1 workers via
SlurmClusterManager. Each task is cgroup-isolated to a single GPU
(--gpus-per-task=1), so CUDA, ONNX *and* VulkanSift all land on that one GPU as
device 0 — no manual pinning. (Do NOT use the whole-allocation --gres + manual
CUDA.device! model: it pins CUDA only, and Vulkan detection would collide on one
GPU.)

Work = one task per media file + one task per board anchor, pmap'd with
work-stealing across the GPU workers. Each task writes its OWN .tracks file (no
shared writer → no concurrent-write corruption); the master then concatenates
them with `merge_track_files`. Resume is per-task-file: a task whose output file
already exists is skipped, and the final merge skips sequences already present.

Required env (from the sbatch wrapper):
  BB_VIDEO_DIR    directory tree of videos/photos
  BB_BOARDS_DIR   directory of board JSONs
  BB_MODEL_PATH   descriptor model (.onnx)
  BB_OUTPUT_DIR   output directory
Optional env (defaults match build_tracks_for_sequences / build_tracks.jl):
  BB_OUTPUT_NAME       base output name        (default "tracks")
  BB_PATCH_TYPE        logpolar|cartesian|original (default "logpolar")
  BB_SCALE             patch half-width in σ   (default 32)
  BB_N_FRAMES          uniformly sample N frames/video (default: unset = all)
  BB_MAX_DETECTIONS    per-frame detection cap (default 15000)
  BB_MAX_IMAGE_PIXELS  frame pixel cap, "none" for native (default 8294400)
  BB_MIN_INLIERS       board-acceptance inliers (default 20)
  BB_CONFUSER_FACTOR   confusers ≤ F×tracked   (default 1.0; "inf" = all)
  BB_SUPERSAMPLE       AA factor               (default 1)
  BB_P_PERSPECTIVE     perspective-aug prob    (default 0.5)
  BB_WRITE_ANCHORS     "1"/"0"                 (default "1")
=#

using Distributed
using SlurmClusterManager
# The master constructs a DescriptorNetwork below (gallery load), and ONNXRunTime's
# :cuda EP checks at call time that CUDA + cuDNN are loaded in the process.
using CUDA, cuDNN

# ── Master-side config ────────────────────────────────────────────────
const VIDEO_DIR   = ENV["BB_VIDEO_DIR"]
const BOARDS_DIR  = ENV["BB_BOARDS_DIR"]
const MODEL_PATH  = ENV["BB_MODEL_PATH"]
const OUTPUT_DIR  = ENV["BB_OUTPUT_DIR"]
const OUTPUT_NAME = get(ENV, "BB_OUTPUT_NAME", "tracks")
const WRITE_ANCHORS = get(ENV, "BB_WRITE_ANCHORS", "1") == "1"

mkpath(joinpath(OUTPUT_DIR, "tracks"))
mkpath(joinpath(OUTPUT_DIR, "anchors"))

# Config threaded to every task (primitives → serializes cheaply through pmap).
_optint(k) = haskey(ENV, k) ? parse(Int, ENV[k]) : nothing
function _optpixels()
    v = lowercase(get(ENV, "BB_MAX_IMAGE_PIXELS", "8294400"))
    v in ("none", "off", "native") ? nothing : parse(Int, v)
end
function _optnum(k, d)
    haskey(ENV, k) || return d
    v = lowercase(ENV[k]); v in ("inf", "all") ? Inf : parse(Float64, v)
end
const CFG = (
    patch_type       = Symbol(get(ENV, "BB_PATCH_TYPE", "logpolar")),
    scale            = parse(Float64, get(ENV, "BB_SCALE", "32")),
    n_frames         = _optint("BB_N_FRAMES"),
    max_detections   = something(_optint("BB_MAX_DETECTIONS"), 15000),
    max_image_pixels = _optpixels(),
    min_inliers      = something(_optint("BB_MIN_INLIERS"), 20),
    confuser_factor  = _optnum("BB_CONFUSER_FACTOR", 1.0),
    supersample      = something(_optint("BB_SUPERSAMPLE"), 1),
    p_perspective    = parse(Float64, get(ENV, "BB_P_PERSPECTIVE", "0.5")),
)

# ── Workers — SlurmClusterManager reads $SLURM_NTASKS, spawns N-1 ─────
@info "Calling addprocs(SlurmManager())..."
addprocs(SlurmManager(); exeflags = ["--project=$(dirname(Base.active_project()))",
                                     "--startup-file=no"])
@info "Got $(nworkers()) worker(s): $(workers())"

# ── Worker imports + resident board gallery ───────────────────────────
@everywhere begin
    using BlobBoards
    # Training tooling is `public`, not exported (see src/api.jl) — import by name.
    using BlobBoards: build_absolute_tracks, save_board_anchors,
                      load_board_gallery, collect_media, merge_track_files
    # CUDA + cuDNN activate BlobBoardsCUDAExt and the ONNX :cuda EP (same as the
    # build_tracks.jl CLI). The ONNX descriptor path needs no Lux/NNlib.
    using CUDA, cuDNN
end

@everywhere const _GALLERY = Ref{NamedTuple}()

@everywhere function _init_worker!(boards_dir, model_path)
    _GALLERY[] = load_board_gallery(model_path, boards_dir)
    n = length(_GALLERY[].boards)
    "ready on worker $(myid()) — $n boards, GPU=$(CUDA.device()) " *
        "(uuid=$(CUDA.uuid(CUDA.device()))) host=$(gethostname())"
end

# One work item is (:media, path) or (:anchor, uid). Each writes its own file.
@everywhere function _process_task(task, output_dir, cfg)
    # Self-init guard (mirrors driver_scm.jl): env vars are inherited via srun,
    # so a worker that missed the master's init loop can recover here.
    isassigned(_GALLERY) ||
        _init_worker!(ENV["BB_BOARDS_DIR"], ENV["BB_MODEL_PATH"])
    g = _GALLERY[]
    kind, key = task
    if kind === :media
        name = first(splitext(basename(key)))
        out  = joinpath(output_dir, "tracks", "$name.tracks")
        isfile(out) && return (:skipped, name)
        sink = BlobBoards._default_sink(out, cfg.patch_type, cfg.scale)
        build_absolute_tracks(
            g.board_by_uid, g.model, key;
            board_feature_dict = g.board_feature_dict,
            board_bundles = g.board_bundles,
            sink,
            n_frames = cfg.n_frames,
            iou_threshold = 0.3,
            max_detections = cfg.max_detections,
            max_image_pixels = cfg.max_image_pixels,
            patch_type = cfg.patch_type,
            scale = cfg.scale,
            min_inliers = cfg.min_inliers,
            confuser_factor = cfg.confuser_factor,
            supersample = cfg.supersample,
            p_perspective = cfg.p_perspective,
            backend = CUDABackend(),
        )
        return (:media_done, name)
    else # :anchor
        uid   = key
        out   = joinpath(output_dir, "anchors", "$uid.tracks")
        isfile(out) && return (:skipped, uid)
        board = g.board_by_uid[uid]
        save_board_anchors(out, board;
            patch_type = cfg.patch_type, scale = cfg.scale,
            inner_extent = BlobBoards.cutoff(board.bundle), backend = CUDABackend())
        return (:anchor_done, uid)
    end
end

# ── Master: build gallery once (uid list), init workers, then pmap ────
@info "Loading board gallery on master..."
const MASTER_GALLERY = load_board_gallery(MODEL_PATH, BOARDS_DIR)
const MEDIA = collect_media(VIDEO_DIR)
@info "Gallery: $(length(MASTER_GALLERY.boards)) boards; media: $(length(MEDIA)) file(s)"

@info "Initializing $(nworkers()) worker(s) (one GPU each via cgroup)..."
@sync for w in workers()
    @async @info remotecall_fetch(_init_worker!, w, BOARDS_DIR, MODEL_PATH)
end

# Work list: anchors first (they seed each worker's file compression while the
# long media jobs run), then all media. pmap work-steals regardless of order.
worklist = Any[]
WRITE_ANCHORS && append!(worklist, [(:anchor, uid) for uid in keys(MASTER_GALLERY.board_by_uid)])
append!(worklist, [(:media, m) for m in MEDIA])
@info "Dispatching $(length(worklist)) task(s) " *
      "($(WRITE_ANCHORS ? length(MASTER_GALLERY.boards) : 0) anchors + $(length(MEDIA)) media)..."

t_start = time()
results = pmap(WorkerPool(workers()), worklist) do task
    try
        _process_task(task, OUTPUT_DIR, CFG)
    catch e
        @warn "Task failed" task = task exception = (e, catch_backtrace())
        (:failed, string(task))
    end
end
elapsed = time() - t_start

n_media = count(r -> r[1] === :media_done,  results)
n_anch  = count(r -> r[1] === :anchor_done, results)
n_skip  = count(r -> r[1] === :skipped,     results)
n_fail  = count(r -> r[1] === :failed,      results)
@info "pmap done in $(round(elapsed; digits = 1))s — " *
      "$n_media media, $n_anch anchors, $n_skip skipped, $n_fail failed"

# ── Merge per-task files into one .tracks on the master ───────────────
final = joinpath(OUTPUT_DIR,
    BlobBoards._tracks_output_name(OUTPUT_NAME, CFG.patch_type, CFG.scale) * ".tracks")
srcs = String[]
for sub in ("anchors", "tracks")
    d = joinpath(OUTPUT_DIR, sub)
    isdir(d) && append!(srcs, filter(f -> endswith(f, ".tracks"), readdir(d; join = true)))
end
@info "Merging $(length(srcs)) per-task file(s) → $final"
merge_track_files(final, srcs)
@info "Done → $final ($n_fail task(s) failed)"
n_fail == 0 || exit(1)
