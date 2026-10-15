#!/usr/bin/env julia
#=
CLI entry point for `build_tracks_for_sequences`.

Builds single-patch-type, single-scale training tracks for every media file in
VIDEO_DIR against every board `.json` in BOARD_DIR, writing one `.tracks` file
(auto-suffixed `_<patch_type>_s<scale>`) plus per-patch board∩in-frame masks and
a per-sequence `is_anchor` flag.

Also writes `<output-name>_view_angles.png` next to it: average detections per
board binned by viewing obliquity, with recall and per-bin support. Recompute it
later with `view_angle_profile(path)` + `save_view_angle_figure`.

Usage:
  julia --project=gpu -t auto examples/training/build_tracks.jl \
      VIDEO_DIR BOARD_DIR MODEL.onnx [options]

`--project=gpu`, not `--project=.`: CUDA and cuDNN are BlobBoards [weakdeps] and
Julia will not load a package's weakdeps from that package's own env, so under
`--project=.` the `using CUDA, cuDNN` below fails outright and `--cpu` is the
only thing that runs. `gpu/` dev's the checkout and adds both as ordinary deps.

Positional (required):
  VIDEO_DIR    directory of videos/photos
  BOARD_DIR    directory of board .json files
  MODEL.onnx   descriptor model path

Options (flag value):
  --patch-type logpolar|cartesian|original   (default logpolar)
  --scale      N            patch half-width in σ            (default 32)
  --iou-threshold    F      min symmetric IoU for a correspondence (default 0.6)
  --min-inliers      N      board-acceptance inlier threshold (default 20)
  --n-frames   N            uniformly sample N frames per video
  --stride     N            process every Nth frame (mutually exclusive w/ n-frames)
  --confuser-factor  F      confusers ≤ F × tracked patches  (default 1.0; "inf" = all)
  --supersample      N      anti-aliasing factor             (default 1)
  --max-detections   N      per-frame detection cap          (default unlimited)
  --max-image-pixels N      frame pixel cap ("none" = native; default ≈8.3 MP)
  --p-perspective    F      perspective-aug probability      (default 0.5)
  --output-dir       DIR    output directory                 (default ".")
  --output-name      NAME   base output name (pre-suffix)    (default "tracks")
  --debug-dir        DIR    save per-(frame,board) detection overlays here,
                            plus a per-media viewing-angle chart
  --uncap                   large-GPU mode: no detection / pixel / confuser caps
  --no-anchors              skip writing per-board GT anchor sequences
  --cpu                     use the CPU backend (default: CUDA)

Example:
  julia --project=gpu -t auto examples/training/build_tracks.jl \
      /data/footage /data/boards /data/model.onnx \
      --patch-type cartesian --scale 16 --n-frames 30 --output-dir /data/out
=#

using BlobBoards
using BlobBoards: build_tracks_for_sequences   # advanced API (unexported)

# Headless Makie backend, so the run-end "detections per board vs. viewing angle"
# chart renders to PNG (BlobBoardsMakieExt). Optional: without it the profile
# still prints as a terminal bar chart.
try
    @eval using CairoMakie
catch e
    @warn "CairoMakie unavailable — viewing-angle profile falls back to a text chart" exception = e
end

# --- Tiny flag parser: positionals first, then `--flag value` / `--bool` -------
function parse_args(argv)
    pos = String[]
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 1
    boolflags = ("--no-anchors", "--cpu", "--uncap")
    while i <= length(argv)
        a = argv[i]
        if startswith(a, "--")
            if a in boolflags
                push!(flags, a); i += 1
            else
                i + 1 <= length(argv) || error("missing value for $a")
                opts[a] = argv[i+1]; i += 2
            end
        else
            push!(pos, a); i += 1
        end
    end
    return pos, opts, flags
end

pos, opts, flags = parse_args(ARGS)
if length(pos) != 3
    println("usage: build_tracks.jl VIDEO_DIR BOARD_DIR MODEL.onnx [options]  (see file header)")
    exit(1)
end
video_dir, board_dir, model_path = pos

# --- Backend: CUDA by default, CPU with --cpu --------------------------------
backend = if "--cpu" in flags
    # KA.CPU(), not VGF's CpuBackend() — the training pipeline dispatches on KA
    # device tokens (it is what `default_device()` falls back to), and only a
    # KA.CPU also switches the ONNX session to the CPU execution provider.
    using KernelAbstractions: CPU
    CPU()
else
    # cuDNN is loaded for its side effect only: together with CUDA it triggers
    # ONNXRunTime's CUDAExt, without which the descriptor session silently
    # drops to the CPU EP.
    using CUDA, cuDNN
    CUDA.functional() ||
        error("CUDA is loaded but not functional — check the driver, or supply --cpu")
    CUDABackend()
end

optint(k)   = haskey(opts, k) ? parse(Int, opts[k]) : nothing
optreal(k, d) = haskey(opts, k) ? parse(Float64, opts[k]) : d
# "inf"/"none"/"off" → Inf / nothing; otherwise parse a number.
function optnum(k, d)
    haskey(opts, k) || return d
    v = lowercase(opts[k])
    v in ("inf", "all") ? Inf : parse(Float64, v)
end

uncap = "--uncap" in flags

kw = (;
    patch_type      = Symbol(get(opts, "--patch-type", "logpolar")),
    scale           = optreal("--scale", 32.0),
    iou_threshold   = optreal("--iou-threshold", 0.6),
    min_inliers     = something(optint("--min-inliers"), 20),
    confuser_factor = uncap ? Inf : optnum("--confuser-factor", 1.0),
    supersample     = something(optint("--supersample"), 1),
    p_perspective   = optreal("--p-perspective", 0.5),
    write_anchors   = !("--no-anchors" in flags),
    output_dir      = get(opts, "--output-dir", "."),
    output_name     = get(opts, "--output-name", "tracks"),
    backend,
)
# n_frames / stride are mutually exclusive — supply at most one.
nf, st = optint("--n-frames"), optint("--stride")
(nf !== nothing && st !== nothing) && error("supply only one of --n-frames / --stride")
md = uncap ? typemax(Int) : optint("--max-detections")
# --max-image-pixels: "none"/"off" (or --uncap) keeps native resolution.
mip = if uncap || lowercase(get(opts, "--max-image-pixels", "")) in ("none", "off")
    :native
else
    optint("--max-image-pixels")
end
debug_dir = get(opts, "--debug-dir", nothing)

@info "build_tracks: $video_dir × $board_dir → $(kw.output_name) [$(kw.patch_type)@$(kw.scale)]" *
      (uncap ? " (uncapped)" : "")
out = build_tracks_for_sequences(board_dir, model_path, video_dir;
    kw...,
    (nf === nothing ? NamedTuple() : (; n_frames = nf))...,
    (st === nothing ? NamedTuple() : (; stride = st))...,
    (md === nothing ? NamedTuple() : (; max_detections = md))...,
    (mip === :native ? (; max_image_pixels = nothing) :
     mip === nothing ? NamedTuple() : (; max_image_pixels = mip))...,
    (debug_dir === nothing ? NamedTuple() : (; debug_dir))...,
)
@info "Done → $out"
