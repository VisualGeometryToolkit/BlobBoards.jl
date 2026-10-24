#=
sweep.jl — the shared find_boards sweep engine.

`go_bmvc2026.jl` (the main image sets) and `go_bmvc2026_occlusion.jl` (the ten
occlusion levels) do the same thing: run the calibrated `find_boards` path over
a list of image directories and write, per directory, the detected poses as
one `vgio/target_poses/v1` file (`target_poses.json`), the per-image timing as
telemetry (`telemetry.jsonl`), and per image a colored overlay. They differ
ONLY in which directories those are.

They used to be two files with 160 of 249 lines byte-identical, and they had
drifted at every point that mattered — the main sweep read its gallery and
frame cap from the config while the occlusion sweep hardcoded
`board_features.jld2` (the config of record names
`board_features_logpolar_s96.jld2`) and read `ENV["BB_LIMIT"]` (the exact
variable whose staleness once truncated a 56-image run to 40 and looked clean).
Neither divergence was reachable by reading either file alone. Sharing the
engine makes them unrepresentable rather than merely fixed.

NOT part of BlobBoardsEval, deliberately. This module needs CUDA + Makie; the
CSV and figure drivers include BlobBoardsEval and must not be made to pay for
a GPU stack to aggregate a table.

    include(".../src/sweep.jl")
    using .BoardSweep: SweepUnit, SweepContext, run_sweep
=#
module BoardSweep
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO

using BlobBoards

import VisualGeometryToolkitIO   # canonical vgio/cameras/v3 camera stores
import VisualGeometryToolkitCore
using BlobBoards: find_boards, load_bundle, uid_from_hash, blobs, cutoff,
    DescriptorNetwork, BoardGallery, Rect, board!,
    FindBoardsConfig, write_config
using CairoMakie: CairoMakie
using Makie: Makie
using FileIO: load as load_image
using Colors: Gray, RGB
using StaticArrays: SMatrix, SVector
using Rotations: RotMatrix
using CUDA: CUDABackend
using JSON
import TOML
using VisualGeometryToolkitEval.Experiments: read_provenance, warn_commit_drift
using VisualGeometryToolkitFeatures: KnnMatch
using VisualGeometryToolkitCore: ustrip, CameraModel, Pinhole, Camera, undistort, imshow, EuclideanMap, CameraFrame,
    BoundedCamera, TargetPoses

# The repo whose commit `tag!` records into the provenance sidecar — same
# expression as provenance.jl's default `repo`.
const _REPO_ROOT = dirname(dirname(@__DIR__))

export SweepUnit, SweepContext, run_sweep

# =============================================================================
# What varies between sweeps
# =============================================================================

"""
    SweepUnit(label, images, poses, val)

One image directory and where its outputs go. `label` appears in progress lines
and prefixes the telemetry frame tag, so branch telemetry can be attributed to
a subset or occlusion level after the fact. All paths absolute.
"""
struct SweepUnit
    label::String
    images::String
    poses::String
    val::String
end

"""
    SweepContext

Everything a sweep needs that does NOT vary between units: the camera, the
descriptor network, the gallery-backed board features, and the config of
record. Built once by `sweep_context`, shared by every unit.
"""
struct SweepContext
    camera::Any
    cam_pinhole::Any
    network::Any
    config::Any
    board_features::Dict
    board_bundles::Dict
    board_colors::Dict
    camera_name::String       # the camera record's name: every frame's `sensor`
    device::Any
    # Device-resident stacked gallery descriptors, built ONCE per sweep:
    # without it every frame's find_boards restages + re-uploads the gallery
    # (~2.5 ms/frame on the 260527 gallery — small, but pure waste × frames).
    gpu_index::Any
    # provenance
    config_path::String
    network_path::String
    gallery_path::String
    config_extra::Dict
end

"""
    sweep_context(; config_path, dataset, run_config, network_path, gallery_path,
                    boards_dir, camera_json, device = CUDABackend()) -> SweepContext

Load the camera, descriptor network, gallery and board bundles once.

`gallery_path` comes from the CONFIG in both sweeps. The gallery is a cache of
(board set × model), not a configuration choice — its key already includes the
model hash, patch size, outer extent and each board's cutoff, so a wrong-model
entry can never be returned. Hardcoding a different filename per driver, as the
occlusion sweep did, silently pairs one sweep's poses with another's features.
"""
function sweep_context(;
        config_path, run_config, network_path, gallery_path,
        boards_dir, camera_json, device = CUDABackend()
    )
    store = VgtkIO.load(Dict{Symbol, BoundedCamera}, camera_json)
    length(store) == 1 || error("$(camera_json): a sweep runs on one camera; the store holds $(sort!(collect(keys(store))))")
    camera_name, bounded = only(store)
    camera = VgtkCore.model(bounded isa BoundedCamera ? bounded : bounded.value)        # raw K + OpenCV-model lens
    cam_pinhole = CameraModel(camera.intrinsics, Pinhole())  # undistorted-frame camera
    network = DescriptorNetwork(network_path)
    gallery = BoardGallery(gallery_path, network)
    println("descriptor model: $network_path (patch $(BlobBoards.patch_size(network)))")

    board_paths = filter(endswith(".json"), readdir(boards_dir; join = true))
    board_features = Dict(
        uid => f for (uid, f) in
            [insert!(gallery, bp, network) for bp in board_paths]
    )
    board_bundles = Dict(
        let (b, _, ch) = load_bundle(bp)
            uid_from_hash(ch) => b
        end
            for bp in board_paths
    )
    println("$(length(board_paths)) boards loaded")

    # One palette authority: stable per board id (sorted) — BlobBoards.board_palette.
    board_colors = BlobBoards.board_palette(keys(board_bundles))

    # The camera is supplied separately because it is DATA (a calibration), not
    # configuration; every other knob comes from [find_boards] in config_path.
    config = FindBoardsConfig(config_path; camera)

    gpu_index = BlobBoards.GalleryIndex(device, board_features)

    extra = Dict(k => v for (k, v) in run_config.config_toml if k != "find_boards")

    return SweepContext(
        camera, cam_pinhole, network, config, board_features,
        board_bundles, board_colors, String(camera_name), device, gpu_index,
        String(config_path),
        String(network_path), String(gallery_path), extra
    )
end

# =============================================================================
# Per-image outputs
# =============================================================================

# A detected board's pose, board → camera optical, millimetres — the value a
# `vgio/target_poses/v1` frame stores per target.
pose_value(model) = EuclideanMap(
    RotMatrix{3, Float64}(SMatrix{3, 3, Float64}(model.R)), SVector{3, Float64}(ustrip.(model.t)); check = false,
)

# Colored-board overlay on the undistorted color image -> raster PNG (one stable
# color per board id).
function save_val_overlay(ctx::SweepContext, out_png, img_color_raw, accepted)
    undist = undistort(ctx.camera, img_color_raw)
    h, w = size(undist)
    fig = Makie.Figure(size = (1600, round(Int, 1600 * h / w)))
    ax, _ = imshow(fig[1, 1], undist)
    for (uid, model, _, _) in accepted
        bundle = ctx.board_bundles[uid]
        col = ctx.board_colors[uid]
        board!(
            ax, Camera(ctx.cam_pinhole, model), Rect(bundle, :world),
            blobs(bundle, :world);
            cutoff = cutoff(bundle), outline_color = col, blob_color = col,
            outline_width = 3.0, blob_width = 1.5, show_frame = true,
            frame_width = 6.0
        )
    end
    Makie.hidedecorations!(ax); Makie.hidespines!(ax)
    return Makie.save(out_png, fig; px_per_unit = 2)   # crisp raster; ~MB not ~100 MB vector PDF
end

# Pure CPU, reads only immutable shared state, so it is safe on any worker.
function write_outputs(ctx::SweepContext, item)
    (; val_dir, fn, color, accepted, idx, ntot, label) = item
    # SMB caches a negative dentry across threads: the producer's mkpath is not
    # visible to this writer thread's open(). Re-mkpath on the writing thread.
    mkpath(val_dir)
    save_val_overlay(ctx, joinpath(val_dir, splitext(fn)[1] * "_val.png"), color, accepted)
    return println("[$label $idx/$ntot] $fn -> $(length(accepted)) boards")
end

# The unit's detections as ONE `vgio/target_poses/v1` file: frame `t` is the
# image's position in the unit's sorted image list — the index the subset's
# motion-tracker trajectories use — so a strided (`limit`) run keeps the join.
# The per-image detection time goes to telemetry, not beside the poses.
const TARGET_POSES_FILE = "target_poses.json"
const TELEMETRY_FILE = "telemetry.jsonl"

function write_target_poses(dir, ctx::SweepContext, frames::Dict{Int, Any}, images::Dict{Int, String})
    tp = TargetPoses{CameraFrame}(
        Dict(t => Dict(Symbol(uid) => pose_value(model) for (uid, model, _, _) in acc) for (t, acc) in frames);
        images, sensors = ctx.camera_name,
    )
    VgtkIO.save(joinpath(dir, TARGET_POSES_FILE), tp)
    return nothing
end

# =============================================================================
# Frame selection
# =============================================================================

# ONE source: the deliberately narrow image predicate (the positional
# mocap↔image pairing is its contract) lives in eval_images.jl, included here
# and in BlobBoardsEval, so the sweep loads without the evaluation module.
include(joinpath(@__DIR__, "eval_images.jl"))

"""
    select_frames(images, limit) -> Vector

Cap a unit's image list to `limit` frames by STRIDING the sequence, never by
taking a prefix.

Frames are captured in station order, so `first(images, n)` samples only the
opening close/frontal stations: a 6-frame smoke run once reported P(det) = 1.000
for all 27 cells — the same saturation that disqualified the calibration set as
a tuning target. A stride spans the difficulty range the slice is meant to probe.
`limit <= 0` means no cap.
"""
function select_frames(images, limit::Integer)
    (limit > 0 && length(images) > limit) || return images
    return images[round.(Int, range(1, length(images), length = limit))]
end

# =============================================================================
# The sweep
# =============================================================================

"""
    run_sweep(units, ctx; limit = 0, driver, write_provenance, extra_provenance = Dict())

Run `find_boards` over every image of every unit and write pose JSON + overlay.

Detection is serial: `find_boards` owns the single GPU + ONNX CUDA session. The
per-image CPU work (overlay render + JSON write) fans out over a thread pool fed
by a bounded channel, so the sweep never holds every 24 MP color image at once.
Start Julia with `JULIA_NUM_THREADS=auto`; on one thread the render will not
parallelize.

`write_provenance` is supplied rather than imported so this module does not
depend on BlobBoardsEval — the caller supplies `BBE.write_provenance`.

ADD-ONLY: a unit whose outputs already exist and are complete is SKIPPED, not
overwritten. Supply `force = true` to redo one. Callers key the unit paths on the
config (src/runs.jl `sweep_dir`) so that a changed config produces a new
directory rather than overwriting the old.
"""
function run_sweep(
        units::AbstractVector{SweepUnit}, ctx::SweepContext;
        limit::Integer = 0, driver::AbstractString,
        write_provenance, extra_provenance::AbstractDict = Dict(),
        force::Bool = false,
        # Experiment instrumentation sinks (`BlobBoards.Telemetry`), stamped
        # with `unit.label/frame` per call; `nothing` = off.
        telemetry = nothing
    )
    nconsumers = max(1, Threads.nthreads() - 1)   # leave one thread for the producer
    Threads.nthreads() == 1 && @warn "Julia started with 1 thread — overlay rendering \
        will not parallelize. Relaunch with JULIA_NUM_THREADS=auto (or julia -t auto)."
    println("Pipeline: 1 GPU producer + $nconsumers render/IO threads")

    chan = Channel{Any}(2 * nconsumers)           # bounded: caps in-flight images
    consumers = map(1:nconsumers) do _
        Threads.@spawn for item in chan
            write_outputs(ctx, item)
        end
    end
    try
        _detect_all(
            chan, units, ctx, limit, driver, write_provenance,
            extra_provenance, force, telemetry
        )
    finally
        close(chan)
    end
    foreach(wait, consumers)                      # rethrows the first consumer failure
    return nothing
end

# Warm up find_boards so per-image timings exclude one-time JIT/GPU-kernel
# compilation. Phase 2's KnnMatch cleanup matcher is skipped whenever phase 1
# already finds every board, so compile it explicitly — otherwise the first
# phase-2-triggering frame absorbs that cost in its timing. Through phase 2's
# OWN call (`match_boards` with the `GalleryIndex`), so the method compiled is
# the one phase 2 runs. Calling `nn_match_stacked` directly cannot be made to
# match it: it wants the index's `StackedDescriptorIndex`, whose one key is the
# stack's Symbol, and a `String`-keyed gallery dict there fails converting it
# (and supplying the `GalleryIndex` itself, as this did, is a MethodError).
function _warmup(ctx::SweepContext, unit::SweepUnit)
    imgs = filter(_is_image, readdir(unit.images))
    isempty(imgs) && return nothing
    gray = Gray.(load_image(joinpath(unit.images, first(imgs))))
    prep = BlobBoards.prepare_image_features(
        ctx.network, gray;
        board_bundles = ctx.board_bundles, uids = keys(ctx.board_features),
        config = ctx.config, device = ctx.device
    )
    find_boards(
        ctx.board_features, prep.image_features;
        config = ctx.config, device = ctx.device,
        board_bundles = ctx.board_bundles, gpu_index = ctx.gpu_index
    )
    BlobBoards.match_boards(
        KnnMatch(3), ctx.board_features, prep.image_features;
        backend = ctx.device, index = ctx.gpu_index, min_similarity = 0.5,
    )
    return nothing
end

# A unit is complete when the provenance sidecar exists — it is written AFTER
# the unit's frame loop (2026-08-11), so its presence genuinely means the run
# finished — AND the JSON count matches the sidecar's recorded n_frames AND
# the recorded frame_limit equals the current one (a finished limit=6 smoke
# run is NOT a complete full run, and must not be silently overwritten
# either). Pre-fix sidecars (no n_frames) fall back to the on-disk image
# count. Code is deliberately NOT keyed (config ⊕ network only, src/runs.jl):
# a completed run under a different source state is still "complete", and
# this function WARNS on a commit mismatch with the sidecar rather than
# invalidating. Restorability comes from provenance — `tag!` records the
# commit + dirty patch at run time — while registered-dep drift is recorded by
# the dependency snapshot.
function _is_complete(unit::SweepUnit, limit::Integer)
    isdir(unit.poses) || return false
    prov_file = joinpath(unit.poses, "_provenance.toml")
    isfile(prov_file) || return false
    # The record through the stack's reader: the current flat layout and the
    # former [provenance] + [config] one read alike.
    prov = try
        read_provenance(unit.poses)
    catch
        return false
    end
    rec_limit = get(prov, "frame_limit", nothing)
    rec_limit !== nothing && Int(rec_limit) != Int(limit) && return false
    tpf = joinpath(unit.poses, TARGET_POSES_FILE)
    n_out = isfile(tpf) ? length(VgtkIO.load(TargetPoses, tpf)) : 0
    rec_n = get(prov, "n_frames", nothing)
    complete = if rec_n !== nothing
        n_out > 0 && n_out == Int(rec_n)
    else
        n_img = count(_is_image, readdir(unit.images))
        n_img > 0 && n_out == n_img
    end
    complete || return false
    # CODE is not keyed — it is RESTORABLE (tag! records commit + dirty
    # patch). A skip under different code
    # is legitimate but worth surfacing: BB_FORCE=1 regenerates.
    warn_commit_drift(prov, unit.label; gitpath = _REPO_ROOT)
    return true
end

function _detect_all(chan, units, ctx, limit, driver, write_provenance, extra_provenance, force, telemetry)
    isempty(units) && return nothing
    all(u -> !force && _is_complete(u, limit), units) && return nothing
    _warmup(ctx, first(units))

    for unit in units
        # ADD-ONLY. This used to `rm` every prior JSON and overlay before
        # writing, which is what made `poses/` a single mutable slot: a
        # superseded run left no trace, so a CSV or a hand-eye fit could
        # reference a snapshot that no longer existed and nothing could tell.
        # The unit path is keyed on the config, so a changed config is a NEW
        # directory and there is nothing to clear; an existing complete
        # directory is a cache hit, exactly like `produce_or_load`.
        if !force && _is_complete(unit, limit)
            println("=== '$(unit.label)' already produced -> $(unit.poses)  (skipping; force = true to redo)")
            continue
        end
        # On force, also drop the finish marker: a re-run killed halfway must
        # not read as complete via the PREVIOUS run's sidecar.
        force && ispath(unit.poses) &&
            foreach(
            f -> (endswith(f, ".json") || endswith(f, ".jsonl") || f == "_provenance.toml") &&
                rm(joinpath(unit.poses, f)), readdir(unit.poses)
        )
        force && ispath(unit.val) &&
            foreach(f -> endswith(lowercase(f), ".png") && rm(joinpath(unit.val, f)), readdir(unit.val))
        mkpath(unit.poses); mkpath(unit.val)

        # The configuration of record, written BESIDE the artifact it produced —
        # the EFFECTIVE config, not a copy of the authored file. A copy records
        # what was WRITTEN DOWN; any parameter the author omitted would reload at
        # whatever the then-current library default happens to be, so an old
        # sidecar could describe a different run than it produced. `write_config`
        # emits every key `config_dict` knows, and `_provenance.toml` still names
        # the source file so the authored copy stays traceable.
        write_config(joinpath(unit.poses, "_config.toml"), ctx.config; extra = ctx.config_extra)

        all_images = sort(filter(_is_image, readdir(unit.images)))
        index = Dict(fn => t for (t, fn) in enumerate(all_images))
        images = select_frames(all_images, limit)
        ntot = length(images)
        detected = Dict{Int, Any}()
        names = Dict{Int, String}()
        println("\n=== '$(unit.label)' -> $(unit.poses) + $(unit.val) : $ntot images ===")
        for (idx, fn) in enumerate(images)
            color = load_image(joinpath(unit.images, fn))
            # Attribute telemetry to the unit as well as the frame; the
            # occlusion sweep visits the same filenames at ten levels.
            tel = telemetry === nothing ? nothing :
                BlobBoards.with_tag(telemetry, string(unit.label, "/", fn))
            gray = Gray.(color)   # convert OUTSIDE the timed region
            local accepted
            elapsed_ms = 1000 * @elapsed accepted = find_boards(
                ctx.board_features, ctx.network, gray;
                config = ctx.config, device = ctx.device,
                board_bundles = ctx.board_bundles, gpu_index = ctx.gpu_index,
                telemetry = tel
            )
            t = index[fn]
            detected[t], names[t] = accepted, fn
            VgtkIO.append_jsonl(
                joinpath(unit.poses, TELEMETRY_FILE),
                (image = fn, t = t, elapsed_time_ms = round(elapsed_ms, digits = 3)),
            )
            put!(chan, (; val_dir = unit.val, fn, color, accepted, idx, ntot, label = unit.label))
        end
        write_target_poses(unit.poses, ctx, detected, names)

        # The FINISH marker — written after the unit's frame loop (2026-08-11;
        # it used to be written up-front, so a run killed mid-unit looked
        # finished). `n_frames` is what _is_complete checks the target-poses
        # frame count against.
        payload = Dict{String, Any}(
            "config_source" => ctx.config_path,
            "descriptor_network" => ctx.network_path,
            "descriptor_network_sha256" => ctx.network.network_hash,
            "gallery" => ctx.gallery_path,
            "unit" => unit.label,
            "frame_limit" => limit,
            "n_frames" => ntot,
        )
        merge!(payload, extra_provenance)
        write_provenance(unit.poses, payload; driver)
    end
    return nothing
end

end # module
