# =============================================================================
# Training Pipeline — Track Generation from Video + GT Homographies
# =============================================================================
#
# Detects and adapts features in each frame, transfers each board's GT blobs
# through the estimated world→image homography, and keys every observation by
# its board-relative blob id.
#
# Pipeline: detect_blobs → adapt → filter_blobs → symmetric_iou_match
#
# detect_blobs + adapt + filter_blobs — composable detection pipeline
# symmetric_iou_match         — symmetric IoU matching through a homography
# build_absolute_tracks       — one media item, every board detected in it
# build_tracks_for_sequences  — the driver: media dir × board dir → one .tracks
# save_tracks                 — .tracks (HDF5) serialization

# =============================================================================
# Imports
# =============================================================================

# EXPLICIT IMPORTS ONLY. This block used to open with blanket `using Images`,
# `using VisualGeometryToolkitCore`, `using StaticArrays`, `using Base`, and even
# `using BlobBoards` (a self-import). Between them they pulled 14 names into
# BlobBoards' namespace implicitly, which is what
# test/test_explicit_imports.jl's `check_no_implicit_imports` gate fails on —
# and `using VisualGeometryToolkitCore` additionally shadowed `model`, so JET read
# pipeline.jl's local `(uid, model, ...)` destructuring as a call to VGC's
# `model`. Everything not listed below is already imported explicitly by
# src/BlobBoards.jl; add names here, never a bare `using`.
using VideoIO: VideoIO
using HDF5: HDF5, h5open
using Statistics: mean, std
using Random: randperm
using LinearAlgebra: I, svdvals          # `inv` is Base's, not LinearAlgebra's
using FileIO: load as load_image
using ImageFiltering: KernelFactors, imfilter
using ImageTransformations: ImageTransformations, imresize
using Colors: RGB                   # overlay palette (blob_dataset viewer)
using VisualGeometryToolkitCore: to_homogeneous, shape_matrix

# =============================================================================
# Backend selection — no hard CUDA dependency
# =============================================================================
#
# training.jl is part of BB's *core* module, so `using CUDA`/`using cuDNN` here
# would drag the whole CUDA stack out of [weakdeps] and into [deps] for every
# consumer. The GPU path arrives through BlobBoardsCUDAExt instead: once the
# caller does `using CUDA`, the extension overrides the two hooks below. With no
# CUDA loaded the training pipeline still runs, on `KA.CPU()`.
#
# cuDNN was only ever an extension trigger here (nothing in this file calls it);
# scripts that need the ONNX `:cuda` EP load it themselves.


# Return the backend's cached pool memory to the driver. No-op for every backend
# except CUDA, where BlobBoardsCUDAExt maps it to `CUDA.reclaim()`. Typed `::Any`
# rather than `::KA.Backend` because callers still supply VGF's `CudaBackend()`.
reclaim_pool!(::Any) = nothing

# ONNX execution provider matching a compute backend. Without this a CPU run
# still asks for the `:cuda` EP, and `DescriptorNetwork` answers with a
# "CUDA execution provider unavailable" warning before falling back to `:cpu`
# on its own — noise that reads like a misconfiguration. Non-KA tokens (VGF's
# `CudaBackend()`) keep the CUDA default.
_execution_provider(backend) = backend isa KA.CPU ? :cpu : :cuda

# `best_iou_match` is VisualGeometryToolkitCore's since 0.21.1 (beside `iou`).
using VisualGeometryToolkitCore: best_iou_match

# Detection pipeline is composed inline in detect_and_extract:
#   detect_blobs → adapt → filter_blobs → canonicalize

# =============================================================================
# Find validated correspondences between two frames
# =============================================================================

const TrainingPair = @NamedTuple{
    frame_i::Int, frame_j::Int,
    idx_i::Int, idx_j::Int,
    iou_forward::Float64, iou_backward::Float64,
}

"""
    symmetric_iou_match(H_ij, frames_i, frames_j;
                        cutoff=2.0, iou_threshold=0.3) -> Vector{TrainingPair}

Symmetric matching of the detections of two frames related by the homography
`H_ij`, scored by the IoU of their `cutoff`-scaled boundary ellipses.

SYMMETRIC: a pair is kept only when both directions succeed. Forward, the
detection of frame i, transferred through `H_ij`, finds its best-IoU match
among frame j's detections. Backward, that match, transferred through `H_ij⁻¹`,
overlaps the original detection. Each IoU must reach `iou_threshold`.

For each detection in frame i:
1. Transfer its boundary ellipse to frame j via H_ij
2. Find best IoU match among frame j detections
3. Transfer that match back to frame i via H_ij⁻¹
4. Check IoU with original detection
5. Accept if min(forward_iou, backward_iou) ≥ iou_threshold

One deterministic sweep: no sampling, no iteration (an algorithm name, not a
`find`: CLAUDE.md's verb conventions).
"""
function symmetric_iou_match(
        H_ij::SMatrix{3, 3},
        frames_i::AbstractVector, frames_j::AbstractVector;
        cutoff = 2.0,
        iou_threshold = 0.3,
        frame_i_idx = 0,
        frame_j_idx = 0,
    )
    H_ji = SMatrix{3, 3}(inv(H_ij))

    ellipses_i = [region(f, cutoff) for f in frames_i]
    ellipses_j = [region(f, cutoff) for f in frames_j]

    # Thread-parallel: each detection independently checked
    n_i = length(ellipses_i)
    results = Vector{Union{TrainingPair, Nothing}}(undef, n_i)

    Threads.@threads for ki in 1:n_i
        ei = ellipses_i[ki]
        # Forward: transfer i→j, find best match in j
        ei_in_j = transfer_ellipse(HomographyMap(HomographyMat(H_ij)), ei)
        if isnothing(ei_in_j)
            results[ki] = nothing
            continue
        end
        idx_j, iou_fwd = best_iou_match(ei_in_j, ellipses_j)
        if iou_fwd < iou_threshold
            results[ki] = nothing
            continue
        end

        # Backward: transfer matched j→i, check IoU with original
        ej = ellipses_j[idx_j]
        ej_in_i = transfer_ellipse(HomographyMap(HomographyMat(H_ji)), ej)
        if isnothing(ej_in_i)
            results[ki] = nothing
            continue
        end
        if !isa(ei, Ellipse)
            ei = Ellipse(ei)
        end
        iou_bwd = iou(ej_in_i, ei)
        if iou_bwd < iou_threshold
            results[ki] = nothing
            continue
        end

        results[ki] = TrainingPair(
            (
                frame_i_idx, frame_j_idx, ki, idx_j, iou_fwd, iou_bwd,
            )
        )
    end

    return TrainingPair[r for r in results if !isnothing(r)]
end

# =============================================================================
# HDF5 I/O
# =============================================================================

"""
    save_tracks(path, tracks; sequence_name, untracked=nothing, patch_type=nothing,
                scale=nothing, is_anchor=false) -> path

Save tracks (and optionally untracked/confuser patches) to a `.tracks` file (HDF5
format). Each media item is stored as a separate sequence group so new items can
be appended incrementally to the same file.

# HDF5 layout

`Dict` (single-mode — one patch type at one scale, written by
[`build_absolute_tracks`](@ref) / `save_board_anchors`):
- `/sequences/<name>` — group attribute `is_anchor` (`UInt8`, 1 for the per-board
  GT anchor sequences, 0 for the per-media tracked sequences)
- `/sequences/<name>/tracks/patches` — `(P, P, N)` float32, chunked+compressed
- `/sequences/<name>/tracks/masks` — `(P, P, N)` float32; per-pixel board∩in-frame
  mask (1 where the patch pixel lands on the board sheet AND on a real image
  pixel; 0 over off-board background, image edges and perspective-warp voids)
- `/sequences/<name>/tracks/affine_shapes` — `(2, 2, N)` float32
- `/sequences/<name>/tracks/frame_id` — `(N,)` int32; source frame of each
  observation. Tracked sequences only (GT-anchor sequences omit it).
- `/sequences/<name>/tracks/homographies` — `(3, 3, F)` float64; ONE world→image
  homography `H_wi` per frame the board was detected in (F frames, not N
  observations). Tracked sequences only.
- `/sequences/<name>/tracks/view_angles` — `(F,)` float64; per-frame viewing
  obliquity in radians (0 = fronto-parallel, → π/2 edge-on), from the local affine
  tilt of `H_wi` at the board centroid (intrinsics-free; metric because the board
  is square). Aligned with `homographies`.
- `/sequences/<name>/tracks/homography_frame_ids` — `(F,)` int32; the frame index
  of each `homographies`/`view_angles` column. Join an observation to its pose via
  `frame_id` → this table.
- `/sequences/<name>/tracks/track_id`, `n_tracks`, `track_lengths`
- `/sequences/<name>/tracks` — group attributes `patch_type` (string) and
  `scale` (float64)
- `/sequences/<name>/untracked/patches`, `untracked/masks` — `(P, P, M)` float32
  (present only when confusers were supplied)

A second, `Vector{Track}` layout used to live here for `build_tracks`; both it
and its reader went with that pipeline (2026-08-19).
"""
function save_tracks(
        path::AbstractString, tracks::Dict;
        sequence_name::AbstractString,
        untracked::Union{Nothing, AbstractVector{<:AbstractMatrix}, NamedTuple} = nothing,
        frame_geometry = nothing,
        patch_type::Union{Nothing, Symbol} = nothing,
        scale::Union{Nothing, Real} = nothing,
        is_anchor::Bool = false
    )
    isempty(tracks) && error("No tracks to save")
    return _save_track_dict(
        path, tracks; sequence_name, untracked, frame_geometry,
        patch_type, scale, is_anchor
    )
end

# Stack a vector of 3×3 homographies into a 3×3×N Float64 array (HDF5-writable).
function _stack_homographies(hs)
    A = Array{Float64}(undef, 3, 3, length(hs))
    for (i, h) in enumerate(hs)
        A[:, :, i] .= h
    end
    return A
end

# Viewing obliquity of a planar homography at world point (x, y): the angle
# between the camera's optical axis and the board-plane normal, recovered WITHOUT
# camera intrinsics from the local affine tilt. Linearize `H` (world→image) at
# (x, y); the 2×2 Jacobian's singular-value ratio t = σ_max/σ_min is the ASIFT
# tilt (same convention as `tilt_matrices`), and the obliquity is `acos(1/t)`
# — 0 for a fronto-parallel view, → π/2 as the board edges on. Returns radians.
# How many of a board's world-space GT blobs land inside an `img_size = (rows,
# cols)` frame when projected through `H` (world→image, detector x=col/y=row
# coords). This is the denominator `n_matched_gt` is measured against: unlike the
# frame-wide detection count it shrinks as the board leaves the frame, so
# `n_matched_gt / n_gt_in_image` is a real per-frame recall.
#
# Centroid-in-bounds, not full-ellipse coverage — a blob whose center is in
# frame is one the matcher had a fair chance at (`filter_blobs` already drops
# detections clipped by more than 20%). Points that cross the horizon flip the
# sign of the homography's third coordinate; requiring the sign to agree with
# the board centroid's keeps those (and any non-finite projection) out.
function _n_gt_in_image(world_frames, H::SMatrix{3, 3}, cx::Real, cy::Real, img_size)
    w_ref = VgtkCore.homogeneous_w(H, SVector(Float64(cx), Float64(cy)))
    rows, cols = img_size
    n = 0
    for f in world_frames
        # (audited) defensive strip: adapted frames are EllipticFrame{Float64},
        # bare px by contract, so ustrip(px, ...) would throw on them.
        c = ustrip.(centroid(feature_frame(f)))
        q = VgtkCore.to_euclidean(H * SVector(Float64(c[1]), Float64(c[2]), 1.0), w_ref)
        q === nothing && continue
        (1 <= q[1] <= cols && 1 <= q[2] <= rows) && (n += 1)
    end
    return n
end

function _view_angle(H::SMatrix{3, 3}, x::Real, y::Real)
    w = H[3, 1] * x + H[3, 2] * y + H[3, 3]
    u = (H[1, 1] * x + H[1, 2] * y + H[1, 3]) / w
    v = (H[2, 1] * x + H[2, 2] * y + H[2, 3]) / w
    J = SMatrix{2, 2, Float64}(
        (H[1, 1] - u * H[3, 1]) / w, (H[2, 1] - v * H[3, 1]) / w,   # column 1
        (H[1, 2] - u * H[3, 2]) / w, (H[2, 2] - v * H[3, 2]) / w,   # column 2
    )
    s = svdvals(J)
    t = s[1] / s[2]                       # ASIFT tilt σ_max/σ_min ≥ 1
    return acos(clamp(inv(t), -1.0, 1.0))
end

# Write the untracked (confuser) patches into an `untracked/` group. No-op when
# there are none. Patch size is taken from the patches themselves.
# Number of confuser patches, regardless of storage shape.
_n_untracked(::Nothing) = 0
_n_untracked(u::AbstractVector) = length(u)
_n_untracked(u::NamedTuple) = length(u.patch)

_write_untracked!(sg, ::Nothing, _) = nothing

# Stack a vector of P×P patches into a P×P×M dataset.
_stack_patches!(ug, name, patches, P) =
    ug[name, chunk = (P, P, 1), compress = 4] = _stack_patches(patches, P)   # annotate.jl's stacker

# Legacy single-mode confusers: one `untracked/patches` dataset.
function _write_untracked!(sg, untracked::AbstractVector, untracked_frame_ids)
    isempty(untracked) && return
    P = size(first(untracked), 1)
    ug = HDF5.create_group(sg, "untracked")
    _stack_patches!(ug, "patches", untracked, P)
    isnothing(untracked_frame_ids) || (ug["frame_id"] = Int32.(untracked_frame_ids))
    return
end

# Single-mode confusers: a NamedTuple `(; patch, mask)` of equal-length patch and
# mask vectors (the chosen patch type at the chosen scale). Written as
# `untracked/patches` + `untracked/masks`, mirroring the tracked layout.
function _write_untracked!(sg, untracked::NamedTuple, _)
    isempty(untracked.patch) && return
    P = size(first(untracked.patch), 1)
    ug = HDF5.create_group(sg, "untracked")
    _stack_patches!(ug, "patches", untracked.patch, P)
    _stack_patches!(ug, "masks", untracked.mask, P)
    return
end

# Single-mode `Dict` format: each track is
#   track_id => Dict("patch"=>[P×P...], "mask"=>[P×P...], "affine_shape"=>[SMatrix{2,2}...],
#                    "frame_id"=>[Int...])
# Stored as `tracks/patches` (P×P×N), `tracks/masks` (P×P×N), `tracks/affine_shapes`
# (2×2×N), plus `track_id` and (tracked sequences only) `frame_id` — the source
# frame of each observation. `frame_geometry` is the sequence's per-frame detection
# geometry `(; frame_ids, homographies, view_angles, n_gt_in_image,
# n_sift_on_board, n_detections_on_board, n_detections_in_frame)`: ONE
# world→image homography + viewing angle + the per-frame counts (see
# `ViewAngleStat`) for each frame the board was detected in, written as
# `tracks/homographies` (3×3×F), `tracks/view_angles` (F),
# `tracks/homography_frame_ids` (F) and one (F) table per count under
# `tracks/<count name>`. An
# observation's `frame_id` indexes into `homography_frame_ids`. GT-anchor sequences
# supply `frame_geometry=nothing` and carry no `frame_id`. The single patch type and
# scale are recorded as `tracks`-group attributes; `is_anchor` marks the whole
# sequence as GT-anchor (true) or tracked (false).
function _save_track_dict(
        path, tracks::Dict; sequence_name, untracked,
        frame_geometry = nothing,
        patch_type = nothing, scale = nothing, is_anchor::Bool = false
    )
    tids = sort(collect(keys(tracks)))
    len(track) = length(track["patch"])

    track_lengths = Int32[len(tracks[tid]) for tid in tids]
    n_tracks = length(tids)
    n_total = Int(sum(track_lengths))
    P = size(tracks[first(tids)]["patch"][1], 1)

    # Per-observation source frame — present on tracked sequences, absent on
    # GT-anchor ones (which have no detection frame).
    has_frame = haskey(tracks[first(tids)], "frame_id")

    all_patches = Array{Float32}(undef, P, P, n_total)
    all_masks = Array{Float32}(undef, P, P, n_total)
    all_affine = Array{Float32}(undef, 2, 2, n_total)
    all_track_ids = Vector{Int32}(undef, n_total)
    all_frame_ids = has_frame ? Vector{Int32}(undef, n_total) : nothing

    idx = 0
    for tid in tids
        track = tracks[tid]
        for k in 1:len(track)
            idx += 1
            all_track_ids[idx] = Int32(tid)
            all_patches[:, :, idx] .= Float32.(track["patch"][k])
            all_masks[:, :, idx] .= Float32.(track["mask"][k])
            all_affine[:, :, idx] .= Float32.(track["affine_shape"][k])
            has_frame && (all_frame_ids[idx] = Int32(track["frame_id"][k]))
        end
    end

    h5open(path, "cw") do fid
        seq_path = "sequences/$sequence_name"
        haskey(fid, seq_path) && HDF5.delete_object(fid, seq_path)
        sg = HDF5.create_group(fid, seq_path)
        HDF5.attributes(sg)["is_anchor"] = UInt8(is_anchor)

        g = HDF5.create_group(sg, "tracks")
        g["patches", chunk = (P, P, 1), compress = 4] = all_patches
        g["masks", chunk = (P, P, 1), compress = 4] = all_masks
        g["affine_shapes"] = all_affine
        g["track_id"] = all_track_ids
        g["n_tracks"] = Int32(n_tracks)
        g["track_lengths"] = track_lengths
        has_frame && (g["frame_id"] = all_frame_ids)   # (N,) per-observation source frame
        # Per-frame detection geometry: one homography + viewing angle per detected
        # frame, joined to observations via `frame_id` → `homography_frame_ids`.
        if frame_geometry !== nothing
            g["homographies"] = _stack_homographies(frame_geometry.homographies)  # (3,3,F) world→image
            g["view_angles"] = frame_geometry.view_angles                        # (F,) radians (obliquity)
            g["homography_frame_ids"] = frame_geometry.frame_ids                          # (F,) frame index per column
            # (F,) per-frame detection counts feeding `view_angle_profile`:
            # (F,) per-frame counts feeding `view_angle_profile`, named for stage
            # and scope — see the `ViewAngleStat` block for what each one means.
            # Optional: callers that build `frame_geometry` without them still
            # save.
            haskey(frame_geometry, :n_gt_in_image) &&
                (g["n_gt_in_image"] = frame_geometry.n_gt_in_image)
            haskey(frame_geometry, :n_sift_on_board) &&
                (g["n_sift_on_board"] = frame_geometry.n_sift_on_board)
            haskey(frame_geometry, :n_detections_on_board) &&
                (g["n_detections_on_board"] = frame_geometry.n_detections_on_board)
            haskey(frame_geometry, :n_detections_in_frame) &&
                (g["n_detections_in_frame"] = frame_geometry.n_detections_in_frame)
        end
        isnothing(patch_type) || (HDF5.attributes(g)["patch_type"] = String(patch_type))
        isnothing(scale) || (HDF5.attributes(g)["scale"] = Float64(scale))

        _write_untracked!(sg, untracked, nothing)
    end

    n_ut = _n_untracked(untracked)
    meta = isnothing(patch_type) ? "" : " $(patch_type)@$(scale)"
    kind = is_anchor ? "anchor" : "tracked"
    println(
        "  Saved $n_total $kind ($n_tracks tracks$meta) " *
            "+ $n_ut untracked to $path [sequence=$sequence_name]"
    )
    return path
end

# =============================================================================
# Main pipeline
# =============================================================================

# The descriptor model is a FILE; the directories beside it in the training
# entry points are not. Every argument there is a string, so this check is what
# turns a call in a retired argument order into an error that names the order.
function _check_model_file(fname::Symbol, path::AbstractString, order::AbstractString)
    isfile(path) && return nothing
    throw(ArgumentError("$fname: the descriptor model `$path` is not a file (is the argument order $order?)"))
end

"""
    collect_media(video_dir) -> Vector{String}

Recursively collect every video/image file under `video_dir` (the unit of
independent work for track building): `walk_media` — the ONE enumeration, in
sorted, reproducible order — behind a loud empty-corpus guard. Shared by
[`build_tracks_for_sequences`](@ref) and the multi-GPU driver.
"""
function collect_media(video_dir::AbstractString)
    media = walk_media(video_dir)
    isempty(media) && error("No video or image files found in $video_dir")
    return media
end

"""
    load_board_gallery(descriptor_model_path, board_dir; device = default_device())
        -> (; boards, board_by_uid, model, board_feature_dict, board_bundles)

Load every board `.json` in `board_dir` (metadata + rendered image), the
descriptor model, and per-board gallery features + bundles (all keyed by uid).
Runs [`check_blob_id_collisions`](@ref) to fail fast on blob-id hash-field
clashes. Shared by [`build_tracks_for_sequences`](@ref) and the multi-GPU driver
(`examples/training/slurm/driver_build_tracks.jl`) so every worker builds an
identical gallery.

`device` is the KA backend used for descriptor extraction — [`default_device`](@ref)
(`CUDABackend()` when CUDA is loaded, otherwise `KA.CPU()`).
"""
function load_board_gallery(
        # The network first — it is the tool the boards are described with.
        # Both are strings, so a call in the retired `(board_dir, model_path)`
        # order would bind: the file check below makes it an ArgumentError.
        descriptor_model_path::AbstractString, board_dir::AbstractString;
        device = default_device()
    )
    _check_model_file(:load_board_gallery, descriptor_model_path, "(descriptor_model_path, board_dir)")
    board_paths = filter(p -> lowercase(splitext(p)[2]) == ".json", readdir(board_dir; join = true))
    isempty(board_paths) && error("No board .json files found in $board_dir")
    # Load + rasterize on all threads (pdftocairo subprocesses and PNG decodes);
    # `fetch` in path order keeps the output order, and each board's result
    # depends on its own file only, so it is the same at any thread count.
    tasks = map(board_paths) do p
        Threads.@spawn begin
            b = load_board(p)
            (; b..., image = _board_raster(b.image_path, b.bundle, b.config_hash; density = b.density))
        end
    end
    boards = map(fetch, tasks)
    board_by_uid = Dict(b.uid => b for b in boards)
    # Fail fast if two boards' hash fields collide (would merge their blob-id
    # ranges); ids are otherwise stable per (board, blob) across all media.
    check_blob_id_collisions(Dict(b.uid => b.config_hash for b in boards))
    model = DescriptorNetwork(
        descriptor_model_path;
        execution_provider = _execution_provider(device)
    )
    board_feature_dict = Dict(b.uid => extract_board_features(model, b; device, b.image) for b in boards)
    board_bundles = Dict(b.uid => b.bundle for b in boards)
    return (; boards, board_by_uid, model, board_feature_dict, board_bundles)
end

"""
    build_tracks_for_sequences(board_dir, descriptor_model_path, video_dir; kwargs...)
        -> output_path

Build multi-modal training tracks for every media file in `video_dir` against
every board `.json` in `board_dir`, accumulating into `\$output_name.tracks`.

Videos (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`) are sampled into frames;
photos (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`) are treated as a
single frame. A
photo yields at most one observation per tracked blob (paired against its GT
anchor via the anchor sequence); `n_frames`/`stride` are ignored for photos.

For each media item the boards are detected + tracked across frames
([`track_boards`](@ref)), then blob correspondences are established per frame by
transferring the board's GT blobs through the estimated world→image homography
and validating by IoU. Each surviving track stores, per observation, a single
patch view (`patch_type`) at a single `scale`, a per-pixel board∩in-frame mask,
and the per-frame affine shape. Tracks hold only tracked observations; the GT
anchor patches are not included.

One `.tracks` file therefore holds exactly one patch type at one scale — the
`output_name` is auto-suffixed `_<patch_type>_s<scale>` so `logpolar`/`cartesian`
runs (or different scales) never collide in the same file.

Each patch carries a `masks` companion (same `P×P×N` shape): 1 where the pixel
back-projects onto the board's world rectangle AND onto a real image pixel, 0
over off-board background, image edges, and perspective-warp gray-fill voids.
Confuser (untracked) patches carry an in-frame-only mask (no board); anchor
patches carry an in-frame mask against the rendered board image.

Two independent augmentations may fire per frame, both before extraction: a
resolution-loss down/up-resample (probability `p_downsample`, coordinate-
preserving) and a perspective warp (`p_perspective`) that strengthens the primary
board's real foreshortening. The perspective warp composes its homography into
every board's world→image map, so correspondences stay valid on the warped frame.
A third, `blur_sigma_max`, adds Gaussian blur and is OFF by default.

Additionally, one anchor sequence per board is written, named by the board `uid`
(rather than `\$(video)_\$(uid)`), holding every GT reference blob keyed by the
same `blob_track_id` used by the per-video tracks — so each track's
GT anchor is recoverable by id.

Boards and their gallery features are keyed by `uid` (the board's SHA-256 hash,
as produced by [`load_board`](@ref)) — the same key [`track_boards`](@ref)
returns — so lookups stay consistent end to end.

Already-processed sequences present in an existing output file are skipped (both
`\$(video)_\$(uid)` video sequences and `\$(uid)` anchor sequences), so the call
is resumable.

# Keyword Arguments
- `n_frames` / `stride`: frame sampling (mutually exclusive; default = every frame)
- `iou_threshold=0.6`: minimum symmetric IoU for a valid blob correspondence.
  Kept fairly strict: a board's homography extrapolates its grid over nearby
  near-coplanar boards in a stack, and a loose threshold lets the dense fractal
  pattern produce coincidental cross-board false matches. `0.6` removes almost
  all of them (a handful survive per board vs hundreds of true matches) at a
  modest recall cost on fully-visible boards.
- `max_detections=typemax(Int)`: per-frame detection cap (set it, e.g. `5000`, on
  high-resolution footage to keep the GPU descriptor-match GEMM in memory)
- `max_image_pixels=nothing`: optional cap on each frame's total pixel count
  (aspect ratio preserved), e.g. `3840*2160` (≈8.3 MP). The default keeps the
  native resolution. The detector allocates a full image pyramid sized to the
  frame, so set this on high-res stills (a 24 MP phone photo). Applied at read
  time so detect/track/adapt/canonicalize all share one coordinate system.
- `min_inliers=20`: board-acceptance threshold applied AFTER tracking — a
  tracked detection is kept only when its inlier count reaches this. It is a
  post-filter on `track_boards`' output, not a `HomographyGeometry` setting;
  the tracker itself runs the config-of-record gate (`min_inliers = 7`).
- `patch_type=:logpolar`: the single patch parameterization written to this file
  — `:logpolar`, `:cartesian`, or `:original` (see [`canonicalize`](@ref)).
- `scale=32`: the single patch scale — half the patch width/height in units of σ.
  Patches are stored `P×P×N`; `patch_type`/`scale` are recorded as attributes on
  each sequence's `tracks` group.
- `confuser_factor=1.0`: cap total confusers (untracked patches) at
  `confuser_factor × tracked patches`, sampled at random. Each confuser stores the
  chosen patch view + its in-frame mask under `untracked/{patches,masks}`. Supply
  `Inf` to keep **all** confusers (no cap).
- `debug_dir=nothing`: when set, save one PNG per processed **frame** into this
  directory — the frame the detector ran on, with *every* board detected in it
  overlaid in its own color: that board's matched GT blobs as projected ellipses
  plus its projected board rectangle. Colors are stable per board across frames
  and printed once at run start. Purely diagnostic; `nothing` disables it.

## Turning off the memory caps (large GPU)

Three knobs bound GPU/host memory on constrained hardware; on a large GPU set
them wide to process everything:
- `max_detections=typemax(Int)` (default) — no per-frame detection cap. Lower it
  (e.g. `5000`) only if the all-boards × all-detections match GEMM OOMs.
- `max_image_pixels=nothing` — keep native frame resolution (the default).
- `confuser_factor=Inf` — keep every confuser patch.
- `supersample=1`: anti-aliasing supersample factor forwarded to
  [`canonicalize`](@ref) for every patch view and scale. At the large scales
  (`64`/`96`/`128`) one output texel spans many source pixels, so the default
  single-tap sampling aliases; `>1` samples on a finer grid and box-averages
  down. Cost scales as `supersample²` and is paid at every scale, including the
  small ones that do not alias. `3`–`4` is the quality sweet spot.
- `p_downsample=0.75`: probability of the resolution-loss augmentation firing on a
  frame (down- then up-sample to simulate blur/scale loss; conditional on firing,
  the three tiers 0.5×/0.25×/0.125× are equally likely). Independent of the
  perspective warp — both can fire on one frame. `0` disables it.
- `p_perspective=0.5`: probability of applying the perspective augmentation to a
  frame. It warps the frame with a homography that strengthens the *primary*
  (highest-quality) board's existing foreshortening, keeping the board centroid
  fixed, and composes the same warp into every board's homography. Independent of
  the resolution-loss augmentation — both can fire on one frame. `0` disables it.
- `perspective_strengthen_max=1.0`: upper bound of the strengthening factor. The
  board plane's vanishing line is scaled by `1+γ` with `γ` drawn uniformly from
  `[0, perspective_strengthen_max]`, so `0` reproduces the frame's real
  perspective and larger values exaggerate it along the board's own tilt.
- `perspective_jitter_max=0.1`: upper bound of the extra random-direction tilt
  added on top of the strengthening, in a board-normalized frame (world extent →
  `[-1,1]²`), so the magnitude is unit- and board-size-invariant.
- `blur_sigma_max=0`: upper bound of the Gaussian blur augmentation. When
  positive, every frame is blurred with `σ ~ U[0, blur_sigma_max]` (px) before
  extraction; `0` disables it. Unlike the other two augmentations this one is
  unconditional once enabled — there is no separate probability — so it shifts
  the whole training distribution rather than sampling from it. Independent of
  the resolution-loss and perspective augmentations.
- `write_anchors=true`: write one per-board GT anchor sequence (named by uid)
  before processing media. Set `false` to skip them — useful for false-board
  diagnostics that only need the `<media>_<uid>` tracked sequences, since the
  anchor galleries are large and slow to compress.
- `writer_buffer=1`: depth of the async-writer queue. A single background task
  owns the output file and does all HDF5 compression/writes while the GPU thread
  builds the next sequence (overlap needs `julia -t ≥2`). Larger values give more
  slack but hold more finished (large) sequences in memory; `0` = rendezvous.
- `async_writer=true`: overlap HDF5 compression with GPU compute via the writer
  task above. Set `false` to write synchronously on the GPU thread (no overlap) —
  for `-t 1` / memory-constrained runs.
- `output_dir="."`, `output_name="tracks"`: output `\$output_name.tracks` location
- `backend=default_device()`: compute backend for adapt/canonicalize/describe
  (`CUDABackend()` when CUDA is loaded, otherwise `KA.CPU()`)
"""
function build_tracks_for_sequences(
        # The gallery and the network first, the media they are run on last
        # (`find_boards(board_feature_dict, descriptor_network, img)`). All three
        # are strings, so a call in the retired `(video_dir, board_dir,
        # model_path)` order would bind: the file check below makes it an
        # ArgumentError before anything loads.
        board_dir::AbstractString,
        descriptor_model_path::AbstractString,
        video_dir::AbstractString;
        n_frames::Union{Nothing, Int} = nothing,
        stride::Union{Nothing, Int} = nothing,
        iou_threshold = 0.6,
        max_detections::Int = typemax(Int),
        max_image_pixels::Union{Nothing, Int} = nothing,
        patch_type::Symbol = :logpolar,
        scale::Real = 32,
        min_inliers::Int = 20,
        confuser_factor::Real = 1.0,
        supersample::Int = 1,
        p_downsample::Real = 0.75,
        p_perspective::Real = 0.5,
        perspective_strengthen_max::Real = 1.0,
        perspective_jitter_max::Real = 0.1,
        blur_sigma_max::Real = 0,
        write_anchors::Bool = true,
        writer_buffer::Int = 1,
        async_writer::Bool = true,
        output_dir::AbstractString = ".",
        output_name::AbstractString = "tracks",
        debug_dir::Union{Nothing, AbstractString} = nothing,
        backend = default_device(),
    )
    _check_model_file(
        :build_tracks_for_sequences, descriptor_model_path,
        "(board_dir, descriptor_model_path, video_dir)"
    )
    _validate_patch_type(patch_type)
    # One file = one patch type at one scale — suffix the name so runs never
    # collide (e.g. `tracks_logpolar_s16.tracks`, `tracks_cartesian_s16.tracks`).
    output_name = _tracks_output_name(output_name, patch_type, scale)
    # --- Boards + descriptor model + per-board gallery features/bundles (by uid) ---
    gallery = load_board_gallery(descriptor_model_path, board_dir; device = backend)
    boards = gallery.boards
    board_by_uid = gallery.board_by_uid
    model = gallery.model
    board_feature_dict = gallery.board_feature_dict
    board_bundles = gallery.board_bundles

    # --- Media: videos (sampled into frames) and photos (single frame each) ---
    media = collect_media(video_dir)

    # --- Resume: sequences already present in the output file ---
    isdir(output_dir) || mkpath(output_dir)
    output_path = joinpath(output_dir, "$output_name.tracks")
    existing_seqs = Set{String}()
    if isfile(output_path)
        h5open(output_path, "r") do fid
            haskey(fid, "sequences") &&
                union!(existing_seqs, keys(fid["sequences"]::HDF5.Group))
        end
        println("Resuming: $(length(existing_seqs)) sequence(s) already present in $output_path")
    end

    # --- Async writer: a single background task owns the output file and does
    # every HDF5 write. zlib compression is single-threaded and leaves the GPU
    # idle; the GPU thread builds each sequence, hands it to the writer via a
    # bounded channel, then proceeds while compression runs concurrently. One
    # writer = one file owner → no concurrent-write corruption. Real overlap
    # needs `julia -t ≥2`; under `-t 1` it degrades to serial (still correct).
    # `writer_buffer` bounds how many finished sequences may queue (memory vs
    # overlap slack) — anchor galleries are large, so keep it small.
    # `async_writer=false` writes synchronously on the GPU thread (no overlap) —
    # an escape hatch for `-t 1` / memory-tight runs and the writer A/B. ---
    write_ch = async_writer ?
        Channel{Tuple{String, Dict{Int32, Dict{String, Vector}}, Any, Bool, Any}}(writer_buffer) : nothing
    writer = async_writer ? (
            Threads.@spawn try
                for (name, seq, confusers, is_anchor, frame_geometry) in write_ch
                    save_tracks(
                        output_path, seq; sequence_name = name, untracked = confusers,
                        patch_type, scale, is_anchor, frame_geometry
                    )
            end
        catch e
                close(write_ch, e)   # unblock a producer waiting on put! with the error
                rethrow()
        end
        ) : nothing
    sink = async_writer ?
        (
            (name, seq, confusers, is_anchor, frame_geometry = nothing) ->
            put!(write_ch, (name, seq, confusers, is_anchor, frame_geometry))
        ) :
        _default_sink(output_path, patch_type, scale)

    # --- Run summary: inputs, config, output target ---
    printstyled("build_tracks_for_sequences\n"; bold = true, color = :cyan)
    println("  boards:      $(length(boards)) from $board_dir")
    println("  media:       $(length(media)) item(s) from $video_dir")
    println("  model:       $descriptor_model_path")
    println("  patch/scale: $patch_type @ s$scale  (backend=$(nameof(typeof(backend))))")
    println("  writer:      $(async_writer ? "async (buffer=$writer_buffer)" : "sync")")
    println("  output:      $output_path")
    isnothing(debug_dir) || println("  debug_dir:   $debug_dir")
    isempty(existing_seqs) || println("  resuming:    $(length(existing_seqs)) sequence(s) already present")
    t_start = time()
    # Run-wide viewing-angle records, accumulated across every media item and
    # charted once the run finishes (see `print_view_angle_profile`).
    view_stats = ViewAngleStat[]

    try
        # Anchor sequences: one per board (named by uid), every GT reference blob,
        # keyed by the same `blob_track_id` as the per-media tracks so
        # each track's anchor stays recoverable. Written once (resume-aware); skip
        # via `write_anchors=false` (large + slow — false-board diagnostics don't
        # need them).
        if write_anchors
            pending_anchors = [uid for uid in keys(board_by_uid) if uid ∉ existing_seqs]
            if !isempty(pending_anchors)
                printstyled(
                    "Writing $(length(pending_anchors)) board anchor sequence(s)\n";
                    bold = true, color = :cyan
                )
                t_anchor = time()
                for (k, uid) in enumerate(pending_anchors)
                    println("  [$k/$(length(pending_anchors))] anchor $uid")
                    save_board_anchors(
                        output_path, board_by_uid[uid];
                        sink, patch_type, scale, inner_extent = cutoff(board_by_uid[uid].bundle), backend
                    )
                end
                println("  anchors done in $(round(time() - t_anchor; digits = 1))s")
            end
        end

        # ONNX inference (inside track_boards) is not thread-safe through one
        # session, so media items are processed serially on the GPU thread.
        for (media_idx, media_path) in enumerate(media)
            media_name = first(splitext(basename(media_path)))
            todo_uids = [
                uid for uid in keys(board_by_uid)
                    if "$(media_name)_$uid" ∉ existing_seqs
            ]
            if isempty(todo_uids)
                println(
                    "[$media_idx/$(length(media))] $(basename(media_path)): " *
                        "all boards already processed; skipping."
                )
                continue
            end

            printstyled(
                "[$media_idx/$(length(media))] $(basename(media_path)): " *
                    "$(length(todo_uids)) board(s) to track\n"; bold = true, color = :green
            )
            t_media = time()

            build_absolute_tracks(
                Dict(uid => board_by_uid[uid] for uid in todo_uids),
                model,
                media_path;
                board_feature_dict = Dict(uid => board_feature_dict[uid] for uid in todo_uids),
                board_bundles = Dict(uid => board_bundles[uid] for uid in todo_uids),
                sink,
                n_frames,
                stride,
                iou_threshold,
                max_detections,
                max_image_pixels,
                patch_type,
                scale,
                min_inliers,
                confuser_factor,
                supersample,
                p_downsample,
                p_perspective,
                perspective_strengthen_max,
                perspective_jitter_max,
                blur_sigma_max,
                output_dir,
                output_name,
                debug_dir,
                on_view_stats = stats -> append!(view_stats, stats),
                backend,
            )

            println(
                "[$media_idx/$(length(media))] $(basename(media_path)): " *
                    "done in $(round(time() - t_media; digits = 1))s"
            )

            # Return CUDA's cached pool memory to the driver between media items so
            # the Vulkan SIFT detector (separate allocator) has room to init its
            # image pyramid on the next frame — otherwise a growing CUDA pool
            # eventually OOMs `vksift_jl_init_ex` on large stills.
            GC.gc()
            reclaim_pool!(backend)
        end
        printstyled(
            "All media processed in $(round(time() - t_start; digits = 1))s → $output_path\n";
            bold = true, color = :cyan
        )
        # Run-wide detection yield vs. board obliquity, charted next to the output
        # file. Resumed runs only cover the media items processed this time;
        # `view_angle_profile(output_path)` recomputes it over the whole file.
        _emit_view_angle_profile(
            joinpath(output_dir, "$(output_name)_view_angles.png"), view_stats;
            title = "run total: avg detections per board vs. viewing angle"
        )
    finally
        # Signal no more sequences, then wait for the writer to drain + surface
        # any write error. (If the writer already failed it closed the channel.)
        # Guard on the handles, NOT on `async_writer`: both are
        # `Union{Nothing, ...}` from the ternaries that built them, and JET
        # cannot see that the flag implies non-`nothing`, so the flag form
        # reports `isopen(::Nothing)` / `wait(::Nothing)`.
        if write_ch !== nothing && writer !== nothing
            isopen(write_ch) && close(write_ch)
            wait(writer)
        end
    end
    return output_path
end

# Random resolution-loss augmentation: with probability `p_downsample` down- then
# up-sample the frame (0.5 / 0.25 / 0.125×) to simulate blur/scale loss, otherwise
# keep it. The resample preserves image coordinates, so a homography fit to the
# original frame stays valid on the augmented one. Conditional on firing, the
# three tiers are equally likely (⅓ each).
function _augment_frame(frame; p_downsample::Real = 0.75)
    rand() > p_downsample && return frame
    r = rand(Float32)
    s = r <= 1 / 3 ? 0.5 : (r <= 2 / 3 ? 0.25 : 0.125)
    small = map(d -> max(1, trunc(Int, d * s)), size(frame))
    return imresize(imresize(frame, small), size(frame))
end

# =============================================================================
# Perspective augmentation
# =============================================================================
#
# Randomly warp a frame with a homography that STRENGTHENS the board's real
# perspective, rather than applying a perspective-agnostic random warp. A board's
# world→image homography `H_wi` already encodes the plane's foreshortening: its
# projective (bottom) row is the board plane's vanishing line in board
# coordinates. Scaling that row up amplifies the existing foreshortening along
# the board's own tilt direction; a small random-direction tilt ("jitter") adds
# pose diversity on top.
#
# The board centroid is a FIXED POINT of the warp: the world center maps to the
# normalized origin `(0,0,1)`, whose image is the third COLUMN of the normalized
# homography — untouched by both the row-scaling and the jitter — so the board
# stays put in the frame while only its foreshortening changes.
#
# Magnitudes are applied in a board-normalized frame (world extent → [-1,1]²) so
# they are unit-invariant. All homographies here are in detector (x=col, y=row)
# pixel coordinates, matching `detect_blobs` / `H_wi`.

# `normalize_map` / `perspective_augmentation` are VisualGeometryToolkitCore.Synthetic's
# since 0.21.1; the RNG is explicit there — this builder draws from the task RNG.
using VisualGeometryToolkitCore.Synthetic: perspective_augmentation

# Swap x↔y (col↔row): converts a homography in detector (x,y) coordinates into
# array (row,col) coordinates for `warp`, which operates in array-index space.
const _XY_SWAP = SMatrix{3, 3, Float64}(
    0.0, 1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 0.0, 1.0
)

# Warp `img` by the image→image homography `H_aug` (detector (x,y) coords),
# cropping to the original axes. `warp` maps OUTPUT indices back to input, so the
# tform is the inverse warp; out-of-frame samples get the interpolant's fill.
function _warp_frame(img::AbstractMatrix, H_aug::SMatrix{3, 3})
    H_rc = _XY_SWAP * H_aug * _XY_SWAP          # (row,col) space
    Minv = inv(H_rc)
    # Blind division ON PURPOSE: ImageTransformations.warp probes every output
    # index; a horizon/degenerate sample must yield non-finite coordinates for
    # the interpolant's fill, never throw (2026-09 review).
    tform = idx -> begin
        v = Minv * SVector(Float64(idx[1]), Float64(idx[2]), 1.0)
        SVector(v[1] / v[3], v[2] / v[3])
    end
    etp = _interpolant(img)
    return clamp.(
        collect(ImageTransformations.warp(etp, tform, axes(img))),
        Gray{Float32}(0), Gray{Float32}(1)
    )
end

# `_IMAGE_EXTS` / `_is_image` come from media.jl — this file used to redefine
# both (a strictly narrower extension set), which is a const + method overwrite
# in the same module and blocked precompilation outright.

# Read a photo as a single-frame sequence, matching the element type
# `_read_video_frames` yields (`Vector{Matrix{Gray{Float32}}}`) so the rest of
# the pipeline is agnostic to whether the source was a video or a still image.
function _read_image_frame(path)
    img = load_image(path)
    return Matrix{Gray{Float32}}[Gray{Float32}.(img)]
end

# Downscale a frame to at most `max_pixels` total pixels (aspect ratio
# preserved); frames already within budget are returned unchanged. Applied at
# read time so the whole pipeline — detect / track / adapt / canonicalize — stays
# in one consistent coordinate system. Guards the Vulkan SIFT detector, which
# allocates a full image pyramid sized to the frame and OOMs on high-resolution
# stills (e.g. a 24 MP phone photo) on a memory-constrained GPU. A pixel budget
# (not a longest-side cap) tames square and wide frames uniformly.
function _limit_resolution(frame, max_pixels)
    npix = prod(size(frame))
    npix <= max_pixels && return frame
    s = sqrt(max_pixels / npix)
    return imresize(frame, map(d -> max(1, round(Int, d * s)), size(frame)))
end

# Read frames from a media file, dispatching on extension: videos → sampled
# frames (`n_frames`/`stride`); images → exactly one frame (sampling ignored).
# `max_image_pixels` (if set) caps each frame's total pixel count (see
# `_limit_resolution`); `nothing` keeps native resolution.
function _read_media_frames(
        path; n_frames = nothing, stride = nothing,
        max_image_pixels = nothing
    )
    frames = _is_image(path) ? _read_image_frame(path) :
        _read_video_frames(path; n_frames, stride)
    isnothing(max_image_pixels) && return frames
    return [_limit_resolution(f, max_image_pixels) for f in frames]
end

# Read sampled frames from a video into a Vector{Matrix{Gray{Float32}}}.
# `n_frames` and `stride` are mutually exclusive (n_frames computes a stride to
# span the clip; default = every frame).
function _read_video_frames(video_path; n_frames = nothing, stride = nothing)
    !isnothing(n_frames) && !isnothing(stride) &&
        throw(ArgumentError("Specify n_frames or stride, not both"))
    n_video_frames = something(VideoIO.get_number_frames(video_path), 0)
    if !isnothing(n_frames)
        stride = _frame_stride(video_path, n_frames)
    elseif isnothing(stride)
        stride = 1
    end
    stride >= 1 || throw(ArgumentError("stride must be ≥ 1, got $stride"))

    frames = Matrix{Gray{Float32}}[]
    VideoIO.openvideo(video_path) do vid
        while !eof(vid)
            push!(frames, Gray{Float32}.(read(vid)))
            !isnothing(n_frames) && length(frames) >= n_frames && break
            for _ in 1:(stride - 1)
                eof(vid) && break
                read(vid)
            end
        end
    end
    return frames
end

# Patch types the single-mode pipeline understands (see `canonicalize`).
const PATCH_TYPES = (:logpolar, :cartesian, :original)
_validate_patch_type(pt::Symbol) = pt in PATCH_TYPES ||
    throw(ArgumentError("patch_type must be one of $(PATCH_TYPES); got :$pt"))

# Scale tag for the output filename: `16` for an integer scale, `16p5` otherwise.
_scale_tag(s::Real) = isinteger(s) ? string(Int(s)) : replace(string(float(s)), "." => "p")
# One file = one patch type at one scale → `<name>_<patch_type>_s<scale>`.
_tracks_output_name(name, patch_type, scale) = "$(name)_$(patch_type)_s$(_scale_tag(scale))"

# =============================================================================
# Per-patch board∩in-frame mask
# =============================================================================
#
# Each canonicalized patch gets a companion P×P mask flagging which output pixels
# genuinely belong to a board. `_sample_grid` evaluates the SAME region map that
# `canonicalize` uses (via `_region_bm`, from canonicalize.jl) on the P×P output
# grid, recovering each pixel's source (row,col) sample location, and classifies
# its in-frame validity. The board test is applied afterwards (`_board_mask`),
# per board, since the sample grid is board-agnostic (shared across all boards
# detected in a frame).

# Return `(sample_xy, inframe)` for one frame's canonicalized patch:
#   sample_xy :: Matrix{SVector{2,Float64}} — per output pixel, the source sample
#                location in detector (x=col, y=row) coordinates of `img`
#   inframe   :: BitMatrix — true where that sample came from a REAL image pixel
# `img_size = (H, W)` is the source image size. `Minv` (or `nothing`) is the
# perspective-warp inverse in (row,col) space — when a warp fired, the sample is
# pulled back through it and tested against the ORIGINAL frame rectangle so warp
# gray-fill voids read as out-of-frame; `nothing` tests the sample directly
# (plain image edges).
function _sample_grid(frame, mode; patch_size, inner_extent, outer_extent, Minv, img_size)
    bm = _region_bm(mode, frame; patch_size, inner_extent, outer_extent)
    H, W = img_size
    xy = Matrix{SVector{2, Float64}}(undef, patch_size, patch_size)
    inf = falses(patch_size, patch_size)
    ctr = centroid(frame)   # (x=col, y=row), warped coords — bare px by the frame contract
    w_ref = Minv === nothing ? 1.0 :
        VgtkCore.homogeneous_w(Minv, SVector(Float64(ctr[2]), Float64(ctr[1])))
    @inbounds for j in 1:patch_size, i in 1:patch_size
        p = bm(SVector(Float64(i), Float64(j)))     # source (row, col)
        r, c = p[1], p[2]
        xy[i, j] = SVector(c, r)                    # detector (x=col, y=row)
        if Minv === nothing
            inf[i, j] = (1.0 <= r <= H) & (1.0 <= c <= W)
        else
            # Guarded: the perspective augmentation makes Minv genuinely
            # projective, so beyond-horizon samples would wrap into bounds
            # and read warp gray-fill as in-frame (2026-09 review). w_ref
            # anchors on the blob's own centroid — a real, visible sample.
            q = VgtkCore.to_euclidean(Minv * SVector(r, c, 1.0), w_ref)
            inf[i, j] = q !== nothing && (1.0 <= q[1] <= H) & (1.0 <= q[2] <= W)
        end
    end
    return xy, inf
end

# Horizon reference for `_in_board`'s sign guard, shared by `_board_mask` and
# `_onboard_mask`: the homogeneous w of the board centre's image under the
# image→world map `H_iw`. Without it, samples beyond the plane's horizon wrap
# into the rectangle from behind the camera and are marked board-valid. The
# two masks drifted on exactly this guard once (`_in_board` carried it, one
# twin did not — 2026-09 audit), so it is spelled ONCE.
function _horizon_ref(H_wi, H_iw, world_rect)
    lo, hi = minimum(world_rect), maximum(world_rect)
    cx, cy = _project_pt(H_wi, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
    return VgtkCore.homogeneous_w(H_iw, SVector(cx, cy))
end

# Board∩in-frame mask for one patch: start from the (board-agnostic) `inframe`
# grid, then zero every pixel whose sample does NOT back-project (via `H_wi_inv`,
# image→world) into the board's world rectangle `world_rect`.
function _board_mask(sample_xy, inframe, H_wi_inv, world_rect)
    P = size(inframe, 1)
    w_ref = _horizon_ref(inv(H_wi_inv), H_wi_inv, world_rect)
    m = Matrix{Float32}(undef, P, P)
    @inbounds for j in 1:P, i in 1:P
        on = inframe[i, j]
        if on
            xy = sample_xy[i, j]
            on = _in_board(H_wi_inv, w_ref, world_rect, xy[1], xy[2])
        end
        m[i, j] = on ? 1.0f0 : 0.0f0
    end
    return m
end

# =============================================================================
# Board-overlay debug images
# =============================================================================
#
# When `build_absolute_tracks` is given a `debug_dir`, each processed frame is
# saved as ONE PNG holding EVERY board detected in it: per board, its accepted
# (matched-to-this-board) GT blobs projected (world→image via `H_wi`) over the
# frame the detector actually ran on, plus its projected world rectangle, all in
# that board's own color. Underneath those, in thin magenta, sit the detections
# that landed on a board — post `adapt` + `filter_blobs`, in image space.
# Reading it:
#
#   magenta, no board  an adapted on-board detection that matched no GT blob
#   board, no magenta  a GT blob the detector missed
#   neither            nothing was found there at all — blur, scale, or the
#                      frame-wide `max_detections` cap evicting it
#
# Off-board clutter is dropped: it is never a candidate for any of those
# readings, and at a few thousand detections a frame it buried the boards. Pure
# array plotting (no Makie) so it works under any backend.

# The raster overlays (`put_pixel!`, `draw_ellipse!`, `draw_projected_ellipse!`,
# `draw_polyline!`) are VisualGeometryToolkitCore's since 0.21.1.
using VisualGeometryToolkitCore: draw_ellipse!, draw_projected_ellipse!, draw_polyline!

@inline function _project_pt(H, x, y)
    w = H * SVector(x, y, 1.0)
    return (w[1] / w[3], w[2] / w[3])
end

# Project the four corners of a world-space rectangle through `H_wi` (world→image,
# detector coords) into image (x,y) points. The homography leads, as in VGC's
# `draw_projected_ellipse!`; typed so the retired `(world_rect, H_wi)` order is a
# MethodError.
function _project_quad(H_wi::AbstractMatrix, world_rect)
    lo, hi = minimum(world_rect), maximum(world_rect)
    corners = ((lo[1], lo[2]), (hi[1], lo[2]), (hi[1], hi[2]), (lo[1], hi[2]))
    return map(((x, y),) -> _project_pt(H_wi, x, y), corners)
end

# 2×2 linear shape matrix of a frame primitive, unitless. World GT blobs are
# isotropic `Disc`s (shape = radius·I); adapted detections are `EllipticFrame`s.
_overlay_shape(f) = SMatrix{2, 2, Float64}(ustrip.(shape_matrix(f)))   # (audited) whole-frame strip: no target-unit form for a shape matrix
_overlay_shape(d::Disc) = (r = ustrip(VgtkCore.scale(d)); SMatrix{2, 2, Float64}(r, 0.0, 0.0, r))   # (audited) whole-Disc strip

# Per-board overlay color, stable for a given board slot `k` across all frames.
# Hues walk the circle by the golden angle starting at green (the single-board
# color this overlay used to have), so any number of boards stays separable.

# Draw one board's accepted (matched-to-this-board) GT blobs plus its projected
# world quad onto `rgb`, all in `color`. `world_frames` are the board's
# world-space GT frames; `matched_ids` is the set of their indices that got a
# correspondence this frame — only those are drawn.
function _draw_board!(rgb, H_wi::AbstractMatrix, world_frames, matched_ids, cutoff, world_rect, color)
    draw_polyline!(rgb, _project_quad(H_wi, world_rect), color)
    for j in matched_ids
        ef = feature_frame(world_frames[j])
        c = ustrip.(centroid(ef))   # (audited) defensive strip over bare-or-quantity centroids
        S = _overlay_shape(ef)
        draw_projected_ellipse!(rgb, H_wi, SVector(c[1], c[2]), S, cutoff, color)
    end
    return rgb
end

# Detections are board-agnostic, so they get one fixed color rather than a
# board's. Magenta: no board hue lands near it until the fifth board in a frame
# (the walk runs 120°, 257.5°, 35°, 172.5°, …), and it survives the gray frame
# and the gray warp fill that a neutral white washed out against.
const _DETECTION_COLOR = RGB{Float32}(1.0f0, 0.0f0, 1.0f0)

# Does the image point (x, y) back-project inside this board? `H_iw` is the
# image→world homography (`inv(H_wi)`) and `w_ref` its third coordinate at the
# board's own image center. Points beyond the plane's horizon flip that sign and
# would otherwise wrap into the rectangle from behind the camera — requiring
# agreement with `w_ref` keeps them out (the same guard `_n_gt_in_image` uses).
_in_board(H_iw, w_ref, world_rect, x, y) =
    VgtkCore.maps_within(H_iw, world_rect, SVector(Float64(x), Float64(y)), w_ref)

# Which of the frame's detections land on this board — a mask over `det_frames`,
# computed once per (board, frame). Over the raw detections it gives
# `n_sift_on_board`; over the adapt+filter survivors it gives
# `n_detections_on_board` AND the mask the overlay draws in magenta, so that
# count and those ellipses can never disagree. `H_wi` is world→image; the rest
# is `_in_board`.
function _onboard_mask(det_frames, H_wi, world_rect)
    H_iw = inv(H_wi)
    w_ref = _horizon_ref(H_wi, H_iw, world_rect)
    m = falses(length(det_frames))
    for (i, d) in enumerate(det_frames)
        c = ustrip.(centroid(feature_frame(d)))   # (audited) defensive strip over bare-or-quantity centroids
        m[i] = _in_board(H_iw, w_ref, world_rect, c[1], c[2])
    end
    return m
end

# Build an RGB overlay of `img` carrying EVERY board detected in this frame.
# `overlays` is a vector of `(; world_frames, matched_ids, cutoff, H_wi,
# world_rect, color, onboard)`, one per board — each drawn in its own color,
# over the magenta detection layer.
#
# `det_frames` is the frame's shared detection set — one detection run per frame,
# board-agnostic — so it covers the whole image, background clutter included.
# Each overlay carries the `onboard` mask of the ones that fall on ITS board
# (`_onboard_mask`, the same mask `n_detections_on_board` counts), and only
# those are drawn, at the cutoff of the board they fell in, so their radii are
# comparable to the GT ellipses they should be sitting on. A detection inside
# two overlapping boards is drawn once, for the first board that claims it.
function _frame_overlay(img, overlays, det_frames)
    rgb = RGB{Float32}.(img)
    drawn = falses(length(det_frames))
    for o in overlays, i in eachindex(o.onboard)
        (o.onboard[i] && !drawn[i]) || continue
        drawn[i] = true
        ef = feature_frame(det_frames[i])
        c = ustrip.(centroid(ef))   # (audited) defensive strip over bare-or-quantity centroids
        draw_ellipse!(
            rgb, SVector(c[1], c[2]), _overlay_shape(ef), o.cutoff,
            _DETECTION_COLOR; w = 1
        )
    end
    for o in overlays
        _draw_board!(
            rgb, o.H_wi, o.world_frames, o.matched_ids, o.cutoff,
            o.world_rect, o.color
        )
    end
    return rgb
end

# Save the all-boards overlay PNG for one frame; best-effort — any failure is
# logged and swallowed so debugging output never aborts a run.
function _save_frame_overlay(debug_dir, name, img, overlays, det_frames)
    return try
        isdir(debug_dir) || mkpath(debug_dir)
        FileIO.save(
            joinpath(debug_dir, "$name.png"),
            _frame_overlay(img, overlays, det_frames)
        )
    catch e
        @warn "frame-overlay save failed for $name" exception = e
    end
end

# =============================================================================
# Viewing-angle detection profile
# =============================================================================
#
# One `ViewAngleStat` per (board, frame) detection, holding the board's viewing
# obliquity in that frame (`_view_angle`, radians) next to the counts along the
# detect → adapt/filter → match chain. Every count is for ONE board in ONE frame,
# and each is named for its stage and its scope:
#
#   n_sift_on_board        RAW `detect_blobs` output — before `adapt` and
#                          `filter_blobs` — whose centers back-project inside
#                          THIS board's rectangle. What the detector actually saw
#                          on the board, and the first number to look at when a
#                          board yields nothing. NB `detect_blobs` is capped
#                          at `max_detections` by |response| over the
#                          WHOLE frame, so heavy background clutter can evict
#                          on-board detections before this is counted.
#   n_detections_on_board  Of those, the ones that survived `adapt` +
#                          `filter_blobs` (aspect ratio, scale ratio, centre
#                          drift, clipping — GT is not involved). This is what
#                          the overlay draws in magenta, and what `n_matched_gt`
#                          is selected from.
#   n_matched_gt           Of those, the ones that won a bidirectional-IoU
#                          correspondence against this board's GT blobs projected
#                          through `H_wi` (`symmetric_iou_match`). These, and
#                          only these, become training observations. NOT the
#                          board detector's descriptor/RANSAC inliers — that is
#                          `n_inliers`, which never gates blob tracking.
#   n_gt_in_image          GT blobs of this board whose projection lands inside
#                          the frame at all, detectable or not. Pure geometry,
#                          the honest denominator: `n_matched_gt / n_gt_in_image`
#                          is the board's per-frame recall, and unlike the
#                          frame-wide count it shrinks as the board leaves view.
#   n_detections_in_frame  Post-`adapt`/`filter_blobs` blobs anywhere in the
#                          frame. Board-agnostic, so every board in a frame
#                          shares it — context, not a denominator. On a small or
#                          blurred board it reads in the thousands while
#                          `n_detections_on_board` is 0; that gap is background
#                          clutter, not a bug.
#
# Binning these by obliquity answers "how fast does the tracked-blob yield fall
# off as a board tilts away from fronto-parallel?" `build_absolute_tracks`
# profiles each media item and `build_tracks_for_sequences` the whole run.
#
# `view_angle_profile(path)` rebuilds the same records from a finished `.tracks`
# file, so a profile can be re-charted (or re-binned) without re-running the
# build. `view_angle_figure` / `save_view_angle_figure` (BlobBoardsMakieExt)
# render them; `print_view_angle_profile` is the no-backend text fallback.

const ViewAngleStat = @NamedTuple{
    uid::String, frame_idx::Int, view_angle::Float64,
    n_matched_gt::Int, n_gt_in_image::Int, n_sift_on_board::Int,
    n_detections_on_board::Int, n_detections_in_frame::Int,
}

# Bin edges in DEGREES. `_view_angle` returns `acos(1/t) ∈ [0, π/2)`, so 0–90°
# spans the range; fixed 10° bins keep profiles comparable across runs.
const VIEW_ANGLE_EDGES = 0.0:10.0:90.0

# Aggregate `stats` into the (uniform) `edges`, given in degrees. One NamedTuple
# per bin: board-frame count, distinct boards, mean/sd matched, the mean of each
# remaining count, and the POOLED recall `Σ matched / Σ gt_in_image` (pooled
# rather than a mean of per-frame ratios so near-edge frames with two visible
# blobs can't swing the bin). Angles outside `[first, last]` clamp into the end
# bins; non-finite ones are dropped.
function _bin_view_angles(stats, edges = VIEW_ANGLE_EDGES)
    nb = length(edges) - 1
    lo, hi = float(first(edges)), float(last(edges))
    w = (hi - lo) / nb
    matched = [Int[] for _ in 1:nb]
    gt = zeros(Int, nb)
    sift = zeros(Int, nb)
    on_board = zeros(Int, nb)
    in_frame = zeros(Int, nb)
    boards = [Set{String}() for _ in 1:nb]
    for s in stats
        deg = rad2deg(s.view_angle)
        isfinite(deg) || continue
        k = clamp(floor(Int, (deg - lo) / w) + 1, 1, nb)
        push!(matched[k], s.n_matched_gt)
        gt[k] += s.n_gt_in_image
        sift[k] += s.n_sift_on_board
        on_board[k] += s.n_detections_on_board
        in_frame[k] += s.n_detections_in_frame
        push!(boards[k], s.uid)
    end
    return [
        (;
                lo = lo + (k - 1) * w, hi = lo + k * w,
                n = length(matched[k]), n_boards = length(boards[k]),
                mean_matched = isempty(matched[k]) ? 0.0 : mean(matched[k]),
                # sd needs ≥2 samples; a single-sample bin gets a zero-width bar.
                sd_matched = length(matched[k]) < 2 ? 0.0 : std(matched[k]),
                mean_gt_in_image = isempty(matched[k]) ? 0.0 : gt[k] / length(matched[k]),
                mean_sift_on_board = isempty(matched[k]) ? 0.0 : sift[k] / length(matched[k]),
                mean_detections_on_board = isempty(matched[k]) ? 0.0 : on_board[k] / length(matched[k]),
                mean_detections_in_frame = isempty(matched[k]) ? 0.0 : in_frame[k] / length(matched[k]),
                recall = gt[k] == 0 ? NaN : sum(matched[k]) / gt[k],
            )
            for k in 1:nb
    ]
end

# Horizontal bar `width` cells wide showing `value/vmax`, at ⅛-cell resolution so
# small-but-nonzero bins stay visible.
const _BAR_EIGHTHS = ('▏', '▎', '▍', '▌', '▋', '▊', '▉', '█')
function _bar(value, vmax, width)
    (vmax > 0 && value > 0) || return ""
    full, rem = divrem(round(Int, clamp(value / vmax, 0, 1) * width * 8), 8)
    return "█"^full * (rem == 0 ? "" : string(_BAR_EIGHTHS[rem]))
end

_fmt1(x) = string(round(x; digits = 1))

_fmt_pct(x) = isnan(x) ? "   —" : string(round(100x; digits = 1)) * "%"

"""
    print_view_angle_profile([io], stats; edges=VIEW_ANGLE_EDGES, width=40, title=...)

Text fallback for [`view_angle_figure`](@ref) — the same binning as a terminal
bar chart, for runs with no Makie backend loaded. `stats` is a collection of
[`ViewAngleStat`](@ref) records, one per (board, frame) detection.

Each row is one angle bin: the mean GT blobs matched per board-frame (the bar),
the recall against the board's blobs visible in those frames, the mean total
frame detections, and the bin's board-frame count. Empty bins are skipped.
"""
print_view_angle_profile(stats; kwargs...) =
    print_view_angle_profile(stdout, stats; kwargs...)

function print_view_angle_profile(
        io::IO, stats;
        edges = VIEW_ANGLE_EDGES, width::Int = 34,
        title = "avg detections per board vs. viewing angle"
    )
    if isempty(stats)
        printstyled(io, "  $title: no board detections\n"; color = :yellow)
        return nothing
    end
    bins = _bin_view_angles(stats, edges)
    vmax = maximum(b -> b.mean_matched, bins)
    n_boards = length(Set(s.uid for s in stats))
    tot_gt = sum(s.n_gt_in_image for s in stats)
    printstyled(io, "  $title\n"; bold = true, color = :cyan)
    println(
        io, "    ($(length(stats)) board-frames over $n_boards board(s); " *
            "bar = mean matched blobs per board-frame)"
    )
    println(
        io, "    on the board: sift = raw detect_blobs · adapted = of those, " *
            "surviving"
    )
    println(io, "    adapt+filter · matched = of those, IoU-matched to a GT blob")
    println(
        io, "    context: gt in img = this board's GT projecting into the frame · " *
            "in frame ="
    )
    println(io, "    adapted detections anywhere in the frame")
    println(io, "     angle°   matched / gt in img   recall    sift   adapted   in frame   frames")
    for b in bins
        b.n == 0 && continue
        println(
            io, "    ", lpad(round(Int, b.lo), 3), "–", lpad(round(Int, b.hi), 3), " ",
            lpad(_fmt1(b.mean_matched), 9), " / ", lpad(_fmt1(b.mean_gt_in_image), 9), " ",
            lpad(_fmt_pct(b.recall), 8), " ", lpad(_fmt1(b.mean_sift_on_board), 7), " ",
            lpad(_fmt1(b.mean_detections_on_board), 9), " ",
            lpad(_fmt1(b.mean_detections_in_frame), 10), " ",
            lpad(b.n, 6), "  ", _bar(b.mean_matched, vmax, width)
        )
    end
    println(
        io, "    overall ", lpad(_fmt1(mean(s.n_matched_gt for s in stats)), 9), " / ",
        lpad(_fmt1(mean(s.n_gt_in_image for s in stats)), 9), " ",
        lpad(_fmt_pct(tot_gt == 0 ? NaN : sum(s.n_matched_gt for s in stats) / tot_gt), 8), " ",
        lpad(_fmt1(mean(s.n_sift_on_board for s in stats)), 7), " ",
        lpad(_fmt1(mean(s.n_detections_on_board for s in stats)), 9), " ",
        lpad(_fmt1(mean(s.n_detections_in_frame for s in stats)), 10), " ",
        lpad(length(stats), 6)
    )
    return nothing
end

"""
    view_angle_profile(tracks_path; sequence=nothing) -> Vector{ViewAngleStat}

Rebuild the per-(board, frame) viewing-angle records from a finished `.tracks`
file, so a profile can be re-charted (or re-binned) offline without re-running
the build:

```julia
stats = view_angle_profile("tracks_logpolar_s32.tracks")
save_view_angle_figure("profile.png", stats)          # needs CairoMakie
save_view_angle_figure("fine.svg", stats; edges = 0:5:90)
print_view_angle_profile(filter(s -> s.uid == uid, stats))   # one board, no backend
```

Reads each tracked sequence's per-frame `view_angles` / `n_gt_in_image` / `n_sift_on_board` /
`n_detections_on_board` / `n_detections_in_frame` table and counts the observations sharing each `frame_id` to
recover how many blobs that board matched in that frame. Anchor sequences (no
frame geometry) are skipped; `sequence` restricts to one sequence name.
"""
function view_angle_profile(
        tracks_path::AbstractString;
        sequence::Union{Nothing, AbstractString} = nothing
    )
    stats = ViewAngleStat[]
    h5open(tracks_path, "r") do fid
        haskey(fid, "sequences") || return
        seqs = fid["sequences"]::HDF5.Group
        for name in (sequence === nothing ? keys(seqs) : [String(sequence)])
            g = (seqs[name]::HDF5.Group)["tracks"]::HDF5.Group
            # Anchor sequences carry no per-frame geometry — nothing to bin.
            (haskey(g, "view_angles") && haskey(g, "frame_id")) || continue
            angles = read(g["view_angles"]::HDF5.Dataset)
            fids = Int.(read(g["homography_frame_ids"]::HDF5.Dataset))
            # The count tables are later additions, and two of them were once
            # stored under shorter names — read the current key, fall back to the
            # legacy one, then to 0 (unknown) so any older file still profiles,
            # only with blank columns where it recorded nothing.
            rd(k, legacy = nothing) =
                haskey(g, k) ? Int.(read(g[k]::HDF5.Dataset)) :
                (
                    legacy !== nothing && haskey(g, legacy) ?
                    Int.(read(g[legacy]::HDF5.Dataset)) : zeros(Int, length(fids))
                )
            n_gt = rd("n_gt_in_image", "n_gt_in_image")
            n_frame = rd("n_detections_in_frame", "n_detections_in_frame")
            n_board = rd("n_detections_on_board")
            n_sift_on_board = rd("n_sift_on_board")
            # Observations per source frame = blobs this board matched there.
            matched = Dict{Int, Int}()
            for f in read(g["frame_id"]::HDF5.Dataset)
                matched[Int(f)] = get(matched, Int(f), 0) + 1
            end
            # Sequence names are "<media>_<uid>"; the uid never contains "_".
            uid = String(last(rsplit(name, '_'; limit = 2)))
            for (k, f) in enumerate(fids)
                push!(
                    stats, (;
                        uid, frame_idx = f, view_angle = Float64(angles[k]),
                        n_matched_gt = get(matched, f, 0),
                        n_gt_in_image = n_gt[k], n_sift_on_board = n_sift_on_board[k],
                        n_detections_on_board = n_board[k],
                        n_detections_in_frame = n_frame[k],
                    )
                )
            end
        end
    end
    return stats
end

"""
    view_angle_figure(stats; edges=VIEW_ANGLE_EDGES, title, size) -> Makie.Figure

Chart the average detections per board against viewing obliquity: mean matched
blobs per board-frame, recall against the board's visible GT blobs, and the
per-bin board-frame support. Requires a Makie backend (BlobBoardsMakieExt).
"""
function view_angle_figure end

"""
    save_view_angle_figure(path, stats; kwargs...) -> path

Render [`view_angle_figure`](@ref) and write it to `path`; the extension picks
the format (`.png`, `.svg`, `.pdf`). Requires a Makie backend that can save —
`using CairoMakie` is the headless choice. Keyword arguments are forwarded to
[`view_angle_figure`](@ref).
"""
function save_view_angle_figure end

# Chart `stats` to `path` if a Makie backend is loaded, otherwise print the text
# profile. Best-effort: a rendering failure never aborts a track build, it just
# degrades to the terminal chart. Used by the build drivers below.
function _emit_view_angle_profile(path, stats; title, edges = VIEW_ANGLE_EDGES)
    if !isempty(stats) && !isempty(methods(save_view_angle_figure))
        try
            # `save_view_angle_figure` is a core stub with ZERO methods until
            # BlobBoardsMakieExt loads, so a direct call is a static
            # MethodError to JET. The `methods(...)` test above is the real
            # guard; `invokelatest` is what makes the dispatch honest.
            Base.invokelatest(save_view_angle_figure, path, stats; title, edges)
            printstyled("  $title → $path\n"; bold = true, color = :cyan)
            return path
        catch e
            @warn "viewing-angle chart failed; falling back to the text profile" exception = e
        end
    end
    print_view_angle_profile(stats; title, edges)
    return nothing
end

# =============================================================================
# Blob (track) identity
# =============================================================================
#
# Every blob needs a GLOBALLY unique id: when many board sequences are loaded
# together each physical blob must be its own contrastive class, and a board's
# per-media tracks must share ids with that board's anchor sequence (same
# board + blob → same id) so a track's GT anchor stays recoverable.
#
# The id packs a per-board hash field ABOVE a reserved blob-index field:
#     id = (board_field << BLOB_ID_BITS) | i          (i is the 1-based blob index)
# so two boards can only collide if their hash fields collide — never because
# their index ranges overlap. The old scheme (`board_offset + i`, with
# `board_offset` just the low 16 bits of the hash) instead ADDED a 16-bit offset
# to the index, so any two boards whose offsets were within a board's blob count
# produced overlapping ranges — near-certain collisions across 50 boards.
#
# Ids must fit a signed Int32 (31 positive bits) for downstream consumers. We
# split it 19 (board) / 12 (blob): up to 4095 blobs/board (3× the ~1330 observed
# max) under 2^19 board buckets. That leaves a ~0.2% birthday chance that two of
# ~50 boards share a hash field, so the driver calls `check_blob_id_collisions`
# up front and errors loudly rather than silently merging two blobs' classes. The
# id is a pure function of (board hash, blob index) — independent of which boards
# are present in a run — so anchors written in one run match the per-media tracks
# written in another (resume-safe).
const BLOB_ID_BITS = Int32(12)                       # blob-index field: up to 2^12−1 blobs/board
const BLOB_ID_MASK = (Int32(1) << BLOB_ID_BITS) - Int32(1)
const BOARD_FIELD_MASK = UInt64((UInt64(1) << (31 - 12)) - 1)    # 19-bit per-board hash field

# 19-bit per-board hash field from the board's STORED identity — the bundle's
# `hashes.config` (`load_board(...).config_hash`; its first 3 hex are the uid) —
# first 16 hex = 64 bits, masked down. Stable per board, independent of the run's
# board set AND of the config's encoding: it used to rehash the loaded preamble,
# which differed from the stored identity for every migrated board and moved
# again with every schema change (v1 → v2).
_board_field(config_hash::AbstractString) =
    Int32(parse(UInt64, config_hash[1:16]; base = 16) & BOARD_FIELD_MASK)

# Global id for blob `i` (1-based) on the board with stored identity `config_hash`. Anchor and per-media paths
# both call this so their ids agree by construction. Fits a signed Int32.
function blob_track_id(config_hash::AbstractString, i::Integer)
    1 <= i <= BLOB_ID_MASK ||
        error("blob index $i out of range 1:$(BLOB_ID_MASK); raise BLOB_ID_BITS")
    return (_board_field(config_hash) << BLOB_ID_BITS) | Int32(i)
end

# Verify no two DISTINCT boards share a hash field (which would collide their
# whole id ranges). Called by the driver once the full board set is known; the
# per-board `blob_track_id` can't see the set, so the check lives here.
function check_blob_id_collisions(config_hashes)
    seen = Dict{Int32, String}()   # board_field => uid
    for (uid, config_hash) in config_hashes
        f = _board_field(config_hash)
        if haskey(seen, f) && seen[f] != uid
            error(
                "blob-id hash-field collision: boards $(seen[f]) and $uid map to " *
                    "field $f — regenerate one board or widen the board field."
            )
        end
        seen[f] = uid
    end
    return
end

# A sink persists a finished sequence — `(sequence_name, tracks, confusers,
# is_anchor, frame_geometry = nothing)`. `frame_geometry` is the per-frame
# homography / view-angle table for tracked sequences (`nothing` for anchors).
# The default writes synchronously; the driver swaps in a channel-backed sink so
# HDF5 compression overlaps GPU compute (see the writer in
# `build_tracks_for_sequences`). `confusers` is `nothing` for anchor sequences.
_default_sink(output_path, patch_type, scale) =
    (name, seq, confusers, is_anchor, frame_geometry = nothing) ->
save_tracks(
    output_path, seq; sequence_name = name, untracked = confusers,
    patch_type, scale, is_anchor, frame_geometry
)

"""
    save_board_anchors(output_path, board; inner_extent, patch_type = :logpolar, scale = 32,
                       patch_size = 64, backend = default_device(), sink = nothing)

Write one anchor sequence for a board to the `.tracks` file at `output_path`,
named by the board's uid: every ground-truth blob (canvas coordinates, sampled
from the rendered board image) as one patch of `patch_type` at `scale`. Track
ids match the video tracks' ids (`blob_track_id`), so a video track's anchor is
found by id; the sequence is flagged `is_anchor = true` and its masks are
in-frame only. `sink` replaces the synchronous writer (the `.tracks` builder
supplies a channel-backed one).
"""
function save_board_anchors(
        output_path::AbstractString,
        board;
        sink = nothing,
        patch_type::Symbol = :logpolar,
        scale::Real = 32,
        patch_size = 64,
        inner_extent,
        backend = default_device(),
    )
    _validate_patch_type(patch_type)
    sink = something(sink, _default_sink(output_path, patch_type, scale))
    board_img = board.image
    img_size = size(board_img)
    N = length(board.canvas_frames)
    patches = canonicalize(
        board.canvas_frames, board_img; mode = patch_type,
        patch_size, inner_extent, outer_extent = scale, device = backend
    )
    masks = [
        Float32.(
                _sample_grid(
                    f, patch_type; patch_size, inner_extent,
                    outer_extent = scale, Minv = nothing, img_size
                )[2]
            )
            for f in board.canvas_frames
    ]
    anchors = Dict{Int32, Dict{String, Vector}}()
    for i in 1:N
        anchors[blob_track_id(board.config_hash, i)] = Dict{String, Vector}(
            "patch" => Any[patches[i]],
            "mask" => Any[masks[i]],
            "affine_shape" => Any[SMatrix{2, 2, Float64}(1.0I)],
        )
    end
    sink(board.uid, anchors, nothing, true)
    return output_path
end

# The perspective augmentation composes ONE collineation into every board's
# world→image map, so it needs the primary board's PLANE HOMOGRAPHY — which the
# homography arm's model is and no other model is: a pose needs a camera this
# driver does not carry, and a radial model (`RadialHomography`,
# `RadialCamera`) is not a collineation of the image. Refuse those by name.
# Until 2026-09-21 this was `forward(_model_to_homography(model, nothing))`,
# which on a radial-arm detection died in a MethodError inside `forward` —
# with `p_perspective = 0.5`, on half the frames (JET).
_augmentation_homography(H::HomographyMat) = SMatrix{3, 3, Float64}(to_homogeneous(_model_to_homography(H, nothing)))
_augmentation_homography(m) = throw(
    ArgumentError(
        "the perspective augmentation needs a homography-arm detection (a plane " *
            "homography); got a $(nameof(typeof(m))) — build these tracks with " *
            "`p_perspective = 0`, or detect with the homography arm"
    )
)

"""
    build_absolute_tracks(boards, model, video_path; board_feature_dict,
                          board_bundles, kwargs...) -> output_path

Build multi-modal tracks for one video against a set of boards (see
[`build_tracks_for_sequences`](@ref) for the higher-level driver).

`boards` is a `Dict{uid => board}` (each a [`load_board`](@ref) NamedTuple with
an extra `image` field), `board_feature_dict` a `Dict{uid => features}` and
`board_bundles` a `Dict{uid => BoardBundle}` — all keyed by the same `uid` that
[`track_boards`](@ref) returns. One `.tracks` sequence is written per board that
gets tracked, named `\$(video)_\$(uid)`.

Sampling, filtering, patch and augmentation keywords (`n_frames`/`stride`,
`iou_threshold`, `max_detections`, `max_image_pixels`,
`patch_type`, `scale`, `confuser_factor`, `supersample`, `p_downsample`,
`p_perspective`, `perspective_strengthen_max`, `perspective_jitter_max`,
`blur_sigma_max`) behave as documented on
[`build_tracks_for_sequences`](@ref), which forwards them here. `patch_size=64`
is local to this function. `output_name` is used verbatim (the higher-level
driver adds the `_<patch_type>_s<scale>` suffix). `sink` receives each finished
sequence as `(name, tracks, confusers, is_anchor)`; the default writes it
synchronously. `on_view_stats`, when given, receives this media item's
viewing-angle records once, as `on_view_stats(stats::Vector{ViewAngleStat})`
(sorted by `(uid, frame_idx)`) — the driver collects them into its run-wide
chart.
"""
function build_absolute_tracks(
        boards,
        model::DescriptorNetwork,
        video_path::AbstractString;
        board_feature_dict::Dict,
        board_bundles::Dict,
        n_frames::Union{Nothing, Int} = nothing,
        stride::Union{Nothing, Int} = nothing,
        iou_threshold = 0.6,
        max_detections::Int = typemax(Int),
        max_image_pixels::Union{Nothing, Int} = nothing,
        patch_size = 64,
        patch_type::Symbol = :logpolar,
        scale::Real = 32,
        min_inliers::Int = 12,
        confuser_factor::Real = 1.0,
        supersample::Int = 1,
        p_downsample::Real = 0.75,
        p_perspective::Real = 0.5,
        perspective_strengthen_max::Real = 1.0,
        perspective_jitter_max::Real = 0.1,
        blur_sigma_max::Real = 0,
        sink = nothing,
        output_dir::AbstractString = ".",
        output_name::AbstractString = "tracks",
        debug_dir::Union{Nothing, AbstractString} = nothing,
        on_view_stats = nothing,
        backend = default_device(),
    )
    _validate_patch_type(patch_type)
    confuser_factor >= 0 || throw(ArgumentError("`confuser_factor` must be ≥ 0"))
    isnothing(debug_dir) || mkpath(debug_dir)
    boards = boards isa Dict ? boards :
        Dict(b.uid => b for b in (boards isa AbstractVector ? boards : [boards]))
    isdir(output_dir) || mkpath(output_dir)
    output_path = joinpath(output_dir, "$output_name.tracks")
    # Persist finished sequences through `sink` — the driver supplies an async
    # channel-backed sink so HDF5 compression overlaps GPU compute; standalone
    # callers get the synchronous default.
    sink = something(sink, _default_sink(output_path, patch_type, scale))
    vid_name = first(splitext(basename(video_path)))

    # Cutoff of the FRAME-LEVEL stages (detect, adapt, both clip filters). These
    # are board-agnostic and stay so: adaptation fits a 1σ frame and does not
    # depend on the sampling extent, which is why `_find_boards_grouped` adapts
    # every board at `DEFAULT_CUTOFF` too (src/pipeline.jl). Matching it here
    # means the frames extraction sees are the frames the tracker matched on.
    # ONLY `canonicalize` / `_sample_grid` consume the board's own cutoff — see
    # `extract_patches`.
    detect_cutoff = DEFAULT_CUTOFF

    # detect + adapt + filter, once per frame and shared by every board in it.
    function detect_frame(img; H_aug = nothing)
        detections = detect_blobs(img; max_detections, device = backend)
        frames = adapt(
            detections, img;
            config = AdaptConfig(; cutoff = detect_cutoff, max_scale_fraction = 0.3), device = backend
        )
        frames = filter_blobs(
            frames, detections;
            cutoff = detect_cutoff, image = img
        )
        # Perspective-augmentation clip filter: `filter_blobs` above checks the
        # core ellipse against the full array, so it's blind to the gray-fill
        # void the warp introduces INSIDE the array. Pull each core ellipse back
        # through `inv(H_aug)` into the original frame and apply the SAME 0.2 clip
        # tolerance against the original rectangle — treating the warp void
        # exactly like the real image edge (a no-op on unwarped frames, and it
        # can't over-filter genuine clipped boards, which are governed by the
        # identical rule). `filter_blobs` already dropped `frames` whose ids we
        # need, so `detections[f.id]` below stays valid after this filter.
        if H_aug !== nothing
            H_inv = inv(H_aug)
            img_rect = Rect(Size2(img))
            frames = filter(frames) do f
                e0 = transfer_ellipse(HomographyMap(HomographyMat(H_inv)), region(f.frame, detect_cutoff))
                e0 === nothing && return false
                (1.0 - coverage(e0, img_rect)) <= 0.2
            end
        end
        return (; detections, frames)
    end

    # The single chosen patch view at the single chosen scale, sampled at ONE
    # board's cutoff. `inner_extent` is the log-polar inner annulus, so it must be
    # the cutoff the board's own anchor patches were rendered at
    # (`save_board_anchors`) — otherwise a track and its anchor cover different
    # annuli and their descriptors are not comparable. `:logpolar`/`:cartesian`
    # sample the *adapted* elliptical frame; `:original` samples the raw
    # *SIFT-detection* disc (`detections[f.id]`), so its scale is relative to the
    # detector's σ rather than the fitted σ. Alongside each patch, `_sample_grid`
    # returns the source sample location per pixel (`sample_xy`) plus its in-frame
    # validity (`inframe`); the per-board board∩in-frame mask is finished in the
    # frame loop below.
    function extract_patches(img, detections, frames; inner_extent, H_aug = nothing)
        # `:original` samples the raw detector discs; the other modes the frames.
        view_frames = patch_type === :original ? [detections[f.id] for f in frames] : frames
        patches = canonicalize(
            view_frames, img; mode = patch_type, patch_size,
            inner_extent, outer_extent = scale, device = backend, supersample
        )
        # Warp inverse in (row,col) space for the in-frame test (see `_sample_grid`).
        Minv = H_aug === nothing ? nothing : inv(_XY_SWAP * H_aug * _XY_SWAP)
        img_size = size(img)
        grids = [
            _sample_grid(
                    vf, patch_type; patch_size, inner_extent,
                    outer_extent = scale, Minv, img_size
                ) for vf in view_frames
        ]
        sample_xy = [g[1] for g in grids]
        inframe = [g[2] for g in grids]
        # (audited) whole-frame strip: no target-unit form for a shape matrix.
        affine = [SMatrix{2, 2, Float64}(ustrip.(shape_matrix(feature_frame(f)))) for f in frames]
        return (; frames, patches, sample_xy, inframe, affine_shapes = affine)
    end

    frames_all = _read_media_frames(video_path; n_frames, stride, max_image_pixels)
    println("  ($(basename(video_path))) read $(length(frames_all)) frame(s)")
    isempty(frames_all) && return output_path

    # Detect + track every board across the sampled frames. Capping detections
    # keeps the GPU descriptor-match GEMM (all boards × all detections) in memory
    # on high-resolution footage.
    board_detections = track_boards(
        board_feature_dict, model, frames_all;
        board_bundles, device = backend,
        config = FindBoardsConfig(
            max_detections = max_detections,
            adapt = AdaptConfig(; cutoff = detect_cutoff)
        )
    )

    # Group quality-gated (board, detection) pairs by frame so each frame is
    # detected ONCE and matched against every board seen in it, rather than
    # re-detecting the frame per board (O(frames) not O(frames×boards)). Patch
    # extraction still runs once per distinct board cutoff in the frame — that
    # is the only stage the cutoff reaches. The `min_inliers` gate is applied
    # here, before any extraction.
    frame_dets = Dict{Int, Vector{Tuple{String, Any}}}()
    for (uid, dets) in board_detections
        (isempty(dets) || !haskey(boards, uid)) && continue
        for det in dets
            if det.n_inliers < min_inliers
                println(
                    "  ($vid_name) frame $(det.frame) [board $(first(uid, 8))]: " *
                        "skipped (inliers=$(det.n_inliers) < $min_inliers; quality=$(round(det.quality; digits = 4)))"
                )
                continue
            end
            push!(get!(() -> Tuple{String, Any}[], frame_dets, det.frame), (uid, det))
        end
    end

    # Per-board accumulators (tracks + confusers), persisting across frames. One
    # entry per board that has ≥1 quality-gated detection. Tracks hold only
    # tracked observations — GT anchor patches live in the board's uid-named
    # anchor sequence, keyed by the same `blob_track_id`.
    detected_uids = unique(uid for dl in values(frame_dets) for (uid, _) in dl)
    # Debug-overlay color per board, fixed by its slot in the (sorted) uid list so
    # a board keeps the same color in every frame's overlay image.
    overlay_colors = board_palette(detected_uids)   # the one palette authority (stable per board id)
    if !isnothing(debug_dir)
        println(
            "  ($vid_name) overlay color on-board adapted " *
                "(survived adapt + filter_blobs): magenta"
        )
        for uid in sort(detected_uids)
            c = overlay_colors[uid]
            println(
                "  ($vid_name) overlay color board $(first(uid, 8)): " *
                    "RGB($(round(c.r; digits = 2)), $(round(c.g; digits = 2)), $(round(c.b; digits = 2)))"
            )
        end
    end
    board_state = Dict{String, Any}()
    for uid in detected_uids
        board = boards[uid]
        N = length(board.canvas_frames)
        tracks = Dict{Int32, Dict{String, Vector}}()
        for i in 1:N
            tracks[Int32(i)] = Dict{String, Vector}(
                "patch" => Any[], "mask" => Any[], "affine_shape" => Any[],
                # Source frame index of each observation — the join key into the
                # sequence's per-frame homography / view-angle table (`frame_geom`
                # below), so the board's world→image geometry is stored once per
                # frame rather than duplicated on every blob.
                "frame_id" => Any[],
            )
        end
        # World-frame GT blobs (unitless) for homography-transfer matching. The
        # per-frame world→image homography maps these into the detected frame.
        world_geom = blobs(board.bundle, :world)   # :world blobs are bare mm by contract
        world_frames = [Detection(d, (; id = j)) for (j, d) in enumerate(world_geom)]
        # World bounding box (center + per-axis half-extent) — normalizes the
        # perspective-augmentation magnitudes to the board's own scale.
        wc = [centroid(g) for g in world_geom]
        wx = [c[1] for c in wc]; wy = [c[2] for c in wc]
        world_extent = (
            (minimum(wx) + maximum(wx)) / 2, (minimum(wy) + maximum(wy)) / 2,
            (maximum(wx) - minimum(wx)) / 2, (maximum(wy) - minimum(wy)) / 2,
        )
        # Board sheet rectangle in the SAME unitless world coords as `world_frames`
        # — the board∩in-frame mask marks samples that back-project inside it.
        world_rect = Rect(board.bundle, :world)
        board_state[uid] = (;
            board, board_cutoff = cutoff(board.bundle), tracks,
            world_frames, world_extent, world_rect,
            # One entry per frame this board is detected in:
            # frame_idx => (world→image homography H_wi, viewing
            # obliquity in radians, blobs matched to this board,
            # this board's GT blobs projecting inside the frame,
            # blobs detected in the frame). Assumes ≤1 detection
            # of a board per frame (track_boards' spatial
            # suppression); a second would overwrite.
            frame_geom = Dict{
                Int, @NamedTuple{
                    H::SMatrix{3, 3, Float64, 9}, view_angle::Float64,
                    n_matched_gt::Int, n_gt_in_image::Int, n_sift_on_board::Int,
                    n_detections_on_board::Int, n_detections_in_frame::Int,
                }
            }(),
            # Confuser (untracked) patches: each unmatched detection
            # contributes one patch + its in-frame mask. Capped per
            # board at save.
            ut_patch = Matrix{Float32}[], ut_mask = Matrix{Float32}[]
        )
    end

    # One extraction per frame, matched against all boards detected in it.
    for frame_idx in sort(collect(keys(frame_dets)))
        dets_here = frame_dets[frame_idx]
        frame_aug = _augment_frame(frames_all[frame_idx]; p_downsample)

        # Perspective augmentation (independent probability, composes with the
        # resolution-loss aug above). Warp the frame with a homography that
        # strengthens the PRIMARY (highest-quality) board's real perspective plus
        # a small random tilt; every board's world→image homography is then
        # composed with the same `H_aug` so correspondences stay valid on the
        # warped frame. Anchored to the primary board because one warp applies to
        # the whole frame — the other boards ride along on a still-valid warp.
        H_aug = SMatrix{3, 3, Float64}(I)
        warped = false
        if p_perspective > 0 && rand() <= p_perspective
            _, pi = findmax(t -> t[2].quality, dets_here)
            prim_uid, prim_det = dets_here[pi]
            H_aug = perspective_augmentation(
                Random.default_rng(),
                # BoardDetection stores only the arm's MODEL; the projection is
                # derived (`_augmentation_homography`), never cached beside it.
                _augmentation_homography(prim_det.model),
                board_state[prim_uid].world_extent;
                strengthen_max = perspective_strengthen_max,
                jitter_max = perspective_jitter_max
            )
            frame_aug = _warp_frame(frame_aug, H_aug)
            warped = true
        end
        # Gaussian blur augmentation — OFF by default (`blur_sigma_max = 0`).
        # σ is drawn uniformly from [0, blur_sigma_max] per frame, so a single
        # knob sets both whether it fires and how hard. Guard on the bound
        # rather than on the draw: `KernelFactors.gaussian(0)` is not the
        # identity kernel, and blurring every frame unconditionally (what this
        # did before) is a training-distribution change, not an augmentation.
        if blur_sigma_max > 0
            sigma = rand() * float(blur_sigma_max)
            frame_aug = imfilter(frame_aug, KernelFactors.gaussian((sigma, sigma)))
        end

        # Detect ONCE for the whole frame — every board in it matches against the
        # same frames. Supply `H_aug` only when a warp fired, so the pullback clip
        # filter runs exactly on the augmented frames (identity warp is a no-op
        # otherwise).
        H_aug_or_nothing = warped ? H_aug : nothing
        detected = detect_frame(frame_aug; H_aug = H_aug_or_nothing)
        # Patches, by contrast, are sampled at the BOARD's cutoff, so they run once
        # per distinct cutoff among the boards in this frame (boards sharing one —
        # and re-visits of the same board — reuse the result).
        patch_cache = Dict{Float64, Any}()

        # One debug overlay per FRAME: every board detected in it is accumulated
        # here and drawn into a single image after the board loop.
        overlays = isnothing(debug_dir) ? nothing : Any[]
        for (uid, det) in dets_here
            st = board_state[uid]
            extracted = get!(patch_cache, Float64(st.board_cutoff)) do
                extract_patches(
                    frame_aug, detected.detections, detected.frames;
                    inner_extent = st.board_cutoff, H_aug = H_aug_or_nothing
                )
            end
            H_wi = H_aug * SMatrix{3, 3, Float64}(to_homogeneous(det.H))  # world → warped image
            H_wi_inv = inv(H_wi)                                  # warped image → world
            # Viewing obliquity of this board in this (possibly warped) frame,
            # evaluated at the board centroid. One record per frame — every
            # observation below just references `frame_idx`.
            view_angle = _view_angle(H_wi, st.world_extent[1], st.world_extent[2])
            csponds = symmetric_iou_match(
                H_wi, st.world_frames, extracted.frames;
                cutoff = st.board_cutoff, iou_threshold,
                frame_i_idx = 0, frame_j_idx = frame_idx,
            )
            # Detections landing ON this board, at both ends of the pipeline:
            # `sift_onboard` over the raw `detect_blobs` output, `onboard` over
            # what survived `adapt` + `filter_blobs`. These are the board-scoped
            # numbers — `n_detections_in_frame` counts the whole frame, so on a
            # distant or blurred board it reads in the thousands while the board
            # itself carries none. The overlay draws exactly the `onboard` mask.
            sift_onboard = _onboard_mask(detected.detections, H_wi, st.world_rect)
            onboard = _onboard_mask(extracted.frames, H_wi, st.world_rect)
            st.frame_geom[frame_idx] = (;
                H = H_wi, view_angle,
                n_matched_gt = length(csponds),
                n_gt_in_image = _n_gt_in_image(
                    st.world_frames, H_wi,
                    st.world_extent[1],
                    st.world_extent[2], size(frame_aug)
                ),
                n_sift_on_board = count(sift_onboard),
                n_detections_on_board = count(onboard),
                n_detections_in_frame = length(extracted.frames),
            )

            # A detection is a confuser for THIS board when it isn't one of the
            # board's correspondences (so a blob matched to another board in the
            # same frame is still a confuser here — preserves per-board semantics).
            tracked = Set{Int}()
            for p in csponds
                t = st.tracks[Int32(p.idx_i)]
                push!(t["patch"], extracted.patches[p.idx_j])
                push!(
                    t["mask"], _board_mask(
                        extracted.sample_xy[p.idx_j],
                        extracted.inframe[p.idx_j],
                        H_wi_inv, st.world_rect
                    )
                )
                push!(t["affine_shape"], extracted.affine_shapes[p.idx_j])
                push!(t["frame_id"], frame_idx)
                push!(tracked, p.idx_j)
            end
            for j in eachindex(extracted.patches)
                j in tracked && continue
                push!(st.ut_patch, extracted.patches[j])
                push!(st.ut_mask, Float32.(extracted.inframe[j]))  # confuser: in-frame only
            end
            println(
                "  ($vid_name) frame $frame_idx [board $(first(uid, 8))]: " *
                    "$(count(sift_onboard)) sift / $(count(onboard)) adapted / " *
                    "$(length(csponds)) matched on board " *
                    "($(length(extracted.frames)) adapted in frame) " *
                    "| quality=$(round(det.quality; digits = 4)) n_inliers=$(det.n_inliers)"
            )

            # Debug overlay contribution: this board's accepted (matched-to-this-
            # board) GT blobs + its projected board quad, in the board's color,
            # plus the detections that fell on it and survived adapt + filter
            # (magenta).
            isnothing(overlays) || push!(
                overlays,
                (;
                    st.world_frames, matched_ids = Set(p.idx_i for p in csponds),
                    cutoff = st.board_cutoff, H_wi, st.world_rect,
                    color = overlay_colors[uid], onboard,
                )
            )
        end

        # All boards of this frame in ONE image, each in its own color, over the
        # on-board adapt + filter survivors in magenta.
        isnothing(overlays) || _save_frame_overlay(
            debug_dir, "$(vid_name)_f$(lpad(frame_idx, 5, '0'))", frame_aug, overlays,
            detected.frames
        )
    end

    # Per-board: keep tracks with ≥ 1 tracked observation, cap confusers, save.
    for (uid, st) in board_state
        filtered = Dict{Int32, Dict{String, Vector}}()
        for (i, t) in st.tracks
            length(t["patch"]) >= 1 && (filtered[blob_track_id(st.board.config_hash, i)] = t)
        end
        if isempty(filtered)
            printstyled(
                "  ($vid_name) board $uid: no tracks; nothing to save\n";
                color = :yellow
            )
            continue
        end
        length(filtered) < 100 &&
            printstyled(
            "  ($vid_name) board $uid: only $(length(filtered)) tracked blobs\n";
            color = :yellow
        )

        # Cap confusers at `confuser_factor × tracked patches`, sampled at random.
        # A non-finite `confuser_factor` (e.g. `Inf`) disables the cap — keep all.
        n_tracked = sum(length(t["patch"]) for t in values(filtered))
        cap = isfinite(confuser_factor) ? round(Int, confuser_factor * n_tracked) : typemax(Int)
        ut_patch, ut_mask = st.ut_patch, st.ut_mask
        if length(ut_patch) > cap
            keep = randperm(length(ut_patch))[1:cap]
            ut_patch = ut_patch[keep]; ut_mask = ut_mask[keep]
        end
        confusers = (; patch = ut_patch, mask = ut_mask)

        # Per-frame detection geometry table for this sequence: one homography +
        # viewing angle per frame the board was detected in, ordered by frame
        # index. Observations join to it via their `frame_id`.
        fids = sort(collect(keys(st.frame_geom)))
        frame_geometry = (;
            frame_ids = Int32.(fids),
            homographies = [st.frame_geom[f].H for f in fids],
            view_angles = Float64[st.frame_geom[f].view_angle for f in fids],
            n_gt_in_image = Int32[st.frame_geom[f].n_gt_in_image for f in fids],
            n_sift_on_board = Int32[st.frame_geom[f].n_sift_on_board for f in fids],
            n_detections_on_board = Int32[st.frame_geom[f].n_detections_on_board for f in fids],
            n_detections_in_frame = Int32[st.frame_geom[f].n_detections_in_frame for f in fids],
        )

        sink("$(vid_name)_$uid", filtered, confusers, false, frame_geometry)
    end

    # Viewing-angle profile for this media item: one record per (board, frame)
    # detection. Handed to the caller's `on_view_stats` sink (the driver
    # aggregates them into a run-wide chart) when one was supplied; charted here
    # only under `debug_dir`, since a per-media chart per media item is a lot of
    # files.
    stats = ViewAngleStat[
        (;
                uid, frame_idx = f, g.view_angle, g.n_matched_gt,
                g.n_gt_in_image, g.n_sift_on_board, g.n_detections_on_board, g.n_detections_in_frame,
            )
            for (uid, st) in board_state for (f, g) in st.frame_geom
    ]
    sort!(stats; by = s -> (s.uid, s.frame_idx))
    isnothing(on_view_stats) || on_view_stats(stats)
    isnothing(debug_dir) || _emit_view_angle_profile(
        joinpath(debug_dir, "$(vid_name)_view_angles.png"), stats;
        title = "($vid_name) avg detections per board vs. viewing angle"
    )
    return output_path
end

"""
    merge_track_files(dest_path, src_paths) -> dest_path

Merge multimodal `.tracks` files into `dest_path` by copying each
`/sequences/<name>` group verbatim (`HDF5.copy_object` — layout + zlib
compression preserved, no decode/re-encode). Sequence names are globally unique
(`<media>_<uid>` tracked, `<uid>` anchor), so this is a plain concatenation;
names already present in `dest_path` are skipped, so it is idempotent (safe to
re-run) and duplicate anchor groups collapse to one.

This is the merge for the multimodal Dict format written by `_save_track_dict`
(patches + masks + affine_shapes + `is_anchor`). It copies groups as-is rather
than decoding and re-encoding, which is what the retired `merge_tracks` did —
that round-tripped through the `Vector{Track}` loader and would have dropped
masks/affine/anchor flags. Used by the multi-GPU driver to combine per-worker
outputs (`examples/training/slurm/driver_build_tracks.jl`).
"""
function merge_track_files(dest_path::AbstractString, src_paths)
    n_copied = 0
    n_skipped = 0
    h5open(dest_path, "cw") do dst
        dsg = haskey(dst, "sequences") ? dst["sequences"]::HDF5.Group :
            HDF5.create_group(dst, "sequences")
        for src_path in src_paths
            abspath(src_path) == abspath(dest_path) && continue
            isfile(src_path) || (@warn "merge: source missing" src_path; continue)
            h5open(src_path, "r") do src
                haskey(src, "sequences") || return
                ssg = src["sequences"]::HDF5.Group
                for name in keys(ssg)
                    if haskey(dsg, name)
                        n_skipped += 1
                        continue
                    end
                    HDF5.copy_object(ssg[name], dsg, name)
                    n_copied += 1
                end
            end
        end
    end
    println(
        "merge_track_files → $dest_path: copied $n_copied sequence(s), " *
            "skipped $n_skipped already present"
    )
    return dest_path
end
