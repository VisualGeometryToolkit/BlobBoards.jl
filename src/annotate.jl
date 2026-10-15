# =============================================================================
# annotate_blobs — GT-labeled canonical patches from board detections
# =============================================================================
#
# Single image:
#   annotate_blobs(board_feature_dict, descriptor_network, img; ...) -> NamedTuple
#
# Video (write per-frame JLD2 files into output_dir):
#   annotate_blobs(board_feature_dict, descriptor_network, video_path;
#                  output_dir, ...)
#
# For each detected board, projects all GT blobs into the image, extracts
# canonical patches, and labels them with (board_uid, blob_id). The result
# is a flat collection of labeled patches suitable for contrastive training.
#
# Storage layout under `output_dir/`:
#   refs/<uid>.jld2                          ← reference patches per board
#   <source_basename>.jld2                   ← image: detections + confusers
#   <video_basename>_frame<NNNN>.jld2        ← video: one file per processed frame
#   <video_basename>.done                    ← marker after all frames done
#   _config.jld2                             ← network_hash, config_hash
#
# Each per-frame/per-image file holds:
#   board_uids::Vector{String}, blob_ids::Vector{Int},
#   patches::Array{Float32,3}  (P×P×N),
#   confusers::Array{Float32,3}(P×P×C)
#
# =============================================================================

using VisualGeometryCore: Rect, region, Ellipse, EllipticFrame, transfer_ellipse,
    Camera, HomographyMap, iou, feature_frame
using CoordinateTransformations: PerspectiveMap
using GeometryBasics: GeometryBasics
using VideoIO: VideoIO
using JLD2: JLD2, jldopen, jldsave
using JSON3: JSON3
using VisualGeometryFeatures: CspondSet
using VisualGeometryCore: bbox

# =============================================================================
# Annotation Config
# =============================================================================

# The patch size is DECLARED by the descriptor network (its ONNX graph fixes the
# input resolution; the Lux backend records it) — there is no second authority.
# Every annotation entry point defaults to it; the `patch_size` keyword remains
# only so a caller can be explicit, and a mismatch fails in `describe`.
_declared_patch_size(network) = patch_size(network)

"""
    config_hash(; config::FindBoardsConfig, cutoff, patch_size, outer_extent, min_iou)

Hash the annotation-pipeline *extraction* settings into a string stored in
`_config.jld2` so the build_annotations driver can detect config drift across
runs.

`config.geometry` is intentionally EXCLUDED: it carries the per-source camera
(`PoseGeometry`) and the homography-vs-P3P choice, both of which legitimately
vary across sources written into the same output directory. EVERY other field
participates: the hash is taken over `config_dict(config)` minus the geometry
table — the same fieldnames-derived serializer the run sidecars use — so a
field added to `FindBoardsConfig` is hashed without anyone remembering to
list it. (Until 2026-09-07 this was a hand-written tuple that omitted
`verify_similarity`, `min_appearance_support`, `ransac_seed` and `refine`:
the board-acceptance gates decide which boards get annotated, so changing
them changed the output while the directory still read as consistent.)

The encoding is the canonical config JSON (sorted keys, Ryu floats), so the
hash is stable across Julia/JSON3 versions — a `string(::Tuple)` was not.

Changing any hashed setting invalidates the existing per-frame JLD2 files in
the output directory (they must be rebuilt). The 2026-09-07 change itself
does that once for every existing annotation directory.
"""
function config_hash(; config::FindBoardsConfig, cutoff, patch_size, outer_extent, min_iou)
    fb = Dict{String, Any}(k => v for (k, v) in config_dict(config) if k != "geometry")
    tree = Dict{String, Any}(
        "find_boards" => fb,
        "cutoff" => Float64(cutoff), "patch_size" => Int(patch_size),
        "outer_extent" => Float64(outer_extent), "min_iou" => Float64(min_iou),
    )
    return bytes2hex(sha256(codeunits(canonical_config_json(tree))))
end

# =============================================================================
# Single-image annotation
# =============================================================================

"""
    annotate_blobs(board_feature_dict, descriptor_network, img;
                   board_bundles, device = default_device(), kwargs...) -> NamedTuple

Detect boards in `img`, project GT blobs into the image, and extract
canonical patches labeled by `(board_uid, blob_id)`.

Also extracts **confuser patches**: image detections that were not matched
to any board blob. These are canonicalized using the same pipeline and
returned separately for use as negatives during contrastive training.

Returns `(; board_uids, blob_ids, patches, confuser_patches)`.
"""
function annotate_blobs(
        # `board_feature_dict::AbstractDict` makes a call in the retired order
        # (`img, board_feature_dict, descriptor_network`) a MethodError.
        board_feature_dict::AbstractDict,
        descriptor_network,
        img::AbstractMatrix;
        board_bundles,
        config::FindBoardsConfig = FindBoardsConfig(),
        cutoff = DEFAULT_CUTOFF,
        patch_size::Int = _declared_patch_size(descriptor_network),
        outer_extent::Float64 = DEFAULT_OUTER_EXTENT,
        min_iou::Float64 = 0.5,
        device = default_device(),
        kwargs...,
    )
    # The SAME feature stages as find_boards(…, img): polarity from the boards,
    # scale correction and the gallery's shared extent all come from the
    # bundles (see `prepare_image_features`). Camera (if any) is read from
    # config.geometry. `cutoff` here is the GT-patch extent, not the query one.
    (; detect_img, adapt_img, image_features) = prepare_image_features(
        descriptor_network, img;
        board_bundles, uids = keys(board_feature_dict), config, device
    )

    # No detections → no boards, return empty
    if isempty(image_features)
        return (;
            board_uids = String[], blob_ids = Int[],
            patches = Matrix{Gray{Float32}}[], confuser_patches = Matrix{Gray{Float32}}[],
        )
    end

    boards = find_boards(
        board_feature_dict, image_features;
        board_bundles, config, device,
        kwargs...
    )

    # The radial arm is not wired into annotation (GT projection below needs
    # a pinhole camera or a plain homography) — same refusal as track_boards.
    config.geometry isa RadialHomographyGeometry && throw(
        ArgumentError(
            "annotate_blobs: the radial arm is not wired into annotation; run find_boards per frame"
        )
    )

    # Project GT blobs, match detections by IoU, extract patches
    img_rect = Rect(Size2(adapt_img))
    cam = _estimation_camera(config.geometry)

    all_uids = String[]
    all_blob_ids = Int[]
    all_patches = Matrix{Gray{Float32}}[]
    matched_ids = Set{Int}()   # .id of image features assigned to a GT blob

    for (uid, model, quality, scored) in boards
        bundle = board_bundles[uid]
        board_cutoff = BlobBoards.cutoff(bundle)

        # Collect matched image feature ids from RANSAC correspondences
        for sc in scored
            push!(matched_ids, last(sc).id)
        end

        # world → image through the ONE spelling (`_model_to_homography`,
        # pipeline.jl). The radial arm hands back a guided TRANSFER, not a map,
        # and annotation has no wiring for it — refuse before dispatch.
        model isa VisualGeometryCore.RadialHomography &&
            throw(ArgumentError("annotate_blobs: the radial arm is not wired into annotation"))
        H = _model_to_homography(model, cam)

        # Project each GT blob and collect visible ones
        world_discs = blobs(bundle, :world)
        gt_frames = EllipticFrame[]   # GT backprojected frames (unit sigma)
        gt_ellipses = Ellipse[]       # cutoff-scaled ellipses for IoU
        valid_ids = Int[]             # blob index into world_discs

        for (j, disc) in enumerate(world_discs)
            ell_vis = transfer_ellipse(H, region(disc, board_cutoff))
            ell_vis === nothing && continue
            ell_vis in img_rect || continue
            ell = transfer_ellipse(H, region(disc, 1.0))
            ell === nothing && continue
            push!(gt_frames, EllipticFrame(ell))
            push!(gt_ellipses, ell_vis)  # cutoff-scaled for IoU
            push!(valid_ids, j)
        end

        isempty(gt_frames) && continue

        # NOTE: we intentionally do NOT emit canonical patches from
        # `gt_frames` (the GT-blobs back-projected through the recovered
        # board pose). The descriptor is matched at inference time against
        # *adapt-refined detections* — never against GT projections — so
        # training on GT-back-projected ellipses would optimize for a
        # blob-shape distribution the matcher never sees. Instead, the
        # `gt_frames` and `gt_ellipses` arrays below are used ONLY as an
        # R-tree index to label which detections fall onto which GT blob
        # (via IoU). Detection patches inherit the GT blob id as their
        # contrastive label, so every emitted patch comes from a real
        # detector output and matches the test-time distribution.

        # --- Detection-augmented patches: match detections to GT by IoU ---
        # Build R-tree of GT ellipses (already at cutoff scale)
        gt_rtree = _new_rtree()
        for (k, ell) in enumerate(gt_ellipses)
            rect = bbox(ell)
            SpatialIndexing.insert!(gt_rtree, rect, k, nothing)
        end

        # For each adapted detection, find overlapping GT blob
        det_frames = EllipticFrame[]
        det_blob_ids = Int[]
        for feat in image_features
            det_ell = Ellipse(region(feature_frame(feat), board_cutoff))
            det_rect = bbox(det_ell)

            best_iou = 0.0
            best_gt = 0
            foreach_intersecting(gt_rtree, det_rect) do elem
                k = SpatialIndexing.id(elem)
                v = iou(det_ell, gt_ellipses[k])
                if v > best_iou
                    best_iou = v
                    best_gt = k
                end
                false  # continue searching
            end

            if best_iou > min_iou && best_gt > 0
                push!(det_frames, feature_frame(feat))
                push!(det_blob_ids, valid_ids[best_gt])
                push!(matched_ids, feat.id)
            end
        end

        # Canonicalize detection-augmented patches
        if !isempty(det_frames)
            det_patches = canonicalize(
                det_frames, adapt_img;
                patch_size, inner_extent = board_cutoff, outer_extent, device
            )
            append!(all_uids, fill(uid, length(det_blob_ids)))
            append!(all_blob_ids, det_blob_ids)
            append!(all_patches, det_patches)
        end
    end

    # --- Confusers: unmatched image detections ---
    confuser_frames = [
        feature_frame(f) for f in image_features
            if f.id ∉ matched_ids
    ]
    confuser_patches = if !isempty(confuser_frames)
        canonicalize(
            confuser_frames, adapt_img;
            patch_size, inner_extent = cutoff, outer_extent, device
        )
    else
        Matrix{Gray{Float32}}[]
    end

    return (;
        board_uids = all_uids, blob_ids = all_blob_ids,
        patches = all_patches, confuser_patches,
    )
end

# =============================================================================
# Reference patch extraction — canonical patches from board PDF
# =============================================================================

"""
    _extract_reference_patches(board_path; patch_size, outer_extent=DEFAULT_OUTER_EXTENT) -> NamedTuple

Extract canonical patches from a board's rendered PDF image.
Returns `(; board_uids, blob_ids, patches)` for all blobs on the board.

`patches` is a 4D `Array{Float32, 4}` of shape `(patch_size, patch_size, 2, N)`:
- channel 1 = intensity (canonicalized PDF rendering)
- channel 2 = validity mask: 1 where the canonical pixel back-projects onto
  the printed board area, 0 where it falls on the 0.5 background fill
  (canvas margin OR outside the canvas image entirely — both manifest as
  ~0.5 in the canonicalized output).

The descriptor network reads both channels and learns to discount the
masked-out region when matching against camera detection patches.
"""
function _extract_reference_patches(
        board_path::AbstractString;
        patch_size::Int, outer_extent = DEFAULT_OUTER_EXTENT, device = default_device()
    )
    board = load_board(board_path)
    bundle = board.bundle
    uid = board.uid
    board_cutoff = cutoff(bundle)
    image = _load_image(board.image_path, bundle; density = board.density)
    mask_image = _board_mask_image(bundle, image, board.density)

    intensity_patches = canonicalize(
        board.canvas_frames, image;
        patch_size, inner_extent = board_cutoff, outer_extent, device
    )
    mask_patches = canonicalize(
        board.canvas_frames, mask_image;
        patch_size, inner_extent = board_cutoff, outer_extent, device
    )
    n = length(intensity_patches)
    patches = _stack_2ch(intensity_patches, mask_patches, patch_size)
    return (; board_uids = fill(uid, n), blob_ids = collect(1:n), patches)
end

"""
Build a binary mask image the same shape as the canvas image: 1 where the
canvas pixel falls inside the printed board rectangle, 0 elsewhere (canvas
margin, if any).

NB two mask-channel semantics exist ON PURPOSE (2026-09 audit): this
annotation path stamps the board rect in canvas space and canonicalizes it
(soft edge, 0.5-fill remap in _stack_2ch); the .tracks builder's
_board_mask (training.jl) back-projects per sample (hard 0/1, horizon
guard). They feed different training pipelines; unifying them changes the
network's inputs and is a training-lineage decision, not a cleanup.

Canonicalizing this alongside the intensity gives a geometrically correct
validity mask per blob — bilinear interpolation produces a soft falloff at
the board edge, and `canonicalize`'s 0.5 out-of-image extrapolation fills
the rest (`_stack_2ch` remaps that 0.5 to 0).
"""
function _board_mask_image(bundle, image, density)
    H, W = size(image)
    board_rect_mm = GeometryBasics.Rect(bundle, :board)
    canvas_rect_mm = GeometryBasics.Rect(bundle, :canvas)
    pmm = Float64(px_per_mm(density))

    dx_mm = ustrip(mm, board_rect_mm.origin[1] - canvas_rect_mm.origin[1])
    dy_mm = ustrip(mm, board_rect_mm.origin[2] - canvas_rect_mm.origin[2])
    bw_mm = ustrip(mm, board_rect_mm.widths[1])
    bh_mm = ustrip(mm, board_rect_mm.widths[2])

    x1 = clamp(round(Int, dx_mm * pmm) + 1, 1, W)
    y1 = clamp(round(Int, dy_mm * pmm) + 1, 1, H)
    x2 = clamp(round(Int, (dx_mm + bw_mm) * pmm), 1, W)
    y2 = clamp(round(Int, (dy_mm + bh_mm) * pmm), 1, H)

    mask = zeros(Gray{Float32}, H, W)
    @inbounds mask[y1:y2, x1:x2] .= Gray{Float32}(1.0f0)
    return mask
end

"""
Stack vectors of P×P canonicalized intensity and mask patches into a
(P, P, 2, N) array. The mask channel is remapped so that `canonicalize`'s
0.5 out-of-image extrapolation fill becomes 0 (invalid), while in-canvas
mask values (0 in canvas margin, 1 inside the printed board, smooth at
the board edge) are pushed into the upper half of the output range.

  m_out = max(0, 2 * (m_canonical - 0.5))

Mapping:
  m_canonical = 0.0  →  m_out = 0    (definitely outside board, in canvas)
  m_canonical = 0.5  →  m_out = 0    (canonicalize extrapolation fill,
                                       or transition halfway through board edge)
  m_canonical = 1.0  →  m_out = 1    (definitely inside board)
"""
function _stack_2ch(intensity_patches, mask_patches, P::Int)
    N = length(intensity_patches)
    @assert length(mask_patches) == N
    A = Array{Float32}(undef, P, P, 2, N)
    @inbounds for i in 1:N
        ip = intensity_patches[i]
        mp = mask_patches[i]
        for c in 1:P, r in 1:P
            A[r, c, 1, i] = Float32(ip[r, c])
            m = Float32(mp[r, c])
            A[r, c, 2, i] = max(0.0f0, 2.0f0 * (m - 0.5f0))
        end
    end
    return A
end

# =============================================================================
# Per-frame JLD2 I/O
# =============================================================================
#
# Each call to `annotate_frame!` writes ONE JLD2 file under output_dir. No
# in-memory buffering, no shared appender. Reference patches are written to
# refs/<uid>.jld2 the first time their UID is detected.
#
# Layout under output_dir:
#   refs/<uid>.jld2                       — reference patches per board
#   <source_basename>.jld2                — image: detections + confusers + meta
#   <video_basename>_frame<NNNNNN>.jld2   — video: one file per processed frame
#   <video_basename>.done                 — marker after all frames done
#   _config.jld2                          — network_hash, config_hash

const _CONFIG_FILE = "_config.jld2"
const _REFS_DIR = "refs"

_refs_path(output_dir, uid) = joinpath(output_dir, _REFS_DIR, "$(uid).jld2")

# uid → board JSON path for a board list — the two annotation drivers used to
# spell this loop separately.
function _board_uid_map(board_paths)
    m = Dict{String, String}()
    for bp in board_paths
        _, _, ch = load_bundle(bp)
        m[uid_from_hash(ch)] = bp
    end
    return m
end
"""
    image_path(output_dir, source_basename) -> String

The annotation file [`annotate_blobs`](@ref)'s driver writes for one source
image: `<output_dir>/<source_basename>.jld2`.
"""
image_path(output_dir, source_basename) =
    joinpath(output_dir, "$(source_basename).jld2")
_frame_path(output_dir, source_basename, frame_idx::Integer) =
    joinpath(output_dir, "$(source_basename)_frame$(lpad(frame_idx, 6, '0')).jld2")
_video_done_marker(output_dir, source_basename) =
    joinpath(output_dir, "$(source_basename).done")

"""Stack a vector of P×P matrices into a P×P×N Float32 array."""
function _stack_patches(patches, P::Int)
    N = length(patches)
    A = Array{Float32}(undef, P, P, N)
    for (i, patch) in enumerate(patches)
        A[:, :, i] .= Float32.(patch)
    end
    return A
end

"""
Write one frame's detections + confusers + meta to a per-frame JLD2 file.

`meta` is a NamedTuple with at least `source`, `image_size`, `wall_seconds`,
and `n_unique_boards`. Stored as fields of the JLD2 for easy bulk analysis.
"""
function _write_frame_jld2(path, result, meta, patch_size::Int)
    P = patch_size
    patches = _stack_patches(result.patches, P)
    confusers = _stack_patches(result.confuser_patches, P)
    mkpath(dirname(path))
    return jldsave(
        path;
        board_uids = collect(String, result.board_uids),
        blob_ids = collect(Int, result.blob_ids),
        patches = patches,
        confusers = confusers,
        meta = meta
    )
end

"""Write reference patches for a single board UID (one file per UID).

`patches` is an `Array{Float32, 4}` of shape `(P, P, 2, N)` — channel 1
intensity, channel 2 validity mask (see `_extract_reference_patches`).
"""
function _write_refs_jld2(
        path, uid::AbstractString, blob_ids,
        patches::AbstractArray{Float32, 4}, patch_size::Int
    )
    mkpath(dirname(path))
    return jldsave(
        path;
        uid = String(uid),
        blob_ids = collect(Int, blob_ids),
        patches = patches
    )
end

"""Set of board UIDs that already have a refs/<uid>.jld2 file."""
function existing_ref_uids(output_dir::AbstractString)
    refs_dir = joinpath(output_dir, _REFS_DIR)
    isdir(refs_dir) || return Set{String}()
    return Set{String}(
        splitext(basename(p))[1]
            for p in readdir(refs_dir; join = true) if endswith(p, ".jld2")
    )
end

# =============================================================================
# Per-frame core — detect → write refs (once per UID) → write frame file
# =============================================================================

"""
Run `annotate_blobs` on one image (or one video frame), insert any
not-yet-written reference patches into refs/, and write the frame's
detections + confusers + meta to its own per-frame JLD2 file.

Returns the meta NamedTuple. Skips writing the frame file when there are
zero detections AND zero confusers.
"""
function annotate_frame!(
        # The one mutated argument first (`refs_inserted` gains the uids whose
        # refs this call wrote), then the output directory the files land in
        # (written, not mutated — a path is not a Julia object), the gallery
        # tools — the descriptor gallery, the network and the board-set-wide
        # uid → bundle path table the refs are read from — then this frame's
        # data. The retired `(output_dir, refs_inserted, …)` order is a
        # MethodError: a path is not a `Set`.
        refs_inserted::Set{String}, output_dir::AbstractString,
        board_feature_dict::AbstractDict, descriptor_network,
        uid_to_path::Dict{String, String},
        frame_path::AbstractString, source_id::AbstractString, img::AbstractMatrix;
        board_bundles,
        config::FindBoardsConfig,
        cutoff, patch_size::Int, outer_extent, min_iou,
        device, kwargs...
    )

    t0 = time()
    result = annotate_blobs(
        board_feature_dict, descriptor_network, img;
        board_bundles, config, cutoff, patch_size, outer_extent,
        min_iou, device, kwargs...
    )
    elapsed = time() - t0

    n_obs = length(result.blob_ids)
    n_conf = length(result.confuser_patches)
    unique_uids = unique(result.board_uids)

    # Write any missing reference files for the UIDs we just detected.
    for uid in unique_uids
        uid in refs_inserted && continue
        haskey(uid_to_path, uid) || continue
        ref = _extract_reference_patches(
            uid_to_path[uid];
            patch_size, outer_extent, device
        )
        _write_refs_jld2(
            _refs_path(output_dir, uid),
            uid, ref.blob_ids, ref.patches, patch_size
        )
        push!(refs_inserted, uid)
        @info "Wrote $(length(ref.blob_ids)) reference patches for board $uid"
    end

    meta = (;
        source = source_id,
        image_size = size(img),
        wall_seconds = elapsed,
        n_obs = n_obs,
        n_confusers = n_conf,
        n_unique_boards = length(unique_uids),
        boards = collect(String, unique_uids),
        timestamp = time(),
    )

    if n_obs > 0 || n_conf > 0
        _write_frame_jld2(frame_path, result, meta, patch_size)
    end

    return meta
end

"""
    annotate_blobs(board_feature_dict, descriptor_network, video_path;
                   output_dir, board_bundles, board_paths, kwargs...)

Process video frames, detect boards, and write per-frame JLD2 files into
`output_dir`. Reference patches from the board PDF are written to
`output_dir/refs/<uid>.jld2` on first detection.

Writes `output_dir/<video_basename>.done` after all frames are processed.
Resumable: existing per-frame files are skipped.
"""
function annotate_blobs(
        board_feature_dict::AbstractDict,
        descriptor_network,
        video_path::AbstractString;
        output_dir::AbstractString,
        board_bundles,
        board_paths::AbstractVector{<:AbstractString},
        config::FindBoardsConfig = FindBoardsConfig(),
        cutoff = DEFAULT_CUTOFF,
        patch_size::Int = _declared_patch_size(descriptor_network),
        outer_extent::Float64 = DEFAULT_OUTER_EXTENT,
        min_iou::Float64 = 0.5,
        device = default_device(),
        stride::Int = 1,
        n_frames::Union{Nothing, Int} = nothing,
        kwargs...,
    )
    mkpath(output_dir)

    uid_to_path = _board_uid_map(board_paths)

    vid = VideoIO.openvideo(video_path)
    frame_count = 0
    processed = 0
    total_obs = 0
    total_confusers = 0
    refs_inserted = existing_ref_uids(output_dir)
    src_base = basename(video_path)

    isnothing(n_frames) || (stride = _frame_stride(video_path, n_frames))

    @info "Annotating: $video_path (stride=$stride)"

    for raw_frame in vid
        frame_count += 1
        (frame_count % stride == 0) || continue
        processed += 1

        fp = _frame_path(output_dir, src_base, frame_count)
        if isfile(fp)
            continue   # resumed; this frame already on disk
        end

        img = Gray{Float32}.(raw_frame)
        meta = annotate_frame!(
            refs_inserted, output_dir,
            board_feature_dict, descriptor_network, uid_to_path,
            fp, "$(src_base)#$(frame_count)", img;
            board_bundles, config, cutoff, patch_size, outer_extent,
            min_iou, device, kwargs...
        )

        total_obs += meta.n_obs
        total_confusers += meta.n_confusers
        append_log(output_dir, meta)
        @info "Frame $frame_count ($processed): $(meta.n_obs) blobs, $(meta.n_confusers) confusers, total=$total_obs obs + $total_confusers conf"
    end

    close(vid)
    touch(_video_done_marker(output_dir, src_base))
    @info "Done: $processed frames, $total_obs observations + $total_confusers confusers → $output_dir"
    return (;
        n_frames = processed, n_observations = total_obs,
        n_confusers = total_confusers, output_dir,
    )
end

# =============================================================================
# Provenance / idempotence — _config.jld2 + per-source file existence
# =============================================================================

"""
Verify the model + config hashes in `output_dir/_config.jld2` match the
current ones. If no _config.jld2 yet, write it. Mismatch → error.
"""
function check_annotation_hashes(
        output_dir::AbstractString,
        m_hash::AbstractString,
        c_hash::AbstractString
    )
    cfg_path = joinpath(output_dir, _CONFIG_FILE)
    return if isfile(cfg_path)
        cfg = JLD2.load(cfg_path)
        for (key, current, label) in [
                ("network_hash", m_hash, "Model"),
                ("config_hash", c_hash, "Config"),
            ]
            if haskey(cfg, key) && cfg[key] != current
                stored = cfg[key]
                error(
                    """$label mismatch in $cfg_path
                      Existing: $(stored[1:min(16, length(stored))])...
                      Current:  $(current[1:min(16, length(current))])...
                    Delete $output_dir and rebuild, or use a different output dir."""
                )
            end
        end
        if !haskey(cfg, "network_hash") || !haskey(cfg, "config_hash")
            mkpath(output_dir)
            jldsave(cfg_path; network_hash = m_hash, config_hash = c_hash)
        end
    else
        mkpath(output_dir)
        jldsave(cfg_path; network_hash = m_hash, config_hash = c_hash)
    end
end

"""
A source is "processed" if its primary output exists.
- Image: `<basename>.jld2`
- Video: `<basename>.done` marker
"""
function is_source_processed(output_dir::AbstractString, source_path::AbstractString)
    return if is_video(source_path)
        isfile(_video_done_marker(output_dir, basename(source_path)))
    else
        isfile(image_path(output_dir, basename(source_path)))
    end
end

# =============================================================================
# Sidecar JSONL log — one line per processed source for fast bulk analysis
# =============================================================================

const _LOG_FILE = "_log.jsonl"

"""
Append one line of structured stats for a processed source to
`output_dir/_log.jsonl`. Designed to be safe under multi-process append:
each line is < PIPE_BUF, so concurrent writes interleave at line boundaries.
"""
append_log(output_dir::AbstractString, meta::NamedTuple) =
    VisualGeometryIo.append_jsonl(joinpath(output_dir, _LOG_FILE), meta)

"""
    summarize_annotations(output_dir) -> Vector{NamedTuple}

Read every per-source meta record from `output_dir/_log.jsonl` and return
them as a vector of NamedTuples. Useful for after-run analysis (which
sources detected no boards, slowest sources, board distribution, etc.).

Falls back to scanning per-file metas if the JSONL log is missing.
"""
function summarize_annotations(output_dir::AbstractString)
    log = joinpath(output_dir, _LOG_FILE)
    if isfile(log)
        return [JSON3.read(line, NamedTuple) for line in eachline(log) if !isempty(line)]
    end
    metas = NamedTuple[]
    for p in readdir(output_dir; join = true)
        endswith(p, ".jld2") || continue
        basename(p) == _CONFIG_FILE && continue
        startswith(basename(p), '_') && continue
        d = JLD2.load(p)
        haskey(d, "meta") && push!(metas, d["meta"])
    end
    return metas
end

# =============================================================================
# Loaders — directory of per-frame JLD2 files
# =============================================================================

"""
    load_annotations(path) -> NamedTuple

Load annotated blob observations. `path` may be:
- A directory written by `annotate_blobs` / `build_annotations` (per-frame
  JLD2 files + `refs/<uid>.jld2`), in which case files are concatenated.
- A single JLD2 split file produced by `split_training_data`.

Returns `(; board_uids, blob_ids, patches, confusers)` — flat shape:
- `patches`   is a P×P×2×N array (refs + per-frame detections concatenated, when reading a dir).
  Channel 1 = intensity. Channel 2 = validity mask (real on the ref side, all-1s on the
  detection side — see `_extract_reference_patches`).
- `confusers` is a P×P×2×C array (detection patches, channel 2 = all-1s).

Group by `(board_uid, blob_id)` to reconstruct CspondSets — `cs[1]` for each
key is the reference patch (always loaded first).
"""
function load_annotations(path::AbstractString)
    if isfile(path) && endswith(path, ".jld2")
        d = JLD2.load(path)
        P = size(d["patches"], 1)
        patches = _promote_to_2ch(d["patches"], P)
        confusers = haskey(d, "confusers") ?
            _promote_to_2ch(d["confusers"], P) :
            Array{Float32}(undef, P, P, 2, 0)
        return (;
            board_uids = String.(d["board_uid"]),
            blob_ids = Int.(d["blob_id"]),
            patches, confusers,
        )
    end
    isdir(path) || error("not a directory or .jld2 file: $path")
    output_dir = path

    refs_dir = joinpath(output_dir, _REFS_DIR)
    ref_files = isdir(refs_dir) ?
        filter(p -> endswith(p, ".jld2"), readdir(refs_dir; join = true)) : String[]
    frame_files = filter(
        p -> endswith(p, ".jld2") &&
            basename(p) != _CONFIG_FILE &&
            !startswith(basename(p), '_'),
        readdir(output_dir; join = true)
    )

    isempty(ref_files) && isempty(frame_files) &&
        error("no annotations found under $output_dir")

    sample = isempty(ref_files) ? first(frame_files) : first(ref_files)
    P = jldopen(sample, "r") do f
        size(f["patches"], 1)
    end

    board_uids = String[]
    blob_ids = Int[]
    patch_chunks = Array{Float32, 4}[]
    confuser_chunks = Array{Float32, 4}[]

    # Refs first — `cs[1]` for each key becomes the reference patch.
    # Refs are stored 4-D (intensity + mask). Detections are stored 3-D
    # (intensity only); we promote them to 4-D with an all-1s mask here,
    # so the downstream training loop sees a uniform (P, P, 2, N) tensor.
    for path in ref_files
        d = JLD2.load(path)
        n = size(d["patches"])[end]
        n == 0 && continue
        append!(board_uids, fill(d["uid"], n))
        append!(blob_ids, d["blob_ids"])
        push!(patch_chunks, _promote_to_2ch(d["patches"], P))
    end

    for path in frame_files
        d = JLD2.load(path)
        n = size(d["patches"])[end]
        if n > 0
            append!(board_uids, d["board_uids"])
            append!(blob_ids, d["blob_ids"])
            push!(patch_chunks, _promote_to_2ch(d["patches"], P))
        end
        c = size(d["confusers"])[end]
        c > 0 && push!(confuser_chunks, _promote_to_2ch(d["confusers"], P))
    end

    patches = isempty(patch_chunks) ?
        Array{Float32}(undef, P, P, 2, 0) : cat(patch_chunks...; dims = 4)
    confusers = isempty(confuser_chunks) ?
        Array{Float32}(undef, P, P, 2, 0) : cat(confuser_chunks...; dims = 4)

    return (; board_uids, blob_ids, patches, confusers)
end

# Promote a 3-D (P, P, N) intensity tensor to 4-D (P, P, 2, N) by stacking
# an all-1s mask channel. Forward 4-D tensors unchanged.
_promote_to_2ch(A::AbstractArray{T, 4}, ::Int) where {T} = A
function _promote_to_2ch(A::AbstractArray{T, 3}, P::Int) where {T}
    N = size(A, 3)
    out = Array{Float32}(undef, P, P, 2, N)
    out[:, :, 1, :] .= A
    out[:, :, 2, :] .= 1.0f0
    return out
end

"""
    load_cspond_sets(path) -> Dict{Tuple{String,Int}, CspondSet}

Load annotations from a directory or split JLD2 file, then group into
CspondSets keyed by `(board_uid, blob_id)`. Each CspondSet contains all
observed patches of the same physical blob. `cs[1]` is the reference patch
(from board PDF), `cs[2:end]` are camera observations.
"""
function load_cspond_sets(path::AbstractString)
    data = load_annotations(path)
    P = size(data.patches, 1)
    N = size(data.patches, 4)   # (P, P, 2, N)

    sets = Dict{Tuple{String, Int}, CspondSet{Vector{Matrix{Gray{Float32}}}}}()
    for i in 1:N
        key = (data.board_uids[i], data.blob_ids[i])
        # CspondSets only carry the intensity channel — used for plotting,
        # not feature extraction. Mask channel is dropped on this path.
        patch = Gray{Float32}.(@view data.patches[:, :, 1, i])
        if haskey(sets, key)
            push!(sets[key], patch)
        else
            sets[key] = CspondSet([patch])
        end
    end
    return sets
end

# =============================================================================
# Build annotations — multi-source annotation into a per-frame JLD2 directory
# =============================================================================

# `is_video` / `_VIDEO_EXTS` live in media.jl — one authority for what
# counts as a video, shared by build_annotations and the media walkers.

"""
    build_annotations(board_feature_dict, descriptor_network, sources;
        board_bundles, board_paths, output_dir="truth/", kwargs...)

Annotate images and/or videos into a directory of per-frame JLD2 files.

`sources` is a vector of input paths. Each element is either:
- A video file (`.mp4`, `.avi`, `.mov`, `.mkv`) — frames extracted with `stride`
- An image file — annotated as a single frame

Resumable: per-source output existence is the "processed" marker; previously
processed sources are skipped. Per-source structured stats are appended to
`output_dir/_log.jsonl` (use `summarize_annotations(output_dir)` to read back).
"""
function build_annotations(
        board_feature_dict::AbstractDict,
        descriptor_network,
        sources::AbstractVector{<:AbstractString};
        board_bundles,
        board_paths::AbstractVector{<:AbstractString},
        output_dir::AbstractString = "truth/",
        config::FindBoardsConfig = FindBoardsConfig(),
        cutoff = DEFAULT_CUTOFF,
        patch_size::Int = _declared_patch_size(descriptor_network),
        outer_extent::Float64 = DEFAULT_OUTER_EXTENT,
        min_iou::Float64 = 0.5,
        device = default_device(),
        stride::Int = 1,
        n_frames::Union{Nothing, Int} = nothing,
        kwargs...,
    )
    mkpath(output_dir)

    # Model + config hash check — error if directory was built differently.
    m_hash = network_hash(descriptor_network)
    c_hash = config_hash(; config, cutoff, patch_size, outer_extent, min_iou)
    check_annotation_hashes(output_dir, m_hash, c_hash)

    uid_to_path = _board_uid_map(board_paths)

    refs_inserted = existing_ref_uids(output_dir)
    frame_kw = (;
        board_bundles, config, cutoff, patch_size,
        outer_extent, min_iou, device, kwargs...,
    )

    n_skipped = 0
    n_failed = 0
    for (si, source_path) in enumerate(sources)
        if is_source_processed(output_dir, source_path)
            @info "[$si/$(length(sources))] Skipping (already processed): $(basename(source_path))"
            n_skipped += 1
            continue
        end

        @info "[$si/$(length(sources))] Processing: $(basename(source_path))"

        try
            if is_video(source_path)
                annotate_blobs(
                    board_feature_dict, descriptor_network, source_path;
                    output_dir, board_bundles, board_paths, config, cutoff,
                    patch_size, outer_extent, min_iou, device, stride, n_frames,
                    kwargs...
                )
                # Refresh refs set after the video may have written new refs.
                union!(refs_inserted, existing_ref_uids(output_dir))
            else
                img = Gray{Float32}.(FileIO.load(source_path))
                fp = image_path(output_dir, basename(source_path))
                meta = annotate_frame!(
                    refs_inserted, output_dir,
                    board_feature_dict, descriptor_network, uid_to_path,
                    fp, basename(source_path), img; frame_kw...
                )
                append_log(output_dir, meta)
                @info "  → $(meta.n_obs) blobs, $(meta.n_confusers) confusers ($(round(meta.wall_seconds; digits = 2))s)"
            end
        catch e
            n_failed += 1
            @warn "[$si/$(length(sources))] Failed; skipping" source = basename(source_path) exception = (e, catch_backtrace())
        end

        # Reclaim CUDA memory between sources: high-confuser images
        # (e.g. iphone_till) leak ~500 MB/frame of GPU tensors otherwise.
        # Runs even on failure path so a partial allocation doesn't poison the next iter.
        GC.gc(false)
        reclaim_pool!(device)                      # no-op on CPU; CUDA.reclaim() via BlobBoardsCUDAExt
    end

    n_skipped > 0 && @info "Skipped $n_skipped already-processed sources"
    n_failed > 0 && @warn "Failed $n_failed source(s) — see prior warnings"

    # Final summary
    metas = summarize_annotations(output_dir)
    n_files = length(metas)
    total_obs = sum(m -> Int(m.n_obs), metas; init = 0)
    total_conf = sum(m -> Int(m.n_confusers), metas; init = 0)
    n_unique_uids = length(existing_ref_uids(output_dir))
    @info "Annotations: $total_obs detection patches across $n_unique_uids boards, $total_conf confusers from $n_files frames → $output_dir"
    return output_dir
end

# =============================================================================
# Split annotations — train/val split by board identity
# =============================================================================

"""
    split_training_data(source_path; output_dir="data/",
        validation_fraction, test_fraction, seed,
        val_boards=nothing, test_boards=nothing) -> NamedTuple

Split an annotation HDF5 into train/val/test files by **board identity**.
All observations of a given board go entirely to one split — no leakage.

# Split strategy
Boards are assigned greedily (largest first, ties broken by a shuffle of the
SORTED board UIDs with `MersenneTwister(seed)`) to test, then val, until each
split reaches its target **patch fraction** (`test_fraction`, then
`validation_fraction`). Remaining boards go to train. Confusers go to
training data. The fractions and the seed are the caller's — there are no
defaults: the split of a training corpus is recorded where it is chosen.

Explicit board lists (`val_boards`, `test_boards`) override the automatic
split for that partition.

# Returns
`(; train_path, val_path, test_path, val_boards, test_boards, stats)`
"""
function split_training_data(
        source_dir::AbstractString;
        output_dir::AbstractString = "data/",
        validation_fraction::Float64,
        test_fraction::Float64,
        seed::Int,
        val_boards::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
        test_boards::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    )
    mkpath(output_dir)
    train_path = joinpath(output_dir, "train.jld2")
    val_path = joinpath(output_dir, "val.jld2")
    test_path = joinpath(output_dir, "test.jld2")
    for p in (train_path, val_path, test_path)
        isfile(p) && rm(p)
    end

    data = load_annotations(source_dir)
    N = size(data.patches, 4)   # data.patches is (P, P, 2, N) after the mask refactor
    P = size(data.patches, 1)

    # Count patches per board UID
    uid_counts = Dict{String, Int}()
    for uid in data.board_uids
        uid_counts[uid] = get(uid_counts, uid, 0) + 1
    end

    # Assign boards to splits: test first, then val, remainder is train.
    test_set, val_set = _assign_splits(
        uid_counts, validation_fraction, test_fraction,
        val_boards, test_boards, seed
    )

    test_uids = sort(collect(test_set))
    val_uids = sort(collect(val_set))
    train_uids = sort(filter(u -> u ∉ val_set && u ∉ test_set, collect(keys(uid_counts))))

    n_test = sum(uid_counts[u] for u in test_uids; init = 0)
    n_val = sum(uid_counts[u] for u in val_uids; init = 0)
    n_train = N - n_val - n_test

    @info "Split: $(length(train_uids)) train boards ($n_train patches), " *
        "$(length(val_uids)) val boards ($n_val patches), " *
        "$(length(test_uids)) test boards ($n_test patches)"

    train_idx = Int[]; val_idx = Int[]; test_idx = Int[]
    for i in 1:N
        uid = data.board_uids[i]
        if uid in test_set
            push!(test_idx, i)
        elseif uid in val_set
            push!(val_idx, i)
        else
            push!(train_idx, i)
        end
    end

    C = size(data.confusers, 4)
    !isempty(train_idx) && _write_split(
        train_path, data, train_idx, P;
        confusers = C > 0 ? data.confusers : nothing
    )
    !isempty(val_idx)   && _write_split(val_path, data, val_idx, P)
    !isempty(test_idx)  && _write_split(test_path, data, test_idx, P)

    stats = (;
        train_patches = length(train_idx), val_patches = length(val_idx),
        test_patches = length(test_idx), confusers = C,
        train_boards = length(train_uids), val_boards_n = length(val_uids),
        test_boards_n = length(test_uids),
        val_frac = n_val / max(N, 1), test_frac = n_test / max(N, 1),
    )

    @info "  Train: $(join(train_uids, ", "))"
    @info "  Val:   $(join(val_uids, ", "))"
    @info "  Test:  $(join(test_uids, ", "))"
    @info "  Fractions — val: $(round(stats.val_frac; digits = 3)), test: $(round(stats.test_frac; digits = 3))"

    return (;
        train_path, val_path, test_path,
        val_boards = val_uids, test_boards = test_uids, stats,
    )
end

"""
Assign boards to test and val splits. Test is carved out first, then val
from the remainder. Explicit lists override automatic assignment.
"""
function _assign_splits(
        uid_counts, validation_fraction, test_fraction,
        val_boards_explicit, test_boards_explicit, seed
    )
    total = sum(values(uid_counts))
    rng = Random.MersenneTwister(seed)
    uids = sort!(collect(keys(uid_counts)))   # a fixed order before the shuffle: Dict order is hash-dependent
    Random.shuffle!(rng, uids)
    sort!(uids; by = u -> uid_counts[u], rev = true)

    # Test split
    test_set = if !isnothing(test_boards_explicit)
        Set{String}(test_boards_explicit)
    else
        _greedy_assign(uids, uid_counts, total, test_fraction)
    end

    # Val split from remaining boards
    remaining = filter(u -> u ∉ test_set, uids)
    remaining_total = sum(uid_counts[u] for u in remaining; init = 0)

    val_set = if !isnothing(val_boards_explicit)
        Set{String}(val_boards_explicit)
    else
        _greedy_assign(
            remaining, uid_counts, remaining_total,
            validation_fraction / max(1.0 - test_fraction, 0.01)
        )
    end

    return test_set, val_set
end

"""Greedily assign boards until patch fraction target is reached."""
function _greedy_assign(uids, uid_counts, total, target)
    assigned = Set{String}()
    count = 0
    for uid in uids
        count + uid_counts[uid] > target * total && !isempty(assigned) && break
        push!(assigned, uid)
        count += uid_counts[uid]
        count >= target * total && break
    end
    if isempty(assigned) && !isempty(uids)
        push!(assigned, uids[end])
    end
    return assigned
end

"""
Write a subset of annotations to a single JLD2 split file. Confusers
(when supplied — only train gets them) are stored in the same file.
"""
function _write_split(path, data, indices, P; confusers = nothing)
    patches = data.patches[:, :, :, indices]   # (P, P, 2, |indices|)
    payload = (;
        patches = patches,
        board_uid = data.board_uids[indices],
        blob_id = Int32.(data.blob_ids[indices]),
        confusers = confusers === nothing ? Array{Float32}(undef, P, P, 2, 0) : confusers,
    )
    return jldsave(path; payload...)
end

# =============================================================================
# Visualization
# =============================================================================

"""
    plot_annotations(sets; max_blobs=20, max_obs=10) -> Figure

Visualize CspondSets as a grid. Each row is one blob: first column is the
reference patch (from board PDF), then a gap, then observation columns
(from camera images). Rows sorted by number of observations (most first).

    plot_annotations(sets)                           # top 20 blobs
    plot_annotations(sets; max_blobs=5, max_obs=20)  # fewer blobs, more obs
    plot_annotations(load_cspond_sets("annotations/"))   # directory of per-image JLD2s
"""
function plot_annotations end
