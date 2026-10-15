# =============================================================================
# Board Pose Pipeline
# =============================================================================
#
# load_board              — parse board JSON metadata (no rasterization)
# extract_board_features  — rasterize + canonicalize + describe one board
# extract_image_features  — detect + adapt + canonicalize + describe
# find_board              — match + estimate for a single board
# find_boards             — detect + loop find_board + verify + suppress
#
# =============================================================================

using VisualGeometryCore:
    centroid, feature_frame,
    Extrinsics3, CameraModel, AbstractIntrinsics, Pinhole, AbstractLensModel,
    undistort, undistort!, Camera, HomographyMat, HomographyMap, EllipticFrame,
    region, Ellipse, project, pinhole, transfer_ellipse,
    sources, Attributed, withattrs, planar_lift, lift, exact   # #21 correspondence layer; `exact`: the board side of a 2D fit

using VisualGeometryFeatures:
    MutualNN, KnnMatch, NNStrategy,
    described_features,
    match, csponds,
    # Hoisted to VGF 0.5: the stacked multi-gallery matcher and the greedy
    # one-to-one reduction. BB owned both until they proved general.
    StackedDescriptorIndex, nn_match_stacked, resolve_one_to_one

using RobustVisualGeometry: RobustVisualGeometry, find_homography, find_absolute_pose, RansacConfig,
    locally_optimize, AbsolutePoseProblem, HomographyProblem, BoundedProblem,
    # The radial arm's OWN search. `find_*` is RANSAC and lives here by the
    # verb table; `fit_*` (inliers given) lives in VisualGeometryCore.
    find_radial_homography, RadialHomographyProblem
# The constraint-problem protocol is VisualGeometryCore's; RVG's problems add
# methods to it.
using VisualGeometryCore: residuals!
# Every fit scores with the Student-t + uniform marginal (Jeffreys scale
# prior) at the geometry's `outlier_halfwidth` — RansacScoringDoneRight's
# scorer, stated explicitly: RVG's own default is MSAC at a threshold.
using RansacScoringDoneRight: StudentTUniformScoring

# =============================================================================
# load_board — parse JSON metadata only (no rasterization)
# =============================================================================

"""
    load_board(board_path; density = nothing) -> NamedTuple

Parse board JSON metadata: extract UID, GT blob positions, and world coordinates.
Nothing is rasterized (a board with no file beside its JSON is rendered into a
cache as its PDF by `BlobBoards.Authoring`).

`density` is the density the board's image is turned into pixels at. A
board has none of its own (a stored v3 board excepted: its layout is its
printer's pixels), so by default a PDF is converted just fine enough to sample
the board's smallest blob well (`configs/authoring.toml`, `[board_raster]`),
and a raster beside the JSON is at the density its sidecar records. Asking for
another density than a raster's, or than a v3 board's own, is an
`ArgumentError`; a raster whose sidecar records no density is an error.

Returns a NamedTuple with:
- `uid::String` — the board's 3-character uid (the first characters of `config_hash`)
- `bundle` — parsed BoardBundle
- `config_hash::String` — the full SHA-256 the uid abbreviates (a caller that
  needs both no longer re-reads the bundle for it)
- `image_path::String` — path to the image/PDF (for deferred rasterization)
- `density` — the density of the board's image in pixels
- `canvas_frames::Vector{Detection}` — GT blob detections in canvas pixel coordinates at `density`
- `world_frames::Vector{Detection}` — GT blobs in world coordinates (mm, unitless)
"""
function load_board(board_path; density::Union{Nothing, Quantity} = nothing)
    bundle, image_path, config_hash = load_bundle(board_path)
    # load_bundle parses either bundle kind; a board pipeline needs the board
    # frames/rects. The assert narrows the union at its source (a pattern
    # JSON here would otherwise die in a confusing MethodError downstream).
    bundle = bundle::BoardBundle
    uid = uid_from_hash(config_hash)
    # the pixels of the board's image: the canvas at this density
    density = _image_density(image_path, bundle; density)
    # :canvas blobs are px by contract: strip them in px.
    gt_discs = ustrip.(px, blobs(bundle, :canvas; density))
    canvas_frames = [Detection(d, (; id = j)) for (j, d) in enumerate(gt_discs)]

    world_discs = blobs(bundle, :world)
    world_frames = [Detection(d, (; id = j)) for (j, d) in enumerate(world_discs)]

    return (; uid, config_hash, bundle, image_path, density, canvas_frames, world_frames)
end

# =============================================================================
# The design resolver — a board uid → its nominal plane points
# =============================================================================
#
# A board is a CONTENT-ADDRESSED design: its uid names its geometry, so a
# stored target rig (VisualGeometryIo's rig/v4) names a board by reference and
# never restates its points. BlobBoards owns the design, so it resolves the
# reference: the nominal points are the board's blob centres in its world
# frame (`blobs(bundle, :world)`: centred on the board, y up, millimetres,
# z = 0), in the bundle's blob order — the order a point index (1-based) of a
# measurement names.

"""
    board_points(bundle::BoardBundle) -> Vector{SVector{2, Float64}}
    board_points(uid::Symbol, boards) -> Vector{SVector{2, Float64}}

A board design's nominal plane points: its blob centres in the board's world
frame (millimetres, centred, y up — the plane `z = 0` of `TargetFrame{uid}`),
in the bundle's blob order, the order a measurement's point index refers to.
`boards` is a directory of board JSON sidecars, or a uid → bundle dictionary;
an unknown uid is an `ArgumentError` naming the uids present.
"""
board_points(bundle::BoardBundle) = SVector{2, Float64}[SVector{2, Float64}(d.center) for d in blobs(bundle, :world)]
board_points(uid::Symbol, boards::AbstractDict) = haskey(boards, uid) ? board_points(boards[uid]) :
    throw(ArgumentError("no board $(uid); the library holds $(sort!(collect(keys(boards))))"))
board_points(uid::Symbol, boards_dir::AbstractString) = board_points(uid, board_library(boards_dir))

"""
    board_library(boards_dir) -> Dict{Symbol, BoardBundle}

Every board design in a directory of board JSON sidecars, keyed by its uid
(`uid_from_hash` of its stored configuration hash); pattern bundles and other
JSON files are skipped. Geometry only: no render is resolved or regenerated.
"""
function board_library(boards_dir::AbstractString)
    lib = Dict{Symbol, BoardBundle}()
    for f in sort!(filter(endswith(".json"), readdir(boards_dir; join = true)))
        bundle, h = try
            _parse_bundle(f)
        catch e
            e isa Union{ErrorException, ArgumentError, KeyError, TypeError} || rethrow()
            continue                                     # not a board sidecar
        end
        bundle isa BoardBundle || continue
        lib[Symbol(uid_from_hash(h))] = bundle
    end
    return lib
end

"""
    board_resolver(boards) -> (design::Symbol -> Vector{SVector{2, Float64}} | nothing)

The resolver a stored target rig takes for its referenced board designs:
`VisualGeometryIo.load(Rig, path; resolve = board_resolver(boards_dir))`.
A design the library does not hold resolves to `nothing` (it stays an
unresolved, pose-only target). `boards` is a directory or a uid → bundle
dictionary; a directory is read once.
"""
board_resolver(boards_dir::AbstractString) = board_resolver(board_library(boards_dir))
board_resolver(lib::AbstractDict) = uid -> haskey(lib, uid) ? board_points(lib[uid]) : nothing

"""
    board_target(uid::Symbol, boards) -> PlanarTarget

The board as a resolved VisualGeometryCore `PlanarTarget`: its design named by
`uid`, its nominal points [`board_points`](@ref), the `Rigid` warp.
"""
board_target(uid::Symbol, boards) = PlanarTarget(uid; nominal = board_points(uid, boards))

# =============================================================================
# extract_board_features — rasterize + canonicalize + describe one board
# =============================================================================

"""
    extract_board_features(model::AbstractDescriptorNetwork, board;
                           patch_size=64,
                           inner_extent=cutoff(board.bundle),
                           outer_extent=DEFAULT_OUTER_EXTENT) -> StructArray{DescribedFeature}

Extract features from a loaded board: rasterize PDF, canonicalize GT blobs to
log-polar patches, and compute descriptors.

`board` is a NamedTuple from [`load_board`](@ref).

Default method assumes a single-channel descriptor (`DescriptorNetwork`,
ONNX): canonicalizes only the rendered PDF intensity and feeds a
`Vector{Matrix{Gray{Float32}}}` to `describe`.

The masked Lux descriptor overrides this method in `BlobBoardsLuxExt`
to additionally canonicalize the board-rect mask and stack into a
`(P, P, 2, N)` array — see `_board_mask_image` / `_stack_2ch` in
`src/annotate.jl`.

`inner_extent` defaults to the board's own Gaussian cutoff (from bundle
metadata); `outer_extent` is the outer sampling radius for the log-polar
patch (in σ units relative to each blob's scale). `image` is the board's
raster when the caller already has it (`load_board_gallery`, the gallery's
batch `insert!`); by default it is read through the raster cache
(`_board_raster`), so the PDF is rasterized at most once per board.

Returns `StructArray{DescribedFeature}` pairing world-coordinate Disc
detections with their descriptor vectors.
"""
function extract_board_features(
        model::AbstractDescriptorNetwork, board;
        patch_size = 64,
        inner_extent = cutoff(board.bundle),
        outer_extent = DEFAULT_OUTER_EXTENT,
        device::KA.Backend = default_device(),
        image::AbstractMatrix = _board_raster(board.image_path, board.bundle, board.config_hash; density = board.density)
    )
    # Same canonicalize+describe stage as the query path (`_canon_describe`):
    # on CUDA the patch cube stays on-device and feeds ORT device tensors —
    # gallery and query descriptors come from ONE code path.
    desc = _canon_describe(
        device, model, board.canvas_frames, image;
        patch_size, inner_extent, outer_extent
    )
    return described_features(board.world_frames, desc)
end

# =============================================================================
# extract_image_features — detect + adapt + canonicalize + describe
# =============================================================================

"""
    extract_image_features(descriptor_network, img; kwargs...) -> StructArray{DescribedFeature}

Full query-image feature extraction pipeline: detect blobs, adapt via posterior
IRLS, filter, canonicalize to log-polar patches, and compute descriptors.

# Backends and image storage
- `device`: the KA device every stage runs on (`KA.CPU()` default,
  `CUDABackend()` for GPU). `img` must match (e.g. `CuArray{Float32}` on
  CUDA for the adapt/canonicalize stages).
- `detect_img`: the image supplied to `detect_blobs`. Defaults to `img`. The
  detector consumes `Matrix{UInt8}` host data. When `detect_img` differs
  from `img` supply both: the detect path uses `detect_img`, the
  adapt/canonicalize path uses `img`.
"""
function extract_image_features(
        descriptor_network, img::AbstractMatrix;
        cutoff = DEFAULT_CUTOFF,
        device::KA.Backend = default_device(),
        detect_img = img,
        adapt_config = AdaptConfig(),
        detect_config::DetectConfig = DetectConfig(),
        max_detections::Int = typemax(Int),
        patch_size::Int = 64
    )
    model = _ensure_network(descriptor_network)
    detections = detect_blobs(
        detect_img;
        config = detect_config, max_detections, device
    )
    return describe_detections(
        model, detections, img;
        cutoff, device, adapt_config, patch_size
    )
end

"""
    describe_detections(model, detections, img; cutoff, device,
                        adapt_config, patch_size) -> StructArray{DescribedFeature}

Adapt → filter → canonicalize → describe a FIXED detection set at one `cutoff`.

Split out of `extract_image_features` because `cutoff` comes from the BOARD
SET, not the image: it is the canonicalization inner extent and therefore
defines what the descriptor IS. `find_boards` resolves ONE shared extent per
gallery (`_shared_image_cutoff` — the uniform board cutoff, else the mean).
Detection itself is cutoff-independent (scale-space extrema), so it is done
once and shared.
"""
function describe_detections(
        # The network first, then the shapes, then the image they are sampled
        # from; typed so the retired `(img, detections, model)` order is a
        # MethodError.
        model, detections::AbstractVector, img::AbstractMatrix;
        cutoff = DEFAULT_CUTOFF,
        device::KA.Backend = default_device(),
        adapt_config = AdaptConfig(),
        patch_size::Int = 64
    )
    frames = adapt_detections(detections, img; cutoff, adapt_config, device)
    return describe_frames(
        model, frames, img; inner_extent = cutoff,
        device, patch_size
    )
end

"""
    adapt_detections(detections, img; cutoff, adapt_config, device) -> frames

Fit elliptic frames to raw detections and drop the unusable ones. Independent
of the descriptor's sampling extent, so it is done ONCE per polarity group and
shared (see `_find_boards_grouped`).

`image` MUST be supplied to `filter_blobs`: without it the clip test
short-circuits (`image_rect === nothing`) and border-clipped ellipses reach
matching. `max_clip_ratio = 0` rejects any ellipse not fully inside the frame —
a partially observed blob has a biased centre and a meaningless descriptor.
"""
adapt_detections(
    detections::AbstractVector, img::AbstractMatrix; cutoff = DEFAULT_CUTOFF,
    adapt_config = AdaptConfig(),
    device::KA.Backend = default_device()
) =
    filter_blobs(
    # ONE extent for all three consumers. `cutoff` is the gallery's
    # (`_shared_image_cutoff`): it sets the adaptation scan window, the clip
    # test below, and — via `describe_detections` — the canonicalization inner
    # extent. It used to reach only the last two while `adapt` read
    # `AdaptConfig.cutoff` from the TOML, so a board set whose printed cutoff
    # differed from that value was FIT at one boundary and then clipped and
    # described at another, three lines apart and with no warning.
    # `scan_padding` is the lever for over/undersizing the window.
    adapt(detections, img; config = ConstructionBase.setproperties(adapt_config, (; cutoff)), device),
    detections;
    cutoff, image = img, max_clip_ratio = 0.0
)

"""
    describe_frames(model, frames, img; inner_extent, device, patch_size)

Canonicalize + describe at ONE sampling extent: `logpolar_map` consumes
`inner_extent / outer_extent`, so gallery and image patches sampled at very
different extents are not comparable. `find_boards` runs this ONCE per
polarity group at the gallery's shared extent (`_shared_image_cutoff`);
per-distinct-cutoff description was removed 2026-08-11 (see the design note
above `_bundle_polarity`).
"""
function describe_frames(
        model, frames::AbstractVector, img::AbstractMatrix; inner_extent = DEFAULT_CUTOFF,
        outer_extent = DEFAULT_OUTER_EXTENT,
        device::KA.Backend = default_device(), patch_size::Int = 64
    )
    isempty(frames) && return described_features(frames, Matrix{Float32}(undef, 0, 128))
    descriptors = _canon_describe(
        device, model, frames, img;
        inner_extent, outer_extent, patch_size
    )
    return described_features(frames, descriptors)
end

# Canonicalize + describe as one dispatchable stage. The generic path is the
# host pipeline (patch download → per-batch host gather → ONNX). The
# CUDABackend + DescriptorNetwork(:cuda) override in BlobBoardsCUDAExt keeps
# the patch cube on-device and feeds ONNX Runtime a device tensor directly —
# at 26k patches the host path moves ~880 MB over PCIe for nothing.
#
# Shapes before the image (`canonicalize(frames, img)`); typed so the retired
# `(device, model, img, frames)` order is a MethodError, here and on the override.
_canon_describe(device, model, frames::AbstractVector, img::AbstractMatrix; inner_extent, outer_extent, patch_size = 64) =
    describe(model, canonicalize(frames, img; patch_size, inner_extent, outer_extent, device))

# =============================================================================
# Dispatch on camera: nothing → homography, CameraModel → P3P,
# RadialHomographyGeometry → radially distorted homography. No separate
# context type: the geometry ITSELF rides the camera slot — it already
# carries the one datum the estimator needs (`image_size`) — so its
# `_find_model`/`_refine` methods live below the GeometryConfig definitions.
# The LO strategy is the geometry config's `local_optimization`
# (`[find_boards.geometry]`, `{ type = "student_t_uniform" }` by default;
# RVG's own find_* defaults are Welsch-HQ). find_boards threads it through
# `fit_kw`; the keyword default re-reads the file for a bare `find_board`.
_default_lo(::Nothing) = _dflt_local_optimization(:homography)
_default_lo(::CameraModel) = _dflt_local_optimization(:absolute_pose)
# The scorer of every fit and refine: the Student-t + uniform marginal at the
# arm's outlier half-width (`P` the arm's problem type).
_scoring(::Type{P}, outlier_halfwidth) where {P} = StudentTUniformScoring(P, outlier_halfwidth)
# The half-width a fit falls back to when `fit_kw` carries none (a bare
# `find_board` call), px. Every engine path supplies the geometry's own value.
# Homography: 5.0. The bmvc2026 outlier half-width CV's run of record
# (bmvc2026/scripts/outlier_halfwidth_cv.jl, run key 3a346460b90dc25d,
# 2026-10-09) selects 6.05 px for the homography arm, but its curve is flat:
# inter-board invariance mAA@10 mm 0.0466-0.0495 over 1-16 px (0.0480 at
# 4.9 px), a spread of 0.003 (an earlier run, on a separately built gallery,
# moved single points by 3e-4). Flat, so the fallback is the package's 5.0
# rather than RobustVisualGeometry 1.0's per-class 20. The radial arm keeps 20
# (not cross-validated).
const _FIT_HALFWIDTH_FALLBACK = (homography = 5.0, absolute_pose = 5.0, radial_homography = 20.0)
_find_model(
    cam::Nothing, pt_csponds; local_optimization = _default_lo(cam),
    outlier_halfwidth = _FIT_HALFWIDTH_FALLBACK.homography, kw...
) = find_homography(
    pt_csponds; scoring = _scoring(HomographyProblem{Float64}, outlier_halfwidth), local_optimization, kw...
)
_find_model(
    cam::CameraModel, pt_csponds; local_optimization = _default_lo(cam),
    outlier_halfwidth = _FIT_HALFWIDTH_FALLBACK.absolute_pose, kw...
) = find_absolute_pose(
    cam, pt_csponds; scoring = _scoring(AbsolutePoseProblem, outlier_halfwidth), local_optimization, kw...
)

# Seeded warm local-optimization refine (no second RANSAC) for the guided final
# fit — the LM-guarded `locally_optimize` (RVG) refines `θ₀` on the guided
# correspondences. Mirrors `_find_model`'s camera dispatch (incl. the LO choice).
# `workspace` is an optional pre-allocated RansacWorkspace (sized ≥ the
# guided set; `set_n!` inside locally_optimize trims it) — the driver supplies
# per-thread warm workspaces so the guided refit stops allocating one per
# candidate fit.
_bounded(p, extent) = isnothing(extent) ? p : BoundedProblem(p, extent)
_refine(
    cam::Nothing, pt_csponds, θ₀; outlier_halfwidth, local_optimization = _default_lo(cam),
    extent = nothing, workspace = nothing
) =
    locally_optimize(
    _bounded(HomographyProblem(pt_csponds), extent), θ₀;
    scoring = _scoring(HomographyProblem{Float64}, outlier_halfwidth), workspace, local_optimization
)
_refine(
    cam::CameraModel, pt_csponds, θ₀; outlier_halfwidth, local_optimization = _default_lo(cam),
    extent = nothing, workspace = nothing
) =
    locally_optimize(
    _bounded(AbsolutePoseProblem(cam, pt_csponds), extent), θ₀;
    scoring = _scoring(AbsolutePoseProblem, outlier_halfwidth), workspace, local_optimization
)

# Endpoint lift per estimator arm (#21): H fits on 2D centroids; pose
# z=0-lifts them to world points. (The radial arm's method — 2D, like H —
# lives below the GeometryConfig definitions.) The board side of a 2D fit is
# `exact` (VisualGeometryCore conventions/measurement-noise.md): board points are
# known, not measured, so the H residual is the transfer error in image pixels —
# as a plain (unit isotropic) point in board mm it was scored in board mm.
_fit_endpoints(::Nothing, scored) = _board_exact(scored)
_fit_endpoints(::CameraModel, scored) = planar_lift(scored)
_board_exact(scored) = lift(scored; source = exact ∘ centroid, target = centroid)

# guided_search projects the board pattern through ONE HomographyMap; arms
# whose transfer is not a collineation opt out (radial method below).
_guided_enabled(_) = true

# =============================================================================
# Undistort — VGC's fused KernelAbstractions warp, device-adaptive
# =============================================================================
#
# Brown-Conrady undistortion routes through VGC's `undistort!`: the fused
# `AffineMap ∘ lens ∘ AffineMap` warp (`distortion_map` + `image_warp`)
# sampled by the shared KA bilinear kernel on whatever backend owns the
# destination array.  The leading `Float32` (`undistort!(dst, Float32, camera,
# src)`) keeps the per-pixel math single-precision on GPU.
# The hand-written `_undistort_bc_ka!` /
# `_undistort_bc_u8_ka!` kernels this file used to carry were copies of
# that math and were deleted in the 2026-07 warp unification (#19).

using KernelAbstractions: KernelAbstractions
using LinearAlgebra: dot

_undistort_detect(::KA.Backend, img, ::Nothing) = img
_undistort_detect(::KA.Backend, img, ::CameraModel{<:AbstractIntrinsics, <:Pinhole}) = img

# Host UInt8 view of an image: N0f8 storage reinterprets, anything else
# quantizes through Gray{N0f8} — the same coercion VGF's `_asift_input`
# applies. One helper, two call sites (here and `_undistorted_pair`).
# The byte-gray input ladder is VisualGeometryFeatures' `gray_u8` since 0.7.1.
using VisualGeometryFeatures: gray_u8

# Any lens with a distortion map — OpenCV's model (the metadata calibration) or
# the division model:
# VGC's fused warp is generic over `AbstractLensModel`, and Pinhole keeps its
# identity method above.
function _undistort_detect(
        dev::KA.Backend, img,
        cam::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel}
    )
    H, W = size(img)
    raw = gray_u8(img)
    d_u8 = KernelAbstractions.allocate(dev, UInt8, H, W)
    copyto!(d_u8, raw)
    d_out = KernelAbstractions.allocate(dev, UInt8, H, W)   # kernel writes every pixel
    # u8→u8: VGC's kernel samples the 8-bit source bilinearly in the 0–255
    # domain (linear ops commute with the scale) and stores saturating UInt8.
    undistort!(d_out, Float32, cam, d_u8)
    return Array(d_out)
end

# Adapt-image undistort: CPU reference is VGC's undistort; the CUDABackend
# method (BlobBoardsCUDAExt) keeps a device-resident CuArray{Float32}.
_undistort_adapt(::KA.Backend, img, camera) = undistort(camera, img)

# =============================================================================
# guided_search — project the verified pattern, re-associate to detections
# =============================================================================

# The one R-tree shape the package indexes points with (here and annotate.jl's GT ellipses).
_new_rtree() = SpatialIndexing.RTree{Float64, 2}(Int, Nothing; leaf_capacity = 25, branch_capacity = 25)

"""
    detection_rtree(detc; skip = nothing) -> SpatialIndexing.RTree

R-tree over detection centroids `detc` (point rects), the `rtree` that
[`guided_search`](@ref) queries; `skip` (a `BitVector` over `detc`) leaves the
detections already claimed out of the index, so they can never be
re-associated. The codebase's standard
spatial index (`pattern.jl` collision detection, `annotate.jl` IoU
matching), built once per phase. `_rtree_nearest` turns the per-projected-
blob nearest-detection scan from O(D) into an O(log D) rect query: the
`point ± radius` box is a superset of the radius disc, so the min-distance
scan over the hits followed by the caller's radius gate is EXACTLY the
brute-force result.
"""
function detection_rtree(detc::AbstractVector; skip::Union{Nothing, BitVector} = nothing)
    rt = _new_rtree()
    # `eachindex` range loads directly — no D-sized collect on the unclaimed
    # build (the per-round rebuilds supply `skip` and pay findall regardless).
    idxs = skip === nothing ? eachindex(detc) : findall(!, skip)
    isempty(idxs) && return rt
    # Bulk load (OMT) instead of n incremental insert!s: one sort-based
    # packing sweep, tighter node MBRs → fewer nodes touched per query.
    # `skip` (already-claimed detections from this phase's earlier commit
    # rounds) are simply not indexed, so guided association can never
    # re-associate one. `bulk_load!` sorts this fresh vector in place
    # (SpatialIndexing's `load!` builds the same tree from sorted copies).
    elems = map(idxs) do j
        p = detc[j]
        x = Float64(p[1]); y = Float64(p[2])
        SpatialIndexing.SpatialElem(SpatialIndexing.Rect((x, y), (x, y)), j, nothing)
    end
    return bulk_load!(rt, elems)
end

# Nearest detection to `(px, py)` within `radius`: VisualGeometryCore's
# branch-and-bound `nearest_neighbor!` (SpatialQueries, written for this call
# site) — subtree pruning by MBR min-distance, no candidate materialization, no
# callback closure. `(dist², index)`, `(Inf, 0)` when nothing is within radius.
@inline function _rtree_nearest(
        rt::SpatialIndexing.RTree, st, px::Float64, py::Float64,
        radius::Float64
    )
    # `st` is a per-CALL `NearestNeighborState(rt)`: one holder per guided
    # search instead of one heap allocation per projected blob.
    # Thread-private — each fit owns its own.
    res = nearest_neighbor!(st, rt, (px, py); max_sqdist = radius * radius)
    return res === nothing ? (Inf, 0) : (res[2], res[1])
end

# A pose model never pairs with the radial token or with `nothing` (those
# geometries fit homographies), so those combinations refuse loudly. Both
# arms exist so JET sees every union member covered — VGC 0.18's
# `Camera(model::AbstractCameraModel, …)` turned the old silent
# StackOverflow on a wrong model into a static MethodError.
"""
    _model_to_homography(model, cam) -> HomographyMap

The world→image map to project through, from an estimator result. Dispatches
on the MODEL, which is the honest discriminator: an `Extrinsics3` is a metric
board→camera pose that knows nothing about pixels and needs the intrinsics to
reach the image, while a `HomographyMat` already IS the map and ignores `cam`.

Both branches are one VGC call — `HomographyMap(::PinholeCamera)` composes
pose ∘ K on the z = 0 plane and returns it LABELED
`TargetPlaneFrame → PixelFrame{:colmap}`. This exists so its call sites (the engine,
`annotate.jl`) do not each spell it out.

`cam` is the ESTIMATION camera (`_estimation_camera`): under `:global` the
pinhole twin, whose plane homography is the exact map; under `:none` the
full camera, for which no homography exists — the guided transfer is then
the `Camera` itself (`transfer_ellipse(::Camera, _)`: the conic through the
plane homography exactly, the lens's affinity at the transferred centre),
and the name stays for continuity with the two radial arms, which also forward
their MODEL through.
"""
_model_to_homography(pose::Extrinsics3, cam::CameraModel{<:AbstractIntrinsics, Pinhole}) =
    HomographyMap(Camera(cam, pose))
_model_to_homography(pose::Extrinsics3, cam::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel}) =
    Camera(cam, pose)
# Untyped on purpose: RadialHomographyGeometry is defined further down this
# file, and a signature naming it here would not precompile.
_model_to_homography(::Extrinsics3, cam) = throw(
    ArgumentError("a pose model needs a pinhole camera to project; got $(typeof(cam))")
)
_model_to_homography(H::HomographyMat, _) = HomographyMap(H)
# The radial arm forwards its MODEL rather than a bare homography: the
# guided transfer needs (H, k, centre) together — the conic goes through H
# exactly, only the undistortion is linearised. Named `_model_to_homography`
# for continuity; what it really returns is the guided TRANSFER.
_model_to_homography(m::VisualGeometryCore.RadialHomography, _) = m


# =============================================================================
# Guided association on the radial arm
# =============================================================================
#
# Guided search used to opt out for `RadialHomographyGeometry` because its
# association projects through ONE HomographyMap and a radially distorted
# transfer is not a collineation. That opt-out cost most of the boards at
# distance — measured on 260527, boards accepted per frame were 23/4/2 on the
# 1/2/3 m rings against 30/30/30 for both the plain-homography and the
# calibrated pose arms — because guided search is exactly the stage that
# recovers boards whose descriptor matching is under-supported, which is what a
# distant board is.
#
# The fix keeps the projective part EXACT and linearises only the distortion.
#
# `RadialHomography` is (H, k, c) with the division model one-sided on the
# TARGET side, so H already maps world points into UNDISTORTED image
# coordinates and only the centre offset is distorted. Inverting the model's
# own forward map (`project`: μ = 2/(z + sqrt(z² − 4kρ²))) gives the classic
# division form for the undistortion,
#
#     ρ = s / (1 + k s²),        s = ‖p − c‖
#
# so with g(s) = 1/(1 + k s²) the undistortion is p ↦ c + g(s)(p − c) and its
# Jacobian at a MEASURED detection centre is
#
#     J = g(s)·I + (g'(s)/s)·(p−c)(p−c)ᵀ
#
# i.e. g(s) tangentially and (1 − k s²)/(1 + k s²)² radially. That anisotropy
# is the whole shape effect distortion has on a blob, and over a blob's support
# — tens of pixels against a field varying over thousands — it is very close to
# affine.
#
# WHY LINEARISE THE UNDISTORTION AND NOT THE WHOLE TRANSFORM. Linearising the
# entire world→pixel map would approximate the homography too, and that carries
# the perspective foreshortening, which is the dominant shape effect on an
# oblique board and is transferred exactly for free. And the Jacobian is
# evaluated at the detection's own centre — a measurement — never at the
# prediction, so model error never enters the shape estimate. Guided search
# exists precisely because the current model is imperfect.

# Forward: undistorted image point -> distorted pixel. Exact (the model's own
# map); `nothing` outside the representable field, as `project` reports.
@inline function _radial_distort(m::VisualGeometryCore.RadialHomography{T}, u) where {T}
    ray = SVector{3, T}(T(u[1]) - T(m.center[1]), T(u[2]) - T(m.center[2]), one(T))
    p = VisualGeometryCore.project(m, ray)
    return p
end

# The radial model's undistortion as a MAP — the pixel-domain division lens
# about `center` — so the local affinity is `transform_deriv` (the chain rule
# through the translation, the lens inverse and the translation back), the
# same derivative every other undistortion in the stack uses.
@inline _radial_undistortion_map(m::VisualGeometryCore.RadialHomography{T}) where {T} =
    Translation(SVector{2, T}(m.center[1], m.center[2])) ∘ inv(VisualGeometryCore.Division(m.k)) ∘
    Translation(SVector{2, T}(-m.center[1], -m.center[2]))

# One detection ellipse carried into the undistorted domain by that affinity —
# VGC's `transfer_ellipse`, the one implementation of `c′ = U(c)`,
# `Σ′ = J·Σ·Jᵀ` (stage by stage; the two translations are exact). `nothing`
# where the division lens has no inverse (its 90° circle), where `J` is not
# finite, or where `Σ′` is not positive definite — at the division fold
# (`r = 1/√k`, `J` singular) and on a degenerate detection. The hand-rolled
# copy it replaces (2026-09-21) threw `ArgumentError` from `Ellipse` there;
# `_guided_accept!` reads `nothing` as a rejected candidate. Elsewhere the
# result is unchanged (measured: centre and axes identical, θ within an ulp).
@inline _undistort_ellipse(m::VisualGeometryCore.RadialHomography, e) =
    transfer_ellipse(_radial_undistortion_map(m), e)

# The two operands guided association needs, per projected blob:
#   (ellipse to COMPARE, centre to SEARCH at)
# Projective: both are the transferred conic — unchanged from before.
# Radial: compare in UNDISTORTED coordinates where the transfer is exact, but
# search at the exactly-distorted centre so the detection index and `max_px`
# stay in distorted pixels and need no remapping.
@inline function _guided_transfer(H::HomographyMap, wreg)
    ep = transfer_ellipse(H, wreg)
    ep === nothing && return nothing
    return (ep, ep.center)
end
@inline function _guided_transfer(m::VisualGeometryCore.RadialHomography, wreg)
    ep = transfer_ellipse(HomographyMap(m.H), wreg)
    ep === nothing && return nothing
    q = _radial_distort(m, ep.center)
    q === nothing && return nothing
    return (ep, q)
end
# Full camera (the :none policy): the transferred conic already lives in the
# RAW image — the plane homography exactly, the lens's affinity at the
# centre (VGC `transfer_ellipse(::Camera, _)`) — so it is compared where
# the detections are and searched at its own centre.
@inline function _guided_transfer(camera::Camera, wreg)
    ep = transfer_ellipse(camera, wreg)
    ep === nothing && return nothing
    return (ep, ep.center)
end

# The detection ellipse in the COMPARISON domain. Projective: as measured.
# Radial: carried into undistorted coordinates by the local affinity at its own
# centre. One Jacobian per accepted candidate, not per detection per board.
@inline _guided_detell(::HomographyMap, det_ells, j) = det_ells[j]
@inline _guided_detell(::Camera, det_ells, j) = det_ells[j]
@inline _guided_detell(m::VisualGeometryCore.RadialHomography, det_ells, j) =
    _undistort_ellipse(m, det_ells[j])

"""
    guided_search(H, bundle, image_features;
                  max_px, min_iou, keep = Pair{Int,Int}[],
                  detc, rtree, det_ells, wdiscs, wregions) -> StructArray

Final-fit correspondence builder. Project every world blob of `bundle`'s
pattern into the image through `H` (build it with
[`_model_to_homography`](@ref)),
then associate each projected blob to its nearest detection within `max_px` px
**only if the projected blob and the adapted detection overlap** — exact
ellipse IoU ≥ `min_iou`, both ellipses at the blob's rendered boundary
(`region(·, cutoff(bundle))`, the same convention `annotate_blobs` uses for
GT labeling). Supply the estimator's `outlier_halfwidth` as `max_px` so guided
association uses the same inlier band as the RANSAC fit (a tighter 3σ band
would drop seed inliers between 3σ and the half-width).

The IoU gate is purely geometric — descriptors are not consulted. It is
what rejects *confuser* boards: a wrong board can be smeared onto a
detection cluster center-wise, but its blobs' projected scales and shapes
won't agree with the adapted frames, so they fail the overlap check. Returns
a world-disc ↔ detection `ScoredCspond` set for a posterior-weighted refit
via `_find_model` (one-to-one, best overlap first), recovering pattern blobs
the descriptor matcher missed without admitting smears.

`H` is the guided TRANSFER: a `HomographyMap` on the projective arms, a
`RadialHomography` on the radial one, where the conic goes through the
homography exactly and only the undistortion is linearised (`_guided_transfer`),
or a VisualGeometryCore `Camera` (a camera model and a board pose), where the
conic goes through the plane homography exactly and the lens's affinity is
taken at the transferred centre.

`keep` (`world blob => detection` index pairs) are admitted first and
unconditionally. The remaining keywords are hoistable context, each computed
here when absent: `detc` the detection centroids, `rtree` an index over them
([`detection_rtree`](@ref), whose `skip` keeps claimed detections out),
`det_ells` the detections' ellipses at the board's rendered boundary, and
`wdiscs`/`wregions` the board's world blobs and their regions.
It is not a bare point map because the work here is CONIC transfer, not
point transfer — the IoU gate tests a blob's projected *shape*, so the map has
to carry the conic, not just its centre.

VGC's contravariant layer (`transforms/ct_contravariant.jl`) makes conics
first-class under `CoordinateTransformations`, so `Translation` and
`LinearMap` map ellipses too and this annotation is narrower than that
protocol allows. It is deliberately not widened: `transfer_ellipse` — which
is what this actually calls — is defined only for `HomographyMap{3}`, so a
wider signature would promise something VGC does not currently back.

`transfer_ellipse` rather than the bare `H(ellipse)` callable, because a
projective map does NOT always send an ellipse to an ellipse: when the conic
crosses the line at infinity the image is a hyperbola or parabola. The
callable returns a conic matrix; `transfer_ellipse` classifies it and returns
`nothing` for the degenerate cases, which `_guided_candidates` skips. Calling
`H(e)` directly would drop that guard.

The standing limit is geometric, not API: guided search assumes a planar
board under a pinhole camera. For a fisheye or double-sphere lens the
world→image map is not projective at all, and no conic transfer is exact —
that case would need a local linearisation at each blob centre.
"""
function guided_search(
        H::Union{HomographyMap, VisualGeometryCore.RadialHomography, Camera}, bundle, image_features;
        max_px::Real, min_iou::Real,
        keep::AbstractVector{Pair{Int, Int}} = Pair{Int, Int}[],
        detc::Union{Nothing, Vector} = nothing,
        rtree::Union{Nothing, SpatialIndexing.RTree} = nothing,
        det_ells::Union{Nothing, Vector} = nothing,
        wdiscs::Union{Nothing, Vector} = nothing,
        wregions::Union{Nothing, Vector} = nothing,
        # Sink for the detection indices this association READ: called
        # `on_read(j::Int32)` once per projected blob that had a neighbour in
        # range (see `_guided_accept!`). Nothing to do with the returned
        # correspondences.
        on_read = nothing,
        # Write-only out-parameter: the `NNMatch`es the returned table was built
        # from, in row order — for a caller that appends to them (`find_board`
        # hands back the arm's rejected descriptor matches beside them) without
        # decompiling the table.
        matches_out::Union{Nothing, Vector} = nothing
    )
    radius = max_px
    # Board-constant and phase-constant context is hoistable — callers that
    # loop over boards (`_claim_boards`) precompute and supply `detc`, an
    # R-tree over it (`detection_rtree`, O(log D) nearest queries), `det_ells`
    # (per distinct cutoff — one Ellipse per detection), and the board's
    # `wdiscs`/`wregions`. Every fallback below computes the identical value.
    wdiscs = wdiscs === nothing ? blobs(bundle, :world) : wdiscs
    board_cutoff = cutoff(bundle)
    detc = detc === nothing ? [centroid(d) for d in image_features] : detc
    # Adapted detections at the rendered-boundary scale — one Ellipse per
    # detection, shared by every projected blob that lands near it.
    det_ells = det_ells === nothing ?
        [Ellipse(region(feature_frame(f), board_cutoff)) for f in image_features] : det_ells
    # World blobs at the same rendered boundary — constant per board.
    wregions = wregions === nothing ?
        [region(d, board_cutoff) for d in wdiscs] : wregions
    cands = rtree === nothing ?
        _guided_candidates(
            H, wregions, detc, det_ells,
            Float64(radius), Float64(min_iou), on_read
        ) :
        _guided_candidates(
            rtree, H, wregions, detc, det_ells,
            Float64(radius), Float64(min_iou), on_read
        )
    # The RANSAC inliers (`keep`) were certified by descriptors AND spatial
    # consensus — guided search may only ADD to them, never drop them. They
    # claim their csponds first, unconditionally (no IoU gate); the IoU
    # candidates then fill in around them, best overlap first, one-to-one.
    used_d = falses(length(image_features)); used_w = falses(length(wdiscs))
    matches = NNMatch[]
    sizehint!(matches, length(keep) + length(cands))
    for (wi, di) in keep
        (used_d[di] || used_w[wi]) && continue
        used_d[di] = true; used_w[wi] = true
        push!(matches, NNMatch(wi, di, 1.0f0, 0.0f0))
    end
    sort!(cands; by = first, rev = true)
    for (v, wi, di) in cands
        (used_d[di] || used_w[wi]) && continue
        used_d[di] = true; used_w[wi] = true
        # score by overlap (higher = better); ratio unused for geometric matches
        push!(matches, NNMatch(wi, di, Float32(v), 0.0f0))
    end
    # Let the VGF intrinsic build the columnar ScoredCspond set — same path
    # `find_board` uses for descriptor matches, so the layout is identical.
    # `det_id` (image-detection index of each cspond) rides along for the
    # appearance gate, which needs the image side of every INLIER — including
    # the guided ones the descriptor matcher never produced.
    # `src_id` (board-blob index) rides along beside `det_id`. Together they
    # let APPEARANCE be checked against a correspondence that geometry alone
    # produced: guided association never consults descriptors, so a false
    # board's inliers pair blobs with unrelated detections. `src_id` indexes
    # `blobs(bundle, :world)`, which is exactly how the gallery features are
    # ordered (`extract_board_features` describes `board.world_frames`), so it
    # indexes `board_feature_dict[uid]` directly.
    matches_out === nothing || append!(empty!(matches_out), matches)
    return _scored_with_ids(wdiscs, image_features, matches)
end

# Scored-cspond table with the `det_id`/`src_id` columns claim bookkeeping needs
# on BOTH return paths (`guided_search`, `find_board`): one spelling.
_scored_with_ids(sources, image_features, matches) = withattrs(
    csponds(sources, image_features, matches);
    det_id = Int[m.second for m in matches],
    src_id = Int[m.first for m in matches]
)

# Shared tail of both _guided_candidates variants: radius gate, dependency
# push, IoU accept. @inline so each caller keeps its function-barrier shape.
#
# THE DEPENDENCY IS RECORDED BEFORE THE IoU GATE. Association is ONE-SHOT:
# nearest surviving detection, then accept/reject on IoU with NO fallback to
# the next-nearest. So a board's guided outcome depends on exactly `bestj`
# per projected blob: claiming anything FARTHER leaves bestj nearest (no
# change), and claiming bestj itself promotes the next-nearest, which may
# satisfy IoU where bestj failed. That last case is why the push precedes the
# gate — a dependency set covers what the computation READ, never what it
# ACCEPTED. Recording only IoU-satisfying csponds would make a blob whose nearest
# fails IoU look independent of that detection: the same class of error that
# killed the match-list re-admission filter (see _run_matching_phase!).
@inline function _guided_accept!(cands, on_read, ep, best2, bestj, det_ells, radius, min_iou, wi, t = nothing)
    bestj == 0 && return nothing
    sqrt(best2) <= radius || return nothing
    on_read === nothing || on_read(Int32(bestj))
    # overlap gate: the adapted detection must agree with THIS blob's
    # projected boundary in scale and shape, not just position. `t` carries the
    # comparison domain — nothing means the detection is already in it.
    de = t === nothing ? det_ells[bestj] : _guided_detell(t, det_ells, bestj)
    de === nothing && return nothing
    v = iou(ep, de)
    v >= min_iou || return nothing
    push!(cands, (v, wi, bestj))
    return nothing
end

# Function barrier for guided_search: `H` arrives concrete here, so the
# transfer + nearest-detection loop compiles allocation-free. Keep this a
# separate function — inlining it back into guided_search reintroduces the
# union-split boxing.
function _guided_candidates(
        H, wregions, detc, det_ells::AbstractVector,
        radius::Float64, min_iou::Float64, on_read = nothing
    )
    cands = Tuple{Float64, Int, Int}[]
    for (wi, wreg) in enumerate(wregions)
        tr = _guided_transfer(H, wreg)
        tr === nothing && continue
        ep, sc = tr
        best2 = Inf; bestj = 0
        @inbounds for j in eachindex(detc)
            dc = detc[j]
            d2 = (sc[1] - dc[1])^2 + (sc[2] - dc[2])^2
            d2 < best2 && (best2 = d2; bestj = j)
        end
        _guided_accept!(cands, on_read, ep, best2, bestj, det_ells, radius, min_iou, wi, H)
    end
    return cands
end

# R-tree-accelerated variant: O(log D) rect-query nearest per projected blob
# instead of the O(D) scan. The `± radius` query box is a superset of the
# radius disc, so the post-radius-gate output is IDENTICAL to the
# brute-force method above.
function _guided_candidates(
        rt::SpatialIndexing.RTree, H, wregions, detc, det_ells::AbstractVector,
        radius::Float64, min_iou::Float64, on_read = nothing
    )
    cands = Tuple{Float64, Int, Int}[]
    st = NearestNeighborState(rt)
    for (wi, wreg) in enumerate(wregions)
        tr = _guided_transfer(H, wreg)
        tr === nothing && continue
        ep, sc = tr
        best2, bestj = _rtree_nearest(rt, st, Float64(sc[1]), Float64(sc[2]), radius)
        _guided_accept!(cands, on_read, ep, best2, bestj, det_ells, radius, min_iou, wi, H)
    end
    return cands
end

# =============================================================================
# find_board — single-board matching + estimation
# =============================================================================

# =============================================================================
# Planar pose ambiguity — resolution (P3P arm)
# =============================================================================
#
# A coplanar board under weak perspective admits TWO pose basins (the normal
# reflected about the viewing ray); they tie on position consensus by
# construction, so RANSAC's winner is a coin flip in the ambiguous regime
# (measured on 260527: ~13% of small-board detections at 3 m). The fix needs
# no special estimator: reflect the winner, refine the reflection with the
# SAME seeded LO, and select on the post-LO likelihood ratio (see the selector
# in `_resolve_pose_ambiguity`).
#
# The ORIGINAL selector was a per-blob log-scale RSS — the imaged scale of a
# circular blob carries a √cos(α) factor that inverts between branches, so the
# scale field looked like the cue the position tie is blind to. Measured on
# 5028 GT-joined decisions it ran at 52.6%: a coin. It was kept for a while as
# a logged field (`rss_inc`/`rss_alt`) and removed 2026-08-17 along with the
# `rw`/`s_meas` arrays that fed it and the non-positive-scale gate that guarded
# them — that gate could refuse to adjudicate a board because of a σ that only
# ever reached telemetry.

# Branch telemetry (experiment instrumentation): when `telemetry.branch_log` is a path,
# _resolve_pose_ambiguity appends one JSON line per board with both branches'
# rotations, the post-LO score margin (the likelihood-ratio test), which branch
# won, and the discriminability d. The driver stamps `telemetry.tag` per frame.
#
# NO BOARD IDENTITY is logged, only the frame — so these records cannot be
# joined to committed poses per board. Every geometric proxy for that join
# fails, each for its own reason (measured 2026-08-17): 67.4% of records have
# their two branches <1 mm apart in translation, so translation cannot identify
# a BRANCH; coplanar boards share an orientation, so rotation cannot identify a
# BOARD; and there are ~30 records per committed pose (mostly rejected
# candidates clustered around real boards), so nearest-neighbour on any metric
# is contaminated. Two separate analyses drew false conclusions from proxy
# joins before this was understood. If per-board attribution is ever needed,
# log `batch[i].uid` (in scope at the `_claim_boards` call site) and join
# on (frame, uid) — do not reach for a tolerance.

# Appearance-gate telemetry (threshold calibration): when `telemetry.appearance_log`
# is a path, the candidate gate in `_claim_boards` appends one JSON line per
# gated candidate — uid, inlier count, appearance support, verdict — so a
# calibration run sees the margins of accepted AND rejected boards alike.
# Same idiom as the branch log; the frame tag is `telemetry.tag`.

# Phase-2 telemetry (does the permissive cleanup phase still pay for itself?):
# when `telemetry.phase2_log` is a path, `find_boards` appends one JSON line per call —
# frame tag, phase-1 count, phase-2 count, and the phase-2 uids. Zero rows are
# logged too: "phase 2 accepted nothing" is affirmative data for the
# delete-phase-2 decision, not absence of evidence. Tagged with `telemetry.tag`.

# Refit-skip telemetry: measures the UPPER BOUND on what a fit cache could save,
# without building one. When `telemetry.refit_skip_log` is a path, `_claim_boards` records each
# agenda entry's dependency set (`_AgendaEntry.deps`) and, after every commit,
# counts how many surviving entries read NO detection the commit claimed —
# those are the fits a sound cache could have skipped.
#
# Why measure before implementing: a quad-intersection version of this cache was
# built and measured on 2026-08-16 and skipped ZERO of 141 fits while running
# 0.742 s against 0.603 s for refit-all. The dependency-set predicate here is
# strictly tighter (two boards' quads can overlap while sharing no read
# detection), so it was worth asking whether tighter beats zero.
#
# IT DOES NOT. Measured 2026-08-17 on testit.jpg (50-board gallery, 4 boards in
# frame, the same 141 fits over 4 rounds the quad attempt used):
#
#     round  fitted  live  skippable   deps
#         1      50    49          0  23210
#         2      48    47          0   3493
#         3      30    29          2   1070
#         4      13    12          0    361
#
# 2 of 137 subsequent-round fits, 1.5% — an upper bound of 1.01x on the fit
# loop, before subtracting the tracking overhead that made the quad version a
# net LOSS. The first commit alone touches 23210 read detections; the committed
# board is large enough that essentially every other board on the agenda has
# read something inside it. FOUR attempts at this optimization have now failed
# (fit_cache, re-admission filter, quad intersection, dependency set), the last
# two for the same underlying reason.
#
# This instrumentation is kept precisely so attempt five costs one run instead
# of one cache. The real lever is the one the refit-all note names: 46 of the 50
# agenda entries are boards that are NOT IN THE FRAME, and they are fitted every
# round. That is an agenda-pruning problem, not a caching one.
#
# NOISE FLOOR, measured the same day and worth knowing before reading any A/B:
# five IDENTICAL find_boards calls on testit.jpg (CUDA, no telemetry) returned
# 1130 / 1146 / 1146 / 1142 / 1146 inliers — a 16-inlier spread, 1.4%. The board
# IDENTITY set was stable across all five (same 4 uids), so it is inlier counts
# that wobble, not decisions. Cause is adapt's CUDA path assigning pixel slots by
# atomics. Consequence: a single-run inlier delta under ~2% is not evidence, and
# an equality assertion on inlier counts across runs is the wrong check — assert
# on the uid set, or average over runs.

# An EMPTY PATH is "no sink" — a sentinel rather than `nothing`, so every field
# is concrete. `nothing` is still accepted at the keyword boundary below, since
# that is what a driver naturally supplies for an unset option.
# NOT `@kwdef`: the keyword constructor below normalizes `nothing` to the empty
# sentinel, and @kwdef's generated 5-arg positional constructor takes `::Any`
# arguments — defining our own alongside it is a method overwrite that fails
# precompilation. Writing both explicitly keeps the default positional
# constructor (String-typed, what `with_tag` uses) and one keyword entry point.
"""
    Telemetry(; tag = "", branch_log, appearance_log, phase2_log, refit_skip_log)

Experiment instrumentation for `find_boards`: four optional JSONL sinks (each
`nothing`, i.e. off, by default) and the frame `tag` every row carries. Supplied
as `find_boards(...; telemetry)` and threaded to the sites that log — until
2026-09 these were four global `Ref`s (`BRANCH_LOG[]`, `APPEARANCE_LOG[]`,
`PHASE2_LOG[]`, `REFIT_SKIP_LOG[]`) that a driver set through `BB_*`
environment variables, so a library call's side effects depended on process
state nothing recorded. Immutable: stamp the frame with [`with_tag`](@ref).

- `branch_log`: `_resolve_pose_ambiguity` appends one row per board with both
  branches, the post-LO likelihood-ratio margin, the winner and the
  discriminability `d` (plus one row per early-return arm).
- `appearance_log`: one row per appearance-gated candidate (uid, inlier count,
  support, verdict, the guided-only similarities).
- `phase2_log`: one row per `find_boards` call with the phase-1 / phase-2
  acceptance counts.
- `refit_skip_log`: per commit, the upper bound on what a fit cache could
  have skipped (see the measurement note above).
"""
struct Telemetry
    tag::String
    branch_log::String
    appearance_log::String
    phase2_log::String
    refit_skip_log::String
end

# Keyword boundary: `nothing` (an unset driver option) and any `AbstractString`
# both normalize to the empty-path sentinel, so callers need not know which
# spelling the struct uses.
_sink(::Nothing) = ""
_sink(s::AbstractString) = String(s)
Telemetry(;
    tag = "", branch_log = nothing, appearance_log = nothing,
    phase2_log = nothing, refit_skip_log = nothing
) = Telemetry(
    String(tag), _sink(branch_log), _sink(appearance_log),
    _sink(phase2_log), _sink(refit_skip_log)
)

"""
    with_tag(t::Telemetry, tag) -> Telemetry

The same sinks with the frame `tag` replaced — the per-frame stamp a driver
applies before each `find_boards` call.
"""
with_tag(t::Telemetry, tag::AbstractString) =
    Telemetry(String(tag), t.branch_log, t.appearance_log, t.phase2_log, t.refit_skip_log)

_refit_skip_log(t) = t === nothing ? "" : t.refit_skip_log

# Telemetry for the resolver's EARLY RETURNS — the arms that hand back a pose
# WITHOUT adjudicating it. They used to be silent, which made "the LRT examined
# this and chose" indistinguishable from "the resolver never looked".
# `reason` names the arm; the other fields are whatever that arm had to decide on.
#
# VERDICT (39321 calls on 260527, 2026-08-17): the bail arms are EXONERATED as a
# source of pose flips, and this instrumentation is what settled it.
#
#     slide_back              700
#     separation_below_gate    16
#     no_sources                0
#     nonpositive_scales        0   (gate since deleted)
#     incumbent_not_converged   0
#     alt_not_converged         0
#     -------------------------------
#     adjudicated           38605
#
# Four arms never fire at all. The two that do CANNOT produce a large flip, by
# their own gate conditions: separation_below_gate fires only when the branches
# are <4 deg apart, slide_back only when they are <1 deg apart, so choosing
# wrong costs at most 4 deg against a 30 deg flip threshold. Any future claim
# that flips come from an unadjudicated board has to get past that bound first.
#
# An earlier reading of this data reported unadjudicated boards flipping at 9.9%
# vs 0.77% — that was a JOIN ARTIFACT, not a bail. See the BRANCH_LOG comment.
# One writer for every telemetry channel: VGIo's append_jsonl (per-path
# lock registry + single-write append) replaced the per-channel locks.
_append_jsonl(path::String, row) = VisualGeometryIo.append_jsonl(path, row)

function _log_branch_bail(
        telemetry, reason; sep_deg = nothing, inc_conv = nothing,
        alt_conv = nothing, n = nothing
    )
    telemetry === nothing && return nothing
    path = telemetry.branch_log
    isempty(path) && return nothing
    return _append_jsonl(
        path, (
            tag = telemetry.tag, bail = String(reason),
            sep_deg = sep_deg, inc_converged = inc_conv,
            alt_converged = alt_conv, n_csponds = n,
        )
    )
end

# Reflect → refine → likelihood-ratio select. Returns the winning RansacEstimate
# (possibly the alternative branch) — the caller's downstream is unchanged.
function _resolve_pose_ambiguity(
        camera, scored, pt_csponds, res;
        outlier_halfwidth, local_optimization, extent, workspace, telemetry = nothing
    )
    bs = sources(scored)
    # Bare strip (audited): `sources` are board-frame blobs, unitless mm by the
    # :world contract, so the target-unit form is not merely unnecessary here —
    # it would THROW. `ustrip` is the defensive identity-or-strip. See the audit
    # in test/test_units_convention.jl.
    Xw = [
        SVector{3, Float64}(ustrip(b.center[1]), ustrip(b.center[2]), 0.0)
            for b in bs
    ]
    isempty(Xw) && (_log_branch_bail(telemetry, :no_sources); return res)

    # VGC mirror_pose is total; a near-frontal view returns ~the same pose —
    # gate on the branch separation (= twice the viewing tilt)
    θm = VisualGeometryCore.mirror(res.estimate, sum(Xw) / length(Xw))
    sep = rad2deg(Rotations.rotation_angle(Rotations.RotMatrix{3}(θm.R' * res.estimate.R)))
    sep > 4.0 ||
        (_log_branch_bail(telemetry, :separation_below_gate; sep_deg = sep, n = length(Xw)); return res)
    # SYMMETRIC LRT: both branches through the IDENTICAL seeded-LO call —
    # same scoring construction, same optimization treatment — so the
    # quality difference is a true log likelihood ratio. (Comparing
    # RANSAC's quality against locally_optimize's is a units error:
    # different scoring objects — measured to flip healthy boards.)
    inc = _refine(
        camera, pt_csponds, res.estimate;
        outlier_halfwidth, local_optimization, extent, workspace
    )
    if isnothing(inc) || !inc.converged
        # Returns the UNREFINED incumbent and never refines the mirror at all.
        # The incumbent's own refinement failing is arguably evidence AGAINST
        # it, so this arm looked like the prime suspect for the large flips —
        # but it fires ZERO times in 39321 calls (see _log_branch_bail), so
        # the argument is moot and there is nothing here to fix.
        _log_branch_bail(
            telemetry, :incumbent_not_converged; sep_deg = sep, inc_conv = false,
            n = length(Xw)
        )
        return res
    end
    alt = _refine(camera, pt_csponds, θm; outlier_halfwidth, local_optimization, extent, workspace)
    if isnothing(alt) || !alt.converged
        _log_branch_bail(
            telemetry, :alt_not_converged; sep_deg = sep, inc_conv = true,
            alt_conv = false, n = length(Xw)
        )
        return inc
    end
    # slide-back: the reflection refined into the incumbent's basin —
    # geometry unambiguous
    if Rotations.rotation_angle(Rotations.RotMatrix{3}(alt.estimate.R' * inc.estimate.R)) < deg2rad(1.0)
        _log_branch_bail(
            telemetry, :slide_back; sep_deg = sep, inc_conv = true, alt_conv = true,
            n = length(Xw)
        )
        return inc
    end

    # Selector: the post-LO likelihood-ratio test — the branches' marginal
    # scores at their optima (measured on 5028 GT-joined decisions: 97.4%
    # correct overall, 82.5% in the d<1 ambiguous core; the retired scale-RSS
    # selector measured 52.6% — a coin). See bmvc2026/scripts/branch_lrt_accuracy.jl.
    #
    # ONLY THE SIGN IS USED, deliberately. Gating the override on a margin
    # threshold was measured and is NET NEGATIVE (61 -> 64 large errors),
    # because "don't override" is not an abstention: it keeps the incumbent,
    # which at low margin is also a coin. There is no safe fallback inside a
    # single frame. The magnitude is not calibrated either — |margin| in
    # [0.5, 2) ran 49.5% correct where a true log-LR would predict 73-88% —
    # so do not convert it to a probability without fitting one. Separating
    # the branches when d is small needs information from OUTSIDE the frame
    # (temporal accumulation in `track_boards`), not a better rule here.
    margin = inc.quality - alt.quality

    blog = telemetry === nothing ? "" : telemetry.branch_log
    if !isempty(blog)
        # discriminability d: inter-branch residual gap over the noise floor
        prob = RobustVisualGeometry.AbsolutePoseProblem(camera, pt_csponds)
        n = length(pt_csponds)
        rA = zeros(n); rB = zeros(n)
        residuals!(rA, prob, inc.estimate)
        residuals!(rB, prob, alt.estimate)
        inl = inc.inlier_mask
        gap = sqrt(sum(i -> inl[i] ? abs2(rA[i] - rB[i]) : 0.0, 1:n))
        d = gap / max(inc.scale, 0.3)
        RA = inc.estimate.R; RB = alt.estimate.R
        _append_jsonl(
            blog, (
                tag = telemetry.tag,
                q_inc = inc.quality, q_alt = alt.quality,
                d = d, n_inl = sum(inl),
                R_inc = collect(vec(Matrix(RA))), R_alt = collect(vec(Matrix(RB))),
                t_inc = collect(inc.estimate.t),
                # WHICH BRANCH WON, and the alternative's translation.
                # Without these a consumer can only join on `t_inc`, so
                # every board the LRT OVERRODE looks like a board the
                # resolver never adjudicated. Necessary but NOT
                # sufficient: adding them did not make the join sound
                # (the residual "unadjudicated" bucket was still a join
                # failure — see the BRANCH_LOG comment), it only removed
                # one of the two ways it was wrong.
                chose = margin >= 0 ? "inc" : "alt",
                t_alt = collect(alt.estimate.t),
            )
        )
    end
    return margin >= 0 ? inc : alt
end

"""
    find_board(camera, board_features, image_features; kwargs...)

Match image features against a single board's features and estimate geometry.

Returns `(model, quality, scored)` or `nothing` if fewer than 4 matches;
the IRLS weights ride `scored` as its `weights` attribute column.
`model` is a `HomographyMat` or `Extrinsics3` depending on `camera`.
"""
function find_board(
        # Typed so a call in the retired order (`image_features, board_features,
        # camera`) is a MethodError: `camera` is `nothing`, a `CameraModel` or a
        # radial geometry, never a vector. (The camera slot stays untyped —
        # the geometry types are defined further down this file.)
        camera, board_features::AbstractVector, image_features::AbstractVector;
        min_similarity = _dflt("min_similarity"),
        # The ENGINE's guided-association overlap gate. 0.5, its tuned value —
        # This gate is what rejects confuser boards (see `guided_search`).
        # Moving it is a tuned-parameter change and wants its own measurement.
        min_iou = 0.5,
        dm = nothing,
        matches = nothing,
        bundle = nothing,
        guided_detc = nothing,
        guided_rtree = nothing,
        guided_det_ells = nothing,
        guided_wdiscs = nothing,
        guided_wregions = nothing,
        refine_workspace = nothing,
        resolve_pose_ambiguity::Bool = false,
        # Sink for the detection indices the GUIDED association read: called
        # `on_guided_read(j::Int32)` once per read (see `guided_search`). A
        # callback, not an out-vector: callers that track fit dependencies
        # collect them beside their own match set.
        on_guided_read = nothing,
        telemetry::Union{Nothing, Telemetry} = nothing,
        fit_kw = (;)
    )
    matches = @something matches match(board_features, image_features, MutualNN(); min_similarity, dm)
    length(matches) < 4 && return nothing

    # `det_id` column: see guided_search — claim bookkeeping needs the
    # image-detection index of every cspond on BOTH return paths.
    scored = _scored_with_ids(board_features, image_features, matches)

    # Lazy endpoint lift (#21): homography path fits on centroids; pose path
    # z=0-lifts the board-frame centroids to world points. The score (and
    # ratio) attribute columns ride the lift, so RVG's problem constructors
    # pick up the PROSAC prior straight from the `:score` column — no
    # sidecar `scores` kwarg.
    pt_csponds = _fit_endpoints(camera, scored)

    res = _find_model(camera, pt_csponds; fit_kw...)
    (!isnothing(res) && res.converged) || return nothing
    # The refines below use the SAME strategy the fit used.
    local_optimization = get(fit_kw, :local_optimization, _default_lo(camera))

    # Planar pose ambiguity (P3P arm; see _resolve_pose_ambiguity): reflect the
    # RANSAC winner, refine the reflection with the same seeded LO, select by
    # the post-LO likelihood ratio. The guided stage downstream is
    # branch-agnostic — it polishes whichever basin wins here.
    if resolve_pose_ambiguity && camera isa CameraModel
        res = _resolve_pose_ambiguity(
            camera, scored, pt_csponds, res;
            outlier_halfwidth = get(fit_kw, :outlier_halfwidth, 5.0), local_optimization,
            extent = get(fit_kw, :extent, nothing),
            workspace = refine_workspace, telemetry
        )
    end

    # Guided final fit: project the full board pattern through the RANSAC
    # model, re-associate every blob to its nearest detection, and refit with
    # posterior-weighted IRLS. Recovers pattern blobs the descriptor matcher
    # missed; falls back to the RANSAC fit if the guided set is too small or
    # the refit fails to converge. Drop `workspace` from fit_kw — the guided
    # set can exceed the descriptor-match count the workspace was sized for.
    if !isnothing(bundle) && _guided_enabled(camera)
        # association band = estimator outlier_halfwidth + per-blob IoU gate
        # at the bundle's rendered-boundary cutoff (fallback matches the
        # PoseGeometry/HomographyGeometry default of 5 px)
        outlier_halfwidth = get(fit_kw, :outlier_halfwidth, 5.0)
        # RANSAC inliers are already certified (descriptor match + spatial
        # consensus): guided search keeps them unconditionally and only ADDS
        # IoU-certified csponds — the union, per-board support can't shrink.
        w = res.weights
        keep = [m.value for (k, m) in enumerate(matches) if is_map_inlier(w[k])]
        # Compose the world->image map HERE, so guided_search receives one
        # concrete HomographyMap instead of a Union to defend against.
        # The world discs the guided search runs on, resolved HERE so the
        # table returned below has the same sources the guided fit used.
        wd = @something guided_wdiscs blobs(bundle, :world)
        g_matches = NNMatch[]
        g_scored = guided_search(
            _model_to_homography(res.estimate, camera), bundle, image_features;
            max_px = outlier_halfwidth, min_iou, keep, detc = guided_detc, rtree = guided_rtree,
            det_ells = guided_det_ells, wdiscs = wd,
            wregions = guided_wregions, on_read = on_guided_read, matches_out = g_matches
        )
        if length(g_scored) >= 4
            g_pt = _fit_endpoints(camera, g_scored)
            # Warm LM-guarded local optimization from res.estimate — no second RANSAC.
            res2 = _refine(
                camera, g_pt, res.estimate;
                outlier_halfwidth, local_optimization, extent = get(fit_kw, :extent, nothing),
                workspace = refine_workspace
            )
            if !isnothing(res2) && res2.converged
                # EVERY correspondence the board was fitted on comes back,
                # weighted: the guided union with the refit's posterior, and
                # the descriptor matches the arm's own band rejected with the
                # RANSAC's (all ≤ ½). Inliers are `is_map_inlier(weight)` on both return
                # paths, so a consumer that reads inliers sees no difference;
                # one that can re-judge a rejected cspond under another model
                # has them.
                rej = findall(!is_map_inlier, w)
                full = withattrs(
                    _scored_with_ids(wd, image_features, vcat(g_matches, matches[rej]));
                    weights = vcat(res2.weights, w[rej])
                )
                return (res2.estimate, res2.quality, full)
            end
        end
    end

    # The IRLS weights join the scored set as an attribute column (#21): they
    # subset in lockstep with the correspondences from here on — no parallel
    # weights vector to keep aligned by hand. Every match comes back, the
    # rejected ones at their (≤ ½) weight.
    full = withattrs(scored; weights = res.weights)
    return (res.estimate, res.quality, full)
end

# NNMatch is used for the guided-search csponds below and rides the agenda's
# `resolved` field. The stacked gallery matcher that used to build these
# dictionaries here now lives in VGF (`nn_match_stacked`).
using VisualGeometryFeatures: NNMatch

# The gallery, concatenated ONCE: the board set is ONE gallery, not 51.
#
# `nn_match_stacked(strategy, galleries::AbstractDict, features)` is documented
# as "per gallery the result is identical to matching that gallery alone —
# MutualNN reverse checks are BLOCK-LOCAL". That is the right contract for
# matching against N independent galleries, and it is not what this pipeline
# wants: a detection belongs to exactly ONE board, so mutuality has to hold over
# the WHOLE board set. Supplying 51 galleries asks 51 independent questions, and a
# detection can win all of them — measured on one 12 MP fisheye image, 629
# detections were claimed by more than one board and 40% of all matches (1685 of
# 4209) were duplicate assignments, which is what fabricated the 29 junk boards
# that sat at the `min_inliers` floor and then wrecked phase 2's shared fit.
#
# So ask the question BB actually has: one gallery, global mutual-NN. Row `g` of
# the stack maps back to (board, local blob) through the prefix sums, which is
# the same layout `StackedDescriptorIndex` builds internally, so nothing is
# recomputed — the stack is concatenated once per gallery and reused for every
# image, and the reverse index the matcher allocates shrinks from
# `N_feat × N_gal` to `N_feat × 1`.
#
# STACKED, not "union": the ORDER is the data structure here. `offsets` is only
# meaningful against a definite ordering of the boards (`uids` is sorted), which
# is the opposite of what a set union connotes — and `Union` already means
# something else entirely in Julia. The name also matches the device half it
# feeds, VGF's `StackedDescriptorIndex` / `nn_match_stacked`.
const _STACK_KEY = :__stack__

struct StackedGallery{F, K}
    features::F            # every board's blobs, concatenated in `uids` order
    uids::Vector{K}
    offsets::Vector{Int}   # 0-based prefix sums; board k owns rows offsets[k]+1 : offsets[k+1]
end

function StackedGallery(board_feature_dict::AbstractDict)
    uids = sort!(collect(keys(board_feature_dict)))
    offsets = Vector{Int}(undef, length(uids) + 1)
    offsets[1] = 0
    for (k, uid) in enumerate(uids)
        offsets[k + 1] = offsets[k] + length(board_feature_dict[uid])
    end
    # A COMPREHENSION, not a generator: Base's allocate-once
    # `reduce(vcat, ::AbstractVector{<:AbstractVector})` matches a vector and
    # not a `Base.Generator`, so a generator falls back to `mapfoldl` — one
    # full copy of a growing ~85 MB array per board. `init` covers the empty
    # gallery, which the engine treats as ordinary (`isempty(gallery_pool) &&
    # return feat_pool`) and which `StackedDescriptorIndex` handled.
    blocks = [board_feature_dict[uid] for uid in uids]
    isempty(blocks) && return StackedGallery(valtype(board_feature_dict)[], uids, offsets)
    return StackedGallery(reduce(vcat, blocks), uids, offsets)
end

# Row of the stack → (board position, board-local blob index). One branchless
# binary search per match; `offsets` is sorted by construction.
@inline function _split_row(sg::StackedGallery, g::Integer)
    k = searchsortedlast(sg.offsets, g - 1)
    return k, g - sg.offsets[k]
end

"""
    GalleryIndex(device, board_feature_dict)

The board gallery staged on `device` as ONE block.

The board set is one gallery, so there is one staging and one matching path
(see `match_boards`). `stacked` is the host half (`StackedGallery`):
the concatenated features and the prefix sums that map a row of the block back
to `(board, local blob)`. `stacked_index` is the device half, VGF's
`StackedDescriptorIndex` over those same descriptors. Built once per
`find_boards` call and threaded as `gpu_index`.

It used to stage the descriptors TWICE, adding a per-board blocking for
`KnnMatch`. That was ~85 MB of duplicate device memory for nothing: in VGF's
stacked matcher the blocking (`offsets`/`sizes`) reaches exactly one kernel,
the REVERSE reduction `_ka_top1_cols_blocked!`. The forward reduction runs per
gallery row over all features, and `KnnMatch`'s `_ka_top4_rows!` likewise —
"the top-4 reduction is per row and there is no column reduction to keep
block-local". So only `MutualNN` can see the blocking at all, and for it the
per-board answer is the wrong question (below).
"""
struct GalleryIndex{G <: StackedGallery, I}
    stacked::G
    stacked_index::I
end

function GalleryIndex(device, board_feature_dict::AbstractDict)
    sg = StackedGallery(board_feature_dict)
    return GalleryIndex(sg, StackedDescriptorIndex(device, Dict(_STACK_KEY => sg.features)))
end

"""
    match_boards(strategy, galleries, features; index, backend, min_similarity)
        -> Dict{uid, Vector{NNMatch}}

Descriptor matching of one image's `features` against a board gallery
(`uid => board features`), keyed per board: one entry per gallery key, empty
vectors included. `index` is a [`GalleryIndex`](@ref) built once over the
board set and reused across images.

ONE path, every strategy: the board set is matched as ONE gallery and the
rows are split back. There is no per-board matching mode, because the
per-board question is the wrong one. `nn_match_stacked`'s reverse test is
BLOCK-LOCAL — documented as "per gallery the result is identical to matching
that gallery alone" — which is the right contract for N INDEPENDENT galleries
and not what this pipeline asks: a detection belongs to exactly one board, so
mutuality has to hold over the whole board set. Asked per board it is N
independent questions and one detection can win several — measured on one
12 MP fisheye frame, 629 detections were claimed by more than one board, 40%
of all matches were duplicates (1685 of 4209), and they fabricated 33 boards
sitting at the `min_inliers` floor with grossly wrong poses. Through the stack
the same frame gives 2496 matches, 2496 unique, zero duplicates.

`KnnMatch` used to take a second, per-board staging on the grounds that it has
no reverse test to correct. It does not need one: its reduction
(`_ka_top4_rows!`) is per gallery ROW over all features, so it returns the
same matches under either blocking, and the per-board staging was only a
duplicate copy of the same descriptors (see [`GalleryIndex`](@ref)).

`galleries` selects which boards may be returned — the engine's pool SHRINKS as
boards are accepted, while the index stays over the full board set. A detection
whose global best belongs to an accepted board is therefore dropped rather than
handed to its runner-up, which is the one-detection-one-board rule the claiming
enforces anyway.
"""
function match_boards(
        strategy, galleries::AbstractDict, features::AbstractVector;
        index::Union{Nothing, GalleryIndex} = nothing, backend, min_similarity,
    )
    # Without a prebuilt index the stack is concatenated here and staged by
    # the matcher itself; the path is otherwise identical.
    sg = index === nothing ? StackedGallery(galleries) : index.stacked
    md = nn_match_stacked(
        strategy, Dict(_STACK_KEY => sg.features), features;
        backend, index = index === nothing ? nothing : index.stacked_index, min_similarity
    )
    return _split_stacked_matches(sg, md, galleries)
end

# Function barrier: `sg`'s type is decided by the branch above, and the loop
# below runs once per match (thousands), so it must not see a small Union.
function _split_stacked_matches(sg::StackedGallery, md, galleries::AbstractDict)
    # ONE ENTRY PER GALLERY KEY, empty vectors included — `nn_match_stacked`'s
    # contract, which the agenda and the serial oracle both rely on: they index
    # `match_dict[uid]` for every uid they were built from. Global mutual-NN
    # makes the empty case the COMMON one (a 51-board gallery on a frame
    # showing five boards leaves ~46 with nothing), so returning only the
    # boards that matched turns an ordinary frame into a `KeyError`.
    out = Dict{keytype(galleries), Vector{NNMatch}}(uid => NNMatch[] for uid in keys(galleries))
    all_matches = get(md, _STACK_KEY, nothing)
    all_matches === nothing && return out
    for m in all_matches
        k, local_i = _split_row(sg, m.value.first)
        board = get(out, sg.uids[k], nothing)
        board === nothing && continue                # not in this pool (accepted, or another polarity)
        push!(board, Attributed(local_i => m.value.second, m.attributes))
    end
    return out
end

# =============================================================================
# FindBoardsConfig — configuration for find_boards pipeline
#
# (DetectConfig + Mxm/EllipseNms live in src/detect_config.jl, loaded
#  before this file so detect.jl can reference them in signatures.)
# =============================================================================

"""
    GeometryConfig

Abstract type for per-estimator geometry knobs (RANSAC outlier half-width,
minimum inlier count, and the local-optimization strategy). Concrete subtypes: `HomographyGeometry`,
`PoseGeometry`, `RadialHomographyGeometry`. Held by `FindBoardsConfig.geometry` — the type chooses the
homography-vs-P3P path inside `find_boards`.
"""
abstract type GeometryConfig end

"""
    HomographyGeometry(; outlier_halfwidth=5.0, min_inliers=7, local_optimization=StudentTUniform())

Homography (no-camera) estimator config. 4 is the geometric minimum for
`find_homography`; the default 7 leaves three outlier-tolerated points above
the minimal sample — homography is 2 DOF looser than P3P, so it needs more
support to reject coincidental 4-point matches against a large board gallery.
(The gate is INCLUSIVE — `nin >= min_inliers` — since 2026-08-11; the default
went 6 → 7 in the same change, so the effective threshold the pipeline was
tuned under is unchanged.) Half-width is tighter than RVG's default (20)
since BB residuals come from adapt-refined sub-pixel detections.

There is no `min_inlier_fraction` anymore. It gated on the fraction of the
correspondence set with posterior weight above ½ (`is_map_inlier`) — a geometric proxy against
dense low-quality match clouds (guided sets of 400+ csponds carrying 6
accidental inliers satisfy any absolute count). The mandatory appearance gate
(`FindBoardsConfig.verify_similarity` / `min_appearance_support`, see
verify.jl) rejects that attack directly and with total measured separation,
so the proxy was removed (2026-08-13) rather than kept as a second, weaker
knob on the same failure mode.
"""
Base.@kwdef struct HomographyGeometry{L} <: GeometryConfig
    outlier_halfwidth::Float64 = _dflt("geometry", "outlier_halfwidth")
    min_inliers::Int = _dflt("geometry", "min_inliers")
    # The RANSAC local-optimization strategy (an RVG `AbstractLocalOptimization`),
    # selected by tag in the file: `local_optimization = { type = "student_t_uniform" }`.
    # A type parameter, not an abstract field: the per-board fit loop reads it.
    local_optimization::L = _dflt_local_optimization(:homography)
end

"""
    PoseGeometry(camera; outlier_halfwidth=5.0, min_inliers=6, local_optimization=StudentTUniform())

Calibrated P3P estimator config. Holds the `camera` (a `CameraModel` carrying
the lens/distortion model used to undistort the image). The camera lives here
— not as a separate `find_boards` argument — so calibrated-ness is encoded in
one place and illegal camera/geometry mismatches are unrepresentable.

`undistortion` selects where the pose is measured: `:none` (the default)
detects, adapts and describes on the RAW image and the estimator verifies
through the FULL camera — RANSAC scoring, the Student-t M-step and the
Welsch lift all measure the reprojection through the lens in the pixels the
detector produced (RobustVisualGeometry ≥ 0.19.9), and guided association
transfers each blob's conic through the plane homography exactly and the
lens's affinity at its centre (`transfer_ellipse(::Camera, _)`). `:global`
undistorts the IMAGE (VGC's fused warp) and runs the whole pipeline on the
result through the pinhole twin — the historical path, and what the BMVC
runs used (their configs state it). Why `:none`: a residual in the
undistorted plane is the detection error magnified by the lens's inverse
slope — 84× at the rim of an 8 mm fisheye — so a 5 px gate there rejected
every rim point; in raw pixels the gate means what it says everywhere.

3 is the geometric minimum for `p3p`; the default 6 leaves three
outlier-tolerated points above the minimal sample. P3P is 2 DOF tighter than
homography, so it needs one fewer support point than the homography path
(min_inliers=7). (The gate is INCLUSIVE — `nin >= min_inliers` — since
2026-08-11; the default went 5 → 6 in the same change, keeping the effective
tuned threshold: a 5-inlier P3P fit against a 50-board gallery measured as a
false accept on DSC02847.) Half-width is 5.0 px — the same inlier band as the
homography path — since BB residuals come from adapt-refined sub-pixel
detections.
"""
struct PoseGeometry{C, L} <: GeometryConfig
    camera::C
    outlier_halfwidth::Float64
    min_inliers::Int
    # Planar pose ambiguity — the two-fold ambiguity of pose from a coplanar
    # target (Schweighofer & Pinz 2006). See _resolve_pose_ambiguity.
    resolve_pose_ambiguity::Bool
    # The RANSAC local-optimization strategy; see HomographyGeometry.
    local_optimization::L
    # Where the pose is measured: :none (raw detection, the full camera in the
    # estimator) or :global (undistort the image, pinhole twin). See the docstring.
    undistortion::Symbol
end
const _UNDISTORTION_POLICIES = (:none, :global)
# DELIBERATE literals, not _dflt: the pose arm is tuned ONE BELOW the
# homography arm's `min_inliers` (a calibrated P3P fit is far more constrained
# than an 8-dof H, so fewer inliers carry the same evidence). It has moved in
# lockstep with every gate change — 5→6 alongside homography's 6→7 when the
# count gate became inclusive (3642fc4, 2026-08-11) — and default.toml's
# [find_boards.geometry] table is homography-scoped by its own header, so
# there is no file entry for it to read. Named as an exception in
# config_defaults.jl.
function PoseGeometry(
        camera; outlier_halfwidth::Float64 = 5.0, min_inliers::Int = 6,
        resolve_pose_ambiguity::Bool = false,
        local_optimization = _dflt_local_optimization(:absolute_pose),
        undistortion::Symbol = :none,
    )
    undistortion in _UNDISTORTION_POLICIES || throw(
        ArgumentError("PoseGeometry: undistortion must be one of $(_UNDISTORTION_POLICIES), got :$undistortion")
    )
    return PoseGeometry(camera, outlier_halfwidth, min_inliers, resolve_pose_ambiguity, local_optimization, undistortion)
end

"""
    RadialHomographyGeometry(image_size; outlier_halfwidth=5.0, min_inliers=7, local_optimization=StudentTUniform())

Uncalibrated radially distorted homography config — the one-sided
division-model `(H, k)` path (RVG `find_radial_homography`; Wadenbäck et
al., 3DV 2026). For captures with an UNKNOWN camera whose lens bends
straight lines: the distortion coefficient is estimated jointly with H per
fit — no calibration, no image undistortion (the detector runs on the raw
image, exactly like the plain homography arm).

`image_size = (width, height)` is DATA, not configuration — the sensor
extent that fixes the distortion centre (the image middle), which the arm
hands its estimator as `center` — so like `PoseGeometry`'s camera it lives in
the geometry and arrives as a `FindBoardsConfig` keyword, never as a TOML value.

`min_inliers` defaults to the homography arm's 7: the model has one extra
DOF, and the count gate stays inclusive (`nin >= min_inliers`).

v1 scope: the guided final fit and `track_boards` are NOT wired for this
arm (both project through a single `HomographyMap`, which cannot express
the distortion); the RANSAC+LO fit is the result.
"""
struct RadialHomographyGeometry{L} <: GeometryConfig
    image_size::Tuple{Int, Int}
    outlier_halfwidth::Float64
    min_inliers::Int
    # The RANSAC local-optimization strategy; see HomographyGeometry.
    local_optimization::L
end
RadialHomographyGeometry(
    image_size;
    outlier_halfwidth::Float64 = _dflt("geometry", "outlier_halfwidth"),
    min_inliers::Int = _dflt("geometry", "min_inliers"),
    local_optimization = _dflt_local_optimization(:radial_homography)
) =
    RadialHomographyGeometry(
    (Int(image_size[1]), Int(image_size[2])), outlier_halfwidth, min_inliers, local_optimization
)

"""
    BoardDetection

Per-frame detection of a single board: estimated model (Extrinsics3 or HomographyMat),
the world→image homography for ellipse transfer, and how the frame was won
(`:detect`, `:init`, or `:track`).
"""
struct BoardDetection{M}
    board::String
    frame::Int
    # The arm's OWN estimate, and the only geometry stored: an `Extrinsics3`
    # pose (P3P + lifted LO), a `HomographyMat` or a `RadialHomography` — or
    # whatever model a caller's own arm estimates. Every map a consumer needs
    # is derived from it (`_model_to_homography` for projection). Caching a
    # derived map beside the model lets the two disagree.
    model::M
    quality::Float64
    n_inliers::Int
    source::Symbol       # :detect | :init | :track
end

# Camera lives inside PoseGeometry; HomographyGeometry is the no-camera path.
_camera(g::PoseGeometry) = g.camera
_camera(::HomographyGeometry) = nothing
# The geometry itself is the token: uncalibrated, but carrying the image
# size the estimator needs — no separate context type.
_camera(g::RadialHomographyGeometry) = g

# The camera the ESTIMATOR and every projection of a fitted model — claim
# quads, guided search, feature claiming — share: the one derivation for the
# six call sites (pipeline ×4, track_boards, annotate_blobs). It follows the
# pose arm's `undistortion` policy: under `:none` the detections are raw
# pixels and the full camera is the model; under `:global` the pipeline runs
# on the undistorted image and the pinhole twin is the model. Mixing them
# misplaces every quad and under-removes features, which lets phase 2
# re-match spurious overlapping boards.
_estimation_camera(::Nothing) = nothing
_estimation_camera(::HomographyGeometry) = nothing
_estimation_camera(g::PoseGeometry) = g.undistortion === :global ? pinhole(g.camera) : g.camera
# Radial arms: the geometry itself is the token — it must SURVIVE this call:
# the engine hands the one `cam` both to `find_board` (estimator dispatch)
# and to `board_quad` (projection). Collapsing it to `nothing` silently
# downgraded the radial arm to the plain homography path (caught on the
# first fisheye run, 2026-08-30).
_estimation_camera(g::RadialHomographyGeometry) = g

# --- The radial homography arm (dispatch on the geometry token) -------------
#
# The arm estimates its own model, the one-sided division homography (H, k), k
# free per fit; the guided final fit transfers through its plain homography.
_default_lo(g::RadialHomographyGeometry) = g.local_optimization
_fit_endpoints(::RadialHomographyGeometry, scored) = _board_exact(scored)   # the board exact, as for H
_guided_enabled(::RadialHomographyGeometry) = true
# Detection runs on the RAW image unless an arm says otherwise — the
# homography arm has no camera, and the radial arm's distortion is part of the
# estimated model. (The pose arm's own method is below.) Declared on the
# abstract type so a `GeometryConfig` defined outside this package — an arm
# whose model `find_board` does not estimate, reached through
# `prepare_image_features` for its features — also detects on the raw image.
_undistort_detect(::KA.Backend, img, ::GeometryConfig) = img
board_quad(::RadialHomographyGeometry, m::HomographyMat, bundle::BoardBundle) = board_quad(nothing, m, bundle)
board_quad(::RadialHomographyGeometry, m::VisualGeometryCore.RadialHomography, bundle::BoardBundle) =
    board_quad(nothing, m, bundle)

# Radially distorted homography: the one-sided division model (H, k), k free
# per fit (RVG `find_radial_homography`; Wadenbäck et al., 3DV 2026).
#
# `pt_csponds` arrives from `_fit_endpoints` already in point form, so it is
# not re-converted.
_find_model(
    g::RadialHomographyGeometry, pt_csponds; local_optimization = g.local_optimization,
    outlier_halfwidth = _FIT_HALFWIDTH_FALLBACK.radial_homography, kw...
) = find_radial_homography(
    pt_csponds; center = _distortion_center(g),
    scoring = _scoring(_RadialProblem5, outlier_halfwidth), local_optimization, kw...
)

# The arm's problem type: the radial 5-point kernel's (`find_radial_homography`'s
# default `minimal = :radial`, minimal sample 5) — the scorer reads its minimal
# sample size and codimension off it.
const _RadialProblem5 = RadialHomographyProblem{Float64, 5}

# The radial arm's distortion centre: the middle of the sensor the pixels were
# measured in (its `image_size`), in those pixels' COLMAP convention.
_distortion_center(g) = VisualGeometryCore.Point2{Float64}(g.image_size[1] / 2, g.image_size[2] / 2)

_refine(g::RadialHomographyGeometry, pt_csponds, θ₀::HomographyMat; local_optimization = g.local_optimization, kw...) =
    _refine(nothing, pt_csponds, θ₀; local_optimization, kw...)
# The radial homography is its own manifold (sl(3) ⊕ k) — polish it there, not
# through a plain homography.
_refine(
    g::RadialHomographyGeometry, pt_csponds, θ₀::VisualGeometryCore.RadialHomography;
    outlier_halfwidth, local_optimization = g.local_optimization,
    extent = nothing, workspace = nothing,
) =
    locally_optimize(
    _bounded(RadialHomographyProblem(pt_csponds; center = _distortion_center(g)), extent), θ₀;
    scoring = _scoring(_RadialProblem5, outlier_halfwidth), workspace, local_optimization
)

# The pose arm: undistort the image under :global, detect raw under :none
# (the estimator then verifies through the full camera).
_undistort_detect(dev::KA.Backend, img, g::PoseGeometry) =
    g.undistortion === :global ? _undistort_detect(dev, img, g.camera) : img

# Per-fit RANSAC stream: the config's base seed XOR the board's uid (3 hex
# chars of its SHA-256 config hash — `uid_from_hash` — unique within any
# working gallery, since it IS the gallery key). RVG routes every sampler
# draw through this rng (`rng` kwarg, RVG ≥ 0.15.7), so a fit's result
# depends only on its INPUTS — not on thread scheduling, call order, or
# ambient RNG state. The same derivation serves the engine, the serial
# oracle, and the warm tracker, which keeps engine ≡ oracle exact. NB CUDA
# feature extraction remains nondeterministic (adapt's atomic pixel-slot
# assignment): end-to-end runs are reproducible GIVEN features; the fit
# stage is bit-reproducible.
"""
    fit_rng(seed, uid) -> Xoshiro

The RANSAC stream of one board's fit: `Xoshiro(seed ⊻ uid)`, the board's uid
read as hexadecimal. `find_boards`, `find_board`'s callers and `track_boards`
all draw from it with `seed = config.ransac_seed`, so a fit's result depends
only on its inputs; a caller that fits boards itself and wants its fits
comparable with the engine's derives its streams here.
"""
fit_rng(seed::Integer, uid::AbstractString) =
    Xoshiro(reinterpret(UInt64, Int64(seed)) ⊻ parse(UInt64, uid; base = 16))

# One board's `fit_kw` outside the threaded engine (which adds its per-thread
# workspace to the same fields): the serial oracle builds it here, so its fits
# stay comparable with the engine's by construction.
_board_fit_kw(config, uid, bundle) = (;
    outlier_halfwidth = config.geometry.outlier_halfwidth,
    local_optimization = config.geometry.local_optimization,
    config = config.ransac_config,
    extent = Rect(bundle, :world),
    rng = fit_rng(config.ransac_seed, uid),
)

"""
    FindBoardsConfig(; kwargs...)

Configuration for `find_boards` pipeline. The `geometry` field
(`HomographyGeometry` or `PoseGeometry`) groups the per-estimator knobs
(outlier half-width, min inliers) and determines which path runs.

# Detection
- `detect::DetectConfig = DetectConfig()`: Detection configuration (single
  source of truth for `detect_blobs` parameters across all backends).
- `max_detections=typemax(Int)`: post-detection response-rank cap. On the
  grouped image path it is applied PER POLARITY GROUP (after
  `_polarity_subset`), so a mixed dark+light board set can process up to
  2× this many detections in total — see `_detect_for_boards` for why the
  shared detection stage must stay uncapped. Single-polarity sets (every
  real set) behave exactly like a `detect_blobs(; max_detections=…)` cap.

# Adaptation (blob fitting)
- `adapt::AdaptConfig = AdaptConfig()`: Adaptation configuration

# Geometry estimator
- `geometry::GeometryConfig = HomographyGeometry()`: per-estimator
  RANSAC + inlier-gate knobs. Use `HomographyGeometry()` for the no-camera
  path or `PoseGeometry(camera)` (the camera lives in the geometry) for the
  calibrated P3P path.

# Matching
- `min_similarity=0.6`: Minimum descriptor cosine similarity. The derived
  gates move with it: phase 2's permissive gate (`_phase2_similarity`,
  0.1 below), which is not configured separately so it cannot drift apart
  under a sweep. The agenda counts every resolved match — there is no second
  similarity bar (see `_n_resolved_matches`).

# Appearance verification (mandatory)
- `verify_similarity=0.6`: an inlier correspondence "agrees in appearance"
  when its gallery and image descriptors reach this cosine similarity.
  Deliberately NOT named `min_similarity` — a different quantity at a
  different stage that happens to share the value (see `appearance_support`
  in verify.jl).
- `min_appearance_support=0.5`: minimum fraction of a candidate board's
  inliers that must agree in appearance for the board to be accepted.

Geometry cannot make this call. Guided association is descriptor-blind, and
at measured detection densities (13,844/MP, 5 px band) 66% of arbitrary
positions have SOME detection in range — a fabricated board fills its inlier
set geometrically, so counts and fractions satisfy it. The inliers' APPEARANCE
separates real from fabricated with zero measured overlap (real ≥ 0.993
support, false = 0.0 at a 0.8 sim gate; see verify.jl). This gate replaced
the geometry structs' `min_inlier_fraction` (a proxy for the same attack,
removed 2026-08-13).

`verify_similarity` was calibrated DOWN twice on the 260527 corpus
(2026-08-14), each time because the real population's frontier sat under
the gate while false boards never came near it (median 0.07–0.37):
  0.8 → 0.7  on the strided 12-frame slice — distant real boards carry
             median inlier sims 0.73–0.80; 0.8 rejected 8 GT-verified
             real boards.
  0.7 → 0.6  on the FULL corpus — the inter-slice 2–3 m frames the stride
             never sampled dip to median sims 0.66–0.71, so 0.7 gated 13
             more real csponds (probe: support 0.30–0.47, deterministic).
At the default support of 0.5 the gate is exactly a MEDIAN test, so the
bar must sit under the real population's frontier median. NB appearance is
structurally blind to POSE FLIPS (correct correspondences, mirrored
rotation): that is `resolve_pose_ambiguity`'s job, not this gate's — and
loosening this bar re-admits flipped poses along with good ones.

# Reproducibility
- `ransac_seed=0x0b10b5eed`: base seed for the per-fit RANSAC streams
  (`Xoshiro(ransac_seed ⊻ board uid)` — see `fit_rng`). Identical inputs
  give identical results, independent of thread scheduling, call order, and
  ambient RNG state; sweep it only to MEASURE fit-stage RNG sensitivity.
  CUDA feature extraction remains nondeterministic (adapt's atomic pixel
  slots), so end-to-end reproducibility is "given features".

Every field is CONCRETE by design: this config is read inside the per-board
fitting loop, so an abstract field (`adapt::AdaptConfig`, `geometry`) would
force boxing and dynamic dispatch on every access. Keep it that way when
adding fields.

# Overriding one knob

The struct is immutable and `find_boards` has no per-knob keyword — you supply
a whole config. To change one setting, use `Accessors.@set`, which reaches
nested fields and rebuilds only what changed:

`Accessors` is NOT a dependency of BlobBoards — add it to your own project
(`pkg> add Accessors`). No library code is needed for this to work:
ConstructionBase's generic `setproperties` already handles these structs.

```julia
using Accessors                       # your dep, not BlobBoards'
cfg = FindBoardsConfig("configs/default.toml"; camera)
cfg = @set cfg.detect.peak_threshold = 0.02
cfg = @set cfg.detect.upsample = false
```

This works through ConstructionBase's generic `setproperties`, which is
correct here only because every one of these structs is a plain `@kwdef`
with no inner constructor — so its positional constructor matches its field
order. Adding a validating inner constructor to `FindBoardsConfig`,
`DetectConfig`, `AdaptConfig` or a `GeometryConfig` silently breaks the
override idiom; `test/test_config_toml.jl` gates that, including the
type-changing case (`@set cfg.detect.nms = EllipseNms(...)`, which alters a
type parameter).

Sweeps should override rather than re-list: a hand-built `FindBoardsConfig`
in a driver duplicates the knobs of the config of record, and the two drift.
See `bmvc2026/scripts/detection_thresholds_cv.jl`.
"""
Base.@kwdef struct FindBoardsConfig{G <: GeometryConfig, N}
    # Detection
    detect::DetectConfig{N} = DetectConfig()
    # BB-side post-detection cap (forwarded to `detect_blobs(; max_detections=…)`
    # via `extract_image_features`). Default `typemax(Int)` ⇒ no cap.
    max_detections::Int = _dflt_max_detections()
    # Adaptation
    adapt::AdaptConfig = AdaptConfig()
    # Geometry estimator — picks homography vs P3P path via dispatch.
    geometry::G = HomographyGeometry()
    # RANSAC trial bounds + success probability — forwarded straight to
    # `find_homography` / `find_absolute_pose` via the `config=` kwarg.
    # min_trials = 100 floors the adaptive stop; the 10_000 ceiling was
    # raised from 2_000 deliberately in 6fa79a8. The agenda's match-count floor
    # prunes most hopeless boards before a single trial, so the ceiling is
    # paid only by plausible candidates where adaptive stopping usually
    # fires in ~100 trials anyway.
    ransac_config::RansacConfig = RansacConfig(
        min_trials = _dflt("ransac", "min_trials"),
        max_trials = _dflt("ransac", "max_trials"),
        success_prob = _dflt("ransac", "success_prob")
    )
    # Base seed for the per-fit RANSAC streams: every board's fit draws from
    # `Xoshiro(ransac_seed ⊻ uid)` (see `fit_rng`), so identical inputs give
    # identical results regardless of thread scheduling, call order, or
    # ambient RNG state. Recorded like every other knob so a run's sidecar
    # names the streams it used; sweep it to MEASURE fit-stage RNG
    # sensitivity, never to tune (decisions were seed-stable across seeds on
    # the 260527 A/B, 2026-08-17). Top-level rather than inside
    # `[find_boards.ransac]`: that table maps 1:1 to RVG's RansacConfig, and
    # the seed is BB's stream-derivation base, not an RVG knob.
    ransac_seed::Int = Int(_dflt("ransac_seed"))
    # Matching
    min_similarity::Float64 = _dflt("min_similarity")
    # Appearance verification (verify.jl): per-inlier descriptor gate and the
    # minimum corroborated fraction. Mandatory in the accept path — replaced
    # the geometry structs' min_inlier_fraction (2026-08-13).
    verify_similarity::Float64 = _dflt("verify_similarity")
    min_appearance_support::Float64 = _dflt("min_appearance_support")
    # Tracker knobs — [find_boards.track]; every track_boards keyword defaults
    # from here and overrides per call.
    track::TrackConfig = TrackConfig()
end

# =============================================================================
# find_boards — multi-board pipeline with verification
# =============================================================================

using VisualGeometryFeatures: DescribedFeature
using StructArrays: StructArray

# =============================================================================
# Board-derived detection settings
# =============================================================================
#
# `polarity` and `cutoff` are properties OF THE BOARD, recorded in its bundle
# at authoring time, so they are derived here rather than configured. What
# makes them awkward is that stages 1-4 (detect, adapt, canonicalize,
# describe) run ONCE per image and are board-agnostic, while only `match` is
# board-aware. The two parameters resolve that at DIFFERENT stages:
#
#   polarity  is a per-DETECTION attribute that survives to match time. VGF
#             keeps intensity > 0 for :dark and < 0 for :light, BB carries it
#             through as `detection.response`, and `DescribedFeature` retains
#             the whole detection. So detection runs at the SUPERSET (:none)
#             and each board applies its own polarity to its own match set —
#             genuinely per board, no extra detection, and a dark board can never
#             match a light blob.
#
#   cutoff    is NOT grouped on. It is `PatternConfig`'s RENDERING cutoff,
#             potentially unique per board — `evaluation/blobboards_original`
#             spans 1.75..1.995 in 0.005 steps, 50 distinct values. The image
#             side is described at ONE shared extent: the gallery's uniform
#             cutoff when the set agrees (enforce this at gallery
#             construction), else the MEAN of the board cutoffs (warned
#             once). Describing per distinct cutoff was tried (6fa79a8) and
#             REMOVED 2026-08-11: each distinct cutoff costs a full
#             descriptor-inference sweep over every detection — measured
#             2.3 s vs 52 ms on the 50-cutoff set, for one borderline inlier
#             on DSC02847 — and fusing the warps into one batch cannot help
#             (51.7 vs 56.7 µs/patch: the cost IS the per-patch convolution,
#             not call overhead). The annulus mismatch |c − mean| ≤ spread/2
#             is tolerated by the descriptor. NB the GEOMETRIC uses of each
#             board's own cutoff (guided-search IoU boundary, board_quad,
#             rendered-boundary ellipses) are physical properties of the
#             print and remain per-board.

_bundle_polarity(b) = b.preamble.pattern_config.polarity

# Superset for the detection stage. NB `:none`, not `:both` — VGF's
# `asift_detect` accepts :none/:dark/:light and throws on anything else.
const _POLARITY_SUPERSET = :none

# Boards partition by polarity ONLY. Polarity partitions the DETECTIONS (so
# adaptation is shared within a polarity); the descriptor sampling extent is
# ONE shared value for the whole gallery (see _shared_image_cutoff).
# Deterministic order so a run is reproducible.
function _board_groups(board_feature_dict, board_bundles)
    groups = Dict{Symbol, Vector{String}}()
    for uid in keys(board_feature_dict)
        haskey(board_bundles, uid) || throw(
            ArgumentError(
                "board_bundles is missing uid $uid; cutoff and polarity are read " *
                    "from the bundle, so every gallery board needs one"
            )
        )
        push!(get!(groups, _bundle_polarity(board_bundles[uid]), String[]), uid)
    end
    return [(pol, sort!(uids)) for (pol, uids) in sort!(collect(groups); by = first)]
end

# The ONE image-side descriptor sampling extent for a gallery: its uniform
# cutoff when every board agrees (enforce this at gallery construction —
# `insert!` warns on mixed batches), else the mean of the board cutoffs,
# which minimizes the worst-case annulus mismatch. Geometric per-board
# cutoffs are unaffected.
function _shared_image_cutoff(board_bundles, uids)
    cs = Float64[Float64(cutoff(board_bundles[uid])) for uid in uids]
    isempty(cs) && return DEFAULT_CUTOFF
    u = unique(cs)
    length(u) == 1 && return u[1]
    m = Statistics.mean(cs)
    @warn "gallery boards carry mixed rendering cutoffs; describing the image " *
        "at their mean — regenerate the gallery with a uniform cutoff to " *
        "remove the annulus mismatch entirely" n_distinct = length(u) mean_cutoff = m maxlog = 1
    return m
end

# Polarity PARTITIONS the detections — VGF keeps intensity > 0 for :dark and
# < 0 for :light, and BB carries that through as `detection.response`. So a
# group describes only its own polarity's detections. Two consequences: the
# per-board semantics are exact (a dark board never sees a light blob), and a
# mixed-polarity board set costs no extra description work, because the two
# subsets are disjoint rather than overlapping.
#
# This is done HERE, before adapt/canonicalize, rather than at match time:
# `filter_blobs` returns adapted frames and `described_features` keeps the
# FRAME, so `response` is gone by the time matches exist. (Attempting it at
# match time is what the annotate smoke test caught — EllipticFrame has no
# `response` field.)
_polarity_subset(detections, ::Val{:none}) = detections
_polarity_subset(detections, ::Val{:dark}) =
    filter(d -> d.metadata.response > 0, detections)
_polarity_subset(detections, ::Val{:light}) =
    filter(d -> d.metadata.response < 0, detections)

# The BB-side response-rank cap, applied to the polarity subset rather than to
# the shared unfiltered detection (see `_detect_for_boards`).
function _cap_detections(detections, max_detections::Int)
    length(detections) > max_detections || return detections
    d = copy(detections)
    sort!(d; by = x -> abs(x.metadata.response), rev = true)
    resize!(d, max_detections)
    return d
end

# The detection stage keeps the NARROWEST polarity that still serves every
# board: one polarity if the set agrees (every real board set — so the
# detector returns exactly what it returned before boards drove this), and
# only the `:none` superset when the set genuinely mixes. Detecting at
# `:none` unconditionally would make VGF convert every feature to a
# `Detection` and BB discard half of them — the kernels are unaffected (VGF
# filters after `asift_detect!`), but the conversion and allocation are not.
function _detection_polarity(board_bundles, uids)
    pols = unique(_bundle_polarity(board_bundles[u]) for u in uids)
    return length(pols) == 1 ? only(pols) : _POLARITY_SUPERSET
end

_detect_config_for_boards(config, polarity::Symbol) =
    ConstructionBase.setproperties(
    config, (;
        detect = ConstructionBase.setproperties(
            config.detect, (; polarity)
        ),
    )
)

# Undistort for the detector: the geometry decides (KA kernel on `device`
# for the pose arm under :global; identity for Pinhole/no camera, for the
# raw-detecting arms, and for the pose arm under :none). The detector
# consumes host UInt8. The second `_undistort_adapt(..., nothing)` is a
# storage convert, not a second kernel.
function _undistorted_pair(img, geometry, device::KA.Backend)
    detect_img = gray_u8(_undistort_detect(device, img, geometry))
    adapt_img = _undistort_adapt(
        device,
        detect_img isa AbstractMatrix{UInt8} ?
            reinterpret(Gray{N0f8}, detect_img) : detect_img,
        nothing
    )
    return detect_img, adapt_img
end

"""
    _detect_for_boards(img, descriptor_network; config, device)
        -> (; model, detect_img, adapt_img, detections)

The board-agnostic half of the pipeline: undistort + detect. Stops before
adapt/canonicalize/describe, which run once per polarity group at the
gallery's shared extent (see `_find_boards_grouped`).
"""
function _detect_for_boards(img, descriptor_network; config, device::KA.Backend)
    model = _ensure_network(descriptor_network)
    detect_img, adapt_img = _undistorted_pair(img, config.geometry, device)
    # NB no `max_detections` here. The cap ranks by |response| over whatever
    # it is given, and this detection is deliberately polarity-UNFILTERED, so
    # capping now would let light blobs evict dark ones (or vice versa) before
    # any board has claimed its own polarity. It is applied per group instead,
    # after `_polarity_subset`, which preserves the original semantics exactly:
    # top-N by |response| WITHIN the polarity the board actually wants.
    detections = detect_blobs(
        detect_img;
        config = config.detect, device
    )
    return (; model, detect_img, adapt_img, detections)
end

# The ONE image-side feature stage after detection: response-rank cap, printed-
# profile scale correction, adapt at the fixed filter boundary, describe at the
# gallery's shared extent. `_find_boards_grouped` runs it once per polarity
# group; `prepare_image_features` (annotate, track, sweeps) runs it once per
# frame. Until 2026-09-07 the seam had its own spelling that skipped the
# rescale, ignored board polarity and described at the caller's cutoff — so
# annotation and tracking matched query descriptors the gallery was never
# sampled like.
function _describe_for_boards(prep, dets, model, config, shared_cutoff, device::KA.Backend)
    dets = _cap_detections(dets, config.max_detections)
    # Detected scales are short by the printed profile's `β(σ_c)` (see
    # `_scale_bias`). Undo it HERE, where the gallery's support cutoff is
    # known — the detector is board-agnostic and cannot know σ_c. Adapt
    # sizes its fitting window from `characteristic_scale`, so this is the
    # last point at which an unbiased scale still buys anything.
    dets = rescale_detections(dets, shared_cutoff)
    # NB the filter boundary is DEFAULT_CUTOFF BY DESIGN, decoupled from
    # `config.adapt.cutoff`: the image-side clip test is a shared,
    # board-agnostic gate, so sweeping adapt.cutoff changes the IRLS
    # window but NOT the border filter. Both are 2.0 at defaults.
    frames = adapt_detections(
        dets, prep.adapt_img;
        cutoff = DEFAULT_CUTOFF,
        adapt_config = config.adapt, device
    )
    # ONE describe at the shared extent — per-cutoff description removed
    # (see the design note above _bundle_polarity).
    feats = describe_frames(
        model, frames, prep.adapt_img;
        inner_extent = shared_cutoff, device,
        patch_size = patch_size(model)
    )
    return feats
end

"""
    prepare_image_features(descriptor_network, img;
        board_bundles, uids = keys(board_bundles), config::FindBoardsConfig, device)
        -> (; detect_img, adapt_img, image_features, shared_cutoff, config)

The camera (for the lens undistort) is read from
`config.geometry` — `PoseGeometry` carries it, `HomographyGeometry` has
none (undistort becomes a no-op).

Detect-once seam for `annotate_blobs`, `track_boards` and the sweeps: the
SAME stages `find_boards(…, img)` runs — undistort (two-step: a KA undistort
kernel for the lens, then a storage conversion via an identity-kernel
`_undistort_adapt`), detect at the polarity the boards imply, cap, printed-
profile scale correction, adapt, describe at the gallery's shared extent
(`_shared_image_cutoff` over `uids`) — producing ONE feature set shared by
every board. There is no `cutoff` keyword: the gallery decides the extent, and
a caller-supplied one could only make query and gallery descriptors silently
incomparable. The returned `config` carries the derived detection polarity;
`shared_cutoff` is the extent the features were described at.

NEVER run two undistort kernels on the same image — the per-stage numerical
drift between kernels (hardware bilinear vs manual bilinear)
propagates through adapt → canonicalize → descriptors and shifts RANSAC
inliers enough to drop borderline boards. The second `_undistort_adapt(…,
nothing)` call here is a storage convert (Pinhole/nothing path), not a
second kernel.
"""
function prepare_image_features(
        descriptor_network, img::AbstractMatrix;
        board_bundles,
        uids = keys(board_bundles),
        config::FindBoardsConfig = FindBoardsConfig(),
        device::KA.Backend = default_device()
    )
    uids = collect(uids)
    det_config = _detect_config_for_boards(config, _detection_polarity(board_bundles, uids))
    prep = _detect_for_boards(img, descriptor_network; config = det_config, device)
    shared_cutoff = _shared_image_cutoff(board_bundles, uids)
    image_features = _describe_for_boards(
        prep, prep.detections, prep.model, det_config, shared_cutoff, device
    )
    return (; prep.detect_img, prep.adapt_img, image_features, shared_cutoff, config = det_config)
end

"""
    find_boards(board_feature_dict, descriptor_network, img::AbstractMatrix;
                board_bundles, config = FindBoardsConfig(), device = default_device(),
                gpu_index = nothing, matching_strategy = MutualNN(),
                exclude_polys = nothing, telemetry = nothing) -> Vector
    find_boards(board_feature_dict, descriptor_network, frames; kwargs...) -> Vector{Vector}
    find_boards(board_feature_dict, image_features; kwargs...) -> Vector

Find known boards in an image and estimate each one's homography or pose; returns
one `(uid, model, quality, scored)` tuple per verified board.

The image method runs the whole pipeline once per image: detect blobs, adapt
them to ellipses, canonicalize and describe them, then for every board match,
fit (RANSAC + local optimization), gate on inlier count and appearance, and
claim the board's image region so that no detection serves two boards.

# Arguments
- `board_feature_dict`: `Dict(uid => features)` of the gallery boards' descriptors,
  as returned by `insert!(::BoardGallery, board_paths, network)`
  ([`BoardGallery`](@ref)). Only these boards are searched for.
- `descriptor_network`: a [`DescriptorNetwork`](@ref) (or the path of its ONNX
  file), the same network that described the gallery.
- `img`: a grey image (`AbstractMatrix`, e.g. `Gray.(load(path))`). Give a
  vector of frames to run each independently with the gallery staged once.
  Give features from `extract_image_features` to skip the image stages.

# Keywords
- `board_bundles` (required): `Dict(uid => BoardBundle)` for every board in
  `board_feature_dict` (`load_board(path).bundle`). Polarity, cutoff and the
  board outline used for spatial claiming come from them.
- `config = FindBoardsConfig()`: every pipeline setting ([`FindBoardsConfig`](@ref)).
  `config.geometry` selects the estimator: [`HomographyGeometry`](@ref) (no
  camera), [`PoseGeometry`](@ref)`(camera)` (calibrated P3P) or
  [`RadialHomographyGeometry`](@ref)`(image_size)` (unknown radially distorted
  camera). Build it from a TOML file with `FindBoardsConfig(path; camera)`.
- `device = default_device()`: the KernelAbstractions backend for detection,
  adaptation and description — `KA.CPU()`, or `CUDABackend()` once CUDA is
  loaded (the default picks the GPU when one is functional).
- `gpu_index = nothing`: a prebuilt [`GalleryIndex`](@ref) of the gallery on
  `device`; built per call when `nothing`. Build it once to reuse across images.
- `matching_strategy = MutualNN()`: the first matching phase's
  nearest-neighbour rule (a permissive `KnnMatch(3)` cleanup phase follows).
- `exclude_polys = nothing`: image polygons whose detections are excluded
  before matching (regions already claimed elsewhere).
- `telemetry = nothing`: a `Telemetry` sink that records per-stage diagnostics.

# Returns
A vector with one `(uid, model, quality, scored)` tuple per accepted board, in
acceptance order:
- `uid`: the board's 3-character uid (the corner glyphs, a key of
  `board_bundles`).
- `model`: with `HomographyGeometry`, the homography mapping the board's
  `:world` plane (mm, z = 0) to image pixels. With `PoseGeometry`, the board →
  camera pose (`model.R`, `model.t` in mm). With `RadialHomographyGeometry`, the
  `(H, k)` division-model fit.
- `quality`: the fit's RANSAC score.
- `scored`: the correspondences (board blob → image detection) with their
  posterior weights in `scored.weights`; `count(is_map_inlier, scored.weights)`
  is the inlier count.

# See also
[`track_boards`](@ref) (video), [`FindBoardsConfig`](@ref), [`BoardGallery`](@ref),
[`DescriptorNetwork`](@ref).
"""
function find_boards(
        # `board_feature_dict::AbstractDict` makes a call in the retired order
        # (`img, board_feature_dict, descriptor_network`) a MethodError.
        board_feature_dict::AbstractDict,
        descriptor_network,
        # `::AbstractMatrix` is load-bearing: the sequence method below takes a
        # vector, tuple or generator of frames, so ONE image must be the MORE
        # SPECIFIC signature or a wrapped image type would route into the
        # sequence method and be iterated pixel by pixel.
        img::AbstractMatrix;
        # Declared, not hidden in kwargs...: it is REQUIRED (cutoff and
        # polarity are read from it), and a required argument that does not
        # appear in the signature is undiscoverable.
        board_bundles,
        config = FindBoardsConfig(),
        device::KA.Backend = default_device(),
        gpu_index = nothing,
        kwargs...,
    )
    return _find_boards_grouped(
        img, board_feature_dict, descriptor_network;
        board_bundles, config, device, gpu_index, kwargs...
    )
end

# Detect ONCE, then one description per POLARITY group at the gallery's
# shared extent. Groups share spatial claims through `exclude_polys` — the
# same channel track_boards uses to hand cross-tier commits to the matcher —
# so a board accepted in one group removes its image region from the others.
# With a single polarity (every real board set) this is one group, one
# description, one call into the tuned matching loop.
function _find_boards_grouped(
        img, board_feature_dict, descriptor_network;
        board_bundles, config, device, gpu_index, prepared = nothing,
        kwargs...
    )
    bundles = board_bundles
    groups = _board_groups(board_feature_dict, bundles)
    det_pol = _detection_polarity(bundles, keys(board_feature_dict))
    det_config = _detect_config_for_boards(config, det_pol)
    prep = @something prepared _detect_for_boards(
        img, descriptor_network;
        config = det_config, device
    ) Some(nothing)
    model = prep.model

    accepted = Tuple[]
    claimed = something(
        get(values(kwargs), :exclude_polys, nothing),
        GeometryBasics.Polygon[]
    )
    claimed = collect(claimed)
    rest = Base.structdiff(NamedTuple(kwargs), NamedTuple{(:exclude_polys,)})
    shared_cutoff = _shared_image_cutoff(bundles, collect(keys(board_feature_dict)))
    for (pol, uids) in groups
        # Polarity partitions the detections; subset only when the detection
        # detection ran at the mixed superset.
        dets = det_pol === _POLARITY_SUPERSET ?
            _polarity_subset(prep.detections, Val(pol)) : prep.detections
        feats = _describe_for_boards(prep, dets, model, config, shared_cutoff, device)
        sub = Dict(uid => board_feature_dict[uid] for uid in uids)
        # ONE call over ALL of this polarity's boards, so the agenda's
        # commit-in-inlier-order arbitration runs BETWEEN them. The prebuilt
        # index covers the FULL gallery; per-board match blocks are computed
        # independently, so extra rows for another polarity's boards are idle
        # GEMM work whose dict entries the agenda never reads.
        got = find_boards(
            sub, feats;
            board_bundles, config, device,
            gpu_index,
            exclude_polys = isempty(claimed) ? nothing : claimed,
            rest...
        )
        append!(accepted, got)
        length(groups) == 1 && break          # nothing left to claim against
        cam = _estimation_camera(config.geometry)
        for (uid, model_g, _, _) in got
            push!(claimed, board_quad(cam, model_g, bundles[uid]))
        end
    end
    return accepted
end

"""
    find_boards(board_feature_dict, descriptor_network,
                frames::AbstractVector{<:AbstractMatrix}; kwargs...) -> Vector{Vector}

Batch entry point: run board detection on a SEQUENCE of frames with one-time
setup amortization — the gallery GPU index (and each frame's detection plan)
is built once and reused. Per-frame results are independent (no temporal
association — see `track_boards` for that); element `i` is exactly what
`find_boards(..., frames[i])` returns.
"""
function find_boards(
        board_feature_dict::AbstractDict,
        descriptor_network,
        frames::Union{AbstractVector, Tuple, Base.Generator};
        board_bundles,
        config = FindBoardsConfig(),
        device::KA.Backend = default_device(),
        gpu_index = nothing,
        kwargs...,
    )
    gpu_index = @something gpu_index GalleryIndex(device, board_feature_dict) Some(nothing)
    model = _ensure_network(descriptor_network)
    return map(frames) do frame
        _find_boards_grouped(
            frame, board_feature_dict, model;
            board_bundles, config, device, gpu_index, kwargs...
        )
    end
end

"""
    find_boards(board_feature_dict, image_features::StructArray{<:DescribedFeature};
                kwargs...) -> Vector

Match pre-extracted image features against all boards, estimate geometry,
gate on inlier count + appearance, and spatially claim/suppress. Runs
**round-based spatial suppression**: every round fits all live boards on the
current claim state, commits the single best, removes it from the gallery,
and claims the detections inside its projected quad; surviving boards
re-resolve their matches against the new claim state and refit, until a
round accepts nothing. Two matching phases run this loop — strict
`MutualNN`, then a permissive `KnnMatch(3)` cleanup on the depleted pool.

# Arguments
- `board_feature_dict`: `Dict{uid => StructArray{DescribedFeature}}` of board features
- `image_features`: Pre-extracted `StructArray{DescribedFeature}` from `extract_image_features`

# Keyword Arguments
- `config`: `FindBoardsConfig`. `config.geometry` selects the path —
  `HomographyGeometry()` (no camera) or `PoseGeometry(camera)` (calibrated P3P).
- `board_bundles`: `Dict{uid => BoardBundle}` — required for spatial claiming
  and guided search (enforced with an `ArgumentError`).
- `device`: KA device (`KA.CPU()` or `CUDABackend()`)
- `gpu_index`: Pre-built [`GalleryIndex`](@ref) for the device-resident board gallery

Returns `(uid, model, quality, scored)` for each verified board
(IRLS weights in `scored.weights`),
concatenated across phases.
"""
function find_boards(
        board_feature_dict::AbstractDict,
        image_features::StructArray{<:DescribedFeature};
        # NB no `img` / `descriptor_network`. They were threaded three levels
        # deep (here -> _run_matching_phase! -> _claim_boards) and read by
        # nothing but `_descriptor_verify`; with that stage gone they were
        # dead weight in the hot path. This method matches pre-extracted
        # features against a gallery — it needs neither the image nor the
        # network to do that.
        board_bundles = nothing,
        gpu_index = nothing,
        config = FindBoardsConfig(),
        device::KA.Backend = default_device(),
        matching_strategy::NNStrategy = MutualNN(),
        exclude_polys = nothing,
        telemetry::Union{Nothing, Telemetry} = nothing,
    )
    # The spatial feature-removal scheme is mandatory: it needs per-board
    # world rects (`board_bundles`) to claim the detections of accepted
    # boards. Refuse to silently degrade to an unclaimed run — that path
    # over-counts boards (features get re-matched to overlapping gallery
    # entries that a spatial mask would have removed).
    isnothing(board_bundles) &&
        throw(
        ArgumentError(
            "find_boards requires `board_bundles`: the spatial " *
                "feature-removal scheme cannot run without per-board world rects."
        )
    )
    # Same coverage check `_board_groups` applies on the image path. Without
    # it a missing bundle surfaces as a KeyError thrown out of the threaded
    # fit loop mid-phase (as a TaskFailedException), not as a usable error.
    for uid in keys(board_feature_dict)
        haskey(board_bundles, uid) || throw(
            ArgumentError(
                "board_bundles is missing uid $uid; every gallery board needs " *
                    "its bundle for spatial claiming and guided search"
            )
        )
    end

    # Build the GPU gallery index ONCE per call. Both phases match through
    # `nn_match_stacked`; without a prebuilt index each call restages (~85 MB
    # host) and re-uploads (~31 MB) the stacked gallery. The index covers the
    # full gallery; phase 2 fits only surviving uids, so the extra rows are
    # idle GEMM work, not correctness risk.
    gpu_index = @something gpu_index GalleryIndex(device, board_feature_dict) Some(nothing)

    # The estimator's camera follows the pose arm's policy — see
    # _estimation_camera.
    cam = _estimation_camera(config.geometry)

    # Greedy spatial claiming, one commit per round (see _claim_boards).
    # Between the two phases, every image feature inside an accepted board's
    # quad is removed from the pool, so phase 2's fresh matcher never sees a
    # claimed feature.
    feat_pool = image_features
    # Pre-claim: features inside externally-committed board quads (e.g.
    # track_boards' warm winners this frame) leave the pool before any
    # matching — the intra-call claim guard cannot see cross-tier commits.
    if exclude_polys !== nothing && !isempty(exclude_polys)
        keep = _features_outside_polys(feat_pool, exclude_polys)
        feat_pool = feat_pool[keep]
    end
    gallery_pool = Dict(uid => bf for (uid, bf) in board_feature_dict)
    all_accepted = Tuple[]
    # Phase 1: strict matching (config strategy + min_similarity) to convergence.
    feat_pool = _run_matching_phase!(
        gallery_pool, all_accepted, feat_pool;
        board_bundles, gpu_index, config, device, cam,
        matching_strategy, min_similarity = config.min_similarity, telemetry
    )
    # Phase 2: permissive cleanup on the depleted pool — KnnMatch(k=3) at the
    # derived permissive gate (`_phase2_similarity`: 0.1 under min_similarity),
    # run to convergence. Recovers boards whose true
    # blobs were out-competed by confusers or sat just under the strict cutoff.
    # Only runs on boards phase 1 missed (empty gallery ⇒ no-op), and the
    # extent + inlier-count + appearance gates guard against false positives.
    #
    # MEASURED ESSENTIAL, not vestigial (260527 slice, 2026-08-14): phase 2
    # accepted 7 boards over 24 frames — zero on easy frames, all 7 on the
    # distant stations, including the GT-verified real boards 952/6ad/575/2a0
    # that the appearance-gate calibration fought to keep. Deleting this phase
    # loses real distant boards; see PHASE2_LOG for re-measurement.
    n_phase1 = length(all_accepted)
    _run_matching_phase!(
        gallery_pool, all_accepted, feat_pool;
        board_bundles, gpu_index, config, device, cam,
        matching_strategy = KnnMatch(3),
        min_similarity = _phase2_similarity(config.min_similarity), telemetry
    )
    p2log = telemetry === nothing ? "" : telemetry.phase2_log
    if !isempty(p2log)
        _append_jsonl(
            p2log, (
                tag = telemetry.tag, n_phase1 = n_phase1,
                n_phase2 = length(all_accepted) - n_phase1,
                phase2_uids = String[r[1] for r in all_accepted[(n_phase1 + 1):end]],
            )
        )
    end
    return all_accepted
end

# Per-thread RANSAC workspace pools, BOTH estimator paths — `(fit, refine)`,
# either `nothing` when its size is below the minimal sample. `set_n!` inside
# ransac()/locally_optimize trims a bigger buffer to the problem's data_size,
# so one workspace sized to the largest set serves every board.
# `fit` serves the initial RANSAC (sized by the largest match list);
# `refine` the guided refit, whose set is bounded by the board's world-blob
# count and can exceed the match count.
#
# Sizing: `Threads.maxthreadid()` (not `nthreads()`) — Julia ≥ 1.10
# separates the default + interactive threadpools, and `threadid()` is
# the GLOBAL id, so a call from the interactive pool can return an id
# above `nthreads(:default)`. `maxthreadid()` is the safe upper bound.
# A workspace is built for the scorer and the local optimizer it serves
# (`outlier_halfwidth`, `local_optimization`: the geometry's).
_fit_workspace_builder(::Nothing, a, lo) =
    n -> RobustVisualGeometry.RansacWorkspace(HomographyProblem{Float64}, n, _scoring(HomographyProblem{Float64}, a), lo)
_fit_workspace_builder(::CameraModel, a, lo) =
    n -> RobustVisualGeometry.RansacWorkspace(AbsolutePoseProblem, n, _scoring(AbsolutePoseProblem, a), lo)
# Radial arm: no pooled workspace yet (it would need K=5 /
# RadialHomography-shaped buffers) — RVG sizes fresh ones per fit.
_fit_workspace_builder(::RadialHomographyGeometry, _a, _lo) = nothing

_pool_kw(g) = (; outlier_halfwidth = g.outlier_halfwidth, local_optimization = g.local_optimization)
function _workspace_pools(cam, max_match::Int, max_blobs::Int; outlier_halfwidth, local_optimization)
    builder = _fit_workspace_builder(cam, outlier_halfwidth, local_optimization)
    builder === nothing && return nothing, nothing
    fit = max_match >= 4 ?
        [builder(max_match) for _ in 1:Threads.maxthreadid()] : nothing
    refine = max_blobs >= 4 ?
        [builder(max_blobs) for _ in 1:Threads.maxthreadid()] : nothing
    return fit, refine
end

# One PHASE of the spatial-claiming engine under a given matching config.
# Mutates `gallery_pool` (deletes accepted) and `all_accepted` (appends); returns
# the (shrunken) feature pool so a subsequent phase can continue from it.
#
# The gallery is matched ONCE per phase (VGF's stacked-GEMM `nn_match_stacked` is
# the dominant cost of the estimation stage). The feature pool stays FIXED for
# the whole phase (indices stable): claimed features are excluded from
# resolution and guided association via the claim mask inside the phase, and the
# pool is physically shrunk only at phase exit for the next phase's matcher.
# What matching-once gives up vs re-matching after every commit: a blob whose
# mutual NN was claimed cannot fall back to a match that was never on its raw
# list — per-round re-resolution recovers only claim-freed candidates already
# ON the KnnMatch list; phase 2's fresh match on the shrunk pool is the
# designed recovery for the rest.
#
# ONE claiming loop per phase (the repeated loop was merged 2026-08-17). With
# one-to-one resolution per ROUND inside `_claim_boards`, a second loop
# over the same claim state re-resolves to exactly the sets the first loop's
# final round already fitted and rejected — extra loops were pure
# confirmation refits, plus a fresh RANSAC lottery on marginal fits (RVG's
# sampler draws from the ambient task RNG), which is variance, not recovery.
# The 260527 A/B that decided the per-round semantics (56 frames × 2 seeds,
# 2026-08-17) measured the per-round serial arm identical to the engine on
# every frame; the recovery the loop boundaries used to provide now happens
# at the commit that frees the matches (board c4f@DSC04564 — a real 3 m
# board at the min_inliers frontier, present in the promoted snapshot — is
# the measured witness: filter-only resolution loses it at every seed).
#
# EVERY round refits EVERY surviving board. There is no cross-round reuse of
# any kind: no result cache, no re-admission filter. That is deliberate and
# was arrived at by deleting two attempts at the optimization on 2026-08-14:
#
#   * a `fit_cache` (result memo keyed on the resolved match set, plus a
#     guided-staleness check) — documented as CONSERVATIVE, and measurably
#     weaker than a fresh refit on the 260527 A/B;
#   * a re-admission filter that agendaed only boards whose match list the
#     boundary `filter!` had shrunk. That one was WRONG. A fit has two
#     claim-sensitive inputs, and the filter tracked one: the guided R-tree
#     is rebuilt from `in_claimed` every round, and `_guided_candidates`
#     takes the single nearest detection per projected blob, so a commit can
#     change what a board's fresh fit would see WITHOUT touching that
#     board's match list. Refit-all inside a loop does not cover it —
#     refit-all only reaches boards already ON that loop's agenda, so a
#     board the filter excluded never saw the loop's claims at all, and the
#     engine could drop a board `_find_boards_serial` accepts. (Caught by
#     the adversarial review, not by test_oracle.jl, whose two-board scene
#     commits everything in one loop and never reaches a second.)
#
# Correctness first — the engine's equality with the serial reference is the
# thing being protected.
function _run_matching_phase!(
        # The mutated arguments first — `gallery_pool` loses the accepted uids,
        # `all_accepted` gains their results — then the feature pool, which is
        # read (the shrunk pool is RETURNED, never written back).
        gallery_pool::AbstractDict, all_accepted::AbstractVector, feat_pool::AbstractVector;
        board_bundles, gpu_index, config, device, cam,
        matching_strategy, min_similarity, telemetry = nothing
    )
    isempty(gallery_pool) && return feat_pool
    # CONTRACT NOTE: `gallery_pool` SHRINKS across phases (accepted boards are
    # deleted) while `gpu_index` is built once over the FULL gallery, so the
    # two disagree by design — rebuilding the index per phase would cost a full
    # restage. `match_boards` owns that disagreement: it matches against the
    # whole staged gallery and keys its output off `keys(galleries)`, so a
    # detection whose global best belongs to an already-accepted board is
    # DROPPED rather than handed to its runner-up. Nothing here depends on
    # VGF's own key handling.
    match_dict = match_boards(
        matching_strategy, gallery_pool, feat_pool;
        backend = device, index = gpu_index, min_similarity
    )
    # Per-thread RANSAC workspace pools, one per PHASE: raw match-list lengths
    # upper-bound the one-to-one-resolved sets the fits actually see, and
    # RVG's set_n! trims an oversized buffer to each problem's data_size, so
    # one pool sized here serves every round.
    fit_workspaces, refine_workspaces = _workspace_pools(
        cam,
        isempty(match_dict) ? 0 : maximum(length, values(match_dict)),
        isempty(gallery_pool) ? 0 :
            maximum(length(bf) for bf in values(gallery_pool));
        _pool_kw(config.geometry)...
    )
    accepted, claimed = _claim_boards(
        feat_pool, gallery_pool;
        board_bundles, gpu_index, config,
        device, matching_strategy, min_similarity,
        match_dict, fit_workspaces, refine_workspaces, telemetry
    )
    append!(all_accepted, accepted)
    for (uid, _, _, _) in accepted
        delete!(gallery_pool, uid)
    end
    # Physically shrink the pool only at phase exit (indices stay stable for
    # the whole phase; `claimed` is the loop's own containment mask — one
    # containment authority, see _mark_inside!). Nothing claimed → alias
    # instead of copying (nothing downstream mutates the pool array).
    any(claimed) || return feat_pool
    return feat_pool[findall(!, claimed)]
end

# =============================================================================
# _find_boards_serial — THE reference implementation (test oracle)
# =============================================================================

"""
    _find_boards_serial(board_feature_dict, image_features; board_bundles,
                        config = FindBoardsConfig(), device = default_device(),
                        gpu_index = nothing, exclude_polys = nothing) -> Vector

The engine's contract, written down. `_run_matching_phase!` +
`_claim_boards` promise results identical to "a fully serial eager
loop"; this IS that loop — fit every live board on the CURRENT claim state,
commit the single best by (inlier count, quality), claim its quad, repeat.
Same two matching phases, same agenda floor, same gates, same commit
ordering — but no speculation, no threading, no dirty vetoes, no refit cache:
the claim state is always current when a fit runs, so "stale" cannot exist,
and every mechanism the engine adds to recover serial semantics under
parallelism is absent BY CONSTRUCTION.

Use it as the oracle when changing the engine (`test/test_oracle.jl` asserts
equality on a synthetic scene; corpus A/Bs use it as the reference arm).
Never call it in production: it refits the surviving gallery once per
commit — O(gallery) more fitting work than the engine per accepted board.
"""
function _find_boards_serial(
        board_feature_dict::AbstractDict, image_features::StructArray{<:DescribedFeature};
        board_bundles, config = FindBoardsConfig(),
        device::KA.Backend = default_device(),
        gpu_index = nothing, exclude_polys = nothing
    )
    cam = _estimation_camera(config.geometry)
    gpu_index = @something gpu_index GalleryIndex(device, board_feature_dict) Some(nothing)
    feat_pool = image_features
    if exclude_polys !== nothing && !isempty(exclude_polys)
        feat_pool = feat_pool[_features_outside_polys(feat_pool, exclude_polys)]
    end
    gallery = Dict(uid => bf for (uid, bf) in board_feature_dict)
    n_min = config.geometry.min_inliers - 1
    accepted = Tuple[]
    for (strategy, s_min) in (
            (MutualNN(), config.min_similarity),
            (KnnMatch(3), _phase2_similarity(config.min_similarity)),
        )
        isempty(gallery) && break
        match_dict = match_boards(
            strategy, gallery, feat_pool;
            backend = device, index = gpu_index,
            min_similarity = s_min
        )
        detc = [centroid(d) for d in feat_pool]
        Di = descriptor_matrix(feat_pool)
        claimed = falses(length(feat_pool))
        while !isempty(gallery)
            rt = detection_rtree(detc; skip = claimed)
            cands = Tuple[]
            for uid in sort!(collect(keys(gallery)))
                # Per-round one-to-one resolution on the claim-filtered RAW
                # match list — the engine's round-top loop, run serially.
                # Filtering a phase-resolved list instead (the pre-2026-08-17
                # reference) cannot promote a blob's next-best raw candidate
                # after a commit claims its winner; the 260527 A/B measured
                # that claim-freed fallback as load-bearing (c4f@DSC04564 — a
                # real 3 m board in the promoted snapshot — is lost without
                # it, at every seed).
                resolved = resolve_one_to_one(
                    filter(m -> !claimed[m.second], match_dict[uid]),
                    length(gallery[uid]), length(feat_pool)
                )
                _n_resolved_matches(resolved) >= n_min || continue
                bdl = board_bundles[uid]
                r = try
                    find_board(
                        cam, gallery[uid], feat_pool;
                        min_similarity = config.min_similarity,
                        matches = resolved, bundle = bdl,
                        guided_detc = detc, guided_rtree = rt,
                        resolve_pose_ambiguity = _pose_ambiguity_enabled(config.geometry),
                        fit_kw = _board_fit_kw(config, uid, bdl),
                    )
                catch err
                    @warn "find_board failed for board $uid (serial reference)" exception = (err, catch_backtrace())
                    nothing
                end
                r === nothing && continue
                model, quality, scored = r
                nin = count(is_map_inlier, scored.weights)
                nin >= config.geometry.min_inliers || continue
                sims = _inlier_sims(descriptor_matrix(gallery[uid]), Di, scored)
                appearance_support(sims; verify_similarity = config.verify_similarity) >=
                    config.min_appearance_support || continue
                push!(cands, (uid, model, quality, scored, nin))
            end
            isempty(cands) && break
            sort!(cands; by = c -> (c[5], c[3]), rev = true)
            (uid, model, quality, scored) = cands[1]
            _mark_inside!(claimed, detc, board_quad(cam, model, board_bundles[uid]))
            push!(accepted, (uid, model, quality, scored))
            delete!(gallery, uid)
            delete!(match_dict, uid)
        end
        any(claimed) || continue
        feat_pool = feat_pool[findall(!, claimed)]
    end
    return accepted
end

# =============================================================================
# Eager multi-board agenda (2026-07-28)
# =============================================================================
#
# Boards compete for image features. Instead of running one full RANSAC per
# gallery board per phase, boards are prioritized by the score mass of their
# resolved match sets and hopeless boards are pruned before a single trial
# (the match-count floor). Fitting is speculative-parallel; committing is serial in
# fitted-support order with claim-and-reweight — an accepted result is always
# a fit computed on claim-cleaned inputs, i.e. result-identical to a fully
# serial eager loop, at the wall time of the old threaded all-boards loop.

"""
    _AgendaEntry

One gallery board's live state in the eager agenda: its one-to-one-resolved
match set (rebuilt at the top of every round against the current claim
state) and the resolved-match count used for prioritization and the fit
floor.
"""
mutable struct _AgendaEntry
    uid::String
    resolved::Vector{NNMatch}
    n_matches::Int
    # Detection indices this entry's LAST fit read: its match set (RANSAC
    # input, rebuilt by the round-top re-resolution) plus the nearest
    # detection per projected blob (guided input, changed by the R-tree's
    # claim skip). Refilled every fitted round; write-only telemetry today
    # (see REFIT_SKIP_LOG).
    deps::Vector{Int32}
end
"""
    _n_resolved_matches(resolved) -> Int

How many resolved matches a board has: the pre-fit scheduling statistic.
Boards that cannot muster enough matches never get a RANSAC trial.

There is NO second similarity bar here. Every match in `resolved` already
cleared `min_similarity`, and this counts all of them.

History, so the two deleted designs are not re-proposed. It replaced
gate-relative score mass `Σ (sᵢ − s_min)/(1 − s_min)` (2026-08-13), which
conflated "many mediocre matches" with "few excellent ones": a floor intended
as "six matches at 0.7" was equally satisfied by TWO perfect matches
(mass 2.0 > 1.5), a board that cannot constrain a homography. That was
replaced by a count above a DERIVED strong bar `s_min + 0.25(1 − s_min)`
(0.7 at the default), which fixed the conflation but bought its dynamic range
with a cliff: a match at 0.699 counted zero, at 0.701 counted one, on the path
to the engine's only irreversible decision — and the 0.25 was inherited
verbatim from the mass floor, never swept.

Counting every resolved match is monotone in the right direction: it can only
RAISE the statistic, so it can only ADMIT more boards. The floor is a resource
heuristic, not a correctness gate — everything downstream can only reject, and
the real gates (fitted `min_inliers`, `appearance_support`) are untouched. The
cost is fit time, not precision.

A COUNT, not a fraction, on purpose. The question at this stage is "is there
enough evidence to be worth fitting", which is inherently absolute: four
correspondences are the minimal sample for a homography, three for P3P. A
fraction has the small-denominator pathology (2-of-2 scores 1.0) and would
penalise exactly the partially-occluded boards the evaluation set is built to
exercise. Fractions belong AFTER the fit, where the denominator is meaningful
— `appearance_support` (verify.jl) over the fitted inlier set.
"""
_n_resolved_matches(resolved) = length(resolved)

"""
    _phase2_similarity(s_min) -> Float64

Phase 2's permissive matching gate, DERIVED from the strict gate: 0.1 below
`s_min`, floored at zero — 0.5 at the default `s_min = 0.6`, preserving the
tuned value. It was a hardcoded 0.5, which inverted the phases' semantics
under sweeps: any `min_similarity < 0.5` made the "permissive cleanup" phase
STRICTER than phase 1. Derived from `s_min` so it cannot.
"""
_phase2_similarity(s_min::Float64) = max(s_min - 0.1, 0.0)

# ONE containment precision for every claim channel: Float64, matching
# `_features_outside_polys` (the pre-claim filter). The former Point2f
# downcast here made the intra-phase masks disagree with the pool filter for
# centroids within ~5e-4 px of a quad edge — fixed 2026-08-10.
_pt64(c) = Point2(Float64(c[1]), Float64(c[2]))   # detections are bare Float64 by contract (adapt.jl)

# Mark detections inside `poly` (function barrier: `poly` arrives concrete
# here, so the containment loop compiles without the Vector{Any} dynamic
# dispatch the per-match closures used to pay).
function _mark_inside!(mask::BitVector, detc, poly)
    @inbounds for j in eachindex(detc)
        mask[j] && continue
        _pt64(detc[j]) in poly && (mask[j] = true)
    end
    return mask
end

# Internal phase implementation: the round loop (fit all live boards, commit
# the single best, re-resolve, repeat). Public callers go through find_boards,
# which runs it once per matching phase.
function _claim_boards(
        image_features::StructArray{<:DescribedFeature},
        board_feature_dict;
        # NB no `img` / `descriptor_network`. They were threaded three levels
        # deep (here -> _run_matching_phase! -> _claim_boards) and read by
        # nothing but `_descriptor_verify`; with that stage gone they were
        # dead weight in the hot path. This method matches pre-extracted
        # features against a gallery — it needs neither the image nor the
        # network to do that.
        board_bundles = nothing,
        gpu_index = nothing,
        config = FindBoardsConfig(),
        device::KA.Backend = default_device(),
        matching_strategy::NNStrategy = MutualNN(),
        min_similarity = config.min_similarity,
        match_dict = nothing,
        # Phase-hoisted per-thread RANSAC workspace pools (see
        # _run_matching_phase!); standalone callers get per-call pools below.
        fit_workspaces = nothing,
        refine_workspaces = nothing,
        telemetry::Union{Nothing, Telemetry} = nothing,
    )
    cfg = config
    cam = _estimation_camera(cfg.geometry)

    # Match dict is normally hoisted by `_run_matching_phase!` (ONE gallery
    # match per phase); standalone callers get a fresh match here. Claimed
    # features stay in `image_features` (stable indices) but are excluded
    # from resolution (round-top filter), guided association (skipped in the
    # R-tree), and the claim masks below.
    match_dict = @something match_dict match_boards(
        matching_strategy, board_feature_dict, image_features;
        backend = device, index = gpu_index, min_similarity
    )

    outlier_halfwidth = cfg.geometry.outlier_halfwidth
    # Hoisted guided_search context: detection centroids are identical for
    # every board in this phase (also reused for claim containment below).
    guided_detc = [centroid(d) for d in image_features]

    # ---- Agenda: per-round one-to-one resolution + match-count floor --------
    n_feats = length(image_features)
    # Sorted for a deterministic tie order in the (n_matches, then fitted-
    # support) sorts below — Dict order is insertion-dependent, and the
    # serial oracle iterates uids sorted, so the engine does too.
    uids = sort!(collect(keys(board_feature_dict)))
    # Entries start UNRESOLVED: one-to-one resolution runs at the top of every
    # ROUND against the current claim state (see the round loop), so a blob
    # whose winning candidate was claimed by a commit falls back to its next
    # raw candidate immediately — not at some later phase boundary. Measured
    # decisive on 260527 (2026-08-17): board c4f@DSC04564, a real 3 m board at
    # the min_inliers frontier, is recovered ONLY via this claim-freed
    # fallback; resolve-once-per-phase (filter-only) loses it at every seed.
    agenda = [_AgendaEntry(uid, NNMatch[], 0, Int32[]) for uid in uids]
    # Agenda floor: a board with too few resolved matches gets no trial THIS
    # ROUND — the hopeless tail of the gallery is skipped before any RANSAC
    # runs. Applied per round (the entry stays on the agenda, mirroring the
    # serial oracle, which reconsiders every uncommitted board each round);
    # claims only shrink raw lists, so in practice a board below the floor
    # does not come back.
    #
    # THE ONE SEMANTICALLY LOAD-BEARING PRUNE. Everything downstream can only
    # reject; a board skipped here is never fitted and cannot appear in the
    # round's result, with no diagnostic. Its job is to skip gallery boards
    # that are NOT IN THE FRAME AT ALL, not to judge the ones that are.
    #
    # It is a HEURISTIC, and deliberately not "could this board reach
    # min_inliers". That comparison looks right — inliers are a subset of
    # correspondences, so `n_matches >= min_inliers` would seem lossless — and
    # it is WRONG here: the guided refit associates by GEOMETRY, and its set is
    # bounded by the board's world-blob count, not by the match count (see the
    # `refine_workspaces` sizing below). A board admitted with `min_inliers-1`
    # matches can finish above the accept gate on correspondences that were
    # never matched. So no floor derived from `min_inliers` is logically
    # justified; the only invariant that matters is that this floor stays
    # STRICTLY LOOSER than the accept gate.
    #
    # `-1` is that slack, and is also what the 2026-08-11 exclusive(>) ->
    # inclusive(>=) migration left: every count default moved up one in
    # lockstep, so writing `min_inliers` here would have silently tightened
    # the pre-fit floor by one match.
    #
    # NB the statistic underneath changed meaning when `_strong_similarity`
    # was removed: this counts ALL resolved matches now, not those above a
    # 0.7 bar, so the floor is strictly looser than the value it was tuned
    # against. It has no telemetry (see the `211@DSC04555` open item) — add a
    # counter before re-tuning it.
    n_min = cfg.geometry.min_inliers - 1

    # Per-thread RANSAC workspaces (see _workspace_pools). Normally hoisted by
    # `_run_matching_phase!` (one pool per PHASE); this fallback serves
    # standalone callers only.
    if fit_workspaces === nothing && refine_workspaces === nothing
        fit_workspaces, refine_workspaces = _workspace_pools(
            cam,
            isempty(agenda) ? 0 : maximum(length(e.resolved) for e in agenda),
            isempty(agenda) ? 0 :
                maximum(length(board_feature_dict[e.uid]) for e in agenda);
            _pool_kw(cfg.geometry)...
        )
    end

    # Guided-search context, hoisted per phase — pure functions of the fixed
    # feature pool and the immutable bundles, so caching them is
    # bit-identical to guided_search's own fallback construction:
    #   det_ells         one Ellipse per detection at each DISTINCT cutoff
    #   (wdiscs, wregions) each board's world discs + boundary regions
    # Built serially; the threaded fit loop only reads them.
    det_ells_by_cutoff = board_bundles === nothing ? nothing :
        Dict(
            c => [Ellipse(region(feature_frame(f), c)) for f in image_features]
            for c in unique(Float64(cutoff(board_bundles[e.uid])) for e in agenda)
        )
    board_geom = board_bundles === nothing ? nothing :
        Dict(
            e.uid => begin
                b = board_bundles[e.uid]
                wd = blobs(b, :world)
                (wdiscs = wd, wregions = [region(d, Float64(cutoff(b))) for d in wd])
            end for e in agenda
        )

    # Appearance-gate context, hoisted per phase: descriptor matrices are pure
    # functions of the fixed feature pool and the (immutable) gallery
    # features. `_inlier_sims` indexes ROWS (feature i = row i), matching
    # `descriptor_matrix`'s layout.
    Di_app = descriptor_matrix(image_features)
    Dg_app = Dict(
        e.uid => descriptor_matrix(board_feature_dict[e.uid])
            for e in agenda
    )

    _quad(uid, model) = board_quad(cam, model, board_bundles[uid])

    committed = Tuple[]
    # Detection-level containment masks: marked once per committed quad
    # (function barrier `_mark_inside!`), then every per-match dirty/touched/
    # filter check is an O(1) BitVector lookup instead of point-in-polygon
    # over a Vector{Any} of quads.
    in_claimed = falses(length(guided_detc))
    # ONE COMMIT PER ROUND (2026-08-14). Fit every live entry in parallel,
    # commit only the single best by (inlier count, quality), claim its quad,
    # refit everything still live. This DELETED the multi-commit round's
    # entire veto apparatus (`in_round`, `guided_dirty`, the two dirty-check
    # channels, the round cap) and with it the engine's one measured
    # divergence from `_find_boards_serial`: a same-round commit could change
    # what a fresh fit would see — a claimed near-miss promoting the
    # next-nearest past the IoU gate — without touching the vetoed inputs,
    # so multi-commit rounds carried fits the serial reference would have
    # recomputed (measured: 8 of 138 boards, ≤ 1.1 mm / ≤ 0.08°, one
    # 1-inlier gap; identity never affected). Every committed fit is now
    # computed on exactly the claim state it is accepted under. Termination
    # needs no cap: each round either commits (that entry leaves the agenda)
    # or breaks, so the loop runs at most |agenda| rounds. Wall time is
    # bounded by the measured serial arm (1.2–2.4 s/frame) and in practice
    # sits below it — rounds still fit in parallel.
    while !isempty(agenda)
        # Detection R-tree for guided_search, rebuilt per round against the
        # current claim state: `skip = in_claimed` covers pre-claims AND this
        # phase's earlier commits, so guided association can never re-associate
        # a claimed detection (documented invariant at detection_rtree).
        # O(D log D) bulk load, a few ms.
        guided_rtree = detection_rtree(guided_detc; skip = in_claimed)
        # Per-round one-to-one resolution on the claim-filtered RAW match
        # lists. Filtering the RESOLVED list instead (the pre-2026-08-17
        # within-phase behavior) cannot promote a blob's next-best candidate
        # when a commit claims its winner — the claim-freed fallback the
        # agenda note above measured as load-bearing. The serial oracle does
        # exactly this, so re-resolution timing can no longer separate them.
        for e in agenda
            e.resolved = resolve_one_to_one(
                filter(m -> !in_claimed[m.second], match_dict[e.uid]),
                length(board_feature_dict[e.uid]), n_feats
            )
            e.n_matches = _n_resolved_matches(e.resolved)
        end
        # ---- Parallel fit of every live agenda entry ------------------------
        # Each fit is independent (read-only image_features, per-thread
        # workspace); static scheduling pins iterations to threads so
        # `threadid()` indexing is stable. Only the best candidate commits;
        # the rest are recomputed next round on the updated claim state, so
        # no fit is ever accepted on inputs a commit has since invalidated.
        # The sort is priority order only — NOT selective: every entry above
        # the floor is fitted; the commit loop re-sorts by fitted inliers.
        batch = sort!([e for e in agenda if e.n_matches >= n_min]; by = e -> -e.n_matches)
        isempty(batch) && break
        slots = Vector{Union{Nothing, Tuple}}(undef, length(batch))
        Threads.@threads :static for i in eachindex(batch)
            e = batch[i]
            ws = fit_workspaces === nothing ? nothing : fit_workspaces[Threads.threadid()]
            rws = refine_workspaces === nothing ? nothing : refine_workspaces[Threads.threadid()]
            # Per-board extent: RVG wraps the problem in `BoundedProblem(_, rect)`
            # so the post-IRLS extent-consensus hook rejects bow-tie fits that
            # fold the board's outer corners.
            extent_i = isnothing(board_bundles) ? nothing :
                Rect(board_bundles[e.uid], :world)
            bdl = isnothing(board_bundles) ? nothing : board_bundles[e.uid]
            geom = board_geom === nothing ? nothing : board_geom[e.uid]
            # ONE NamedTuple shape — RVG's find_* declare `workspace = nothing`
            # as their default, so supplying nothing explicitly is identical and
            # keeps fit_kw type-stable across the ws/no-ws cases.
            fit_kw = (;
                outlier_halfwidth, workspace = ws,
                local_optimization = cfg.geometry.local_optimization,
                config = cfg.ransac_config, extent = extent_i,
                rng = fit_rng(cfg.ransac_seed, e.uid),
            )
            # Dependency capture: seed with the RANSAC input — the match set,
            # inliers AND outliers, since the round-top re-resolution rebuilds
            # it from the claim-filtered raw list — then `find_board` reports
            # the guided reads through `on_guided_read`. Each entry is written
            # only by the thread that owns its index, so no synchronisation is
            # needed.
            track_deps = !isempty(_refit_skip_log(telemetry))
            if track_deps
                empty!(e.deps)
                for m in e.resolved
                    push!(e.deps, Int32(m.second))
                end
            end
            slots[i] = try
                find_board(
                    cam, board_feature_dict[e.uid], image_features;
                    min_similarity = cfg.min_similarity,
                    matches = e.resolved,
                    bundle = bdl,
                    on_guided_read = track_deps ? (j -> push!(e.deps, j)) : nothing,
                    telemetry,
                    guided_detc, guided_rtree,
                    guided_det_ells = det_ells_by_cutoff === nothing ? nothing :
                        det_ells_by_cutoff[Float64(cutoff(bdl))],
                    guided_wdiscs = geom === nothing ? nothing : geom.wdiscs,
                    guided_wregions = geom === nothing ? nothing : geom.wregions,
                    refine_workspace = rws,
                    resolve_pose_ambiguity = _pose_ambiguity_enabled(cfg.geometry),
                    fit_kw
                )
            catch err
                @warn "find_board failed for board $(e.uid); skipping this board for this frame" exception = (err, catch_backtrace())
                nothing
            end
        end
        # ---- Gates ----------------------------------------------------------
        candidates = Tuple[]
        for i in eachindex(batch)
            r = slots[i]; r === nothing && continue
            model, quality, scored = r
            nin = count(is_map_inlier, scored.weights)
            # >= : min_inliers is the documented MINIMUM (inclusive), matching
            # the >= fraction gate below. Was `>` — an off-by-one against all
            # three docstrings, fixed 2026-08-10.
            nin >= cfg.geometry.min_inliers || continue
            # Appearance gate (mandatory; replaced min_inlier_fraction,
            # 2026-08-13): do the inliers LOOK like the board they claim?
            # Guided association is descriptor-blind, so a fabricated board
            # fills its inlier set geometrically — counts and fractions satisfy
            # it; the inlier descriptors don't (measured separation is total;
            # see verify.jl). One dot product per inlier, matrices hoisted.
            # NB the padding here is REAL but load-bearing: match-derived
            # inliers clear the bar by construction (verify_similarity ==
            # min_similarity), so a board with <= 2x the minimal sample
            # cannot fail this gate. Judging ONLY the guided correspondences
            # was tried (2026-08-16) and REVERTED: it rejected 109 boards on
            # the 260527 sweep and added none, dropping detection rate 76.5
            # -> 70.4 pp, including GT-verified real boards (952/6ad/575/2a0).
            # Distant real boards get FEW matches and MANY guided csponds whose
            # sims sit at 0.66-0.71 against a 0.6 bar, so judged alone half
            # fall below and the board dies. The padding is not only masking
            # false boards, it is carrying real ones — and
            # min_appearance_support = 0.5 is calibrated against the PADDED
            # denominator. Any narrowing must re-derive that threshold.
            sims = _inlier_sims(Dg_app[batch[i].uid], Di_app, scored)
            support = appearance_support(sims; verify_similarity = cfg.verify_similarity)
            alog = telemetry === nothing ? "" : telemetry.appearance_log
            if !isempty(alog)
                # Guided-only view, LOGGED BUT NOT GATED: the padded support
                # above is what decides. Judging the guided subset alone was
                # tried and reverted (see the comment on the gate); redoing
                # it needs its OWN threshold, re-derived from the guided-sim
                # distribution rather than inherited from verify_similarity.
                # These columns are that distribution: `g_sims` is every
                # guided inlier's similarity, so a threshold can be chosen
                # offline against GT instead of by another 8-minute sweep.
                matched = Set(
                    (Int(m.first), Int(m.second)) for m in batch[i].resolved
                )
                g_sims = _inlier_sims(
                    Dg_app[batch[i].uid], Di_app, scored; skip = matched
                )
                _append_jsonl(
                    alog, (
                        tag = telemetry.tag, uid = batch[i].uid,
                        nin = nin, n_judged = length(sims), support = support,
                        median_sim = isempty(sims) ? 0.0 :
                            Float64(Statistics.median(sims)),
                        succeeded = support >= cfg.min_appearance_support,
                        n_matched = nin - length(g_sims),
                        n_guided = length(g_sims),
                        g_sims = Float64.(g_sims),
                    )
                )
            end
            support >= cfg.min_appearance_support || continue
            # nin rides along so the commit-order sort below doesn't recount
            # the weights column O(n log n) times through its by-closure.
            push!(candidates, (batch[i].uid, model, quality, scored, nin))
        end

        # ---- Commit the single best — inlier count, quality tie-break -------
        # NOT raw quality alone: |q| grows with support size, so a 6-cspond
        # junk fit (q ≈ −7) would outrank a real 110-cspond board (q ≈ −100)
        # and steal its features (measured: board 336 collapsed 93/110 →
        # 18/18 under a pure-q sort, 2026-07-28). And NOT agenda mass — mass
        # is pre-fit information; the fitted support is the honest priority.
        #
        # NB no soft cspond suppression against previously-committed quads
        # here — that channel (transplanted from the deleted `spatial_verify`)
        # was provably dead once every input of a committed fit was
        # claim-cleaned; an assertion run recorded zero firings across ~38
        # engine invocations (removed 2026-08-14).
        isempty(candidates) && break
        sort!(candidates; by = c -> (c[5], c[3]), rev = true)
        (uid, model, quality, scored) = candidates[1]
        _mark_inside!(in_claimed, guided_detc, _quad(uid, model))
        push!(committed, (uid, model, quality, scored))

        # ---- Rebuild agenda: EVERY surviving entry refits next round on the
        # updated claim state (per-round re-resolution at the loop top, floor
        # re-applied).
        # Refit-all is what makes the commit above safe without vetoes — and
        # it makes any boundary re-admission through a guided channel
        # redundant, since no fit ever outlives a claim it didn't see.
        #
        # MEASURED NECESSARY (2026-08-16), the third attempt to avoid it and
        # the first to fail for a geometric rather than a bookkeeping reason.
        # Caching each entry's fit and re-fitting only those a commit could
        # have reached — tested against the committed quad, which is the
        # GUIDED channel the two earlier attempts ignored — skipped ZERO of
        # 141 fits over 4 rounds on a close-range frame, at margin 0.25 AND at
        # margin 0, and ran 0.742 s against 0.603 s for refit-all. The
        # committed board covers enough of the frame that every other board's
        # projected quad intersects it, so no sound predicate can discriminate
        # exactly where the cost is worst. The waste is not the re-fitting; it
        # is the 46 hopeless boards being on the agenda at all.
        # Upper bound on cache savings, measured on the round that just ran:
        # an entry whose deps contain NO detection this commit claimed would
        # have refit to a bit-identical result. Counted before the next
        # round's re-resolution rebuilds `resolved` (what deps described).
        rlog = _refit_skip_log(telemetry)
        if !isempty(rlog)
            # Fresh-deps entries only: this round's FITTED set minus the
            # commit (below-floor agenda entries carry stale deps).
            live = [e for e in batch if e.uid != uid]
            skippable = count(e -> !any(i -> in_claimed[i], e.deps), live)
            _append_jsonl(
                rlog, (
                    tag = telemetry.tag, committed = uid,
                    n_fitted = length(batch), n_live = length(live),
                    n_skippable = skippable,
                    n_deps = isempty(live) ? 0 :
                        sum(length(e.deps) for e in live),
                )
            )
        end
        # Drop only the committed entry. Every survivor RE-RESOLVES next round
        # against the updated claim state (round-top loop), so refit-all — the
        # thing that makes the single-commit round safe without vetoes — now
        # also carries the claim-freed fallback.
        agenda = [e for e in agenda if e.uid != uid]
    end

    # `in_claimed` rides back so the phase runner shrinks the pool from the
    # loop's own containment mask instead of recomputing it from the accepted
    # models — one containment authority (2026-08-14; see _run_matching_phase!).
    return committed, in_claimed
end

# Indices of image features whose centroid lies OUTSIDE every polygon —
# the pre-claim filter for externally-committed quads (`exclude_polys`).
# The containment test is `Base.in(::Point2, ::Polygon)` (VGC operators.jl)
# at `_pt64` precision, the same predicate `_mark_inside!` uses, so the two
# claim channels cannot disagree. (`_features_outside_verified_quads`, which
# re-derived the between-phase mask from accepted models through this helper,
# was removed 2026-08-14 — the loop now returns its own mask.)
function _features_outside_polys(image_features, polys)
    keep = trues(length(image_features))
    @inbounds for i in eachindex(image_features)
        # Defensive identity-or-strip: image features arrive bare (adapted
        pt = _pt64(centroid(image_features[i]))
        for p in polys
            if pt in p
                keep[i] = false
                break
            end
        end
    end
    return findall(keep)
end

# PoseGeometry opts into resolving the planar pose ambiguity; every other
# geometry is branch-free BY CONSTRUCTION, not by omission: the plane-induced
# homography is a unique 8-DOF map fixed by the correspondences, and the
# two-fold ambiguity is a property of its DECOMPOSITION into (R, t, n) given K
# (Faugeras & Lustman 1988: 8 solutions → 4 by plane-in-front → 2 by
# cheirality, and those 2 are this ambiguity). Nothing in BB decomposes an H,
# so the uncalibrated arm has no branch to select. A future consumer that does
# decompose one inherits the ambiguity at that point — and none of this
# machinery, since the likelihood ratio below requires both branches refined
# through the same scoring object.
_pose_ambiguity_enabled(g::PoseGeometry) = g.resolve_pose_ambiguity
_pose_ambiguity_enabled(::GeometryConfig) = false
