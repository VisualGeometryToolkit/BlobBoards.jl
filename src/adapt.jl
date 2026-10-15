using VisualGeometryCore: EllipticFrame, feature_frame

# =============================================================================
# Blob Adaptation — Unified Interface
# =============================================================================
#
# adapt(detections, img; config=AdaptConfig(), device)
#   Paraboloid IRLS in log-intensity space — Student-t/uniform mixture with
#   whitened normal equations, backend-generic via KernelAbstractions
#   (src/adapt_ka.jl): KA.CPU(), CUDABackend(), Metal, ROCm.
#
# filter_blobs(frames, detections; ...)
#   Geometric quality filters on fitted frames.
#
# =============================================================================

# =============================================================================
# AdaptConfig — all adaptation hyperparameters in one struct
# =============================================================================

"""
    AdaptConfig(; kwargs...)

Configuration for blob adaptation (paraboloid fitting).

# IRLS
- `spatial_weighting::Bool = true` — adaptive spatial weights from current shape
- `cutoff::Float64 = 2.0` — pixel window in units of σ
- `max_irls_iter::Int = 20` — maximum IRLS iterations
- `rtol::Float64 = 1e-6` — relative parameter convergence threshold

# Filtering
- `min_intensity::Float64 = 1e-6` — minimum inverted intensity to include pixel
- `max_scale_fraction::Float64 = 0.1` — skip detections whose σ exceeds this
  FRACTION of the image's √area, i.e. `σ > f·√(W·H)`. A resource guard: a
  blob's fitted support is ~`(2·cutoff·σ)²` pixels, so an unbounded σ is an
  OOM. It is a fraction rather than a pixel count because the same absolute
  cap means different things on different sensors — 200 px was ~5.8% of √area
  at 12 MP and larger than the whole frame at VGA, so an absolute value could
  not transfer across cameras. √area (the side of the equivalent square) is
  used rather than min(W,H) or the diagonal so the cap is aspect-neutral.
- `max_scale_ratio::Float64 = 4.0` — reject if adapted σ / detection σ exceeds this
- `scan_padding::Float64 = 1.0` — per-blob scan box halfwidth = `cutoff · scan_padding · σ` (decoupled from `max_scale_ratio`)
- `max_clip_ratio::Float64 = 0.2` — reject if blob is clipped by image boundary
- `max_aspect_ratio::Float64 = 10.0` — reject if ellipse axis ratio exceeds this

Accumulation and the 6×6 solve run in Float64 wherever the backend supports
it (Float32 on Metal); per-pixel staging is pinned Float32.
(`store` was deleted 2026-09: inert since the KA port — no kernel read it.)
"""
Base.@kwdef struct AdaptConfig
    spatial_weighting::Bool = _dflt("adapt", "spatial_weighting")
    # DATA, not configuration: the printed blob's boundary in units of sigma.
    # `find_boards` overrides it from the gallery (`_shared_image_cutoff`), so
    # the scan window, the clip test and the canonicalization extent cannot
    # disagree. This default serves only the gallery-less callers (`adapt` used
    # directly, `annotate_blobs`). NOT a TOML key — tune `scan_padding` instead.
    cutoff::Float64 = DEFAULT_CUTOFF
    max_irls_iter::Int = _dflt("adapt", "max_irls_iter")
    rtol::Float64 = _dflt("adapt", "rtol")
    min_intensity::Float64 = _dflt("adapt", "min_intensity")
    max_scale_fraction::Float64 = _dflt("adapt", "max_scale_fraction")
    max_scale_ratio::Float64 = _dflt("adapt", "max_scale_ratio")
    scan_padding::Float64 = _dflt("adapt", "scan_padding")
    max_clip_ratio::Float64 = _dflt("adapt", "max_clip_ratio")
    max_aspect_ratio::Float64 = _dflt("adapt", "max_aspect_ratio")
end

# =============================================================================
# adapt — Unified entry point
# =============================================================================

# Resolve the image-relative σ cap to pixels. Both solver arms already have
# W and H in scope for their clip test, so this stays a host-side scalar.
_max_scale_px(cfg::AdaptConfig, W, H) = cfg.max_scale_fraction * sqrt(float(W) * float(H))

"""
    adapt(detections, img; config=AdaptConfig(), device = default_device()) -> Vector{BlobDetection}

Fit elliptical frames to raw blob detections in `img` — the detections are the
shapes being refined, so they lead. Both are typed, so a call in the retired
`(img, detections)` order is a MethodError.

Uses a unified IRLS loop with Student-t posterior weights.
The solve step is the Student-t/uniform mixture: WLS in log-space with f²
correction and the sort-and-sweep posterior partition. It is not configurable.

Accepts `Vector{Detection}` (from `detect_blobs`) or `Vector{Disc}` (ground truth).

# Returns
`Vector{BlobDetection{Float64}}` — fitted elliptical frames with `.id` indexing
into the original detection vector.
"""
function adapt(detections::AbstractVector, img::AbstractMatrix; config::AdaptConfig = AdaptConfig(), device = default_device())
    # When the detection geometry is already an EllipticFrame (e.g. the
    # GPU ASIFT pipeline back-projects each per-warp feature to
    # an anisotropic input-frame ellipse), forward the frames and seed
    # the paraboloid IRLS from S.S' instead of collapsing to a Disc.
    # Image detections are px by regime: px-united frames strip in px, bare
    # ones are kept (`strip_or_bare`).
    sites = if !isempty(detections) && feature_frame(first(detections)) isa EllipticFrame
        [strip_or_bare(px, feature_frame(d)) for d in detections]
    else
        [strip_or_bare(px, Disc(d)) for d in detections]
    end
    return _adapt(img, sites, config, device)
end

function adapt(discs::AbstractVector{<:Disc}, img::AbstractMatrix; config::AdaptConfig = AdaptConfig(), device = default_device())
    # px regime: px-united discs strip in px, bare ones are kept.
    return _adapt(img, strip_or_bare.(px, discs), config, device)
end

# Backend-generic KernelAbstractions adaptation (src/adapt_ka.jl): threaded
# host implementation on KA.CPU(), cpu=false KA kernels on any GPU backend.
# The CUDA extension specializes only the hooks `_solve_6x6_batched!`
# (cuBLAS batched LU) and `_device_lgamma` (libdevice `__nv_lgamma`) —
# there is no CUDA-specific `_adapt` method anymore.
_adapt(
    img, sites::AbstractVector{<:Union{Disc, EllipticFrame}},
    cfg::AdaptConfig, device::KA.Backend = default_device()
) =
    _adapt_ka(img, sites, cfg, device)

# =============================================================================
# filter_blobs — Geometric quality filters
# =============================================================================

"""
    filter_blobs(frames, detections; kwargs...) -> Vector{BlobDetection{Float64}}

Apply geometric quality filters to fitted blob detections.

Uses `det.id` to look up the original detection for scale/drift comparisons.

# Arguments
- `frames`: Fitted detections from `adapt`
- `detections`: Original detections (indexed by `frame.id`)

# Keyword Arguments
- `cutoff`: Sigma multiplier for boundary ellipse
- `image=nothing`: Image for clipping check (skipped if `nothing`)
- `max_aspect_ratio=10.0`: Maximum allowed aspect ratio
- `max_scale_ratio=4.0`: Maximum σ ratio (fitted/initial)
- `max_center_drift=10.0`: Maximum center drift in units of initial σ
- `max_clip_ratio=0.2`: Maximum fraction of ellipse clipped by image boundary

# Returns
- `Vector{BlobDetection{Float64}}` — filtered subset
"""
function filter_blobs(
        frames, detections;
        cutoff = DEFAULT_CUTOFF,
        image = nothing,
        max_aspect_ratio = 10.0,
        max_scale_ratio = 4.0,
        max_center_drift = 10.0,
        max_clip_ratio = 0.2
    )
    image_rect = image !== nothing ? Rect(Size2(image)) : nothing

    return filter(frames) do det
        e = region(det.frame, cutoff)
        orig = detections[det.id]
        # Bare-by-contract (px regime): `asift_detect` emits plain Float64
        # σ/centroids, so there is nothing to strip — the target-unit
        # `ustrip(px, …)` forms the 8fc127c units audit put here had no
        # method for bare input and broke the dataset-gated smokes. A future
        # quantity-emitting producer will fail loudly in the arithmetic below.
        σ_orig = orig.σ
        max(e.a, e.b) / min(e.a, e.b) <= max_aspect_ratio || return false
        σ_fit = characteristic_scale(det.frame)
        sr = σ_fit / σ_orig
        1 / max_scale_ratio <= sr <= max_scale_ratio || return false
        norm(centroid(det.frame) .- centroid(orig)) <= max_center_drift * σ_orig || return false
        image_rect === nothing || (1.0 - coverage(e, image_rect)) <= max_clip_ratio || return false
        true
    end
end
