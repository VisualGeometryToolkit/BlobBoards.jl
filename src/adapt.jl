using VisualGeometryToolkitCore: EllipticFrame, feature_frame, aspect_ratio
using VisualGeometryToolkitCore: EnsembleProblem, EnsembleKernel, EnsembleResult, StoppingCriterion, StopWhenAny,
    StopWhenIncrementLess, StopAfterIteration

# =============================================================================
# Blob Adaptation — an ensemble solve
# =============================================================================
#
# solve(EnsembleProblem(BlobBatch(detections, img), length(detections)), BlobAdaptation(; …), EnsembleKernel(device);
#       stops) -> EnsembleResult
#   Paraboloid IRLS in log-intensity space — Student-t/uniform mixture with
#   whitened normal equations, one item per blob, backend-generic via
#   KernelAbstractions (src/adapt_ka.jl): KA.CPU(), CUDABackend(), Metal, ROCm.
#   `adapted_blobs(result)` is the kept blobs.
#
# filter_blobs(frames, detections; ...)
#   Geometric quality filters on fitted frames.
#
# =============================================================================

"""
    BlobAdaptation(; kwargs...)

The blob adaptation (paraboloid fitting) of an ensemble of blobs, a VisualGeometryToolkitCore `FitKind`:

    solve(EnsembleProblem(BlobBatch(detections, img), length(detections)), BlobAdaptation(; …), EnsembleKernel(device);
          stops = blob_adaptation_stops()) -> EnsembleResult

Each detection (a `Detection` of `detect_blobs`, or a `Disc` of ground truth) is one item, fitted by the Student-t /
uniform mixture IRLS (WLS in log-space with the f² correction and the sort-and-sweep posterior partition; not
configurable), all of them in one KernelAbstractions launch per iteration on the `EnsembleKernel`'s backend. The model
follows the blob's sample count: 20 or more fit the anisotropic paraboloid (6 parameters), 10 to 19 the isotropic one
(4 parameters: a round blob, `C = c·I`), and a blob of fewer than 10 is dropped. A detection whose geometry is an
`EllipticFrame` (the ASIFT pipeline's back-projected ellipses) seeds the IRLS from its shape.

`stops` is a VisualGeometryToolkitCore `StoppingCriterion` evaluated per blob in the kernel after every accepted step
(`stop_reason(stops, nothing, nothing, nothing, θ, θ_old, iteration)` on the blob's model parameters), and it must
hold one `StopAfterIteration(max_iter)`, the budget (an `ArgumentError` otherwise). [`blob_adaptation_stops`](@ref)
is the configured one: `StopWhenIncrementLess(rtol) | StopAfterIteration(max_iter)`, the increment test first, so a
blob that converges on the last iteration is converged.

The result is an `EnsembleResult`, one entry per detection in order: `u[i]` the blob's [`BlobDetection`](@ref)
(`id = i`, its outcome), or `nothing` where a gate dropped it; `retcode[i]` and `stop_rule[i]` its termination record —
`(Success, IncrementSize)` converged, `(MaxIters, IterationCap)` capped (returned, flagged, unconverged), `(Failure,
ShapeGate)` / `(InitialFailure, ShapeGate)` refused by the shape gate (`rejection` says why: `:not_peak` or
`:aspect_ratio`; the blob keeps its previous iterate), and for a dropped blob `(InitialFailure, NoRule)` (oversize,
clipped or too few samples: never fitted), `(Failure, NonFinite)` (a singular solve) or `(Failure, ShapeGate)` (the
frame gates: not a peak, an invalid shape, a scale ratio beyond `max_scale_ratio`); `nsteps[i]` its IRLS iterations;
`rms` NaN (the fit reports none). [`adapted_blobs`](@ref) is the kept blobs; [`count_unconverged`](@ref) and
[`count_rejected`](@ref) count them.

# IRLS
- `spatial_weighting::Bool = true` — adaptive spatial weights from current shape
- `cutoff::Float64 = 2.0` — pixel window in units of σ

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

The defaults are `[find_boards.blob_adaptation]` of configs/default.toml. Accumulation and the 6×6 solve run in
Float64 wherever the backend supports it (Float32 on Metal, where `stops`' tolerances compare in Float32 too);
per-pixel staging is pinned Float32.
"""
Base.@kwdef struct BlobAdaptation <: VgtkCore.AbstractEstimationAlgorithm
    spatial_weighting::Bool = toml_default(_TOML_DEFAULTS, "blob_adaptation", "spatial_weighting")
    # DATA, not configuration: the printed blob's boundary in units of sigma.
    # `find_boards` overrides it from the gallery (`_shared_image_cutoff`), so
    # the scan window, the clip test and the canonicalization extent cannot
    # disagree. This default serves only the gallery-less callers (a solve run
    # directly, `annotate_blobs`). NOT a TOML key — tune `scan_padding` instead.
    cutoff::Float64 = DEFAULT_CUTOFF
    min_intensity::Float64 = toml_default(_TOML_DEFAULTS, "blob_adaptation", "min_intensity")
    max_scale_fraction::Float64 = toml_default(_TOML_DEFAULTS, "blob_adaptation", "max_scale_fraction")
    max_scale_ratio::Float64 = toml_default(_TOML_DEFAULTS, "blob_adaptation", "max_scale_ratio")
    scan_padding::Float64 = toml_default(_TOML_DEFAULTS, "blob_adaptation", "scan_padding")
    max_clip_ratio::Float64 = toml_default(_TOML_DEFAULTS, "blob_adaptation", "max_clip_ratio")
    max_aspect_ratio::Float64 = toml_default(_TOML_DEFAULTS, "blob_adaptation", "max_aspect_ratio")
end
VgtkCore.estimator_kind(::BlobAdaptation) = VgtkCore.FitKind()

"""
    BlobBatch(detections, img)

The data of a [`BlobAdaptation`](@ref) ensemble: the blobs to fit (`detections`, a `Vector{Detection}` of
`detect_blobs` or a `Vector{Disc}` of ground truth; the shapes lead) and the image they are fitted in. One
`EnsembleProblem(BlobBatch(detections, img), length(detections))` is every blob, item `i` the `i`-th detection.
Image detections are pixels by regime: px-united frames are stripped in px, bare ones kept.
"""
struct BlobBatch{D <: AbstractVector, I <: AbstractMatrix}
    detections::D
    img::I
end

"""
    blob_adaptation_stops(; max_iter, rtol) -> StoppingCriterion

The stops of a [`BlobAdaptation`](@ref) solve. They are `StopWhenIncrementLess(rtol) | StopAfterIteration(max_iter)`
(VisualGeometryToolkitCore's criteria): a blob is converged once its step relative to its new parameters is below
`rtol`, capped after `max_iter` iterations. The defaults are `[find_boards.blob_adaptation.stops]` of
configs/default.toml (`max_iter = 100`, `rtol = 1e-6`).
"""
blob_adaptation_stops(; max_iter::Integer = toml_default(_TOML_DEFAULTS, "blob_adaptation.stops", "max_iter"),
                        rtol::Real = toml_default(_TOML_DEFAULTS, "blob_adaptation.stops", "rtol")) =
    StopWhenIncrementLess(Float64(rtol)) | StopAfterIteration(Int(max_iter))

# The budget of a stops criterion: its one `StopAfterIteration` (the host loop's iteration count).
_stops_budget(c::StopAfterIteration) = c.n
_stops_budget(c::StopWhenAny) = (n = [_stops_budget(x) for x in c.criteria if _has_budget(x)]; length(n) == 1 ? only(n) : _no_budget(c))
_stops_budget(c::StoppingCriterion) = _no_budget(c)
_has_budget(c::StopAfterIteration) = true
_has_budget(c::StopWhenAny) = any(_has_budget, c.criteria)
_has_budget(::StoppingCriterion) = false
_no_budget(c) = throw(ArgumentError(
    "BlobAdaptation: `stops` must hold one StopAfterIteration(max_iter), the iteration budget; got $(c)"))

# The criterion at the accumulation precision `Ta` (a Float32 kernel compares in Float32).
_stops_at(::Type{Ta}, c::StopWhenIncrementLess) where {Ta} = StopWhenIncrementLess(Ta(c.rtol))
_stops_at(::Type{Ta}, c::StopWhenAny) where {Ta} = StopWhenAny(map(x -> _stops_at(Ta, x), c.criteria))
_stops_at(::Type, c::StoppingCriterion) = c

function VgtkCore.__solve(ens::EnsembleProblem{<:BlobBatch}, alg::BlobAdaptation, e::EnsembleKernel;
                          stops::StoppingCriterion = blob_adaptation_stops())
    (; detections, img) = ens.f
    ens.n == length(detections) || throw(ArgumentError(
        "BlobAdaptation: the ensemble's $(ens.n) items are not the batch's $(length(detections)) detections"))
    return _adapt_ka(img, _adapt_sites(detections), alg, stops, e.backend)
end

# When the detection geometry is already an EllipticFrame (e.g. the GPU ASIFT pipeline back-projects each per-warp
# feature to an anisotropic input-frame ellipse), forward the frames and seed the paraboloid IRLS from S.S' instead of
# collapsing to a Disc. Image detections are px by regime: px-united frames strip in px, bare ones are kept
# (`strip_or_bare`).
_adapt_sites(discs::AbstractVector{<:Disc}) = strip_or_bare.(px, discs)
_adapt_sites(detections::AbstractVector) =
    if !isempty(detections) && feature_frame(first(detections)) isa EllipticFrame
        [strip_or_bare(px, feature_frame(d)) for d in detections]
    else
        [strip_or_bare(px, Disc(d)) for d in detections]
    end

"""
    adapted_blobs(result::EnsembleResult) -> Vector{BlobDetection{Float64}}

The blobs a [`BlobAdaptation`](@ref) solve kept, in detection order (each `id` its detection's index): the
result's `u` without the dropped items.
"""
adapted_blobs(r::EnsembleResult{<:AbstractVector{Union{Nothing, BlobDetection{Float64}}}}) =
    BlobDetection{Float64}[d for d in r.u if d !== nothing]

# Resolve the image-relative σ cap to pixels. Both solver arms already have
# W and H in scope for their clip test, so this stays a host-side scalar.
_max_scale_px(alg::BlobAdaptation, W, H) = alg.max_scale_fraction * sqrt(float(W) * float(H))

"""
    count_unconverged(frames) -> Int

The number of adapted frames whose IRLS was capped by its iteration budget before it
converged (`retcode = ReturnCode.MaxIters`). Rejected blobs are not counted here
(see [`count_rejected`](@ref)). See [`BlobAdaptation`](@ref).
"""
count_unconverged(frames::AbstractVector) = count(f -> f.retcode == VgtkCore.ReturnCode.MaxIters, frames)

"""
    count_rejected(frames) -> Int
    count_rejected(frames, rejection::Symbol) -> Int

The number of adapted frames whose IRLS step the shape gate refused. Those are
`retcode = ReturnCode.Failure` or `InitialFailure`, `stop_rule = ShapeGate`; with `rejection`, only those
refused for that reason, `:not_peak` or `:aspect_ratio`. See [`BlobAdaptation`](@ref).
"""
count_rejected(frames::AbstractVector) = count(f -> f.stop_rule == VgtkCore.ShapeGate, frames)
function count_rejected(frames::AbstractVector, rejection::Symbol)
    rejection in (:not_peak, :aspect_ratio) ||
        throw(ArgumentError("rejection $(repr(rejection)): the shape gate refuses for :not_peak or :aspect_ratio"))
    return count(f -> f.rejection === rejection, frames)
end

# =============================================================================
# filter_blobs — Geometric quality filters
# =============================================================================

"""
    filter_blobs(frames, detections; kwargs...) -> Vector{BlobDetection{Float64}}

Apply geometric quality filters to fitted blob detections.

Uses `det.id` to look up the original detection for scale/drift comparisons.

# Arguments
- `frames`: Fitted detections, [`adapted_blobs`](@ref) of a [`BlobAdaptation`](@ref) solve
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
        max_aspect_ratio = toml_default(_TOML_DEFAULTS, "blob_adaptation", "max_aspect_ratio"),
        max_scale_ratio = toml_default(_TOML_DEFAULTS, "blob_adaptation", "max_scale_ratio"),
        max_center_drift = 10.0,
        max_clip_ratio = toml_default(_TOML_DEFAULTS, "blob_adaptation", "max_clip_ratio")
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
        aspect_ratio(e) <= max_aspect_ratio || return false
        σ_fit = characteristic_scale(det.frame)
        sr = σ_fit / σ_orig
        1 / max_scale_ratio <= sr <= max_scale_ratio || return false
        norm(centroid(det.frame) .- centroid(orig)) <= max_center_drift * σ_orig || return false
        image_rect === nothing || (1.0 - coverage(e, image_rect)) <= max_clip_ratio || return false
        true
    end
end
