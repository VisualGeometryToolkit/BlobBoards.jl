# =============================================================================
# Blob Detection
# =============================================================================
#
# detect_blobs  — scale-space extrema + NMS → Vector{Detection}
#
# =============================================================================

using VisualGeometryToolkitFeatures: VisualGeometryToolkitFeatures, Detection
using VisualGeometryToolkitCore: EllipticFrame, feature_frame

# The float-gray input ladder is VisualGeometryToolkitFeatures' `gray_f32` since 0.7.1
# (a Float32 device array is forwarded untouched).
using VisualGeometryToolkitFeatures: gray_f32

# =============================================================================
# detect_blobs — Raw scale-space detection
# =============================================================================

"""
    detect_blobs(img; config=DetectConfig(),
                 max_detections=typemax(Int)) -> Vector{Detection}

Detect blobs in an image via the KernelAbstractions ASIFT pipeline, then apply
a BB-side response-rank cap.

Returns raw detections. Use `adapt` for shape fitting and `filter_blobs`
for quality filtering.

# Keyword Arguments
- `config::DetectConfig`: detector-stage parameters (see `DetectConfig`).
  These are read by the detector backend.
- `max_detections=typemax(Int)`: Final cap by response magnitude.
  Default is effectively no cap; supply a finite integer to enable.

`max_detections` is a BB-side post-processing knob, not an input the
detector backend consumes — that's why it lives here rather than on
`DetectConfig`.

There is deliberately no `cutoff`: detection is scale-space extrema and is
cutoff-INDEPENDENT. The kwarg existed "for API parity" and was a documented
no-op, which made `cutoff` look like a detection parameter when it is
actually a property of the BOARD, consumed downstream by `adapt`,
`filter_blobs` and `canonicalize`. That misreading is what let a run config
state a cutoff that disagreed with the boards.

Cross-warp NMS (ellipse-aware) is configured through `config.nms`
(`MxmNms` / `EllipseNms`); there is no BB-side geometric NMS stage over
the detector's elliptic output (the legacy `suppress_overlapping` path
only knows `Disc` bbox IoU and is bypassed).

# Example
```julia
detections = detect_blobs(img)
frames = adapt(detections, img; cutoff=2.0)
frames = filter_blobs(frames, detections; cutoff=2.0)
```
"""
function detect_blobs(
        img;
        config::DetectConfig = DetectConfig(),
        # BB-side post-detection knobs — these don't belong in DetectConfig
        # because the detector backend doesn't consume them; the response-rank
        # cap (max_detections) is applied to the Vector{Detection} returned by
        # `_detect`. Default `typemax(Int)` ⇒ no cap; supply a finite value to enable.
        max_detections::Int = typemax(Int),
        # Detector device: VGF's KernelAbstractions ASIFT runs on any KA
        # device. Defaults to `default_device()` — the GPU when one is
        # functional. It used to default to `KA.CPU()`, which made a caller
        # that simply forgot the kwarg run the 25-warp pyramid in HOST RAM,
        # where VGF sizes it at 80% of `Sys.free_memory()` and the process
        # gets OOM-killed rather than erroring.
        device::KA.Backend = default_device(),
    )
    detections = _detect(img, device; config)

    if length(detections) > max_detections
        sort!(detections; by = d -> abs(d.metadata.response), rev = true)
        resize!(detections, max_detections)
    end

    return detections
end

# =============================================================================
# _detect — KernelAbstractions ASIFT backend (VisualGeometryToolkitFeatures)
# =============================================================================
#
# Routes detect_blobs through VGF's backend-agnostic KA IMAS-25 pipeline.
# Emits the Detection{EllipticFrame} contract: COLMAP coords,
# (response, warp_idx) metadata.
#
# Polarity is gated BEFORE the cross-warp NMS splat (VGF >= 0.6.1): a
# light-polarity feature never splats, so it can never suppress a dark blob.
# Through VGF 0.6.0 the filter ran on the NMS survivors instead, and the white
# gaps between a board's dark blobs (|response| up to 1.5x the blobs') killed
# ~20% of the true blobs on dense boards before being discarded themselves.
# This matters for the DEFAULT config: `nms` is `EllipseNms(2.0)`, not `nothing`.

function _detect(
        img, device::KA.Backend;
        config::Union{Nothing, DetectConfig} = nothing
    )
    cfg = something(config, DetectConfig())
    nms_sym, wh, kc = cfg.nms isa MxmNms ? (:mxm, cfg.nms.window_half, 3.0f0) :
        cfg.nms isa EllipseNms ? (:ellipse, 3, cfg.nms.cutoff) :
        (:none, 3, 3.0f0)
    bo, br = cfg.nms === nothing ? (0.0f0, 1) : (cfg.nms.band_octaves, cfg.nms.band_reach)
    return VgtkFeatures.asift_detect(
        img;
        device,
        nms = nms_sym, window_half = wh, cutoff = kc,
        band_octaves = bo, band_reach = br,
        intensity_thr = cfg.peak_threshold,
        edge_thr = cfg.edge_threshold,
        polarity = cfg.polarity,
        # VGF spells "the IMAS-25 schedule" as `tilts = nothing`; BB spells it as
        # the empty vector (see `_tilts_from_toml`). Translate at the boundary.
        tilts = isempty(cfg.tilts) ? nothing : cfg.tilts,
        # 2x upsample: pyramid starts at octave -1. Live as of VGF 0.4.0,
        # which is the first release to implement it — the `upsample` kwarg
        # existed as a no-op placeholder since the original KA ASIFT import,
        # so every detection before 0.4.0 ran at native resolution.
        upsample = cfg.upsample
    )
end

# =============================================================================
# Printed-profile scale bias — `_scale_bias` / `rescale_detections`
# =============================================================================
#
# The detector reports the scale-space extremum of the blob it actually sees,
# and the blob a board prints is NOT an ideal Gaussian: `blob_profile`
# truncates at `cutoff·σ` and renormalises intensity across the truncated
# disc, which removes the pedestal `p = exp(-cutoff²/2)`. Truncation alone is
# scale-neutral (the extremum stays at σ₀ to within 1.5% for cutoff ≥ 2), but
# removing the pedestal subtracts a disc of constant height over the whole
# support, and that disc carries its own scale-space signature: the extremum
# moves to `β(cutoff)·σ₀`.
#
# β is fixed by the cutoff alone — scale covariance is retained, the reported
# scale is simply short by a known constant. Measured against this detector on
# synthetic profiles it is accurate to ≤0.003 over cutoff ∈ [1.5, 4].
#
# NB the OTHER DoG scale bias — assigning the lower Gaussian's σ to a level
# instead of the geometric mean, a factor √k with k = 2^(1/nb_scales) — is
# already corrected upstream in VGF (`asift/extract.jl`, the `+0.5f0` in the
# level-σ exponent). An ideal untruncated Gaussian measures 1.000 here.

# Scale-normalised LoG response at the centre of the printed profile
#   u(r) = (exp(-r²/2σ₀²) - p)/(1 - p),  r ≤ cutoff·σ₀,  p = exp(-cutoff²/2)
# in dimensionless ρ = σ/σ₀. Reduces to -2ρ²/(1+ρ²)² (the ideal Gaussian) as
# cutoff → ∞.
@inline function _profile_log_response(ρ::Float64, cutoff::Float64)
    p = exp(-cutoff^2 / 2)
    κ = cutoff^2 * (1 + ρ^2) / (2ρ^2)
    q = exp(-κ)
    return (
        2 * (1 - (1 + κ) * q) / (1 + ρ^2)^2 - 2 * (1 - q) / (1 + ρ^2) +
            cutoff^2 * q / ρ^2
    ) / (1 - p)
end

const _SCALE_BIAS_CACHE = Dict{Float64, Float64}()
# Guards the cache: `rescale_detections` runs inside callers' threaded frame
# loops, and a concurrent first-miss `get!` on a plain Dict is a data race.
const _SCALE_BIAS_CACHE_LOCK = ReentrantLock()

"""
    _scale_bias(cutoff) -> β

Ratio of the scale the detector reports to the true `σ₀` of a blob printed at
support `cutoff` (see `blob_profile`). `β < 1`: reported scales are short.

Boards in one gallery need not share a cutoff — `β(1.75) = 0.81` while
`β(2.0) = 0.88` — so this is resolved per gallery from the board set, not
assumed global.
"""
function _scale_bias(cutoff::Real)
    c = Float64(cutoff)
    c >= 6 && return 1.0                      # pedestal below float noise
    return lock(_SCALE_BIAS_CACHE_LOCK) do
        get!(_SCALE_BIAS_CACHE, round(c; digits = 6)) do
            best_ρ, best = 1.0, -Inf
            for i in 0:20_000                  # ρ ∈ [0.2, 2.0], step 9e-5
                ρ = 0.2 + i * (1.8 / 20_000)
                v = abs(_profile_log_response(ρ, c))
                v > best && (best = v; best_ρ = ρ)
            end
            best_ρ
        end
    end
end

"""
    rescale_detections(detections, cutoff) -> Vector{Detection}

Undo the printed profile's scale bias, mapping each detection's scale from
what the detector reports to `σ₀`. The shape matrix is `S = σ·A⁻¹` (VGF's
back-projection), so scaling `S` scales σ and leaves the warp's tilt/angle
untouched.

This is consumed by `adapt`, which sizes its per-blob fitting window from
`characteristic_scale` — an unbiased scale gives a correctly sized window.
It does NOT correct the scale adaptation converges to: that fit re-estimates
scale from the pixel data and is insensitive to its seed (measured: adapted
frames agree to three decimals with and without this rescale). What it buys is
the window, and through it the blobs that survive at all.
"""
function rescale_detections(detections, cutoff::Real)
    β = _scale_bias(cutoff)
    β ≈ 1 && return detections
    return [
        Detection(EllipticFrame(feature_frame(d).center, feature_frame(d).S ./ β), d.attributes)
            for d in detections
    ]
end

# NOTE: the ASIFT tilt-warp helpers that lived here (_warp_tilt!,
# _alloc_warp_buf, _warp_view, _to_raw) were dead code — detection routes
# through VisualGeometryToolkitFeatures.asift_detect, which owns its warp kernels.
# The CPU and CUDA copies had even diverged on the shear sign ((t-1) vs
# (1-t)).  Deleted in the 2026-07 warp unification (#19).
