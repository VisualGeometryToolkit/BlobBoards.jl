# =============================================================================
# DetectConfig — single source of truth for detect_blobs parameters
#
# Lives in its own file so detect.jl (loaded before pipeline.jl) can reference
# DetectConfig in its method signatures.
# =============================================================================

"""
    AbstractNmsConfig

Marker supertype for cross-warp NMS strategies (ASIFT-only). `nothing`
disables cross-warp NMS entirely (default — downstream RANSAC handles
duplicates).
"""
abstract type AbstractNmsConfig end

# Scale banding, shared by both strategies. A feature may only be suppressed by
# a stronger feature whose geometric-mean scale √det S lies within
# `band_reach` bands of its own, bands being `band_octaves` wide (band =
# ⌊log2 √det S / band_octaves⌋); `band_octaves = 0` is the whole-pool rule.
# Without it, a board- or panel-sized DoG extremum and the tilted warps'
# merged-blob streaks — both legitimate extrema, both far stronger — erased the
# 1–3 px blobs they were built from: on a panel of six 70×100 px boards the
# whole-pool rule kept 171 of ~500 true blobs and left one board empty; 1 octave
# ± 1 (scale ratio < 4) kept 634 with every board populated (VGF 0.6.2,
# 2026-08-25). A tilted copy of a blob has the blob's own geometric-mean scale
# (σ√t), so it still lands in the same band and is still deduplicated.
"""
    MxmNms(; window_half=3, band_octaves=1.0f0, band_reach=1)

M×M centroid-window cross-warp NMS. Two ASIFT features collide if their
centers fall inside an `M = 2*window_half + 1` pixel box in input-frame
coordinates; the lower-response one is dropped. Cheap; ignores ellipse
shape. `band_octaves`/`band_reach` restrict suppression to features of
comparable scale (see `EllipseNms`).
"""
Base.@kwdef struct MxmNms <: AbstractNmsConfig
    window_half::Int = 3
    band_octaves::Float32 = 1.0f0
    band_reach::Int = 1
end

"""
    EllipseNms(; cutoff=2.0f0, band_octaves=1.0f0, band_reach=1)

Mahalanobis-footprint cross-warp NMS. Two ASIFT features collide if their
`cutoff`-σ footprints share a pixel; the lower-response one is dropped.
Respects ellipse shape. Suppression is SCALE-BANDED: a feature can only be
suppressed by one whose geometric-mean scale band `⌊log2 √det S /
band_octaves⌋` is within `band_reach` of its own (1 octave ± 1 ⇒ scale ratio
< 4), so board-sized extrema and tilted-warp streaks cannot erase the small
blobs inside them. `band_octaves = 0` restores the whole-pool rule.
"""
Base.@kwdef struct EllipseNms <: AbstractNmsConfig
    cutoff::Float32 = 2.0f0
    band_octaves::Float32 = 1.0f0
    band_reach::Int = 1
end

"""
    DetectConfig(; kwargs...)

Single source of truth for `detect_blobs` parameters across all backends
(`KA.CPU()`, `CUDABackend()`, …). Each device implementation reads the
fields it understands and ignores the rest; ASIFT-specific fields are
no-ops for non-ASIFT backends.

Defaults match `examples/pipeline/find_boards.jl` — the canonical tuned
invocation.

# Generic fields (all backends)
- `peak_threshold = 0.01`: DoG/intensity response threshold
- `edge_threshold = 5.0`: edge-rejection ratio
- `polarity = :dark`: which polarity to keep (`:dark`, `:light`, or `:none`
  for the unfiltered superset). NB `:none` — VGF throws on `:both`.

The per-call detection cap (`max_detections`) lives as a top-level
kwarg of `detect_blobs`, not here — it's a BB-side post-NMS cap on the
returned vector, not something the detector device consumes.

# ASIFT-specific (no-ops for non-ASIFT backends)
- `tilts = []`: ASIFT tilt schedule; EMPTY means the IMAS-25 schedule (the
  same convention `configs/default.toml` uses)
- `nms = EllipseNms(cutoff = 2.0f0)`: cross-warp NMS strategy (`MxmNms()`,
  `EllipseNms()`, or `nothing` to disable and let downstream RANSAC handle
  the duplicates).

Parameterized on the NMS config type `N` (`Nothing`, `MxmNms`, or
`EllipseNms`) so the field stays concrete rather than an abstract union.
"""
Base.@kwdef struct DetectConfig{N}
    peak_threshold::Float64 = _dflt("detect", "peak_threshold")
    edge_threshold::Float64 = _dflt("detect", "edge_threshold")
    polarity::Symbol = :dark
    # NB: there is no `nms_mode`. It selected 2D-vs-3D scale-space extremum
    # suppression in VGF's retired WORKSPACE pipeline; detection now routes
    # through `asift_detect`, which has no such parameter. The field survived
    # the migration unread — settable, round-tripping through write_config,
    # and inert. `_check_keys` could not catch it because the key was KNOWN;
    # that validator rejects unknown keys, not dead ones. Removed 2026-08.
    # ASIFT-specific (no-ops for non-ASIFT backends):
    tilts::Vector{Tuple{Float64, Float64}} = _dflt_tilts()   # empty ⇒ IMAS-25
    # NB no `fill_value`. Like `nms_mode` it is a leftover of VGF's retired
    # workspace pipeline: `asift_detect` has no such parameter, so BB never
    # forwarded it and could not have. It was briefly re-exposed in the config
    # while making the writer exhaustive — a struct field is not evidence of a
    # consumer. Removed 2026-08.
    # Tuned suppression radius, NOT the blob-boundary convention: 2.0 keeps
    # more detections than a wider footprint. It equals DEFAULT_CUTOFF by
    # coincidence — deriving it from the constant would let a future change to
    # the boundary convention silently re-tune the detector.
    nms::N = let n = _dflt("detect", "nms")
        # Three-way, mirroring the loader's NMS_TAGS — the file format admits
        # "none", and the old two-way branch indexed n["window_half"] for it,
        # a KeyError at precompile time.
        if n["type"] == "none"
            nothing
        else
            band = (band_octaves = Float32(n["band_octaves"]), band_reach = Int(n["band_reach"]))
            n["type"] == "ellipse" ? EllipseNms(; cutoff = Float32(n["cutoff"]), band...) :
                n["type"] == "mxm" ? MxmNms(; window_half = Int(n["window_half"]), band...) :
                error("configs/default.toml [find_boards.detect] nms type must be none/ellipse/mxm, got $(repr(n["type"]))")
        end
    end
    # 2× upsample before building the scale space: starts the pyramid at
    # octave -1 instead of 0 (VGF `o_first = upsample ? -1 : 0`), adding one
    # octave and recovering small blobs that vanish at native resolution.
    #
    # This used to be UNSETTABLE from BlobBoards: `_detect` simply never
    # forwarded it, so runs inherited VGF's own `upsample = true` default.
    # It was real enough to be encoded in a directory name (`poses/bb.upsample`)
    # and declared in the detection-threshold CV config's (then tuning_grid.toml) [fixed] block as "recorded so the
    # sweep is reproducible" — while nothing read either. A setting that
    # changes results, is named in artifacts, and cannot be set or recorded is
    # the same failure the descriptor-model consolidation removed.
    upsample::Bool = _dflt("detect", "upsample")
end

"""
    TrackConfig

The tracker's knobs — `[find_boards.track]` in the config of record; every
`track_boards` keyword defaults from here and overrides per call. Until
2026-09 these four sat as bare keyword literals on `track_boards`, so a
tracking run's sidecar could not name them.

- `iou_threshold`: per-blob predicted↔detected ellipse IoU gate for the warm
  association (ellipse overlap at the board's own cutoff).
- `support_fraction`: fraction of the predicted blobs that must associate for
  the warm bet to proceed to estimation.
- `patience`: consecutive failed frames before a track stops being re-targeted
  every frame (it stays eligible for discovery keyframes).
- `keyframe_every`: discovery cadence for boards not currently tracked; `0`
  disables periodic discovery after frame 1.
"""
Base.@kwdef struct TrackConfig
    iou_threshold::Float64 = _dflt("track", "iou_threshold")
    support_fraction::Float64 = _dflt("track", "support_fraction")
    patience::Int = Int(_dflt("track", "patience"))
    keyframe_every::Int = Int(_dflt("track", "keyframe_every"))
end

