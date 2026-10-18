# =============================================================================
# Pipeline inspection — stage capture for qualitative comparison
# =============================================================================
#
# `trace_pipeline` runs the standard detect → adapt → filter stages ONCE and
# keeps every intermediate, with the adaptation optionally fanned out over
# several arms (e.g. Student-t vs HQ-Welsch) that all share the identical
# detections — the A/B precondition baked into the API.
#
# The viewer (`inspect_pipeline(trace; …)`) lives in BlobBoardsMakieExt: a grid of
# linked axes — columns = pipeline steps, rows = adaptation arms — composed
# from the canonical primitives (VGC `imshow`/`ellipses!`, VGF `detections!`).

using JLD2: JLD2

# The MANDATORY half of a trace — what every capture has, whatever produced it.
# Every field is a type parameter: the image type follows the caller, and a bare
# `img`/`detect_img` field would be `Any`, while annotating `arms` as
# `Vector{Pair{Symbol, NamedTuple}}` widens each arm's concrete NamedTuple back
# to the UnionAll.
"""
    PipelineCapture

The mandatory part of a [`trace_pipeline`](@ref) record: the original image
(`img`), the undistorted 8-bit image the detector saw (`detect_img`), the raw
ASIFT detections shared by every adaptation arm (`detections`), each arm's
frames and filtered frames (`arms`, `name => (frames, filtered)`), and the
blob `cutoff`. [`PipelineTrace`](@ref) attaches the optional provenance.
"""
struct PipelineCapture{I, DI, DE, A}
    img::I                 # original image
    detect_img::DI         # undistorted UInt8 (what the detector saw)
    detections::DE         # raw ASIFT frames (shared by every arm)
    arms::A                # Vector{Pair{Symbol, NamedTuple}} — name => (frames, filtered)
    cutoff::Float64
end

# QUALIFIED: this file is included before pipeline.jl, where `Attributed` is
# imported, and a `const` is evaluated at load time.
"""
    PipelineTrace

A [`PipelineCapture`](@ref) with its OPTIONAL provenance attached as
`Attributed` attributes — the idiom VisualGeometryToolkitCore uses to carry a record's
optional payload, and for the same reason: what a trace carries beyond the
capture genuinely varies, and absence should be a fact about the TYPE rather
than a `nothing` every reader has to test.

Attributes, each present only when it was actually captured:

- `registration`: `(; accepted, bundles, cam)` — `accepted` is `find_boards`'
  output, so the inspector can draw each verified board's reprojected model and
  its inlier correspondences next to the detections that produced them. Absent
  when the trace stopped at filtering, and always absent on a LOADED trace: the
  models and bundle references are cheap to recompute but awkward to persist.
- `config`, `network_hash`: the configuration of record, on the same terms as
  `poses/_config.toml` — `config_dict` output plus the descriptor checkpoint's
  SHA-256. Without them a trace told you WHAT happened but not UNDER WHAT
  SETTINGS. Absent on traces written before they were recorded.

Field access forwards, so `trace.arms`, `trace.cutoff` and `trace.detections`
read exactly as before. Use `_has_attr` / `_attr` for the optional three.

NB no constructor is defined on this alias: it would be a method on
VisualGeometryToolkitCore's `Attributed`, i.e. piracy. Build one with
`Attributed(PipelineCapture(...), attrs)`.
"""
const PipelineTrace = VgtkCore.Attributed{V, A} where {V <: PipelineCapture, A}

# Presence is a COMPILE-TIME property of the attribute NamedTuple's type, so
# both of these constant-fold: a reader's branch on an absent attribute is
# eliminated rather than tested per call.
@inline _has_attr(::VgtkCore.Attributed{V, A}, s::Symbol) where {V, A} = s in fieldnames(A)
@inline _attr(t::VgtkCore.Attributed{V, A}, s::Symbol, default) where {V, A} =
    s in fieldnames(A) ? getproperty(t, s) : default

# The optional half, assembled from whatever was actually captured. Only the
# entries that exist reach the NamedTuple, which is what puts absence in the type.
function _trace_attrs(; registration = nothing, config = nothing, network_hash = nothing)
    a = NamedTuple()
    registration === nothing || (a = (; a..., registration))
    config === nothing || (a = (; a..., config))
    network_hash === nothing || (a = (; a..., network_hash))
    return a
end

"""
    trace_pipeline(cam, img; detect_config, cutoff=DEFAULT_CUTOFF,
                   device=KA.CPU(),
                   adapt_arms=[:default => AdaptConfig(cutoff = cutoff)])
        -> PipelineTrace

Run detect → adapt → filter, capturing every intermediate. All `adapt_arms`
consume the same detections. Supply `cam = nothing` to skip undistortion.

Registration (when a gallery + bundles + network are given) runs under ONE
`FindBoardsConfig` — geometry from `cam`, `detect = detect_config`, `adapt` =
the FIRST arm of `adapt_arms` — and that same config is what the trace records.
"""
function trace_pipeline(
        # Typed so a call in the retired `(img, cam)` order is a MethodError.
        # Qualified: `CameraModel` is imported by pipeline.jl, after this file.
        cam::Union{Nothing, VgtkCore.CameraModel}, img::AbstractMatrix;
        detect_config::DetectConfig = DetectConfig(),
        cutoff::Real = DEFAULT_CUTOFF,
        device = default_device(),
        adapt_arms::AbstractVector{<:Pair{Symbol, <:AdaptConfig}} =
            [:default => AdaptConfig(cutoff = cutoff)],
        board_feature_dict = nothing,
        board_bundles = nothing,
        descriptor_network = nothing,
        find_config = nothing
    )
    u8, adapt_img = _undistorted_pair(img, cam, device)   # the pipeline's one undistort-for-detect/adapt helper
    detections = detect_blobs(u8; config = detect_config, device)
    # Left UNTYPED and narrowed at the end: a `Pair{Symbol, NamedTuple}[]`
    # annotation widens each arm's concrete NamedTuple back to the UnionAll,
    # which is exactly the abstractness `PipelineCapture{…, A}` exists to keep
    # out of the record.
    arms = []
    for (name, acfg) in adapt_arms
        frames = adapt(detections, adapt_img; config = acfg, device)
        filtered = filter_blobs(
            frames, detections; cutoff,
            image = adapt_img, max_clip_ratio = 0.0
        )
        push!(arms, name => (; frames, filtered))
        # Each arm allocates its own flat-packed device buffers on top of the
        # resident detector plan; without collecting between arms a multi-arm
        # sweep exhausts the card (measured: 4 arms → 14.4 GiB pooled → OOM).
        GC.gc()
    end
    arms = identity.(arms)          # → Vector{Pair{Symbol, <concrete NamedTuple>}}
    # Registration is captured in the SAME trace so the inspector can show
    # detections → adaptation → inliers → reprojected board in one view.
    # Skipped unless the caller supplies a gallery + bundles + model.
    # ONE config: the one registration runs under is the one the trace records
    # (it used to record a geometry-less config while running PoseGeometry).
    cfg = @something(
        find_config, FindBoardsConfig(
            geometry = cam === nothing ? HomographyGeometry() : PoseGeometry(cam),
            detect = detect_config,
            adapt = last(first(adapt_arms))
        )
    )
    registration = if board_feature_dict === nothing || board_bundles === nothing ||
            descriptor_network === nothing
        nothing
    else
        accepted = find_boards(board_feature_dict, descriptor_network, img; config = cfg, device, board_bundles)
        (; accepted, bundles = board_bundles, cam)
    end
    net = descriptor_network === nothing ? nothing : _ensure_network(descriptor_network)
    return VgtkCore.Attributed(
        PipelineCapture(img, u8, detections, arms, Float64(cutoff)),
        _trace_attrs(;
            registration,
            config = config_dict(cfg),
            network_hash = net === nothing ? nothing : network_hash(net),
        )
    )
end

"""
    inspect_pipeline(trace::PipelineTrace; region=nothing) -> Figure

Linked-axes inspection grid: column 1 = raw detections over the detector
image; per adaptation arm, one column of fitted frames and one of the
post-filter survivors. Pan/zoom is synchronized across every panel.
`region = (xrange, yrange)` crops the initial view.

Implemented in BlobBoardsMakieExt (needs a Makie backend).
"""
function inspect_pipeline end

# =============================================================================
# Trace persistence — capture on the GPU, plot CUDA-free
# =============================================================================
#
# GL context creation fails in a process whose CUDA allocation has the card
# nearly full (the detector plan alone is ~13.5 GiB at 12 MP). Plain GLMakie
# renders the full 5-panel 12 MP grid without trouble — just not there. So
# the workflow splits: capture where the GPU is, plot where it isn't.

"""
    save_trace(path, trace::PipelineTrace)

Persist a trace for plotting in a separate, CUDA-free process. Device
arrays are brought to the host first.
"""
function save_trace(path::AbstractString, trace::PipelineTrace)
    JLD2.jldsave(
        path;
        detect_img = Array(trace.detect_img),
        detections = trace.detections,
        arm_names = [String(n) for (n, _) in trace.arms],
        arm_frames = [a.frames for (_, a) in trace.arms],
        arm_filt = [a.filtered for (_, a) in trace.arms],
        cutoff = trace.cutoff,
        config = _attr(trace, :config, nothing),
        network_hash = _attr(trace, :network_hash, nothing)
    )
    return path
end

"""
    _load_pipeline_trace(path) -> PipelineTrace

The FILE branch of `load_trace` (src/track_trace.jl): a JLD2 pipeline trace
written by `save_trace(::PipelineTrace)`.
"""
function _load_pipeline_trace(path::AbstractString)
    d = JLD2.load(path)
    arms = [
        Symbol(n) => (; frames = f, filtered = g)
            for (n, f, g) in zip(d["arm_names"], d["arm_frames"], d["arm_filt"])
    ]
    # Registration is not serialised: `accepted` carries pose/homography models
    # and bundle references that are cheap to recompute but awkward to persist.
    # A loaded trace therefore inspects detect→adapt→filter only — and now SAYS
    # so in its type, since the attribute is simply not attached.
    # `get` rather than indexing: traces written before configuration was
    # recorded still load, and those attributes stay absent too.
    return VgtkCore.Attributed(
        PipelineCapture(d["detect_img"], d["detect_img"], d["detections"], arms, d["cutoff"]),
        _trace_attrs(;
            config = get(d, "config", nothing),
            network_hash = get(d, "network_hash", nothing),
        )
    )
end
