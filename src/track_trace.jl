# =============================================================================
# TrackTrace — capture/replay records for track_boards visualization
# =============================================================================
#
# One record schema, three consumers (design 2026-07-28): a live `on_frame`
# sink, an in-memory/streamed `TrackTrace`, and the on-disk trace bundle.
# `track_boards` has ONE sink, the `on_frame` callback: it receives a
# `TrackFrame` (the record, the frame's pixels, the configuration of record),
# and a caller that wants a trace pushes it — `on_frame = f -> push!(trace, f)`.
# The trace is never a keyword the tracker fills in (2026-09-17: an object the
# function mutates is not an output channel). Headless is structurally free —
# with `on_frame === nothing` (the default) no record is ever built. There is
# deliberately NO `visualize=` flag on the entry points: sinks compose (one
# closure can feed a trace and a live view), flags don't. Rendering
# (`plot_tracks`, `replay`, `record_tracks`) lives in BlobBoardsMakieExt.
#
# On-disk layout (directory bundle, annotate-pipeline precedent — streaming
# append, random access, crash-tolerant):
#
#   x.trace/
#   ├── trace.json       manifest: schema_version, source, keep_frames,
#   │                    n_frames, gallery uids
#   ├── frames/000001.jld2   one TrackFrameRecord per frame
#   └── images/000001.png    only for keep_frames ∈ (:thumbs, :full)
# =============================================================================

const _TRACE_SCHEMA_VERSION = 1

"""
    BoardFrameEntry

Per-board diagnostics for one frame of a `TrackFrameRecord`: how the frame
was won, the fit, the warm bet's evidence (associations + predictions), and
the COMMITTED model's board projection — the artifacts each panel of the
replay triptych (detections | association | boards) plays back from.
"""
struct BoardFrameEntry{M}
    uid::String
    source::Symbol                       # :detect | :init | :track
    model::M
    quality::Float64
    n_inliers::Int
    associations::Vector{NTuple{3, Float64}}   # (world_idx, det_idx, iou); empty for non-warm
    predicted::Vector{NTuple{5, Float64}}      # predicted ellipses (cx, cy, a, b, θ); empty for non-warm
    projected::Vector{NTuple{5, Float64}}      # committed-model board blob projections (cx, cy, a, b, θ)
end

"""
    TrackFrameRecord

Everything needed to render (and debug) one tracked frame without re-running
the pipeline: the per-board entries, the frame's detection ellipses, and
stage timings. Frame PIXELS are governed by the trace's `keep_frames`
policy, not stored here.
"""
struct TrackFrameRecord
    frame_idx::Int
    boards::Vector{BoardFrameEntry}
    detections::Vector{NTuple{5, Float64}}     # detected ellipses (cx, cy, a, b, θ) at DEFAULT_CUTOFF
    t_extract::Float64
    t_estimate::Float64
end

"""
    TrackFrame

One frame of a `track_boards` run, as its `on_frame` sink receives it: the
`record` (a `TrackFrameRecord`), the frame `image` it was measured on, and the
run's configuration of record — `config` (`config_dict` output, AFTER the
board-derived polarity is folded in, so it is what actually ran) and the
descriptor `network_hash`. The last two are the same for every frame of a call.
`push!(trace::TrackTrace, frame)` records it.
"""
struct TrackFrame{I}
    record::TrackFrameRecord
    image::I
    config::Dict{String, Any}
    network_hash::String
end

# Every optional field carries an EMPTY SENTINEL rather than `nothing` — an
# empty path/hash is "unset", an empty `Dict` is "no configuration recorded",
# and a zero `Size2` is "no frame seen yet". The struct is mutable and these are
# stamped after construction, so a type parameter cannot express the absence
# (it would freeze `Nothing` at construction); a sentinel can, and keeps every
# field concrete. The `trace.json` manifest still writes `null` for an unset
# field, so the on-disk format is unchanged and old traces load.
"""
    TrackTrace(; keep_frames = :none, dir = nothing, source = nothing)

Capture sink for `track_boards`: push the frames its `on_frame` callback
receives, `track_boards(…; on_frame = f -> push!(trace, f))`. Records
accumulate in memory; when `dir` is given each record is ALSO streamed to the
trace bundle as it is produced (bounded memory on long videos, crash-tolerant).

- `keep_frames`: `:none` (replay re-decodes pixels from `source`),
  `:thumbs` (¼-scale PNGs), `:full` (full-resolution PNGs). Non-`:none`
  policies require `dir`.
- `source`: the video path (or `nothing` for in-memory frame sequences) —
  recorded in the manifest so `:none` replays can re-decode.
"""
mutable struct TrackTrace
    keep_frames::Symbol
    dir::String
    source::String
    records::Vector{TrackFrameRecord}
    # Stamped from the pushed `TrackFrame`s, not by the caller: the same
    # configuration of record that poses/_config.toml carries (config_dict
    # output + the checkpoint hash), as `track_boards` resolved it, so a
    # replayed trace can say what produced it.
    # `Dict{String, Any}`, not a bare `Dict`: the latter is a UnionAll, which
    # makes the field abstract for no gain — `config_dict` returns exactly this
    # type, and the replay path (`_read_meta!`) already constructs it. A type
    # PARAMETER would be wrong here: the struct is mutable and this field is
    # stamped after construction, so `Nothing` would freeze into the type.
    config::Dict{String, Any}
    network_hash::String
    # Extent of the FULL-resolution frames, captured on the first append.
    # Recorded artifacts live in full-resolution pixel coordinates, so a
    # renderer needs this to scale a :thumbs image and to pin axis limits —
    # size-of-thumb × stride is only exact when the frame divides evenly.
    frame_size::Size2{Int}
    function TrackTrace(;
            keep_frames::Symbol = :none,
            dir::Union{Nothing, AbstractString} = nothing,
            source::Union{Nothing, AbstractString} = nothing
        )
        keep_frames in (:none, :thumbs, :full) ||
            throw(ArgumentError("keep_frames must be :none, :thumbs or :full"))
        keep_frames === :none || dir !== nothing ||
            throw(ArgumentError("keep_frames = $keep_frames requires dir"))
        if dir !== nothing
            mkpath(joinpath(dir, "frames"))
            keep_frames === :none || mkpath(joinpath(dir, "images"))
        end
        return new(
            keep_frames, dir === nothing ? "" : String(dir),
            source === nothing ? "" : String(source), TrackFrameRecord[],
            Dict{String, Any}(), "", Size2(0, 0)
        )
    end
end

Base.length(t::TrackTrace) = length(t.records)
Base.getindex(t::TrackTrace, i::Int) = t.records[i]
Base.iterate(t::TrackTrace, s...) = iterate(t.records, s...)

const _THUMB_STRIDE = 4

"""
    push!(trace::TrackTrace, frame::TrackFrame) -> trace

Record one `track_boards` frame: stamp the run's configuration of record, keep
the record in memory and, when the trace has a `dir`, stream it (and its image,
per `keep_frames`) to the bundle.
"""
function Base.push!(trace::TrackTrace, frame::TrackFrame)
    trace.config = frame.config
    trace.network_hash = frame.network_hash
    _trace_append!(trace, frame.record, frame.image)
    return trace
end

# Append a record: memory + optional streamed frame file + optional image.
function _trace_append!(trace::TrackTrace, rec::TrackFrameRecord, img)
    push!(trace.records, rec)
    iszero(trace.frame_size.width) && img !== nothing &&
        (trace.frame_size = Size2(img))
    dir = trace.dir
    if !isempty(dir)
        JLD2.jldsave(
            _trace_frame_path(dir, rec.frame_idx);
            record = rec
        )
        if trace.keep_frames !== :none && img !== nothing
            out = trace.keep_frames === :thumbs ?
                img[1:_THUMB_STRIDE:end, 1:_THUMB_STRIDE:end] : img
            FileIO.save(
                _trace_image_path(dir, rec.frame_idx),
                _clamp_patch(out)
            )
        end
    end
    return nothing
end

# The single emit point track_boards calls. Compiles to a no-op when the sink
# is `nothing`; `on_frame` failures must not kill tracking.
@inline function _emit_track_frame(on_frame, frame::TrackFrame)
    on_frame === nothing && return nothing
    try
        on_frame(frame)
    catch err
        @warn "track_boards on_frame sink failed" exception = err maxlog = 3
    end
    return nothing
end

# On-disk names of a trace bundle's per-frame files — five sites spelled them.
_trace_frame_path(dir, idx) = joinpath(dir, "frames", string(lpad(Int(idx), 6, '0'), ".jld2"))
_trace_image_path(dir, idx) = joinpath(dir, "images", string(lpad(Int(idx), 6, '0'), ".png"))
# Clamp any float patch into the stored Gray{N0f8} range (also annotate.jl's).
_clamp_patch(p) = Gray{N0f8}.(clamp.(Float32.(p), 0.0f0, 1.0f0))

"""
    save_trace(dir, trace::TrackTrace)

Write the trace bundle manifest (records already stream to `dir` when the
trace was constructed with one; for in-memory traces this writes everything).
"""
function save_trace(dir::AbstractString, trace::TrackTrace)
    mkpath(joinpath(dir, "frames"))
    if trace.dir != dir            # in-memory trace, or copying to a new dir
        for rec in trace.records
            JLD2.jldsave(
                _trace_frame_path(dir, rec.frame_idx);
                record = rec
            )
        end
    end
    manifest = Dict(
        "schema_version" => _TRACE_SCHEMA_VERSION,
        "keep_frames" => String(trace.keep_frames),
        # `null` for an unset field, exactly as before the sentinels landed —
        # the manifest schema is unchanged.
        "source" => isempty(trace.source) ? nothing : trace.source,
        "config" => isempty(trace.config) ? nothing : trace.config,
        "network_hash" => isempty(trace.network_hash) ? nothing : trace.network_hash,
        "frame_size" => iszero(trace.frame_size.width) ? nothing :
            Dict("width" => trace.frame_size.width, "height" => trace.frame_size.height),
        "n_frames" => length(trace.records),
        "frame_indices" => [r.frame_idx for r in trace.records],
    )
    open(joinpath(dir, "trace.json"), "w") do io
        _write_json(io, manifest)
    end
    return dir
end

"""
    load_trace(path) -> TrackTrace | PipelineTrace

One generic loader for both trace kinds: a DIRECTORY is a `track_boards`
trace bundle (`TrackTrace`); a FILE is an `inspect.jl` pipeline trace
(`PipelineTrace`, JLD2).
"""
load_trace(path::AbstractString) =
    isdir(path) ? _load_track_trace(path) : _load_pipeline_trace(path)

function _load_track_trace(dir::AbstractString)
    manifest = JSON.parse(read(joinpath(dir, "trace.json"), String))::_ParsedObject
    schema = manifest["schema_version"]::Int64
    schema == _TRACE_SCHEMA_VERSION ||
        error("trace schema version $schema ≠ $_TRACE_SCHEMA_VERSION")
    keep = manifest["keep_frames"]::String
    src = manifest["source"]::Union{Nothing, String}
    trace = TrackTrace(;
        keep_frames = Symbol(keep),
        dir = keep == "none" ? nothing : dir,
        source = src
    )
    trace.dir = String(dir)
    # Restore the configuration of record. `save_trace` writes these; without
    # reading them back the provenance survived only in memory, which defeats
    # the point of recording it. `get` so traces written before this still load.
    let c = get(manifest, :config, nothing)
        trace.config = c === nothing ? Dict{String, Any}() : Dict{String, Any}(String(k) => v for (k, v) in c)
    end
    let h = get(manifest, :network_hash, nothing)
        trace.network_hash = h === nothing ? "" : String(h)
    end
    let fs = get(manifest, :frame_size, nothing)
        # `isa`, not `=== nothing`: JSON.parse types the value as a wide union and
        # JET union-splits the getindex — the typed check narrows it.
        trace.frame_size = fs isa _ParsedObject ?
            Size2(Int(fs["width"]::Real), Int(fs["height"]::Real)) : Size2(0, 0)
    end
    for idx in manifest["frame_indices"]::Vector{Any}
        p = _trace_frame_path(dir, Int(idx))
        push!(trace.records, JLD2.load(p, "record"))
    end
    return trace
end

"""
    _trace_frame_image(trace, i) -> Matrix{Gray} | nothing

Pixels for the i-th record, per the `keep_frames` policy: stored PNG when
kept, re-decoded from the source video for `:none` (exact — the manifest
records the frame index), `nothing` when neither is available (viewers draw
overlays on a blank canvas).
"""
function _trace_frame_image(trace::TrackTrace, i::Int)
    rec = trace.records[i]
    dir, src = trace.dir, trace.source
    if trace.keep_frames !== :none && !isempty(dir)
        p = _trace_image_path(dir, rec.frame_idx)
        isfile(p) && return FileIO.load(p)
    end
    if !isempty(src) && isfile(src)
        v = VideoIO.openvideo(src)
        local img = nothing
        try
            for (n, raw) in enumerate(v)
                if n == rec.frame_idx
                    img = Gray{Float32}.(raw)
                    break
                end
            end
        finally
            close(v)
        end
        return img
    end
    return nothing
end
