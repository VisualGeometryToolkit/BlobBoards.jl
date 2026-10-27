# =============================================================================
# Migrating a board or pattern file of an earlier schema — the one reader of
# the retired `{value, unit}` quantity form
# =============================================================================
#
# A board file is read in ONE format, `blobboards/board/v5` (a pattern file,
# `blobboards/pattern/v3`): every quantity a number in the unit the file's
# `units` block declares (types.jl, json.jl). The schemas before it stored each
# quantity as a `{"value": …, "unit": …}` object. Nothing reads those at run
# time; a stored file is MIGRATED once, with VisualGeometryToolkitIO's
# `vgio_migrate` (or the `bin/vgio_migrate.jl` command), which finds the file's
# schema and calls the method of `migrate_schema` this file adds for it:
#
#     VisualGeometryToolkitIO.vgio_migrate("blob_board_ab1_c2d3e.json")
#
# A migration reads the old file with the LEGACY lift below (a vendored copy of
# the reader the board schemas v1–v4 were read with, `{value, unit}` and the
# v1/v2 ruler and `max_diameter_fraction` spellings included), writes the
# bundle through the v5 writer (`save_json`, with the file's own stored
# `hashes.config` and its `render` block), READS THE RESULT BACK with the v5
# reader and compares every value with what the old file read as, bit for bit.
# Any difference refuses the file and removes the output. The identity is the
# stored `hashes.config`, carried over unchanged; the bundle's identity
# recomputed from the migrated file is checked to be the one recomputed from
# the old file (`_bundle_identity`, which covers the identity schema and the
# placed blobs, never the file format, so a format bump moves no uid).
#
# What a migration does NOT carry: a key the old reader already ignored
# (`saddle_points`, `oversampling`, `sigma_cutoff`) and a v4 point-grid
# `pattern_resolution` / `pixel_type` / `raster_aligned` (derived, or constant,
# on a point grid).

# A quantity object of the retired form: both keys present.
_is_legacy_quantity(x) = x isa _JSONObject && _hasfield(x, "value") && _hasfield(x, "unit")

# The retired form's number or `{value, unit}` object, read: a number as it is,
# an object as the quantity (any unit name `parse_unit` reads, so every string a
# stored board holds loads: mm, px, dpi, pd, pt).
function _legacy_scalar(x)
    x isa _JSONObject || return x
    u = parse_unit(String(_getfield(x, "unit")); compound = true)
    v = _getfield(x, "value")
    return (dimension(u) == dimension(mm) ? Float64(v) : _snap_integer(v)) * u
end
_legacy_size2(x) = Size2(width = _legacy_scalar(_getfield(x, "width")), height = _legacy_scalar(_getfield(x, "height")))

function _legacy_ruler_tick(x)
    return RulerTick(_legacy_scalar(_getfield(x, "spacing")), _legacy_scalar(_getfield(x, "width")))
end

function _legacy_ruler_spec(x)
    major_tick = _legacy_ruler_tick(_getfield(x, "major_tick"))
    minor_tick = _legacy_ruler_tick(_getfield(x, "minor_tick"))
    # A pre-v3 ruler (board schema v1/v2) carries `width` and half-extent tick
    # widths: drop the one, double the others.
    if _getfield(x, "width", nothing) !== nothing
        major_tick = RulerTick(major_tick.spacing, 2 * major_tick.width)
        minor_tick = RulerTick(minor_tick.spacing, 2 * minor_tick.width)
    end
    spacing_promoted = promote(major_tick.spacing, minor_tick.spacing)
    width_promoted = promote(major_tick.width, minor_tick.width)
    return RulerSpec(
        RulerTick(spacing_promoted[1], width_promoted[1]),
        RulerTick(spacing_promoted[2], width_promoted[2]),
    )
end

function _legacy_pattern_config(x)
    pixel_type = _pixel_type_from_name(String(_getfield(x, "pixel_type")))
    pattern_resolution = _legacy_size2(_getfield(x, "pattern_resolution"))
    min_scale = _legacy_scalar(_getfield(x, "min_scale"))
    seed_raw = _getfield(x, "seed")
    seed = seed_raw isa AbstractString ? parse(UInt64, seed_raw) : UInt64(seed_raw)
    polarity_raw = _getfield(x, "polarity", nothing)
    polarity = polarity_raw !== nothing ? Symbol(polarity_raw) : :dark
    # v2 stores `max_scale`; a v1 file stores `max_diameter_fraction`, converted
    # to the top rung it always produced (same ladder, so the blobs re-place
    # bit-identically; the stored hash stays the identity).
    max_scale_raw = _getfield(x, "max_scale", nothing)
    max_fraction_raw = _getfield(x, "max_diameter_fraction", nothing)
    (max_scale_raw === nothing) == (max_fraction_raw === nothing) && throw(
        ArgumentError("pattern_config must carry exactly one of max_scale (v2) or max_diameter_fraction (v1)")
    )
    cutoff = Float64(something(_getfield(x, "cutoff", _getfield(x, "sigma_cutoff", nothing)), DEFAULT_CUTOFF))
    scales_per_octave = Int(_getfield(x, "scales_per_octave"))
    max_scale = if max_scale_raw !== nothing
        _legacy_scalar(max_scale_raw)
    else
        σ_min = Float64(ustrip(px, min_scale))
        k = _ladder_top_from_fraction(ustrip.(px, pattern_resolution), σ_min, Float64(max_fraction_raw), scales_per_octave, cutoff)
        σ_min * 2^(k / scales_per_octave) * px
    end
    alpha_raw = _getfield(x, "alpha", nothing)
    alpha_raw === nothing && throw(ArgumentError("pattern_config missing alpha"))
    return PatternConfig(
        pattern_resolution = pattern_resolution, min_scale = min_scale, seed = seed, pixel_type = pixel_type,
        max_scale = max_scale, scales_per_octave = scales_per_octave, max_trials = Int(_getfield(x, "max_trials")),
        α = Float64(alpha_raw), cutoff = cutoff, raster_aligned = Bool(_getfield(x, "raster_aligned")), polarity = polarity,
    )
end

function _legacy_grid(x)
    g = _getfield(x, "layout_grid", nothing)
    if g !== nothing
        (Float64(_getfield(g, "value")) == 0.01 && String(_getfield(g, "unit")) == "pt") ||
            throw(ArgumentError("board_config.layout_grid must be {value: 0.01, unit: \"pt\"}, got $(g)"))
        return PointGrid()
    end
    return PixelGrid(_legacy_scalar(_getfield(x, "print_density")))
end

function _legacy_board_config(x)
    canvas_size = _legacy_size2(_getfield(x, "canvas_size"))
    layout_grid = _legacy_grid(x)
    sized = layout_grid isa PixelGrid ? (; board_size = _legacy_size2(_getfield(x, "board_size"))) :
        (; margin = _legacy_size2(_getfield(x, "margin")))
    q(k) = _legacy_scalar(_getfield(x, k))
    frame_padding = q("frame_padding")
    ruler_spec = _legacy_ruler_spec(_getfield(x, "ruler_spec"))
    marks = if layout_grid isa PixelGrid
        (; ruler_width = q("frame_width"), frame_style = Symbol(_getfield(x, "frame_style")), _tick_kwargs(ruler_spec)...)
    else
        rw = q("ruler_width")
        (; ruler_width = rw, corner_size = q("corner_size"), (iszero(rw) ? (;) : _tick_kwargs(ruler_spec))...)
    end
    return BoardConfig(; canvas_size, sized..., layout_grid, frame_padding, marks...)
end

# A point grid's pattern config in the retired form: lengths `{value, "pt"}`
# (the stored px, declared as the points they are), the pattern size, pixel type
# and `raster_aligned` derived or constant (a v4 file written before 2026-10-14
# records them).
function _legacy_pattern_from_points(x, bc::BoardConfig)
    function stored(q)
        u = String(_getfield(q, "unit"))
        u == "pt" || throw(ArgumentError("a point-grid board's pattern lengths are in \"pt\", got \"$u\""))
        return Dict{String, Any}("value" => _getfield(q, "value"), "unit" => "px")
    end
    res = _getfield(x, "pattern_resolution", nothing)
    d = Dict{String, Any}(string(k) => v for (k, v) in pairs(x))
    d["pattern_resolution"] = if res === nothing
        p = _board_pattern_points(bc)
        Dict{String, Any}(k => Dict{String, Any}("value" => ustrip(px, q), "unit" => "px") for (k, q) in (("width", p.width), ("height", p.height)))
    else
        Dict{String, Any}("width" => stored(_getfield(res, "width")), "height" => stored(_getfield(res, "height")))
    end
    for k in ("min_scale", "max_scale")
        v = _getfield(x, k, nothing)
        v === nothing || (d[k] = stored(v))      # (a v1 config has no max_scale)
    end
    t = get(d, "pixel_type", _pixel_type_name(DEFAULT_PIXEL_TYPE))
    t == _pixel_type_name(DEFAULT_PIXEL_TYPE) ||
        throw(ArgumentError("a point-grid board has no pixel type; this one records $(repr(t))"))
    d["pixel_type"] = t
    a = get(d, "raster_aligned", false)
    a === false || throw(ArgumentError("a point-grid board's centres sit on its grid: raster_aligned must be false, got $(repr(a))"))
    d["raster_aligned"] = false
    return d
end

# The retired columns declared their own `unit`: it must be the grid's.
function _legacy_blobs(data, unit::String)
    data isa AbstractVector && return _construct_blobs(data)     # (the retired row form: refused there)
    String(_getfield(data, "unit")) == unit ||
        throw(ArgumentError("columnar blobs must declare unit \"$unit\", got \"$(_getfield(data, "unit"))\""))
    return _construct_blobs((centers = _getfield(data, "centers"), sigmas = _getfield(data, "sigmas")))
end

function _legacy_board_bundle(x)
    pre = _getfield(x, "preamble")
    bc = _legacy_board_config(_getfield(pre, "board_config"))
    raw = _getfield(pre, "pattern_config")
    pc = _legacy_pattern_config(bc.layout_grid isa PointGrid ? _legacy_pattern_from_points(raw, bc) : raw)
    blobs = _legacy_blobs(_getfield(x, "blobs"), _blob_unit(bc.layout_grid))
    return BoardBundle(BoardPreamble(bc, pc), blobs)
end

function _legacy_pattern_bundle(x)
    return PatternBundle(_legacy_pattern_config(_getfield(x, "pattern_config")), _legacy_blobs(_getfield(x, "blobs"), "px"))
end

# -----------------------------------------------------------------------------
# The check: two bundles hold the same values, bit for bit
# -----------------------------------------------------------------------------

# Same type and the same bits in every leaf (a signed zero, an `Int` quantity
# and its `Float64` twin all differ). A struct is compared field by field, an
# array element by element.
_same_value(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(_same_value(x, y) for (x, y) in zip(a, b))
_same_value(a::Union{Type, Symbol, AbstractString}, b::Union{Type, Symbol, AbstractString}) = isequal(a, b)
_same_value(a::T, b::T) where {T} =
    isbitstype(T) ? a === b : (fieldcount(T) > 0 ? all(_same_value(getfield(a, f), getfield(b, f)) for f in fieldnames(T)) : isequal(a, b))
_same_value(a, b) = false

# A render block's quantity objects as the numbers a v5 file holds, key by key
# in the file's order; every other entry (the format, the profile, the id label,
# the shading function, even a legacy key name) as it is.
function _migrated_render(r::_JSONObject)
    out = _ParsedObject()
    for (k, v) in pairs(r)
        out[String(k)] = _is_legacy_quantity(v) ? _file_number(_legacy_scalar(v)) : v
    end
    return out
end

# The migrated block against the old one: a quantity object became the number
# that reads back as that very quantity; the rest is equal.
function _render_migrated_ok(old::_JSONObject, new)
    new isa _JSONObject && Set(String.(keys(old))) == Set(String.(keys(new))) || return false
    return all(pairs(old)) do (k, v)
        w = _getfield(new, String(k))
        if _is_legacy_quantity(v)
            q = _legacy_scalar(v)
            kind = dimension(q) == dimension(mm) ? :length : dimension(q) == dimension(dpi) ? :density : :pixel
            w isa Real && _same_value(_file_quantity(w, kind), q)
        else
            isequal(v, w)
        end
    end
end

function _migrate_bundle_file(src::AbstractString, dst::AbstractString, from::AbstractString; after_write = identity)
    try
        return _migrate_bundle_file_checked(src, dst, from, after_write)
    catch
        rm(dst; force = true)         # a refused migration leaves no output at `dst`
        rethrow()
    end
end

# (`after_write(dst)`: a test hook run on the written file before it is read back)
function _migrate_bundle_file_checked(src::AbstractString, dst::AbstractString, from::AbstractString, after_write)
    obj = JSON.parse(read(src, String))::_ParsedObject
    get(obj, :schema, nothing) == from ||
        throw(ArgumentError("$(src): declares schema $(repr(get(obj, :schema, nothing))), not $(from)"))
    old = if haskey(obj, :preamble)
        _legacy_board_bundle(obj)
    elseif haskey(obj, :pattern_config)
        _legacy_pattern_bundle(obj)
    else
        throw(ArgumentError("$(src): neither a board (preamble) nor a pattern (pattern_config)"))
    end
    stored = _stored_config_hash(obj)
    stored === nothing && throw(
        ArgumentError("$(src): stores no hashes.config; a migration carries a stored identity over and mints none")
    )
    # The uid must not move: the stored `hashes.config` is authoritative for every board kind and is carried over
    # unchanged, and the identity recomputed from the migrated file must be the one recomputed from the old file
    # (under the current, frozen encoding). Whether either equals the stored hash is information, never a refusal:
    # boards minted under an older encoding (all 50 example boards) recompute to something else.
    identity_old = _bundle_identity(old)
    identity_old == stored ||
        @info "migrating $(basename(src)): the identity recomputed under the current encoding ($(identity_old[1:8])…) is not the stored hashes.config ($(stored[1:8])…); the stored hash is carried over"
    render = haskey(obj, :render) ? _migrated_render(obj[:render]::_JSONObject) : nothing
    save_json(dst, old; overwrite = true, config_hash = stored, render)
    after_write(dst)
    new, config_hash = _parse_bundle(dst)
    _same_value(old, new) ||
        throw(ArgumentError("$(src): the migration does not read back as the file read (a number moved); the output is discarded"))
    config_hash == stored ||
        throw(ArgumentError("$(src): the migration does not keep the stored identity; the output is discarded"))
    _bundle_identity(new) == identity_old ||
        throw(ArgumentError("$(src): the identity recomputed from the migration differs from the old file's; the output is discarded"))
    if render !== nothing
        written = JSON.parse(read(dst, String))::_ParsedObject
        _render_migrated_ok(obj[:render]::_JSONObject, get(written, :render, nothing)) ||
            throw(ArgumentError("$(src): the migrated render block does not read back as the old one; the output is discarded"))
    end
    return (; to = String(get(JSON.parse(read(dst, String))::_ParsedObject, :schema, "")), uid = uid_from_hash(stored), config_hash = stored)
end

"""
    VisualGeometryToolkitIO.migrate_schema(::Val{Symbol("blobboards/board/v4")}, src, dst)

BlobBoards' migrations of its retired `{value, unit}` schemas — board v1 to v4
and pattern v1 and v2 — into `blobboards/board/v5` and `blobboards/pattern/v3`,
by the contract of [`VisualGeometryToolkitIO.migrate_schema`]: write `dst`
through the current writer, read it back, compare every value bit for bit, keep
the stored identity. Run by `VisualGeometryToolkitIO.vgio_migrate` (and its
command), which stages the result and replaces the file only once verified.
Returns `(; to, uid, config_hash)`.
"""
VgtkIO.migrate_schema(::Val{Symbol("blobboards/board/v4")}, src::AbstractString, dst::AbstractString) = _migrate_bundle_file(src, dst, "blobboards/board/v4")
VgtkIO.migrate_schema(::Val{Symbol("blobboards/board/v3")}, src::AbstractString, dst::AbstractString) = _migrate_bundle_file(src, dst, "blobboards/board/v3")
VgtkIO.migrate_schema(::Val{Symbol("blobboards/board/v2")}, src::AbstractString, dst::AbstractString) = _migrate_bundle_file(src, dst, "blobboards/board/v2")
VgtkIO.migrate_schema(::Val{Symbol("blobboards/board/v1")}, src::AbstractString, dst::AbstractString) = _migrate_bundle_file(src, dst, "blobboards/board/v1")
VgtkIO.migrate_schema(::Val{Symbol("blobboards/pattern/v2")}, src::AbstractString, dst::AbstractString) = _migrate_bundle_file(src, dst, "blobboards/pattern/v2")
VgtkIO.migrate_schema(::Val{Symbol("blobboards/pattern/v1")}, src::AbstractString, dst::AbstractString) = _migrate_bundle_file(src, dst, "blobboards/pattern/v1")
