# =============================================================================
# Migrating a board or pattern file of an earlier schema — the one reader of
# the retired `{value, unit}` quantity form
# =============================================================================
#
# A board file is written in ONE format, `blobboards/board/v7.0` (a pattern
# file, `blobboards/pattern/v5.0`): every quantity a number in the unit the
# file's `units` object declares (IO 0.10's envelope; types.jl, json.jl), in
# the vgio file shape. A record of the retired units layout (board v5/v6,
# pattern v3/v4) has its `units` read through IO's `migrate_units` first
# (`_with_current_units`).
# Every earlier schema is still READ — published boards keep their schemas (a
# schema change must not force the published data to be uploaded again):
# `load_bundle` runs the version step of its schema in memory
# (`migrate_schema(::Val{schema}, tree)`, through VisualGeometryToolkitIO's
# `read_record`), and the file is untouched. A stored file can also be
# MIGRATED once, with VisualGeometryToolkitIO's `vgio_migrate` (or the
# `bin/vgio_migrate.jl` command), which finds the file's schema and calls the
# file form of `migrate_schema` this file adds for it:
#
#     VisualGeometryToolkitIO.vgio_migrate("blob_board_ab1_c2d3e.json")
#
# Every step reads the old record into its bundle and gives the tree the current
# writer writes for it (the step in memory, the file form on disk), by one of two
# lifts:
#
# - board v5 and pattern v3 (`PRINTED_SCHEMAS`) hold the current record under
#   an earlier name (they were printed by a JSON3.pretty clone, which printed
#   an integral float as an integer): the current lift.
# - board v1–v4 and pattern v1–v2 (`MIGRATED_SCHEMAS`) hold the `{value, unit}`
#   form: they are read with the LEGACY lift below (a vendored copy of the
#   reader those schemas were read with, `{value, unit}` and the v1/v2 ruler
#   and `max_diameter_fraction` spellings included).
#
# A file migration writes the bundle through the current writer (`save_json`,
# with the file's own stored `hashes.config` and its `render` block), READS
# THE RESULT BACK with the current reader and compares every value with what
# the old file read as, bit for bit. Any difference refuses the file and
# removes the output. The identity is the stored `hashes.config`, carried over
# unchanged; the bundle's identity recomputed from the migrated file is checked
# to be the one recomputed from the old file (`_bundle_identity`, which covers
# the identity schema and the placed blobs, never the file format, so a format
# bump moves no uid).
#
# What a migration does NOT carry: a key the old reader already ignored
# (`saddle_points`, `oversampling`, `sigma_cutoff`) and a v4 point-grid
# `pattern_resolution` / `pixel_type` / `raster_aligned` (derived, or constant,
# on a point grid).

# A quantity object of the retired form: both keys present.
_is_legacy_quantity(x) = x isa AbstractDict && haskey(x, :value) && haskey(x, :unit)

# The retired form's number or `{value, unit}` object, read: a number as it is,
# an object as the quantity (any unit name `parse_unit` reads, so every string a
# stored board holds loads: mm, px, dpi, pd, pt).
function _legacy_scalar(x)
    x isa AbstractDict || return x
    u = parse_unit(String(x.unit); compound = true)
    v = x.value
    return (dimension(u) == dimension(mm) ? Float64(v) : VgtkIO.snap_integer(v)) * u
end
_legacy_size2(x) = Size2(width = _legacy_scalar(x.width), height = _legacy_scalar(x.height))

_legacy_ruler_tick(x) = RulerTick(_legacy_scalar(x.spacing), _legacy_scalar(x.width))

function _legacy_ruler_spec(x)
    major_tick = _legacy_ruler_tick(x.major_tick)
    minor_tick = _legacy_ruler_tick(x.minor_tick)
    # A pre-v3 ruler (board schema v1/v2) carries `width` and half-extent tick
    # widths: drop the one, double the others.
    if get(x, :width, nothing) !== nothing
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
    pixel_type = _pixel_type_from_name(String(x.pixel_type))
    pattern_resolution = _legacy_size2(x.pattern_resolution)
    min_scale = _legacy_scalar(x.min_scale)
    seed_raw = x.seed
    seed = seed_raw isa AbstractString ? parse(UInt64, seed_raw) : UInt64(seed_raw)
    polarity_raw = get(x, :polarity, nothing)
    polarity = polarity_raw !== nothing ? Symbol(polarity_raw) : :dark
    # v2 stores `max_scale`; a v1 file stores `max_diameter_fraction`, converted
    # to the top rung it always produced (same ladder, so the blobs re-place
    # bit-identically; the stored hash stays the identity).
    max_scale_raw = get(x, :max_scale, nothing)
    max_fraction_raw = get(x, :max_diameter_fraction, nothing)
    (max_scale_raw === nothing) == (max_fraction_raw === nothing) && throw(
        ArgumentError("pattern_config must carry exactly one of max_scale (v2) or max_diameter_fraction (v1)")
    )
    cutoff = Float64(something(get(x, :cutoff, get(x, :sigma_cutoff, nothing)), DEFAULT_CUTOFF))
    scales_per_octave = Int(x.scales_per_octave)
    max_scale = if max_scale_raw !== nothing
        _legacy_scalar(max_scale_raw)
    else
        σ_min = Float64(ustrip(px, min_scale))
        k = _ladder_top_from_fraction(ustrip.(px, pattern_resolution), σ_min, Float64(max_fraction_raw), scales_per_octave, cutoff)
        σ_min * 2^(k / scales_per_octave) * px
    end
    alpha_raw = get(x, :alpha, nothing)
    alpha_raw === nothing && throw(ArgumentError("pattern_config missing alpha"))
    return PatternConfig(
        pattern_resolution = pattern_resolution, min_scale = min_scale, seed = seed, pixel_type = pixel_type,
        max_scale = max_scale, scales_per_octave = scales_per_octave, max_trials = Int(x.max_trials),
        α = Float64(alpha_raw), cutoff = cutoff, raster_aligned = Bool(x.raster_aligned), polarity = polarity,
    )
end

function _legacy_grid(x)
    g = get(x, :layout_grid, nothing)
    if g !== nothing
        (Float64(g.value) == 0.01 && String(g.unit) == "pt") ||
            throw(ArgumentError("board_config.layout_grid must be {value: 0.01, unit: \"pt\"}, got $(g)"))
        return PointGrid()
    end
    return PixelGrid(_legacy_scalar(x.print_density))
end

function _legacy_board_config(x)
    canvas_size = _legacy_size2(x.canvas_size)
    layout_grid = _legacy_grid(x)
    sized = layout_grid isa PixelGrid ? (; board_size = _legacy_size2(x.board_size)) :
        (; margin = _legacy_size2(x.margin))
    q(k) = _legacy_scalar(x[k])
    frame_padding = q(:frame_padding)
    ruler_spec = _legacy_ruler_spec(x.ruler_spec)
    marks = if layout_grid isa PixelGrid
        (; ruler_width = q(:frame_width), frame_style = Symbol(x.frame_style), _tick_kwargs(ruler_spec)...)
    else
        rw = q(:ruler_width)
        (; ruler_width = rw, corner_size = q(:corner_size), (iszero(rw) ? (;) : _tick_kwargs(ruler_spec))...)
    end
    return BoardConfig(; canvas_size, sized..., layout_grid, frame_padding, marks...)
end

# A point grid's pattern config in the retired form: lengths `{value, "pt"}`
# (the stored px, declared as the points they are), the pattern size, pixel type
# and `raster_aligned` derived or constant (a v4 file written before 2026-10-14
# records them).
function _legacy_pattern_from_points(x, bc::BoardConfig)
    function stored(q)
        u = String(q.unit)
        u == "pt" || throw(ArgumentError("a point-grid board's pattern lengths are in \"pt\", got \"$u\""))
        return _object("value" => q.value, "unit" => "px")
    end
    res = get(x, :pattern_resolution, nothing)
    d = _object(x)
    d["pattern_resolution"] = if res === nothing
        p = _board_pattern_points(bc)
        _object((k => _object("value" => ustrip(px, q), "unit" => "px") for (k, q) in (("width", p.width), ("height", p.height)))...)
    else
        _object("width" => stored(res.width), "height" => stored(res.height))
    end
    for k in ("min_scale", "max_scale")
        v = get(x, k, nothing)
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
    String(data.unit) == unit ||
        throw(ArgumentError("columnar blobs must declare unit \"$unit\", got \"$(data.unit)\""))
    return _construct_blobs((centers = data.centers, sigmas = data.sigmas))
end

function _legacy_board_bundle(x)
    pre = x.preamble
    bc = _legacy_board_config(pre.board_config)
    raw = pre.pattern_config
    pc = _legacy_pattern_config(bc.layout_grid isa PointGrid ? _legacy_pattern_from_points(raw, bc) : raw)
    blobs = _legacy_blobs(x.blobs, _blob_unit(bc.layout_grid))
    return BoardBundle(BoardPreamble(bc, pc), blobs)
end

_legacy_pattern_bundle(x) = PatternBundle(_legacy_pattern_config(x.pattern_config), _legacy_blobs(x.blobs, "px"))

# -----------------------------------------------------------------------------
# The check: two bundles hold the same values, bit for bit
# -----------------------------------------------------------------------------

# VisualGeometryToolkitIO's `same_value` at `length_ulps = 0`: the same type and
# the same bits in every leaf (a signed zero, an `Int` quantity and its
# `Float64` twin all differ). IO's migrations compare a length within 2 ulp, the
# decimal-exact codec's guarantee; a board's numbers are products, read back
# bit for bit, so its check is exact.

# The render block's quantities, by key, with their kinds: a PDF's bleed, a
# raster's density (and the key it had before 2026-10-14). Every other entry is
# not a quantity.
const _RENDER_QUANTITIES = (bleed = :length, density = :density, print_density = :density)

# A render entry's quantity and its kind, read under the units its file declares
# (`_FILE_UNITS`): a retired `{value, unit}` object (its kind its dimension's), or
# a number under a quantity key; `nothing` for any other entry.
function _render_quantity(k, v)
    if _is_legacy_quantity(v)
        q = _legacy_scalar(v)
        return (q, dimension(q) == dimension(mm) ? :length : dimension(q) == dimension(dpi) ? :density : :pixel)
    end
    kind = get(_RENDER_QUANTITIES, Symbol(k), nothing)
    return kind !== nothing && v isa Real && !(v isa Bool) ? (_file_quantity(v, kind), kind) : nothing
end

# A render block as a current file in the world unit `unit` holds it, key by key
# in the file's order. A number is carried VERBATIM, never through the unit
# conversion: a v5 file's numbers are in the unit the migrated file keeps (a
# length only takes the Float64 form the writer gives every length, so v5's
# `"bleed": 0` is `0.0`; a density stays as stated, `52.4` included). A retired
# `{value, unit}` object becomes its `value` when its unit is the file's for its
# kind (mm, dpi, px), and is converted only when it is not. Every other entry
# (the format, the profile, the id label, the shading function, even a legacy
# key name) is as it is.
function _migrated_render(r::AbstractDict, unit)
    out = _ParsedObject()
    for (k, v) in pairs(r)
        out[String(k)] = _migrated_render_value(k, v, unit)
    end
    return out
end
function _migrated_render_value(k, v, unit)
    qk = _render_quantity(k, v)
    qk === nothing && return v
    q, kind = qk
    if _is_legacy_quantity(v)
        stated = parse_unit(String(v.unit); compound = true)
        own = kind === :length ? unit : kind === :density ? dpi : px
        stated == own || return _file_number(q; unit)        # a unit the file does not declare: converted
        v = v.value
    end
    return kind === :length ? Float64(v) : VgtkIO.snap_integer(v)
end

# The migrated block against the old one, each read under the units its own file
# declares (`old_units`, `new_units`): every quantity reads back as that very
# quantity, the rest is equal.
function _render_migrated_ok(old::AbstractDict, new, old_units, new_units)
    new isa AbstractDict && Set(String.(keys(old))) == Set(String.(keys(new))) || return false
    return all(pairs(old)) do (k, v)
        w = new[String(k)]
        qk = @with _FILE_UNITS => old_units _render_quantity(k, v)
        qk === nothing && return isequal(v, w)
        q, kind = qk
        w isa Real && VgtkIO.same_value(@with(_FILE_UNITS => new_units, _file_quantity(w, kind)), q; length_ulps = 0)
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

# A record of the earlier schema `from`: the bundle, its stored `hashes.config`, the world `unit` the migrated
# file declares and its render block as that file holds it (`nothing` when it has none) — a printed schema's read
# with the current lift, in the unit the file declared (its numbers, the render block's too, carried in it); a
# retired one's with the legacy lift, in the unit its lengths read back in. `units` is what the old file declares
# (world, density). `where_` names the record in a refusal. With `require_hash` (a file migration: it carries a
# stored identity over and mints none) a record storing no hash is refused; without it (a read) `stored` is
# `nothing`, and the reader recomputes the identity with a warning, as it does for a current file storing none.
function _legacy_record(obj::_ParsedObject, from::AbstractString, where_::AbstractString; require_hash::Bool = true)
    get(obj, :schema, nothing) == from ||
        throw(ArgumentError("$(where_): declares schema $(repr(get(obj, :schema, nothing))), not $(from)"))
    from in PRINTED_SCHEMAS && (obj = _with_current_units(obj))
    old = if from in PRINTED_SCHEMAS
        _lift_bundle(obj, where_)
    elseif haskey(obj, :preamble)
        _legacy_board_bundle(obj)
    elseif haskey(obj, :pattern_config)
        _legacy_pattern_bundle(obj)
    else
        throw(ArgumentError("$(where_): neither a board (preamble) nor a pattern (pattern_config)"))
    end
    stored = _stored_config_hash(obj)
    stored === nothing && require_hash && throw(
        ArgumentError("$(where_): stores no hashes.config; a migration carries a stored identity over and mints none")
    )
    units = _declared_units(obj; required = false)
    unit = from in PRINTED_SCHEMAS ? first(units) : _bundle_length_unit(old, nothing)
    render = haskey(obj, :render) ?
        (@with _FILE_UNITS => units _migrated_render(obj[:render]::AbstractDict, unit)) : nothing
    return (; old, stored, unit, units, render)
end

# The STEP of an earlier schema, run on load by VisualGeometryToolkitIO's `read_record` (the published boards keep
# their old schemas and are read, not rewritten): the record read with its lift, as the current record — exactly
# the tree `save_json` writes for it, the stored `hashes.config` and the render block carried over as the file form
# carries them. Nothing is written.
function _migrate_bundle_record(tree::AbstractDict, from::AbstractString)
    (; old, stored, unit, render) = _legacy_record(_object(tree), from, "a $(from) record"; require_hash = false)
    (; tree, style) = _bundle_file(old, stored, render; unit)
    return merge(tree, (; units = _units_object(style)))   # in memory the record carries the units its file would
end

# An earlier record's `units`, in the retired UnitContract layout (`{world, density, lengths}`), as the 0.10
# units object (IO's `migrate_units`); a record without any (the {value, unit} schemas) as it is.
function _with_current_units(obj::_ParsedObject)
    haskey(obj, :units) || return obj
    o = copy(obj)
    o["units"] = _object(VgtkIO.migrate_units(obj.units))
    return o
end

# The units object the file written with `style` declares (`_bundle_style`'s), for a record a version step builds
# in memory: the style's contract stated in the layout IO's `migrate_units` reads (IO has no public writer of a
# style's units object short of writing the file).
function _units_object(style::VgtkIO.VgioStyle)
    c = style.contract
    roles = Base.structdiff(c.lengths, NamedTuple{(:world,)})
    return VgtkIO.migrate_units((world = unit_string(VgtkIO.role_unit(c)), density = unit_string(c.density), lengths = map(unit_string, roles)))
end

# (`after_write(dst)`: a test hook run on the written file before it is read back)
function _migrate_bundle_file_checked(src::AbstractString, dst::AbstractString, from::AbstractString, after_write)
    obj = JSON.parse(read(src, String))::_ParsedObject
    (; old, stored, unit, units, render) = _legacy_record(obj, from, src)
    # The uid must not move: the stored `hashes.config` is authoritative for every board kind and is carried over
    # unchanged, and the identity recomputed from the migrated file must be the one recomputed from the old file
    # (under the current, frozen encoding). Whether either equals the stored hash is information, never a refusal:
    # boards minted under an older encoding (all 50 example boards) recompute to something else.
    identity_old = _bundle_identity(old)
    identity_old == stored ||
        @info "migrating $(basename(src)): the identity recomputed under the current encoding ($(identity_old[1:8])…) is not the stored hashes.config ($(stored[1:8])…); the stored hash is carried over"
    (; tree, style) = _bundle_file(old, stored, render; unit)
    _write_record(dst, tree; style)
    after_write(dst)
    written = _read_record(dst)                     # the file as written, read once
    new, config_hash = _bundle_and_hash(written, dst)
    VgtkIO.same_value(old, new; length_ulps = 0) ||
        throw(ArgumentError("$(src): the migration does not read back as the file read (a number moved); the output is discarded"))
    config_hash == stored ||
        throw(ArgumentError("$(src): the migration does not keep the stored identity; the output is discarded"))
    _bundle_identity(new) == identity_old ||
        throw(ArgumentError("$(src): the identity recomputed from the migration differs from the old file's; the output is discarded"))
    if render !== nothing
        _render_migrated_ok(obj[:render]::AbstractDict, get(written, :render, nothing), units, _declared_units(written)) ||
            throw(ArgumentError("$(src): the migrated render block does not read back as the old one; the output is discarded"))
    end
    return (; to = String(get(written, :schema, "")), uid = uid_from_hash(stored), config_hash = stored)
end

for s in (PRINTED_SCHEMAS..., MIGRATED_SCHEMAS...)
    @eval VgtkIO.migrate_schema(::Val{Symbol($s)}, tree::AbstractDict) = _migrate_bundle_record(tree, $s)
    @eval VgtkIO.migrate_schema(::Val{Symbol($s)}, src::AbstractString, dst::AbstractString) = _migrate_bundle_file(src, dst, $s)
end

@doc """
    VisualGeometryToolkitIO.migrate_schema(::Val{Symbol("blobboards/board/v5")}, tree)
    VisualGeometryToolkitIO.migrate_schema(::Val{Symbol("blobboards/board/v5")}, src, dst)

BlobBoards' migrations of its earlier schemas into `blobboards/board/v7.0` and
`blobboards/pattern/v5.0`, by the contract of
[`VisualGeometryToolkitIO.migrate_schema`], in its two forms: board v6/v5 and
pattern v4/v3 (the current body under the retired units layout: its units
through IO's `migrate_units`, then read with the current lift),
board v1 to v4 and pattern v1 and v2 (the retired `{value, unit}` form: read
with the legacy lift). Either is written as the current writer writes it.

The STEP (`tree`, a parsed record): the record as the current one, in memory,
its stored identity kept — what `load_bundle` runs, through
`VisualGeometryToolkitIO.read_record`, on an old file, so a published board
keeps loading as it is.

The FILE form: write `dst` through the current writer, read it back, compare
every value bit for bit, keep the stored identity. Run by
`VisualGeometryToolkitIO.vgio_migrate` (and its command), which stages the
result and replaces the file only once verified. Returns `(; to, uid, config_hash)`.
""" VgtkIO.migrate_schema(::Val{Symbol("blobboards/board/v5")}, ::AbstractDict)
