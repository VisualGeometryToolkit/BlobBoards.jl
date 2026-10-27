# =============================================================================
# JSON (JSON.jl 1.x): field access, and the one writer of every JSON file
# =============================================================================
#
# Each serialized type gets JSON.jl's two hooks (types.jl): `JSON.lower(::JSON.JSONStyle, x)`
# returns the plain value written, `JSON.lift(::Type{T}, obj)` builds `T` from a
# parsed object, and `StructUtils.structlike(::Type{T}) = false` makes
# `JSON.parse(str, T)` hand `lift` the whole object. VisualGeometryToolkitFeatures
# does the same for `Size2` and quantities.
#
# THE WRITER IS BYTE-PINNED (test/suites/identity_pins.jl). Until 2026-10 every
# file was written by JSON3 (`JSON3.pretty`, or `JSON3.write`), and a file written
# now must be byte-identical to the one that writer wrote for the same board.
# The compact form is: JSON.jl and JSON3 print every value the same way (keys
# in the lowered order — a Dict's in its iteration order, `sort_keys = false` —,
# floats in Base's shortest round-trip form, the same characters escaped). The
# pretty form is not, so `_write_pretty` reproduces JSON3.pretty exactly,
# quirks included:
#
#   - 4-space indent, `"key": value`, one element per line;
#   - an empty object or array spans two lines (`{` newline, indent, `}`);
#   - JSON3.pretty re-read the compact text with JSON3.read, which reads an
#     integral float (`3.0`) as an integer, so it printed `3` — except inside an
#     array whose earlier elements were all floats (or nulls) with at least one
#     float: there it stayed `3.0` (`[30.25, 40.0]` but `[3, 4.5]`), and an
#     integer was read, and printed, as a float (`[1.5, 2]` as `[1.5, 2.0]`).
#
# Checked against JSON3.pretty on 20 000 random trees (nested objects and
# arrays, empties, nulls, Bools, integers, integral and non-integral floats,
# -0.0, 1e16, escaped strings) when it was written: no byte differed.

# A parsed JSON object (`JSON.Object`, any AbstractDict) or a NamedTuple standing
# in for one, accessed by field name. Symbol-keyed dictionaries are accepted
# too: a caller may build the input by hand.
_hasfield(x::AbstractDict, key::String) = haskey(x, key) || haskey(x, Symbol(key))
_hasfield(x::NamedTuple, key::String) = haskey(x, Symbol(key))

_getfield(x::AbstractDict, key::String) = haskey(x, key) ? x[key] : x[Symbol(key)]
_getfield(x::NamedTuple, key::String) = getproperty(x, Symbol(key))
_getfield(x, key::String, default) = _hasfield(x, key) ? _getfield(x, key) : default

const _JSONObject = Union{AbstractDict, NamedTuple}

# A value's JSON lowering (types.jl: each BlobBoards type's `JSON.lower`; a
# quantity or a `Size2` through `_lower`/`_jv` below), the value itself when it
# has none: what the writer serializes and what the canonical identity encoding
# (`_plain`) is computed over. The style is private; every lowering method takes
# any `JSON.JSONStyle`.
struct _LowerStyle <: JSON.JSONStyle end
_lower(x) = JSON.lower(_LowerStyle(), x)

# -----------------------------------------------------------------------------
# A quantity in a board file, and in a board's identity
# -----------------------------------------------------------------------------
#
# FILE (board/v5, pattern/v3): a quantity is a NUMBER in the unit the file's
# `units` block declares — VisualGeometryToolkitIO's quantity format, the one
# every stack file uses, and the ONLY place a file declares a unit: a length in
# the declared `world` unit (mm), a density in the declared `density` unit
# (dpi), a pixel count in pixels; the point-grid board's own numbers (the σ
# range, the layout grid, the blobs) are plain numbers in the declared `pattern`
# length role (pt). A pixel count or a density is an integer when it is one.
#
# TEMPORARY: every conversion between a quantity and a file number goes through
# ONE pair, `_file_number` / `_file_quantity`, and that pair is `ustrip(unit, q)`
# and `number * unit`, NOT VisualGeometryToolkitIO's decimal-exact
# `write_quantity`/`read_quantity`. The reason is an IO gap: its writer is not
# exact for a quantity built as `v * unit` — a board is built from literals
# (`9.0mm`, which is 0.009000000000000001 m), so IO would print
# `9.000000000000002` for it, and its reader would hold a value an ulp from the
# one the board was built with (identity, byte pins and placement were computed
# with `v * unit`). When IO's writer writes `v` back for `v * unit`, this pair is
# replaced by `write_quantity`/`read_quantity` under the file's `UnitContract`
# (roles included) and nothing else changes: no other code converts a file
# number. (`test/suites/identity_pins.jl` reads a written board with IO's reader.)
#
# IDENTITY: a board's identity (`hashes.config`, the uid printed in its corners
# and carried by its QR code) is the SHA-256 of a canonical tree built from the
# same lowering, in the form every board's identity has ever been minted from:
# each quantity an object `{value, unit}`. That form is the identity's own
# encoding, frozen — not a file format; nothing reads it back — and it is what
# keeps a uid from moving when the file format does. `_plain` turns it on for
# the duration of one canonical encoding (`_IDENTITY`); everything else is a file.
# `_lowered` is the one function every lowering calls for a quantity, and
# `_identity_form` the one place the frozen form is written.
const _IDENTITY = ScopedValue(false)

# The units a file being READ declares (`units`: world, density). The default is
# what every file this package writes declares.
const _FILE_UNITS = ScopedValue((mm, dpi))

_snap_integer(v::Integer) = Int(v)
_snap_integer(v::AbstractFloat) = isinteger(v) ? Int(v) : v
_snap_integer(v) = v

# A quantity as the number a file holds. `role = :world` (the default): by
# dimension, a length in mm, a density in dpi, a pixel count in px. `role =
# :pattern`: a point-grid board's own number (σ range, layout grid, blob
# coordinate), the stored-unit magnitude — the points the print master holds —
# always a Float64.
function _file_number(q::Quantity; role::Symbol = :world)
    role === :pattern && return Float64(ustrip(px, q))
    d = dimension(q)
    d == dimension(mm) && return Float64(ustrip(mm, q))
    d == dimension(dpi) && return _snap_integer(ustrip(dpi, q))
    d == dimension(px) && return _snap_integer(ustrip(px, q))
    throw(ArgumentError("no board-file unit for a quantity of dimension $(d)"))
end

"""
    _identity_form(q::Quantity; role = :world) -> NamedTuple

The FROZEN canonical form of a quantity in a board's identity tree: the object
`(value, unit)` every board's identity was minted from (a length in "mm", a
density in "dpi", a pixel count in "px", a `:pattern` number in "pt"). It is
not a file format and nothing reads it back; changing it re-uids every board.
"""
function _identity_form(q::Quantity; role::Symbol = :world)
    role === :pattern && return (value = Float64(ustrip(px, q)), unit = unit_string(pt))
    d = dimension(q)
    u = d == dimension(mm) ? mm : d == dimension(dpi) ? dpi : d == dimension(px) ? px :
        throw(ArgumentError("no board-file unit for a quantity of dimension $(d)"))
    v = ustrip(u, q)
    return (value = u === mm ? Float64(v) : _snap_integer(v), unit = unit_string(u))
end

# The one function every lowering calls for a quantity: the file's number, or
# (inside `_plain`) the identity's frozen object.
_lowered(q::Quantity; role::Symbol = :world) = _IDENTITY[] ? _identity_form(q; role) : _file_number(q; role)

# A field's value as the writer writes it.
_jv(q::Quantity) = _lowered(q)
_jv(s::Size2) = (width = _jv(s.width), height = _jv(s.height))
_jv(x) = x

_lower(q::Quantity) = _jv(q)
_lower(s::Size2) = _jv(s)

# A file's number as the quantity of its kind (`:length`, `:pixel`, `:density`;
# `:pattern` and `:blob`, the point-grid board's own numbers, stored-unit pixels:
# `:pattern` snaps an integral value to an integer, `:blob` is always a Float64)
# under the units it declares. A `{value, unit}` object is the retired form,
# which only a migration reads (migrate_boards.jl).
function _file_quantity(x, kind::Symbol)
    (x isa Real && !(x isa Bool)) || throw(
        ArgumentError(
            "a board file's quantity is a number in the file's declared unit, got $(repr(x)) " *
                "(the {value, unit} form belongs to board schemas up to v4 and pattern schemas up to v2: " *
                "migrate the file once with VisualGeometryToolkitIO.vgio_migrate)"
        )
    )
    world, density = _FILE_UNITS[]
    kind === :length && return Float64(x) * world
    kind === :density && return _snap_integer(x) * density
    (kind === :pixel || kind === :pattern) && return _snap_integer(x) * px
    kind === :blob && return Float64(x) * px
    throw(ArgumentError("a quantity's kind is :length, :pixel, :density, :pattern or :blob, got $(repr(kind))"))
end
_file_size2(x, kind::Symbol) =
    Size2(width = _file_quantity(_getfield(x, "width"), kind), height = _file_quantity(_getfield(x, "height"), kind))

# The `units` block a file declares, read: a VisualGeometryToolkitIO
# `UnitContract`. A board declares a world length (mm), a density (dpi) and, on a
# point grid, the length role `pattern` (pt); any other role is refused.
function _declared_units(x; required::Bool = true)
    if !_hasfield(x, "units")
        required || return (mm, dpi)
        schema = _getfield(x, "schema", nothing)
        schema in MIGRATED_SCHEMAS && throw(
            ArgumentError(
                "this is a $(schema) file, the retired {value, unit} quantity form; this BlobBoards reads $(join(READABLE_SCHEMAS, ", ")). " *
                    "Migrate the file once with VisualGeometryToolkitIO.vgio_migrate (it keeps the stored identity)"
            )
        )
        throw(ArgumentError("a board or pattern file declares its `units` (world, density); this one has none"))
    end
    c = VgtkIO.from_json(VgtkIO.UnitContract, _getfield(x, "units"))
    extra = setdiff(keys(c.lengths), (:world, :pattern))
    isempty(extra) || throw(ArgumentError("a board file's units declare no length role $(first(extra)); only `pattern`"))
    haskey(c.lengths, :pattern) && (c.lengths.pattern == pt || throw(ArgumentError("the length role `pattern` is in pt, got $(c.lengths.pattern)")))
    return (VgtkIO.role_unit(c), c.density)
end

# The type `JSON.parse(str)` returns for an object.
const _ParsedObject = JSON.Object{String, Any}

"""
    _write_json(io, x; pretty = true)

Write `x` as JSON: JSON.jl's compact form, or (`pretty`) the 4-space layout
every BlobBoards file has always had (see the header of src/json.jl).
"""
function _write_json(io::IO, x; pretty::Bool = true)
    s = JSON.json(x; sort_keys = false)    # a Dict in its iteration order, as JSON3 wrote it (JSON.jl sorts a Dict's keys by default)
    pretty || return write(io, s)
    _write_pretty(io, JSON.parse(s), 0, false)    # (a bare scalar: JSON3.pretty wrote it as it was)
    return nothing
end

const _INDENT = 4

# `checkint`: whether JSON3.read would have read an integral float here as an
# integer (true everywhere but late in an all-float array).
function _write_pretty(io::IO, x::AbstractDict, level::Int, ::Bool)
    write(io, "{\n")
    isempty(x) && return write(io, ' '^(_INDENT * level), "}")
    for (i, (k, v)) in enumerate(x)
        write(io, ' '^(_INDENT * (level + 1)), JSON.json(String(k)), ": ")
        _write_pretty(io, v, level + 1, true)
        write(io, i == length(x) ? "\n" * ' '^(_INDENT * level) * "}" : ",\n")
    end
    return nothing
end

function _write_pretty(io::IO, x::AbstractVector, level::Int, ::Bool)
    write(io, "[\n")
    isempty(x) && return write(io, ' '^(_INDENT * level), "]")
    floats = false     # an element read as a float so far
    others = false     # an element of any other kind (an integer, a string, a container, …)
    for (i, v) in enumerate(x)
        write(io, ' '^(_INDENT * (level + 1)))
        checkint = !floats || others
        _write_pretty(io, v, level + 1, checkint)
        if v === nothing
        elseif (v isa Real && !(v isa Bool)) && !(checkint && _reads_as_integer(v))
            floats = true
        else
            others = true
        end
        write(io, i == length(x) ? "\n" * ' '^(_INDENT * level) * "]" : ",\n")
    end
    return nothing
end

# A number: an integral one in integer form where JSON3.read checked for
# integers, every one in float form where it did not (it read `-2` there as
# -2.0).
_write_pretty(io::IO, x::Real, ::Int, checkint::Bool) =
    write(io, checkint && _reads_as_integer(x) ? string(Int64(x)) : JSON.json(Float64(x)))
_write_pretty(io::IO, x::Bool, ::Int, ::Bool) = write(io, JSON.json(x))
_write_pretty(io::IO, x, ::Int, ::Bool) = write(io, JSON.json(x))

# JSON3.read's test: integral and within Int64's range (an integer literal past
# it was read as a float).
_reads_as_integer(x::Real) = isinteger(x) && -0x1p63 <= x < 0x1p63
