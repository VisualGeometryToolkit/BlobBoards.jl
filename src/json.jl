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
# quantity or a `Size2` through `_lower`/`_quantity_json` below), the value itself
# when it has none: what the writer serializes and what the canonical identity
# encoding (`_plain`) is computed over. The style is private; every lowering
# method takes any `JSON.JSONStyle`.
struct _LowerStyle <: JSON.JSONStyle end
_lower(x) = JSON.lower(_LowerStyle(), x)

# A board file's quantity is a `{value, unit}` object in the unit its dimension
# names: a length in "mm", a density in "dpi", a pixel count (dimensionless) in
# "px"; a pixel count or a density snaps to an integer when it is one. The
# stack's own quantity format (a number in the record's declared unit) is not
# written here: board files keep the form every stored board has, and
# VisualGeometryToolkitIO reads it (`read_quantity` accepts the object).
function _quantity_json(q::Quantity)
    d = dimension(q)
    u = d == dimension(mm) ? mm : d == dimension(dpi) ? dpi : d == dimension(px) ? px :
        throw(ArgumentError("no board-file unit for a quantity of dimension $(d)"))
    v = ustrip(u, q)
    return (value = u === mm ? Float64(v) : _snap_integer(v), unit = unit_string(u))
end
_snap_integer(v::Integer) = Int(v)
_snap_integer(v::AbstractFloat) = isinteger(v) ? Int(v) : v
_snap_integer(v) = v

# a board file's number or quantity, as the writer writes it
_json_scalar(q::Quantity) = _quantity_json(q)
_json_scalar(x) = x
_json_size2(s::Size2) = (width = _json_scalar(s.width), height = _json_scalar(s.height))

_lower(q::Quantity) = _quantity_json(q)
_lower(s::Size2) = _json_size2(s)
_jv(s::Size2) = _json_size2(s)   # a field's value as the writer writes it
_jv(x) = _json_scalar(x)

# A board file's number or `{value, unit}` object, read: a number as it is, an
# object as the quantity (any unit name `parse_unit` reads, so every string a
# stored board holds loads: mm, px, dpi, pd, pt).
function _lift_scalar(x)
    x isa _JSONObject || return x
    u = parse_unit(String(_getfield(x, "unit")); compound = true)
    v = _getfield(x, "value")
    return (dimension(u) == dimension(mm) ? Float64(v) : _snap_integer(v)) * u
end
_lift_size2(x) = Size2(width = _lift_scalar(_getfield(x, "width")), height = _lift_scalar(_getfield(x, "height")))

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
