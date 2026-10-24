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

# A value's JSON lowering (types.jl, VisualGeometryToolkitFeatures' for `Size2` and
# quantities), the value itself when it has none: what the writer serializes
# and what the canonical identity encoding (`_plain`) is computed over. The
# style is private; every lowering method takes any `JSON.JSONStyle`.
struct _LowerStyle <: JSON.JSONStyle end
_lower(x) = JSON.lower(_LowerStyle(), x)

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
