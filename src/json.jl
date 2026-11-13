# =============================================================================
# JSON: the board's numbers, and its two lowerings (a file, an identity)
# =============================================================================
#
# Every JSON file is written and read at VisualGeometryToolkitIO's file
# boundary, `VgtkIO.write_record(path, record; style)` and
# `VgtkIO.read_record(path)` (the vgio file shape; an earlier schema read
# through the version steps of src/migrate_boards.jl). Each BlobBoards type
# is lowered and lifted on IO's style, `JSON.lower(::VgtkIO.VgioStyle, x)` and
# `JSON.lift(::VgtkIO.VgioStyle, T, record)` (types.jl): no method lands on
# JSON.jl's own style, so no other package's JSON changes when BlobBoards loads.
# A lift reads a parsed record by property access (`x.key`, `get(x, :key,
# default)`): a `JSON.Object`, or a NamedTuple built in memory.
#
# What stays here is what is the board's own: the conversion of its numbers
# and the frozen form its identity is minted in.

# -----------------------------------------------------------------------------
# A quantity in a board file, and in a board's identity
# -----------------------------------------------------------------------------
#
# FILE (board/v7.0, pattern/v5.0): a quantity is a NUMBER in the unit the file's
# `units` object declares — a length in the declared `length` unit, a density in
# the declared `density` unit (dpi), a pixel count in pixels; the point-grid
# board's own numbers (the σ range, the layout grid, the blobs) are plain
# numbers in the declared `pattern` length role (pt). A pixel count or a
# density is an integer when it is one.
#
# Every conversion between a quantity and a file number goes through ONE
# pair, `_file_number` / `_file_quantity`, and that pair is `ustrip(u, q)` and
# `number * u` in the file's declared unit `u`, NOT VisualGeometryToolkitIO's
# decimal-exact `write_quantity`/`read_quantity`: every stored board's numbers
# were written with the pair and read back with it (IO's reader puts
# `read_length(v, mm) !== v * mm` for 24% of millimetre decimals, and reads a
# density as a Float64 product where a board holds the exact `300 * dpi`).
#
# THE WORLD UNIT IS THE FILE'S. A board in memory holds its lengths as they were
# built (`210.0mm`, `8.5inch`, a `0.33inch` margin), and no one unit's file
# number reads back to every metre double (measured 2026-11-06, FlexUnits
# 0.6.16, on v·mm and v·inch for v in 0:0.01:200: the pair in millimetres misses
# 486 inch-built values, in inches 1785 millimetre-built ones; IO's codec misses
# 4498 millimetre- and 1737 inch-built ones). So a board file declares, as its
# `world`, the first of millimetres, inches and metres in which EVERY length it
# holds (a PDF's bleed included) reads back bit for bit (`_exact_length_unit`):
# millimetres for every board built in them (every stored and published board),
# inches for one built in inches, metres — exact for any double — for one that
# mixes values neither holds. Memory → file → memory is then exact for every
# board; the in-memory values, so every identity and every QR code, are what
# they were. The reader has always honoured the declared `world`. A file that
# this writer did not write (an earlier schema, another unit) may change bytes
# when re-saved, never values.
#
# IDENTITY: a board's identity (`hashes.config`, the uid printed in its corners
# and carried by its QR code) is the SHA-256 of a canonical tree built from the
# same lowering, in the form every board's identity has ever been minted from:
# each quantity an object `{value, unit}`. That form is the identity's own
# encoding, frozen — not a file format; nothing reads it back — and it is what
# keeps a uid from moving when the file format does. It is a STYLE of its own,
# `_IdentityStyle` (`_identity_lowering` lowers on it); a file is lowered on IO's
# `VgioStyle`. `_lowered(style, q)` is the one function every lowering calls
# for a quantity, and `_identity_form` the one place the frozen form is written.

"""
    _IdentityStyle <: JSON.JSONStyle

The style a board's identity tree is lowered on (`_identity_lowering`): every BlobBoards
type's lowering is the file's, except that each quantity is the frozen
`{value, unit}` object (`_identity_form`).
"""
struct _IdentityStyle <: JSON.JSONStyle end

"""
    _LengthProbe <: JSON.JSONStyle

A style that lowers a value only to collect the lengths a file of it holds
(`_exact_length_unit`): exactly the quantities a file's lowering writes in its
`world` unit, gathered in `lengths`.
"""
struct _LengthProbe <: JSON.JSONStyle
    lengths::Vector{Quantity}
end
_LengthProbe() = _LengthProbe(Quantity[])

# The styles a BlobBoards type is lowered on: a file's (IO's), the identity's, a probe's.
const _Style = Union{VgtkIO.VgioStyle, _IdentityStyle, _LengthProbe}

# A file's style: its declared `world` unit (the quantities themselves go
# through `_lowered`, never through the style's own length codec). Lifting
# reads the units a file declares (`_FILE_UNITS`), whatever style it is given.
_file_style(u) = VgtkIO.VgioStyle(; length_unit = u)
const _FILE_STYLE = _file_style(mm)

# Every BlobBoards file goes through IO's `write_record` with BlobBoards named as the
# calling package in its `generator` (`with_caller`).
_write_record(path, tree; style) = VgtkIO.with_caller(() -> VgtkIO.write_record(path, tree; style), BlobBoards)

# The world units a file may declare, in the order they are tried.
const _FILE_LENGTH_UNITS = (mm, inch, m)

# A length as the number a file in `u` holds, read back (`_file_quantity`'s product).
_holds_exactly(q::Quantity, u) = Float64(ustrip(u, q)) * u === q

"""
    _exact_length_unit(lengths) -> Units

The world unit a file holding `lengths` declares: the first of mm, inch and m
in which every one reads back bit for bit (see the header). Metres hold any
length exactly.
"""
function _exact_length_unit(lengths)
    for u in _FILE_LENGTH_UNITS
        all(q -> _holds_exactly(q, u), lengths) && return u
    end
    return m
end

# The units a file being READ declares (`units`: world, density). The default is
# what every file this package writes declares.
const _FILE_UNITS = ScopedValue((mm, dpi))


# A quantity as the number a file holds. `role = :world` (the default): by
# dimension, a length in the file's world unit `unit`, a density in dpi, a pixel
# count in px. `role = :pattern`: a point-grid board's own number (σ range,
# layout grid, blob coordinate), the stored-unit magnitude — the points the
# print master holds — always a Float64.
function _file_number(q::Quantity; role::Symbol = :world, unit = mm)
    role === :pattern && return Float64(ustrip(px, q))
    d = dimension(q)
    d == dimension(mm) && return Float64(ustrip(unit, q))
    d == dimension(dpi) && return VgtkIO.snap_integer(ustrip(dpi, q))
    d == dimension(px) && return VgtkIO.snap_integer(ustrip(px, q))
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
    return (value = u === mm ? Float64(v) : VgtkIO.snap_integer(v), unit = unit_string(u))
end

# The one function every lowering calls for a quantity: the file's number (in the
# style's world unit), the identity's frozen object, or a probe's record of a length.
_lowered(s::VgtkIO.VgioStyle, q::Quantity; role::Symbol = :world) = _file_number(q; role, unit = s.length_unit)
_lowered(::_IdentityStyle, q::Quantity; role::Symbol = :world) = _identity_form(q; role)
function _lowered(p::_LengthProbe, q::Quantity; role::Symbol = :world)
    role === :world && dimension(q) == dimension(mm) && push!(p.lengths, q)
    return 0
end

# A field's value as the style writes it.
_jv(s::_Style, q::Quantity) = _lowered(s, q)
_jv(s::_Style, z::Size2) = (width = _jv(s, z.width), height = _jv(s, z.height))
_jv(::_Style, x) = x

# A quantity or a size left in an identity tree (a run configuration's,
# `config_dict`) lowers in the identity's form. (A file's lowerings convert
# their quantities themselves, through `_jv`.)
JSON.lower(s::_IdentityStyle, q::Quantity) = _jv(s, q)
JSON.lower(s::_IdentityStyle, z::Size2) = _jv(s, z)

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
    kind === :density && return VgtkIO.snap_integer(x) * density
    (kind === :pixel || kind === :pattern) && return VgtkIO.snap_integer(x) * px
    kind === :blob && return Float64(x) * px
    throw(ArgumentError("a quantity's kind is :length, :pixel, :density, :pattern or :blob, got $(repr(kind))"))
end
_file_size2(x, kind::Symbol) = Size2(width = _file_quantity(x.width, kind), height = _file_quantity(x.height, kind))

# The `units` object a file declares (the envelope's, VisualGeometryToolkitIO 0.10),
# read through IO's style: a board declares a world length (mm), a density (dpi)
# and, on a point grid, the length role `pattern` (pt); any other role is refused.
function _declared_units(x; required::Bool = true)
    if !haskey(x, :units)
        required || return (mm, dpi)
        schema = get(x, :schema, nothing)
        schema in MIGRATED_SCHEMAS && throw(
            ArgumentError(
                "this is a $(schema) record, the retired {value, unit} quantity form, lifted without its version step; " *
                    "read the file with load_bundle, which runs the step in memory, or rewrite it once with " *
                    "VisualGeometryToolkitIO.vgio_migrate (both keep the stored identity)"
            )
        )
        throw(ArgumentError("a board or pattern file declares its `units` (world, density); this one has none"))
    end
    c = VgtkIO.VgioStyle(x).contract
    extra = setdiff(keys(c.lengths), (:world, :pattern))
    isempty(extra) || throw(ArgumentError("a board file's units declare no length role $(first(extra)); only `pattern`"))
    haskey(c.lengths, :pattern) && (c.lengths.pattern == pt || throw(ArgumentError("the length role `pattern` is in pt, got $(c.lengths.pattern)")))
    return (VgtkIO.role_unit(c), c.density)
end

# The type a parsed JSON object is (`VgtkIO.read_record`, `JSON.parse`): read
# by property access, its keys in order.
const _ParsedObject = JSON.Object{String, Any}

# A JSON object built in code, in the form a lift reads: the pairs given, or a
# dictionary or NamedTuple with every nested one converted (a version step's
# tree is plain `Dict`s, which have no property access).
function _object(kv::Pair...)
    o = _ParsedObject()
    for (k, v) in kv
        o[String(k)] = v
    end
    return o
end
_object(x) = _object((String(k) => _as_record(v) for (k, v) in pairs(x))...)
_as_record(x::Union{AbstractDict, NamedTuple}) = _object(x)
_as_record(x::AbstractVector) = Any[_as_record(v) for v in x]
_as_record(x) = x
