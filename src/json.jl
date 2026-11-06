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
# FILE (board/v6, pattern/v4): a quantity is a NUMBER in the unit the file's
# `units` block declares — a length in the declared `world` unit (mm), a
# density in the declared `density` unit (dpi), a pixel count in pixels; the
# point-grid board's own numbers (the σ range, the layout grid, the blobs) are
# plain numbers in the declared `pattern` length role (pt). A pixel count or a
# density is an integer when it is one.
#
# Every conversion between a quantity and a file number goes through ONE
# pair, `_file_number` / `_file_quantity`, and that pair is `ustrip(mm, q)`
# and `number * mm` (a density, a pixel count likewise in its unit), NOT
# VisualGeometryToolkitIO's decimal-exact `write_quantity`/`read_quantity`.
#
# EXACT ROUND TRIP. No file number in millimetres reads back to EVERY metre
# double, under either codec: some metre values are not the image of any
# millimetre double (measured 2026-11-06, FlexUnits 0.6.16, on v·mm and v·inch
# for v in 0:0.01:200 and 200 000 random lengths up to 4 m: the product pair
# misses 486 inch-built values and 3.4% of the random ones, with the
# neighbouring file numbers tried too; IO's `write_length`/`read_length` misses
# 4498 millimetre- and 1737 inch-built ones). So a board's lengths are STORED on
# the millimetre grid, `_on_mm_grid(q) = ustrip(mm, q) * mm` (types.jl, `_vgc`),
# on which the product pair is exact: 0 misses on all 240 002 values, and the
# grid moves no identity (`ustrip(mm, q)` is the same number on and off it,
# which is all the identity tree reads). A length built in inches (`8.5inch`,
# a 0.33 inch margin) is held one ulp from the bare product in about one value
# in forty; every board read from a file or a QR code already held this form.
# The product pair, not IO's codec, because it is what every stored board's
# numbers were written with and read back as (IO's reader puts `read_length(v,
# mm) !== v * mm` for 24% of millimetre decimals, and reads a density as a
# Float64 product where a board holds the exact `300 * dpi`).
#
# IDENTITY: a board's identity (`hashes.config`, the uid printed in its corners
# and carried by its QR code) is the SHA-256 of a canonical tree built from the
# same lowering, in the form every board's identity has ever been minted from:
# each quantity an object `{value, unit}`. That form is the identity's own
# encoding, frozen — not a file format; nothing reads it back — and it is what
# keeps a uid from moving when the file format does. It is a STYLE of its own,
# `_IdentityStyle` (`_plain` lowers on it); a file is lowered on IO's
# `VgioStyle`. `_lowered(style, q)` is the one function every lowering calls
# for a quantity, and `_identity_form` the one place the frozen form is written.

"""
    _IdentityStyle <: JSON.JSONStyle

The style a board's identity tree is lowered on (`_plain`): every BlobBoards
type's lowering is the file's, except that each quantity is the frozen
`{value, unit}` object (`_identity_form`).
"""
struct _IdentityStyle <: JSON.JSONStyle end

# The styles a BlobBoards type is lowered on: a file's (IO's) and the identity's.
const _Style = Union{VgtkIO.VgioStyle, _IdentityStyle}

# The style every BlobBoards file is written on: lengths in mm, the unit the
# file's `units` block declares (the quantities themselves go through
# `_lowered`, never through the style's own length codec).
const _FILE_STYLE = VgtkIO.VgioStyle(; length_unit = mm)

# The units a file being READ declares (`units`: world, density). The default is
# what every file this package writes declares.
const _FILE_UNITS = ScopedValue((mm, dpi))

_snap_integer(v::Integer) = Int(v)
_snap_integer(v::AbstractFloat) = isinteger(v) ? Int(v) : v
# an exact quantity's magnitude (`300px` is a Rational): an integer, else its Float64
_snap_integer(v::Rational) = isinteger(v) ? Int(v) : Float64(v)
_snap_integer(v) = v

# A quantity in the form a board stores it: a length on the millimetre grid (the
# value its file number reads back as: see the header), any other quantity as
# VisualGeometryToolkitCore's `base_quantity`.
_on_mm_grid(q::Quantity) = dimension(q) == dimension(mm) ? base_quantity(Float64(ustrip(mm, q)) * mm) : base_quantity(q)

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
# the identity's frozen object.
_lowered(::VgtkIO.VgioStyle, q::Quantity; role::Symbol = :world) = _file_number(q; role)
_lowered(::_IdentityStyle, q::Quantity; role::Symbol = :world) = _identity_form(q; role)

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
    kind === :density && return _snap_integer(x) * density
    (kind === :pixel || kind === :pattern) && return _snap_integer(x) * px
    kind === :blob && return Float64(x) * px
    throw(ArgumentError("a quantity's kind is :length, :pixel, :density, :pattern or :blob, got $(repr(kind))"))
end
_file_size2(x, kind::Symbol) = Size2(width = _file_quantity(x.width, kind), height = _file_quantity(x.height, kind))

# The `units` block a file declares, read: a VisualGeometryToolkitIO
# `UnitContract`. A board declares a world length (mm), a density (dpi) and, on a
# point grid, the length role `pattern` (pt); any other role is refused.
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
    c = VgtkIO.from_json(VgtkIO.UnitContract, x.units)
    extra = setdiff(keys(c.lengths), (:world, :pattern))
    isempty(extra) || throw(ArgumentError("a board file's units declare no length role $(first(extra)); only `pattern`"))
    haskey(c.lengths, :pattern) && (c.lengths.pattern == pt || throw(ArgumentError("the length role `pattern` is in pt, got $(c.lengths.pattern)")))
    return (VgtkIO.role_unit(c), c.density)
end

# The type a parsed JSON object is (`VgtkIO.read_record`, `JSON.parse`): read
# by property access, its keys in order.
const _ParsedObject = JSON.Object{String, Any}
# What a lift reads (its type keeps the lift off JSON.jl's own methods for its
# lazy values): a parsed object, or a NamedTuple built in memory.
const _Record = Union{JSON.Object, NamedTuple}

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
