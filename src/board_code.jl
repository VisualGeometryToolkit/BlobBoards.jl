# =============================================================================
# Board code — a board's definition as the short text its QR code carries
# =============================================================================
#
# The QR in a board's corner carries the board's DEFINITION, so a print alone
# rebuilds the board: every value its identity is computed from, positionally,
# '/'-separated, in QR's alphanumeric character set (0-9 A-Z . / and a few
# more; no lower case, no comma), which packs 5.5 bits a character against
# byte mode's 8. ~120-150 characters: a 41 × 41 code, 0.24 mm modules in a
# 10 mm frame corner — read by any phone off a print. (The default A4 board's
# code is 126 characters; in the mm/IN spelling it was 104, 37 × 37, 0.27 mm.)
# (Until 2026-10 the QR held the preamble as JSON, ~880 characters, 113 × 113
# modules: 0.18 mm in A4's 22 mm corner, too fine to read off paper.)
#
#   BB4 / canvas w / h / margin w / h / ruler width / corner size / frame padding /
#   major spacing / minor spacing / major length / minor length / seed /
#   smallest σ / largest σ / scales per octave / max trials / α / cutoff /
#   polarity / check
#
# A length is a number and its unit, the unit symbol of VisualGeometryToolkitCore's
# FlexUnits registry (`PixelUnits`) upper-cased for QR's alphanumeric set:
# `210MM`, `8.5INCH`. It is written in mm or in inches, whichever is shorter and
# still decodes to the identical double (Letter's 8.5 inch is
# 215.89999999999998 mm: `8.5INCH`), and read back by lower-casing the symbol and
# resolving it through the registry — any registered length unit reads. A ruler
# width or corner size of 0 turns that mark off. (Until 2026-10-15 a length was
# a bare mm number or a number with the suffix IN; such codes are refused.) The board is centred: the code carries its margin, and the decoder
# derives the board exactly as `BoardConfig` does (`_board_from_margin`). σ in
# points, the seed in hex, polarity D or L. Every number is written as the
# shortest decimal that reproduces its double — the canonical encoding the
# identity is computed from — so it parses back to the identical value.
# Defaults are written too: leaving them out would rebuild a different board
# the day a default changes. `check` is the first 8 hex of the identity: the
# decoder recomputes the identity from the decoded configuration and refuses a
# code whose check disagrees, so a misread or shifted field cannot rebuild a
# different board. `BB4` is the format (board schema v4, the point grid).
# (While v4 was unreleased its code changed: until 2026-10-13 it carried a bit
# depth, until 2026-10-14 no corner size, and its identity did not cover the
# blobs. Such codes are refused — by their field count, or by their check.)

# The tag IS the board schema's version (`BOARD_SCHEMA`, "blobboards/board/v4" →
# "BB4"), derived from it so the two cannot disagree. A schema is the meaning of
# a configuration — its fields AND how they place the blobs and the frame's
# marks — so any change to placement or the marks is a new schema, and a new tag.
# The identity covers the placed blobs (`_board_identity`), so a code decoded by
# software that places differently fails its check rather than rebuilding
# another board.
const BOARD_CODE_TAG = "BB" * last(split(BOARD_SCHEMA, "/v"))
const _BOARD_CODE_FIELDS = 21

# A number exactly as the identity encodes it (`canonical_json`: an integral
# double bare, others shortest round-trip, a non-finite one refused), upper case
# (an exponent's `e` is not in QR's alphanumeric set).
_code_number(x::Real) = uppercase(canonical_json(Float64(x)))

# A length's code spelling: the number in a unit, then the unit's registry
# symbol (FlexUnits' own printing, `string(u)`) upper-cased. The shortest of mm
# and inch that decodes (`_parse_length`) to the identical value; mm always does.
_spell_length(q, u) = _code_number(ustrip(u, q)) * uppercase(string(u))
function _code_length(q)
    best = _spell_length(q, mm)
    inches = _spell_length(q, inch)
    _parse_length(inches) == q && length(inches) < length(best) && (best = inches)
    return best
end
# A number, then an upper-cased unit symbol. The symbol is lower-cased and read
# by VisualGeometryToolkitCore's `parse_unit` (the stack's one guarded `uparse`); it
# must be a unit of length. `nothing` for anything else.
function _parse_length(s::AbstractString)
    m = Base.match(r"^([0-9.+\-]+(?:E[+\-]?[0-9]+)?)([A-Z]+)$", s)
    m === nothing && return nothing
    v = tryparse(Float64, m[1]::SubString{String})   # both groups are mandatory: they matched
    u = try
        parse_unit(lowercase(m[2]::SubString{String}))
    catch e
        e isa ArgumentError || rethrow()
        nothing
    end
    (v === nothing || u === nothing) && return nothing
    return dimension(u) == dimension(mm) ? v * u : nothing
end
_code_stored(q) = _code_number(ustrip(px, q))


"""
    board_code(preamble::BoardPreamble) -> String
    board_code(bundle::BoardBundle) -> String

The board's definition as the text its QR code carries. The format, `BB4`,
holds every value the identity is computed from, positionally, in QR's
alphanumeric characters, ending in the first 8 hex of the identity as a check.
[`parse_board_code`](@ref) is the inverse. Only a point-grid (v4) board has one.

```julia
board_code(bundle.preamble)
# "BB4/210MM/297MM/10MM/10MM/10MM/10MM/0MM/10MM/5MM/10MM/5MM/…"   (A4)
# "BB4/8.5INCH/11INCH/10MM/10MM/10MM/10MM/0MM/10MM/5MM/10MM/5MM/…"   (Letter)
```
"""
board_code(preamble::BoardPreamble) = _board_code(preamble, hash_config(preamble))
board_code(bundle::BoardBundle) = _board_code(bundle.preamble, _bundle_identity(bundle))   # from its own blobs: no placement
# The code with the identity it checks against given (placement already done).
function _board_code(preamble::BoardPreamble, identity::AbstractString)
    bc, pc = preamble.board_config, preamble.pattern_config
    bc.layout_grid isa PointGrid ||
        throw(ArgumentError("board_code: only a point-grid (schema v4) board has a board code; this one is laid out on $(bc.layout_grid)"))
    # the two fields the code leaves out, as every point-grid board has them
    # (`_board_preamble`); any other value would print a code that rebuilds
    # another board
    (pc.raster_aligned == false && pc.pattern_resolution == _board_pattern_points(bc)) ||
        throw(ArgumentError("board_code: the pattern is not this board's pattern cell placed on its grid (`_board_preamble`)"))
    rs = bc.ruler_spec
    fields = String[
        BOARD_CODE_TAG,
        _code_length(bc.canvas_size.width), _code_length(bc.canvas_size.height),
        _code_length(bc.margin.width), _code_length(bc.margin.height),
        _code_length(bc.ruler_width), _code_length(bc.corner_size), _code_length(bc.frame_padding),
        _code_length(rs.major_tick.spacing), _code_length(rs.minor_tick.spacing),
        _code_length(rs.major_tick.width), _code_length(rs.minor_tick.width),
        uppercase(string(pc.seed; base = 16)),
        _code_stored(pc.min_scale), _code_stored(pc.max_scale),
        string(pc.scales_per_octave), string(pc.max_trials), _code_number(pc.α), _code_number(pc.cutoff),
        pc.polarity === :dark ? "D" : "L",
        uppercase(identity[1:8]),
    ]
    return join(fields, "/")
end

"""
    parse_board_code(code::AbstractString) -> BoardPreamble

The board a [`board_code`](@ref) describes. This is what a board's QR code reads
as, checked: the identity recomputed from the decoded configuration must begin with
the code's check, or this is an `ArgumentError` (a misread, shifted or edited
code never rebuilds a different board). `hash_config` of the result is the
board's identity; `blob_board(preamble)` rebuilds the board
under it.
"""
function parse_board_code(code::AbstractString)
    f = split(strip(code), '/')
    bad(why) = throw(ArgumentError("parse_board_code: $why — got $(repr(code))"))
    if f[1] != BOARD_CODE_TAG
        tag = Base.match(r"^BB(\d+)$", f[1])
        tag === nothing && bad("not a board code (no BB<schema> tag)")
        bad("a board schema v$(tag[1]) code; this BlobBoards reads schema $(BOARD_SCHEMA) ($(BOARD_CODE_TAG)) codes")
    end
    length(f) == _BOARD_CODE_FIELDS || bad("a $(BOARD_CODE_TAG) code has $(_BOARD_CODE_FIELDS) '/'-separated fields, this one $(length(f))")
    function num(i)
        v = tryparse(Float64, f[i])
        return v === nothing ? bad("field $i ($(repr(f[i]))) is not a number") : v
    end
    function len(i)
        v = _parse_length(f[i])
        return v === nothing ? bad("field $i ($(repr(f[i]))) is not a length (a number and a registered length unit, e.g. 210MM, 8.5INCH)") : v
    end
    function int(i)
        v = tryparse(Int, f[i])
        return v === nothing ? bad("field $i ($(repr(f[i]))) is not an integer") : v
    end
    seed = tryparse(UInt64, f[13]; base = 16)
    seed === nothing && bad("seed $(repr(f[13])) is not hex")
    f[20] in ("D", "L") || bad("polarity $(repr(f[20])) is not D or L")

    board_config = BoardConfig(;
        canvas_size = Size2(width = len(2), height = len(3)),
        margin = Size2(width = len(4), height = len(5)),
        layout_grid = PointGrid(),
        frame_padding = len(8), corner_size = len(7), ruler_width = len(6),
        # the ticks while the ruler is on (with it off they are zero, `_frame_fields`)
        (
            iszero(len(6)) ? (;) : (;
                    major_tick_spacing = len(9), minor_tick_spacing = len(10),
                    major_tick_width = len(11), minor_tick_width = len(12),
                )
        )...,
    )
    preamble = _board_preamble(
        board_config;
        min_scale = num(14) * px, max_scale = num(15) * px, seed,
        scales_per_octave = int(16), max_trials = int(17), α = num(18), cutoff = num(19),
        polarity = f[20] == "D" ? :dark : :light,
    )
    check = lowercase(f[_BOARD_CODE_FIELDS])
    occursin(r"^[0-9a-f]{8}$", check) || bad("the check $(repr(f[_BOARD_CODE_FIELDS])) is not 8 hex digits")
    identity = hash_config(preamble)                 # places the blobs: the check proves the rebuild
    startswith(identity, check) || bad("the decoded board's identity $(identity[1:8]) does not match the check $(check)")
    return preamble
end
