# =============================================================================
# The sheet's text — set in one embedded font
# =============================================================================
#
# The uid characters and the id label are TEXT in the print master: set in the
# sheet font, which the file embeds (pdf.jl `_font_objects`), so the page is
# searchable and the label selectable — the board's hash can be copied off
# the sheet — and a RIP draws it from the font program the file carries, as
# PDF/X requires. The metrics are FreeTypeAbstraction's, the ones Makie lays
# text out with. (For a few days in 2026-10 the text was glyph outlines, with
# no font in the file.)

# The one typeface a sheet is set in: TeX Gyre Heros, as Makie ships it (its
# default sans, until 2026-10-14 looked up through `Makie.to_font`), vendored in
# assets/fonts (VENDORED.toml). Opened once, on first use: a font handle is a
# FreeType pointer, which must not be made at precompile time. Its path is found
# at run time (`pkgdir`), like the ICC profile's: an `@__DIR__` path is fixed
# when the package precompiles and dangles once the depot moves.
function _sheet_font_file()
    path = pkgdir(@__MODULE__, "assets", "fonts", "TeXGyreHerosMakie-Regular.otf")
    path === nothing && error("BlobBoards: cannot locate the package directory for the sheet font")
    return path
end
const _SHEET_FONT = Ref{Union{Nothing, FTFont}}(nothing)
function _sheet_font()
    f = _SHEET_FONT[]
    f === nothing || return f
    font = FTFont(_sheet_font_file())
    _SHEET_FONT[] = font
    return font
end

# A character's glyph index in the sheet font; a character it lacks is refused
# (it would print as the font's "missing" box).
function _glyph_index(font::FTFont, ch::Char)
    index = Int(FreeTypeAbstraction.glyph_index(font, ch))
    iszero(index) && throw(ArgumentError("the sheet font ($(_SHEET_FONT)) has no glyph for $(repr(ch))"))
    return index
end

# The glyph as the two-byte code the embedded font is addressed by
# (Identity-H: the code IS the glyph index).
_glyph_code(font::FTFont, ch::Char) = string(_glyph_index(font, ch); base = 16, pad = 4)

# A glyph's advance in thousandths of an em, as a PDF font states widths.
_glyph_width(font::FTFont, ch::Char) =
    round(Int, 1000 * Float64(FreeTypeAbstraction.hadvance(FreeTypeAbstraction.get_extent(font, ch))))

# The centre of a glyph's ink, in em from its origin (y up).
function _ink_centre(font::FTFont, ch::Char)
    box = FreeTypeAbstraction.inkboundingbox(FreeTypeAbstraction.get_extent(font, ch))
    return Tuple(Float64.(GeometryBasics.origin(box) .+ GeometryBasics.widths(box) ./ 2))
end

# A line's advance width in em — the sum of its glyphs' widths, as the font
# object states them (no kerning): what a PDF viewer advances by.
_line_width(font::FTFont, text::AbstractString) = sum(ch -> _glyph_width(font, ch), text; init = 0) / 1000

# Text rendering modes (the `Tr` operator): fill the glyphs, or fill and stroke them.
const _TEXT_FILL, _TEXT_FILL_STROKE = 0, 2

# `text` with its origin at (x, y), one em `size` units, in the page font `/F`,
# as PDF operators. `chars` collects what the sheet sets: the font's widths and
# its ToUnicode map list exactly those.
function _text_ops!(io::IO, chars::Set{Char}, font::FTFont, text::AbstractString, x::Real, y::Real, size::Real; mode::Int = _TEXT_FILL)
    union!(chars, text)
    print(io, "BT /F ", _pdf_num(size), " Tf ", mode, " Tr ", _pdf_num(x), ' ', _pdf_num(y), " Td <")
    for ch in text
        print(io, _glyph_code(font, ch))
    end
    print(io, "> Tj ET\n")
    return nothing
end
