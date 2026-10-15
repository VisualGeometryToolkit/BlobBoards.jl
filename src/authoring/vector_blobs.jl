# =============================================================================
# Vector blobs — the pattern as radial shadings instead of a raster
# =============================================================================
#
# A board's blobs are vector graphics — its ONE source of truth: every blob is a
# PDF radial shading (`ShadingType 3`) whose colour function is
# `BlobBoards.blob_profile`, so the printer's RIP evaluates the Gaussian at its
# own device resolution, and the file is ~150 bytes per blob (a 1 m² board at
# 1200 dpi: 0.25 MB, vs ~83 MB as a raster). A board's raster outputs — TIFF,
# PNG, the gallery's images — are conversions of this PDF (pdftocairo). The
# raster pattern engine was removed in 2026-10 after an A/B: the conversion
# agrees with `render_pattern` (the same profile, sampled) to one 8-bit level,
# and fitted blob centres agree to < 1 µm.
#
# The colour function is `:calculator` → a PostScript calculator function (Type 4,
# the exact `exp`) or `:sampled` → a 1024-entry 16-bit table (Type 0,
# linear interpolation, error ~1e-4 level): the fallback for a RIP that
# mishandles calculator functions. Both are PDF 1.3, as is the radial shading
# itself.
#
# How the blobs reach the page: the writer names ONE shading in the page's
# resources (`/B`: the unit disc, its colour function the profile) and the
# content stream paints it once per blob, under a matrix that scales it to the
# blob's cutoff radius at its centre (`_pattern_content`).
#
# Checked 2026-10-10 by rendering with pdftocairo (Cairo evaluates a shading
# per pixel): within one level of the profile at every σ from 2 px up. Other
# renderers decompose shadings into ~1-device-pixel bands and step small blobs
# (Ghostscript: 44 levels at σ = 2 px, 3 at σ ≥ 47 px), which is why
# `rasterize_pdf` uses pdftocairo.

# The shading's colour function — `BlobBoards.blob_profile` on t ∈ [0, 1]
# (inverted for `:light`), as (its dictionary's entries, its stream's data): a
# calculator program, the profile's formula written in PostScript,
# `(1 − e^(−k t²)) / (1 − e^(−k))` with `k = cutoff²/2`; or a table of
# `blob_profile` itself. The literals are ≤ 10 significant digits (`_pdf_num`):
# Ghostscript 10.02 silently drops a calculator function holding a 16-digit
# literal (paints nothing), and RIPs may have such limits too.
function _calculator_function(cutoff::Real, light::Bool)
    k = cutoff^2 / 2
    body = "dup mul $(_pdf_num(-k)) mul 2.7182818285 exch exp 1 exch sub $(_pdf_num(1 / (1 - exp(-k)))) mul" *
        (light ? " 1 exch sub" : "")
    return "/FunctionType 4 /Domain [ 0 1 ] /Range [ 0 1 ]", Vector{UInt8}(codeunits("{ $body }"))
end
const _PROFILE_SAMPLES = 1024

function _sampled_function(cutoff::Real, light::Bool)
    n = _PROFILE_SAMPLES
    samples = UInt8[]
    for i in 0:(n - 1)
        v = BlobBoards.blob_profile(i / (n - 1), cutoff)
        light && (v = 1 - v)
        s16 = round(UInt16, v * 65535)
        push!(samples, s16 >> 8, s16 & 0xff)                  # big-endian, as PDF reads samples
    end
    return "/FunctionType 0 /Domain [ 0 1 ] /Range [ 0 1 ] /Size [ $n ] /BitsPerSample 16 /Order 1", samples
end
# `name`: `Pdf.shading_function`, one of `BlobBoards.SHADING_FUNCTIONS` (checked when the output was made).
function _profile_function(name::Symbol, pattern_config)
    cutoff, light = pattern_config.cutoff, pattern_config.polarity == :light
    name === :calculator && return _calculator_function(cutoff, light)
    name === :sampled && return _sampled_function(cutoff, light)
    throw(ArgumentError("no writer for shading_function = $(repr(name))"))
end

# The content that paints the pattern onto the page, in default user space:
# points, origin bottom-left. `page_units` is the page height in figure (grid)
# units. The pattern's corner is `Translation(:pattern, :canvas)`, the layout's
# own offset. Its area is first filled with the background (white, or black for
# `:light`), then each blob is the unit-disc shading `/B` scaled to its cutoff
# radius at its centre (in the board's stored unit, :colmap: the grid's corner
# at 0).
function _pattern_content(bundle::BoardBundle, page_units::Integer)
    pattern_config = bundle.preamble.pattern_config
    grid = bundle.preamble.board_config.layout_grid
    light = pattern_config.polarity == :light
    cutoff = pattern_config.cutoff
    res = ustrip.(px, pattern_config.pattern_resolution)
    x0, y0 = Translation(bundle.preamble.board_config, :pattern, :canvas).translation
    coord = _page_coords(grid, page_units, x0, y0)
    pt_per_stored = BlobBoards._pt_per(BlobBoards.stored_density(grid))
    io = IOBuffer()
    x, y = coord(0.0, res.height)                              # the pattern's lower-left corner
    print(
        io, light ? "0" : "1", " g ", x, " ", y, " ",
        _pattern_length(grid, res.width), " ", _pattern_length(grid, res.height), " re f\n"
    )
    for blob in bundle.blobs
        X, Y = coord(ustrip.(px, blob.center)...)
        r = _pdf_num(ustrip(px, blob.σ) * cutoff * pt_per_stored)
        print(io, "q ", r, " 0 0 ", r, " ", X, " ", Y, " cm /B sh Q\n")
    end
    return String(take!(io))
end

# Lengths and positions on the page, in points, as the file is to hold them.
# A point-grid board is laid out in whole grid units, so each is k/100 pt
# written as exactly its two decimals — the stored value, character for
# character, never a product. A v3 board (whole printer pixels): the product,
# as it was always written.

# A position in the pattern (stored units, y down) → its page coordinates (y up).
function _page_coords(g::PixelGrid, page_units, x0, y0)
    pt_per_unit = _pt_per_unit(g)
    page_h = page_units * pt_per_unit
    return (cx, cy) -> (_pdf_num((x0 + cx) * pt_per_unit), _pdf_num(page_h - (y0 + cy) * pt_per_unit))
end
function _page_coords(g::PointGrid, page_units, x0, y0)
    gx0, gy0 = BlobBoards._grid_units(g, x0), BlobBoards._grid_units(g, y0)
    return (cx, cy) -> (
        _hundredths(gx0 + BlobBoards._grid_units(g, cx)),
        _hundredths(page_units - (gy0 + BlobBoards._grid_units(g, cy))),
    )
end
# A length of the pattern (stored units).
_pattern_length(g::PixelGrid, len) = _pdf_num(len * _pt_per_unit(g))
_pattern_length(g::PointGrid, len) = _hundredths(BlobBoards._grid_units(g, len))
# A page length of `units` whole figure (grid) units.
_page_length(::PointGrid, units::Integer) = _hundredths(units)
_page_length(g::PixelGrid, units::Integer) = _pdf_num(units * _pt_per_unit(g))
