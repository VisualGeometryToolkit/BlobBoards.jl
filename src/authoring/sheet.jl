# =============================================================================
# The sheet around the pattern — rulers, QR code, uid characters, id label
# =============================================================================
#
# Everything a board's page holds besides its blobs, as PDF content: black
# rectangles (the QR code), stroked lines (the ruler ticks) and text in the
# embedded sheet font (the uid characters, the id label). Coordinates are in figure (grid) units —
# whole units of the board's layout grid — with y UP from the page's
# bottom-left corner, as a PDF page has it; the writer scales the lot to points
# with one matrix (`_page_content`). A cell is its bottom-left corner
# `(ox, oy)` and its size.

# The frame's line weight, in figure (grid) units: one printer pixel on a v3
# board, as it was printed; 0.1 mm on a point-grid board, whatever the density.
const _FRAME_LINE_WIDTH = 0.1mm
_frame_line(::PixelGrid) = 1
_frame_line(g::PointGrid) = ustrip(px, _FRAME_LINE_WIDTH * BlobBoards.grid_density(g))

# A frame position snapped to the grid: a v3 board's lines on pixel centres
# (n + ½, one pixel wide), a point-grid board's on whole grid units.
_snap(::PixelGrid, x) = floor(x) + 0.5
_snap(::PointGrid, x) = round(x)

# What a board's page holds around its pattern: the frame's content
# (`_frame_content`) and the characters its text sets (none: no font is
# embedded). The page itself is the board's layout, which the writer reads
# from the bundle.
struct Sheet
    frame::String                  # PDF operators, in grid units, y up
    chars::Vector{Char}            # sorted
end

# --- rulers ---------------------------------------------------------------------

# A ruler band's ticks as segments (x1, y1, x2, y2) in its cell, in figure
# (grid) units: ticks at exact multiples of their spacing (mm), centred along
# the band, snapped to the grid (`_snap`).
function _ruler_ticks(
        spec, grid::LayoutGrid, cell_size::Size2{<:Quantity};
        orientation, ruler_width::Quantity, frame_padding::Quantity
    )
    density = BlobBoards.grid_density(grid)
    # tick spacings in plain grid units: mm × units/mm (`_length_to_px`)
    major_spacing_px = BlobBoards._length_to_px(spec.major_tick.spacing, density)
    minor_spacing_px = BlobBoards._length_to_px(spec.minor_tick.spacing, density)
    padding = Int(ustrip(px, BlobBoards._grid_count(grid, frame_padding)))   # whole grid units

    ruler_width_phys = ruler_width
    ruler_width, frame_length = orientation ==
        vruler ? (cell_size.width, cell_size.height) : (cell_size.height, cell_size.width)

    # The padded band in whole pixels; a tick's drawn length is its physical
    # length (its full extent across the band, see RulerSpec) as a FRACTION of
    # the physical padded band, times the band's pixels — so the default (the
    # major tick = the whole band) lands on the band's pixel edges exactly.
    content_width = ruler_width - 2 * padding * px
    @assert ustrip(px, content_width) > 0 "frame_padding too large: no space left for ruler content (width)"
    band = ruler_width_phys - 2 * frame_padding
    major_tick_width = content_width * (ustrip(mm, spec.major_tick.width) / ustrip(mm, band))
    minor_tick_width = content_width * (ustrip(mm, spec.minor_tick.width) / ustrip(mm, band))

    # Horizontal extent: also apply padding
    ruler_length_ubound = frame_length - 2 * padding * px
    @assert ustrip(px, ruler_length_ubound) > 0 "frame_padding too large: no space left for ruler content (length)"

    ruler_length_ubound_px = ustrip(px, ruler_length_ubound)
    major_tick_count = floor(Int, ruler_length_ubound_px / major_spacing_px)
    ruler_length_px = major_spacing_px * major_tick_count
    minor_tick_count = floor(Int, ruler_length_px / minor_spacing_px)

    # Compute exact floating-point offset (no premature rounding)
    centering_offset = (ruler_length_ubound_px - ruler_length_px) / 2
    offset = padding + centering_offset

    # Compute exact positions, then snap to the grid (`_snap`: whole grid units
    # on a point grid; half-pixels, on pixel centres, on a v3 board's pixel grid)

    major_positions = [_snap(grid, offset + i * major_spacing_px) for i in 0:major_tick_count]
    minor_positions = [_snap(grid, offset + i * minor_spacing_px) for i in 0:minor_tick_count]
    setdiff!(minor_positions, major_positions)

    # Tick widths and y-positions, snapped the same way
    ruler_width_px = ustrip(px, ruler_width)
    major_tick_width_px = ustrip(px, major_tick_width)
    minor_tick_width_px = ustrip(px, minor_tick_width)

    majorymin = _snap(grid, (ruler_width_px - major_tick_width_px) / 2)
    majorymax = _snap(grid, majorymin + major_tick_width_px)
    minorymin = _snap(grid, (ruler_width_px - minor_tick_width_px) / 2)
    minorymax = _snap(grid, minorymin + minor_tick_width_px)

    # along the band, across it → the cell's x, y (swapped for a vertical ruler)
    tick(p, lo, hi) = orientation === vruler ? (lo, p, hi, p) : (p, lo, p, hi)
    return NTuple{4, Float64}[
        (tick(p, majorymin, majorymax) for p in major_positions)...,
        (tick(p, minorymin, minorymax) for p in minor_positions)...,
    ]
end

function _ruler_ops(io::IO, ox::Real, oy::Real, ticks)
    isempty(ticks) && return nothing
    for (x1, y1, x2, y2) in ticks
        print(io, _pdf_num(ox + x1), ' ', _pdf_num(oy + y1), " m ", _pdf_num(ox + x2), ' ', _pdf_num(oy + y2), " l\n")
    end
    print(io, "S\n")
    return nothing
end

# --- QR code --------------------------------------------------------------------

# The QR code as vector rectangles (x, y, w, h) in its corner cell: one per
# horizontal run of dark modules (`true`), each module `rat` × `rat` grid
# units, the code centred in the cell (on a v3 board's pixel grid, the
# whole-pixel geometry the bitmap version had), sharp at any resolution. Runs,
# not single modules: an anti-aliasing viewer would show hairlines where
# separate squares touch. The cell's background (the light modules and the
# quiet zone) is the paper.
function _qr_rects(qrmat::AbstractMatrix{Bool}, cell_size::Size2{<:Quantity}; padding = 0)
    sz = ustrip.(px, cell_size)
    # Content area after padding (padding on all sides)
    content_sz = sz .- 2 * padding
    rat = minimum(floor.(Int, content_sz ./ Size2(qrmat)))
    rects = NTuple{4, Int}[]
    if rat >= 1
        h, w = size(qrmat)
        # centring offset in the cell, whole pixels from its top-left corner
        x_off, y_off = floor.(Int, (sz - Size2(w * rat, h * rat)) / 2)
        for i in 1:h
            j = 1
            while j <= w
                qrmat[i, j] || (j += 1; continue)
                k = j
                while k < w && qrmat[i, k + 1]
                    k += 1
                end
                # y up from the cell's bottom edge
                push!(rects, (x_off + (j - 1) * rat, Int(sz.height) - (y_off + i * rat), (k - j + 1) * rat, rat))
                j = k + 1
            end
        end
    else
        @warn "QR code is larger than allocated frame space, QR code will be omitted"
    end
    return rects
end

function _qr_ops(io::IO, ox::Real, oy::Real, rects)
    isempty(rects) && return nothing
    for (x, y, w, h) in rects
        print(io, _pdf_num(ox + x), ' ', _pdf_num(oy + y), ' ', _pdf_num(w), ' ', _pdf_num(h), " re\n")
    end
    print(io, "f\n")
    return nothing
end

# --- uid characters ---------------------------------------------------------------

# A uid character's em, as a share of its padded corner cell's shorter side.
const _UID_GLYPH_SCALE = 0.9

# One uid character in its corner cell: the glyph's ink centred in the cell,
# one em `_UID_GLYPH_SCALE` of the padded cell's shorter side (whole grid units), filled
# and stroked at the frame's line weight.
function _uid_ops!(
        io::IO, chars::Set{Char}, font::FTFont, ch::Char, ox::Real, oy::Real, cell_size::Size2{<:Quantity}; padding = 0
    )
    sz = ustrip.(px, cell_size)
    em = round(Int, _UID_GLYPH_SCALE * minimum(sz .- 2 * padding))
    cx, cy = _ink_centre(font, ch)
    _text_ops!(io, chars, font, string(ch), ox + sz.width / 2 - em * cx, oy + sz.height / 2 - em * cy, em; mode = _TEXT_FILL_STROKE)
    return nothing
end

# --- id label -----------------------------------------------------------------------

# The label's printed size and placement: 6 pt text centred under the board, a
# 1.5 mm gap below it, and its lower edge kept 5 mm from the sheet edge (clear of
# a printer's unprintable border, typically 4.2 mm). With no room — a board that
# fills the sheet, a margin under 9.25 mm, or a sheet too narrow for the line even
# at 3 pt — the label is skipped (with a warning when the board is frameless and
# so left unidentified); it is never squeezed onto the board.
const _ID_LABEL_PT = 6.0
const _ID_LABEL_MIN_PT = 3.0
const _ID_LABEL_GAP = 1.5mm
const _ID_LABEL_EDGE = 5.0mm

# Returns whether the label was drawn. `grid`: the layout grid the sheet is
# drawn in (its units are grid units).
function _id_label_ops!(io::IO, chars::Set{Char}, font::FTFont, text::AbstractString, board_sizes, grid::LayoutGrid; framed::Bool)
    density = BlobBoards.grid_density(grid)
    to_px(q) = BlobBoards._length_to_px(q, density)
    sz = ustrip.(px, board_sizes.canvas_size)
    _, _, bottom, _ = board_sizes.margin
    edge, gap = to_px(_ID_LABEL_EDGE), to_px(_ID_LABEL_GAP)
    pt_px = 1 / _pt_per_unit(grid)                            # grid units per typographic point
    # ≈ 0.6 em per character for the sheet's sans; shrink to fit the printable width
    fit_pt = (sz.width - 2 * edge) / (0.6 * length(text) * pt_px)
    font_pt = min(_ID_LABEL_PT, fit_pt)
    font_px = font_pt * pt_px
    # the line hangs from its ASCENDER, so the ink reaches ~1.3 em below the
    # anchor (TeX Gyre Heros): budget that, not 1 em, against the edge guard
    if font_pt < _ID_LABEL_MIN_PT || bottom - gap - 1.3 * font_px < edge
        # A framed board still carries its uid and QR, so only a FRAMELESS board,
        # left with no printed identity at all, is worth a warning.
        framed || @warn "id_label skipped: no room for it in the sheet margin under the " *
            "board (margin $(round(ustrip(mm, bottom / pt_px * pt); digits = 1)) mm), and a " *
            "frameless board then carries no printed identity; give the board a smaller " *
            "board_size than the sheet, or set id_label = false"
        return false
    end
    # centred on the sheet, its ascender line `gap` under the board
    x0 = sz.width / 2 - font_px * _line_width(font, text) / 2
    baseline = bottom - gap - font_px * Float64(FreeTypeAbstraction.ascender(font))
    _text_ops!(io, chars, font, text, x0, baseline, font_px)
    return true                                                  # drawn
end

# --- the frame ----------------------------------------------------------------------

# The frame's content for a board: its marks (`BlobBoards._frame_marks`, the one
# list placement keeps clear of too) — the QR code and the uid characters in the
# corner cells, and the ruler bands between them (a stored v3 `:none` board's
# band is blank) — and, when `label` is a text, the id label under the board.
# A board without marks has only the label. It is drawn over the pattern, which
# on a point-grid board covers the whole board: a `:light` board's keep-outs are
# painted white first, over its black background.
# Returns (the sheet, whether the label was drawn).
function _frame_content(
        preamble::BoardPreamble, board_sizes, config_hash::AbstractString, qr_text::AbstractString, label::Union{Nothing, AbstractString}
    )
    board_config = preamble.board_config
    grid = board_config.layout_grid
    font = _sheet_font()
    io = IOBuffer()
    chars = Set{Char}()
    print(io, "0 g 0 G ", _pdf_num(_frame_line(grid)), " w\n")
    marks = BlobBoards._frame_marks(board_config)
    framed = !isempty(marks)
    if framed
        uid = uid_from_hash(String(config_hash))   # the characters printed on the corner squares
        left, _, _, top = board_sizes.margin
        H = Int(ustrip(px, board_sizes.canvas_size.height))
        page(m) = (left + m.x, H - top - m.y - m.h)              # a mark's bottom-left corner on the page (y up)
        padding = Int(ustrip(px, BlobBoards._grid_count(grid, board_config.frame_padding)))

        # a point-grid board's pattern covers the board: on a `:light` one the
        # marks' reserved areas are painted white over its black background
        if grid isa PointGrid && preamble.pattern_config.polarity == :light
            print(io, "1 g\n")
            for (x0, y0, x1, y1) in BlobBoards._keepout_boxes(board_config)
                print(io, left + x0, " ", H - top - y1, " ", x1 - x0, " ", y1 - y0, " re f\n")
            end
            print(io, "0 g\n")
        end

        ticks(cell; orientation) = _ruler_ticks(
            board_config.ruler_spec, grid, cell; orientation,
            ruler_width = board_config.ruler_width, frame_padding = board_config.frame_padding
        )
        for m in marks
            x, y = page(m)
            cell = Size2(m.w * px, m.h * px)
            if m.kind === :qr
                _qr_ops(io, x, y, _qr_rects(make_qr_code(qr_text), cell; padding))
            elseif m.kind === :uid
                _uid_ops!(io, chars, font, uid[m.index], x, y, cell; padding)
            elseif m.kind === :ruler                              # (a `:blank` band draws nothing)
                _ruler_ops(io, x, y, ticks(cell; orientation = m.index <= 2 ? hruler : vruler))   # top, bottom: horizontal
            end
        end
    end
    labelled = label !== nothing && _id_label_ops!(io, chars, font, label, board_sizes, grid; framed)
    return Sheet(String(take!(io)), sort!(collect(chars))), labelled
end
