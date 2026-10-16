# =============================================================================
# Documentation figures, drawn from BlobBoards itself when the docs are built
# =============================================================================
#
# make.jl calls `write_diagrams(dir)` before `makedocs`. Every size, box, mark
# and origin in a figure is what the package produces — a board it lays out, a
# PDF it writes and reads back, the maps it converts with — so a figure cannot
# drift from the code the way hand-drawn ASCII did. The SVGs land in
# docs/src/assets/generated/ (gitignored) and the pages show them as images.

using BlobBoards, CairoMakie
using BlobBoards: ustrip                     # BlobBoards' unit stripping (not exported)

const EDGE = :black
const TRIM = :crimson
const BOARD = :royalblue
const MARK = (:darkorange, 0.35)
const NOTE = :gray25
const FONT = 12

# Grid units of a board's layout (0.01 pt on a point-grid board) → mm.
_mm(units, bc) = ustrip(mm, units * px / BlobBoards.grid_density(bc.layout_grid))
# Points (a PDF's unit, and a board's stored unit) → mm.
_pt(v) = ustrip(mm, v * px / BlobBoards.POINT_DENSITY)

_fmt(x) = (r = round(x; digits = 1); isinteger(r) ? string(Int(r)) : string(r))

# A board on `paper` for a 600 dpi printer (`kw`: blob_board's), its PDF page
# rendered at `density_dpi`, and the PDF's page boxes in mm, read back from the
# file (y up, as PDF).
function _page(paper, output; density_dpi = 40, kw...)
    mktempdir() do dir
        bundle = blob_board(paper; blob_sizes = suggest_blob_sizes(600dpi, paper), seed = 0xB10B, output, dir, kw...)
        pdf = only(filter(endswith(".pdf"), readdir(dir; join = true)))
        s = read(pdf, String)
        box(k) = _pt.(parse.(Float64, split(match(Regex("/$k \\[ ([^\\]]*) \\]"), s)[1])))
        img = BlobBoards.rasterize_pdf(pdf, density_dpi)
        return bundle, img, Dict(k => box(k) for k in ("MediaBox", "BleedBox", "TrimBox"))
    end
end

# The board's rectangle on the canvas, in mm (x, y from the top-left, y down).
function _board_rect(bc)
    L = BlobBoards.BoardLayout(bc)
    left, top = _mm(L.margin[1], bc), _mm(L.margin[4], bc)
    return left, top, (_mm(ustrip(px, s), bc) for s in L.board_size)...
end

# A figure of one page in mm, y down, the page image under it, with `pad`
# (left, right, top, bottom) mm of room around the page for the labels.
function _page_figure(img, w, h; pad = (10, 10, 10, 10), scale = 2.4)
    l, r, t, b = pad
    fig = Figure(; size = (round(Int, (w + l + r) * scale), round(Int, (h + t + b) * scale)),
        backgroundcolor = :white, figure_padding = 2)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax); hidespines!(ax)
    image!(ax, 0 .. w, 0 .. h, permutedims(img); interpolate = true)
    limits!(ax, -l, w + r, h + b, -t)
    return fig, ax
end

_rect!(ax, x, y, w, h; kw...) = lines!(ax, [x, x + w, x + w, x, x], [y, y, y + h, y + h, y]; kw...)

# A dimension line from a to b at `at` (a y for a horizontal line, an x for a
# vertical one), its label on `side` (:above, :below, :left, :right).
function _dim!(ax, a, b, at, label; vertical = false, side = vertical ? :right : :above, color = NOTE)
    xs, ys = vertical ? ([at, at], [a, b]) : ([a, b], [at, at])
    lines!(ax, xs, ys; color, linewidth = 1)
    scatter!(ax, xs, ys; marker = vertical ? :hline : :vline, markersize = 8, color)
    mid = vertical ? (at, (a + b) / 2) : ((a + b) / 2, at)
    align, offset = Dict(:above => ((:center, :bottom), (0, 3)), :below => ((:center, :top), (0, -3)),
        :left => ((:right, :center), (-5, 0)), :right => ((:left, :center), (5, 0)))[side]
    text!(ax, mid...; text = label, align, offset, fontsize = FONT, color)
end

# A label at (x, y) with a leader line to the point it names.
function _callout!(ax, label, (x, y), (px_, py); color = NOTE, align = (:right, :center))
    lines!(ax, [x, px_], [y, py]; color, linewidth = 0.8)
    text!(ax, x, y; text = label, align, offset = (align[1] === :right ? -3 : 3, 0), fontsize = FONT, color)
end

# -----------------------------------------------------------------------------
# The frame's marks on a default board: corner cells and ruler bands
# -----------------------------------------------------------------------------
function frame_marks_figure()
    bundle, img, boxes = _page("A5", Pdf(); density_dpi = 60)
    bc = bundle.preamble.board_config
    L = BlobBoards.BoardLayout(bc)
    w, h = boxes["MediaBox"][3], boxes["MediaBox"][4]
    left, top, bw, bh = _board_rect(bc)
    fig, ax = _page_figure(img, w, h; pad = (26, 26, 12, 12))
    _rect!(ax, 0, 0, w, h; color = :gray60, linewidth = 1)
    for m in BlobBoards._frame_marks(bc)
        x, y, mw, mh = left + _mm(m.x, bc), top + _mm(m.y, bc), _mm(m.w, bc), _mm(m.h, bc)
        poly!(ax, Rect2f(x, y, mw, mh); color = MARK, strokecolor = :darkorange, strokewidth = 1)
        m.kind === :ruler && continue
        label = m.kind === :qr ? "QR code" : "uid $(m.index)"
        onleft = x + mw / 2 < w / 2
        _callout!(ax, label, (onleft ? -3 : w + 3, y + mh / 2), (x + mw / 2, y + mh / 2);
            align = (onleft ? :right : :left, :center))
    end
    _rect!(ax, left, top, bw, bh; color = BOARD, linewidth = 1.5)
    cs = _mm(ustrip(px, L.corner_size.width), bc)
    rw = _mm(ustrip(px, L.ruler_width), bc)
    _dim!(ax, left, left + cs, -3, "corner_size $(_fmt(cs)) mm")
    _dim!(ax, left, left + rw, h + 3, "ruler_width $(_fmt(rw)) mm"; side = :below)
    return fig
end

# -----------------------------------------------------------------------------
# Page boxes: MediaBox = BleedBox, TrimBox, the board inside its margin
# -----------------------------------------------------------------------------
function page_boxes_figure()
    bleed = 2mm
    bundle, img, boxes = _page("A4", Pdf(; bleed); density_dpi = 40)
    bc = bundle.preamble.board_config
    W, H = boxes["MediaBox"][3], boxes["MediaBox"][4]
    t = boxes["TrimBox"]                                     # PDF y up → page y down
    tx, ty, tw, th = t[1], H - t[4], t[3] - t[1], t[4] - t[2]
    left, top, bw, bh = _board_rect(bc)
    left, top = tx + left, ty + top                          # the board sits on the trimmed sheet
    fig, ax = _page_figure(img, W, H; pad = (8, 46, 14, 26), scale = 2.0)
    _rect!(ax, 0, 0, W, H; color = EDGE, linewidth = 1.5)
    _rect!(ax, tx, ty, tw, th; color = TRIM, linewidth = 1.5, linestyle = :dash)
    _rect!(ax, left, top, bw, bh; color = BOARD, linewidth = 1.5)
    _dim!(ax, 0, W, -4, "MediaBox = BleedBox  $(_fmt(W)) × $(_fmt(H)) mm"; color = EDGE)
    _dim!(ax, tx, tx + tw, H + 4, "TrimBox (the cut)  $(_fmt(tw)) × $(_fmt(th)) mm"; side = :below, color = TRIM)
    _dim!(ax, left, left + bw, H + 15, "board  $(_fmt(bw)) × $(_fmt(bh)) mm"; side = :below, color = BOARD)
    # down the right edge: the bleed (page edge → trim), then the margin (trim → board)
    _dim!(ax, 0, ty, W + 4, ""; vertical = true, color = TRIM)
    _callout!(ax, "bleed $(_fmt(ty)) mm", (W + 9, -6), (W + 4, ty / 2); color = TRIM, align = (:left, :center))
    _dim!(ax, ty, top, W + 4, "margin $(_fmt(top - ty)) mm"; vertical = true)
    return fig
end

# -----------------------------------------------------------------------------
# Image origins: where each convention puts the first pixel's centre and corner
# -----------------------------------------------------------------------------
function image_origins_figure()
    EM, PF = BlobBoards.EuclideanMap, BlobBoards.PixelFrame
    fig = Figure(; size = (780, 290), backgroundcolor = :white)
    for (i, conv) in enumerate((:colmap, :opencv, :julia))
        to = EM(PF{:colmap}, PF{conv})                 # the map the package converts with
        centre, corner = to(Vec2(0.5, 0.5)), to(Vec2(0.0, 0.0))
        ax = Axis(fig[1, i]; aspect = DataAspect(), yreversed = true, title = ":$conv", titlesize = 15)
        hidedecorations!(ax); hidespines!(ax)
        for k in 0:3
            lines!(ax, [0, 3], [k, k]; color = :gray70); lines!(ax, [k, k], [0, 3]; color = :gray70)
        end
        poly!(ax, Rect2f(0, 0, 1, 1); color = (BOARD, 0.15))
        scatter!(ax, [0.5], [0.5]; color = BOARD, markersize = 10)
        scatter!(ax, [0.0], [0.0]; color = TRIM, markersize = 10, marker = :rect)
        text!(ax, 0.5, 0.5; text = "centre\n($(_fmt(centre[1])), $(_fmt(centre[2])))", align = (:center, :top),
            offset = (0, -8), fontsize = FONT, color = BOARD)
        text!(ax, 0.0, 0.0; text = "corner ($(_fmt(corner[1])), $(_fmt(corner[2])))", align = (:left, :bottom),
            offset = (4, 4), fontsize = FONT, color = TRIM)
        limits!(ax, -0.4, 3.2, 3.2, -0.7)
    end
    return fig
end

# -----------------------------------------------------------------------------
# Reference frames: the :canvas, :board and :pattern origins on a real board
# -----------------------------------------------------------------------------
function reference_frames_figure()
    bundle, img, boxes = _page("A5", Pdf(); density_dpi = 50)
    bc = bundle.preamble.board_config
    W, H = boxes["MediaBox"][3], boxes["MediaBox"][4]
    _, _, bw, bh = _board_rect(bc)
    # each frame's origin on the canvas, by the map the package uses
    origin(frame) = Tuple(_pt.(BlobBoards.Translation(bc, frame, :canvas).translation))
    frames = (:canvas, :board, :pattern)
    at = Dict{Tuple{Float64, Float64}, Vector{Symbol}}()
    foreach(f -> push!(get!(at, origin(f), Symbol[]), f), frames)     # frames that share an origin
    bx, by = origin(:board)
    fig, ax = _page_figure(img, W, H; pad = (64, 6, 12, 22))
    _rect!(ax, 0, 0, W, H; color = EDGE, linewidth = 1.5)
    _rect!(ax, bx, by, bw, bh; color = BOARD, linewidth = 1.5)
    for (k, ((x, y), fs)) in enumerate(sort(collect(at); by = first))
        color = :canvas in fs ? EDGE : BOARD
        scatter!(ax, [x], [y]; color, markersize = 10, strokecolor = :white, strokewidth = 1)
        _callout!(ax, join((":$f" for f in fs), " = ") * " origin\n($(_fmt(x)), $(_fmt(y))) mm", (-4, y + 6k), (x, y); color)
    end
    _dim!(ax, 0, W, -4, "canvas  $(_fmt(W)) × $(_fmt(H)) mm"; color = EDGE)
    _dim!(ax, bx, bx + bw, H + 4, "board = pattern  $(_fmt(bw)) × $(_fmt(bh)) mm"; side = :below, color = BOARD)
    _dim!(ax, 0, bx, H + 4, "margin $(_fmt(bx)) mm"; side = :below, color = NOTE)
    return fig
end

const DIAGRAMS = (
    "frame_marks" => frame_marks_figure,
    "page_boxes" => page_boxes_figure,
    "image_origins" => image_origins_figure,
    "reference_frames" => reference_frames_figure,
)

function write_diagrams(dir)
    mkpath(dir)
    for (name, figure) in DIAGRAMS
        save(joinpath(dir, "$name.svg"), figure())
    end
    return dir
end
