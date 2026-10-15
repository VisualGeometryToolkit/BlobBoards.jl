# =============================================================================
# Board export — the print master (PDF/X-4) and its TIFF/PNG conversions
# (split from src/io.jl; the JSON/load side stays in the core package)
# =============================================================================

# Points per figure unit: the sheet is laid out in whole grid units.
_pt_per_unit(grid::LayoutGrid) = BlobBoards._pt_per(BlobBoards.grid_density(grid))

# The canvas of a board, in whole grid units: (width, height).
function _canvas_units(bundle::BoardBundle)
    c = ustrip.(px, BoardLayout(bundle.preamble.board_config).canvas_size)
    return Int(c.width), Int(c.height)
end

# The page's content stream: the pattern, in points (`_pattern_content`: a blob
# centre is written as the very decimal it is stored as, never a product), then
# the frame over it, scaled from figure (grid) units to points by one matrix.
function _page_content(sheet::Sheet, bundle::BoardBundle)
    s = _pdf_num(_pt_per_unit(bundle.preamble.board_config.layout_grid))
    # the pattern first, the frame over it (a `:corners` board's cells lie on its pattern)
    return _pattern_content(bundle, _canvas_units(bundle)[2]) * "q $s 0 0 $s 0 0 cm\n" * sheet.frame * "Q\n"
end

# A file is never left half-made: `write_to(temp)` writes beside its place,
# then it is moved there.
function _write_whole(write_to, path::AbstractString)
    temp_path = path * ".tmp"
    try
        write_to(temp_path)
        mv(temp_path, path; force = true)
    finally
        isfile(temp_path) && rm(temp_path)
    end
    return nothing
end

# Write the file `o` is: `sheet` is the frame around the pattern, `bundle`
# supplies the layout and the blobs.
#
# PDF: the print master, written directly (pdf.jl) — PDF/X-4 and PDF/A-2b,
# lossless Flate, DeviceGray, no transparency, no image, text in one embedded
# font; plain objects (no object streams / xref streams), readable by older RIPs.
#
# The page is EXACTLY the canvas, grid units · points per unit — a whole number
# of grid units (A4 on the point grid: 59527 × 84188 units = 595.27 × 841.88 pt,
# the grid's floor of 210 × 297 mm; a v3 board at 300 dpi: 2480 px = 595.2 pt) —
# and the TrimBox the board area inside the margins.
function _write_output(o::Pdf, path, sheet::Sheet, bundle::BoardBundle; title::AbstractString)
    grid = bundle.preamble.board_config.layout_grid
    len(units) = _page_length(grid, units)
    w, h = _canvas_units(bundle)
    left, right, bottom, top = BoardLayout(bundle.preamble.board_config).margin
    bytes = _print_master(;
        page = (len(w), len(h)), trimbox = (len(left), len(bottom), len(w - right), len(h - top)),
        content = _page_content(sheet, bundle),
        shading = _profile_function(o.shading_function, bundle.preamble.pattern_config),
        sheet.chars, o.icc_profile, title
    )
    return _write_whole(temp -> write(temp, bytes), path)
end

# A raster output is the vector sheet CONVERTED — the one source of truth: the
# print master (blobs as shadings) rasterized by pdftocairo at the raster's density.
# The image is the page on Poppler's pixel grid: anchored at the page's top-left
# corner, rounded UP to whole pixels (a pixel coordinate is the page's points ×
# dpi/72); a page that is a whole number of pixels (a v3 board at its own
# density) is exactly the canvas — floating point can add one blank row or
# column, cropped.
function _sheet_raster(sheet::Sheet, bundle::BoardBundle, density; title)
    render_dpi = BlobBoards._render_dpi(density)                # a whole number (`Tiff`, `Png`)
    return mktempdir() do dir
        pdf = joinpath(dir, "sheet.pdf")
        _write_output(Pdf(), pdf, sheet, bundle; title)
        page = BlobBoards.rasterize_pdf(pdf, render_dpi)
        w, h = _raster_size(_canvas_units(bundle), bundle.preamble.board_config.layout_grid, render_dpi)
        0 <= size(page, 1) - h <= 1 && 0 <= size(page, 2) - w <= 1 ||
            error("the converted sheet is $(size(page)) px, the canvas $((h, w))")
        size(page) == (h, w) ? page : page[1:h, 1:w]
    end
end

# The canvas (whole grid units) in whole pixels at `dpi_val`: rounded up, as
# Poppler sizes the page — a hair of slack, so a page that IS a whole number of
# pixels stays one (8.5 inch at 300 dpi is 2550 px, not 2551).
_raster_size(units, grid, dpi_val) =
    Tuple(ceil.(Int, units .* (dpi_val / ustrip(dpi, BlobBoards.grid_density(grid))) .- 1.0e-6))

# TIFF: the converted sheet, 8-bit grey, with the density in the resolution
# tags (so the print shop's software sizes it exactly) and the ICC profile
# embedded; LZW-compressed by libtiff (`write_tiff`, tiff.jl).
function _write_output(o::Tiff, path, sheet::Sheet, bundle::BoardBundle; title)
    grey = _sheet_raster(sheet, bundle, o.density; title)
    icc = read(first(_icc_profile(o.icc_profile)))
    return _write_whole(temp -> write_tiff(temp, reinterpret(UInt8, grey), ustrip(dpi, o.density), icc), path)
end

# PNG, for digital use: the converted sheet as single-channel grey with linear
# gamma (file_gamma = 1.0) for an accurate intensity representation.
function _write_output(o::Png, path, sheet::Sheet, bundle::BoardBundle; title)
    grey = _sheet_raster(sheet, bundle, o.density; title)
    return _write_whole(temp -> BlobBoards.PNGFiles.save(temp, grey; file_gamma = 1.0), path)
end

# The board's file and its sidecar, for a checked output; an existing pair is replaced.
function _save_board(
        o::BoardOutput, sheet::Sheet, bundle::BoardBundle; dir, id_label::Bool, config_hash::AbstractString
    )
    valid_dir = validate_dir(dir; create = true)
    base = make_filebase(bundle; config_hash)
    fig_path = joinpath(valid_dir, base * "." * BlobBoards._format(o))
    _write_output(o, fig_path, sheet, bundle; title = "BlobBoards board $base")
    @info "Saved Board" image = fig_path

    json_path = joinpath(valid_dir, base * ".json")
    save_json(
        json_path, bundle; overwrite = true, pretty_json = true, render = BlobBoards._render_record(o, id_label), config_hash
    )
    return fig_path, json_path
end
