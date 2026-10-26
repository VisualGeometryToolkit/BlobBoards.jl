"""
    board.jl

Generate calibration boards with blob patterns for computer vision.

Demonstrates all board variants: standard paper sizes (A4, A5, Letter),
custom physical dimensions, negative polarity, and PNG output format.

Usage (CLI), from the checkout:
    julia --project=. examples/generate/board.jl                     # A4, for a 300 dpi printer, PDF
    julia --project=. examples/generate/board.jl A5                  # A5, 300 dpi
    julia --project=. examples/generate/board.jl Letter 200          # Letter, 200 dpi
    julia --project=. examples/generate/board.jl A4 600 png          # A4, 600 dpi, PNG at 600 dpi
    julia --project=. examples/generate/board.jl A4 300 pdf light    # Negative polarity

A board has no pixel density. The `density` these functions take is the
PRINTER's: it sizes the smallest disc (`suggest_blob_sizes(density, extent)`:
3 mm at 1200 dpi, 6 mm at 600, 12 mm at 300) and is the density of a TIFF or
PNG. The largest disc follows from the board: 20% of its shorter side, at most
64 mm.

From REPL:
    include("examples/generate/board.jl")
    generate_board("A4", 300)
    generate_board("A5", 300; ruler_width=20mm)
    generate_board("A4", 300; polarity=:light)
    generate_custom_board(150mm, 200mm, 300)
    generate_png_board("A4", 300)

Blob sizes are physical lengths, the printed disc diameters, and required:
    blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"))
    blob_board("A4"; blob_sizes = (2mm, 12mm))
    blob_board("A4"; blob_sizes = (4.0mm, 0.2short_side))

Files Created:
    blob_board_<UID>.pdf (or .tiff, .png)  - Calibration board
    blob_board_<UID>.json           - Metadata for bit-exact reproduction
"""

using BlobBoards            # re-exports the units: mm, inch, dpi, …

# The output a format name and a density describe: a PDF has no density, a
# raster is the sheet converted at `density`.
function board_output(format::String, density)
    format == "pdf" && return Pdf()
    format in ("tiff", "tif") && return Tiff(density)
    format == "png" && return Png(density)
    throw(ArgumentError("format must be \"pdf\", \"tiff\" or \"png\", got $(repr(format))"))
end

# The default board's extent: the page less the default margin on each side.
board_extent(width, height) = (width - 2 * BlobBoards.DEFAULT_BOARD_MARGIN, height - 2 * BlobBoards.DEFAULT_BOARD_MARGIN)

# =============================================================================
# Standard paper-size board
# =============================================================================

"""
    generate_board(paper_size="A4", density=300; dir=".", format="pdf",
                   polarity=:dark, ruler_width=nothing)

Generate a calibration board for a standard paper size.

# Arguments
- `paper_size`: Paper size string ("A4", "A5", "Letter")
- `density`: the printer's density in DPI (e.g., 300, 600, 1200): sizes the
  smallest disc (`suggest_blob_sizes`) and is the density of a TIFF or PNG
- `dir`: Output directory
- `format`: Output format ("pdf", "tiff" or "png")
- `polarity`: `:dark` (dark blobs on light) or `:light` (light blobs on dark)
- `ruler_width`: Optional override for the ruler bands' width (0mm: no ruler)
"""
function generate_board(paper_size::String="A4", density::Int=300;
                        dir::String=".",
                        format::String="pdf",
                        polarity::Symbol=:dark,
                        ruler_width=nothing)
    printer_density = density * dpi
    blob_sizes = suggest_blob_sizes(printer_density, paper_size)

    println("Generating $paper_size calibration board...")
    println("  Printer density: $printer_density")
    println("  Blob sizes: $(blob_sizes[1]) … $(blob_sizes[2])")
    println("  Format: $format")
    println("  Polarity: $polarity")
    println("  Output directory: $dir")

    kwargs = Dict{Symbol,Any}(:dir => dir, :cutoff => 2.0, :blob_sizes => blob_sizes,
                               :output => board_output(format, printer_density),
                               :polarity => polarity)
    if ruler_width !== nothing
        kwargs[:ruler_width] = ruler_width
        println("  Ruler width: $ruler_width")
    end

    bundle = blob_board(paper_size; kwargs...)

    num_blobs = length(bundle.blobs)
    println("\nGenerated $paper_size board with $num_blobs blobs")
    println("Files saved to: $dir")

    return bundle
end

# =============================================================================
# Custom physical dimensions
# =============================================================================

"""
    generate_custom_board(width, height, density=300; dir=".",
                          ruler_width=15mm,
                          major_tick_spacing=10mm,
                          minor_tick_spacing=5mm)

Generate a calibration board with custom physical dimensions.

# Arguments
- `width`: Board width with units (e.g., `150mm`)
- `height`: Board height with units (e.g., `200mm`)
- `density`: the printer's density in DPI (sizes the smallest disc)
- `dir`: Output directory
- `ruler_width`: The ruler bands' width (0mm: no ruler)
- `major_tick_spacing`: Ruler major tick spacing
- `minor_tick_spacing`: Ruler minor tick spacing
"""
function generate_custom_board(width::BlobBoards.Quantity, height::BlobBoards.Quantity,
                               density::Int=300;
                               dir::String=".",
                               ruler_width=15mm,
                               major_tick_spacing=10mm,
                               minor_tick_spacing=5mm)
    printer_density = density * dpi

    println("Generating custom calibration board...")
    println("  Dimensions: $width x $height")
    println("  Printer density: $printer_density")
    println("  Ruler width: $ruler_width")
    println("  Output directory: $dir")

    bundle = blob_board(width, height;
                        blob_sizes=suggest_blob_sizes(printer_density, board_extent(width, height)),
                        ruler_width=ruler_width,
                        major_tick_spacing=major_tick_spacing,
                        minor_tick_spacing=minor_tick_spacing,
                        cutoff=2.0,
                        dir=dir)

    num_blobs = length(bundle.blobs)
    println("\nGenerated $(width) x $(height) board with $num_blobs blobs")
    println("Files saved to: $dir")

    return bundle
end

# =============================================================================
# Convenience wrappers for common variants
# =============================================================================

"""
    generate_png_board(paper_size="A4", density=300; dir=".")

Generate a board as a PNG at `density` instead of a PDF. Useful for training data or display.
"""
generate_png_board(paper_size::String="A4", density::Int=300; dir::String=".") =
    generate_board(paper_size, density; dir=dir, format="png")

"""
    generate_negative_board(paper_size="A4", density=300; dir=".", ruler_width=25mm)

Generate a board with light blobs on a dark background.
"""
generate_negative_board(paper_size::String="A4", density::Int=300;
                        dir::String=".", ruler_width=25mm) =
    generate_board(paper_size, density; dir=dir, polarity=:light,
                   ruler_width=ruler_width)

# =============================================================================
# CLI entry point
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    paper_size = length(ARGS) >= 1 ? ARGS[1] : "A4"
    density    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
    format     = length(ARGS) >= 3 ? ARGS[3] : "pdf"
    polarity   = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :dark
    generate_board(paper_size, density; format=format, polarity=polarity)
end
