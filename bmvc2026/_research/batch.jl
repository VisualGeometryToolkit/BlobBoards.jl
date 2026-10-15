"""
    batch.jl

Generate multiple calibration boards in batch -- sweep paper sizes, printer
densities or pattern sizes. A board has no density of its own (a PDF prints on
any printer): a printer's density sizes its blobs through
`suggest_blob_sizes(density, paper)`.

Usage (CLI):
    julia --project=. bmvc2026/_research/batch.jl                 # A4, A5, Letter for a 200 dpi printer
    julia --project=. bmvc2026/_research/batch.jl sizes           # Same as above
    julia --project=. bmvc2026/_research/batch.jl densities       # A4 for 150, 300, 600 dpi printers

From REPL:
    include("bmvc2026/_research/batch.jl")
    generate_paper_sizes()
    generate_paper_sizes(; density=300, dir="output")
    generate_densities()
    generate_densities(; paper_size="Letter", densities=[150, 300], dir="output")

Files Created:
    Multiple blob_board_<UID>.pdf and blob_board_<UID>.json files
"""

using BlobBoards   # re-exports mm, dpi
using BlobBoards: ustrip


# =============================================================================
# Pattern-size sweep
# =============================================================================

"""
    generate_pattern_sizes(; density=200, dir=".")

Generate boards of several sizes (12, 9 and 5 cm square) centred on a 15 cm
square page, with no ruler (blobs fill the board around the corner cells).
`density` is not used: these boards take explicit blob sizes.
"""
function generate_pattern_sizes(; density::Int=1200, dir::String=".")
    canvas_size = Size2(width=150mm, height=150mm)
    sizes = [Size2(width=120mm, height=120mm), Size2(width=90mm, height=90mm), Size2(width=50mm, height=50mm)]

    println("Generating calibration boards for multiple pattern sizes...")
    println("  Output directory: $dir")
    println()

    for pattern_size in sizes
        name = "$(ustrip(mm, pattern_size.width))×$(ustrip(mm, pattern_size.height)) mm"
        println("Generating $name...")
        # (was a blank 10 mm band, frame_style = :none; a board's marks are now switched by their sizes)
        bundle = blob_board(canvas_size; board_size=pattern_size, cutoff=2.0, α=1.6, blob_sizes=(0.8mm, 0.9short_side), ruler_width=0mm, frame_padding=2mm, polarity=:dark, dir=dir)  # was max_diameter_fraction = 0.9 at cutoff 2
        num_blobs = length(bundle.blobs)
        println("  $name: $num_blobs blobs")
    end

    println("\nGenerated $(length(sizes)) boards")
    println("Files saved to: $dir")
end

# =============================================================================
# Paper-size sweep
# =============================================================================

"""
    generate_paper_sizes(; density=200, dir=".")

Generate boards for standard paper sizes (A4, A5, Letter), their blobs sized
for a printer at `density` dpi.
"""
function generate_paper_sizes(; density::Int=200, dir::String=".")
    sizes = ["A4", "A5", "Letter"]

    println("Generating calibration boards for multiple paper sizes...")
    println("  Printer density: $(density) dpi")
    println("  Output directory: $dir")
    println()

    for paper_size in sizes
        println("Generating $paper_size...")
        bundle = blob_board(paper_size; blob_sizes=suggest_blob_sizes(density * dpi, paper_size), cutoff=2.0, dir=dir)
        num_blobs = length(bundle.blobs)
        println("  $paper_size: $num_blobs blobs")
    end

    println("\nGenerated $(length(sizes)) boards")
    println("Files saved to: $dir")
end

# =============================================================================
# Density sweep
# =============================================================================

"""
    generate_densities(; paper_size="A4", densities=[150, 300, 600], dir=".")

Generate boards for printers at different densities (each density sizes the
blobs), with timing.
"""
function generate_densities(; paper_size::String="A4",
                              densities::Vector{Int}=[150, 300, 600],
                              dir::String=".")
    println("Generating calibration boards at multiple densities...")
    println("  Paper size: $paper_size")
    println("  Densities: $(join(densities, ", ")) DPI")
    println("  Output directory: $dir")
    println()

    for density in densities
        println("Generating for a $density dpi printer...")
        start_time = time()

        bundle = blob_board(paper_size; blob_sizes=suggest_blob_sizes(density * dpi, paper_size), cutoff=2.0, dir=dir)

        elapsed = time() - start_time
        num_blobs = length(bundle.blobs)
        println("  $(density)dpi: $num_blobs blobs ($(round(elapsed, digits=2))s)")
    end

    println("\nGenerated $(length(densities)) boards")
    println("Files saved to: $dir")
end

# =============================================================================
# CLI entry point
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? ARGS[1] : "sizes"
    if mode == "sizes"
        generate_paper_sizes()
    elseif mode == "densities"
        generate_densities()
    elseif mode == "patterns"
        generate_pattern_sizes(; dir = get(ENV, "BB_OUT", joinpath("output", "boards", "training")))
    else
        println("Usage: julia examples/generate/batch.jl [sizes|densities|patterns]")
    end
end
