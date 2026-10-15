"""
    pattern.jl

Generate blob patterns for computer vision calibration targets.

Two modes: simple (dimensions only) and explicit (full PatternConfig control).
Patterns are saved as PNG images with a JSON sidecar for bit-exact reproduction.

Usage (CLI):
    julia --project=. examples/generate/pattern.jl                # 800x600 (default)
    julia --project=. examples/generate/pattern.jl 400 300        # 400x300
    julia --project=. examples/generate/pattern.jl 1920 1080      # Full HD

From REPL:
    include("examples/generate/pattern.jl")
    generate_pattern(800, 600)
    generate_pattern(400, 300; dir="output")
    generate_pattern_bundle(Size2(width=600, height=400); blob_sizes=(12px, nothing), seed=0x1234)

Files Created:
    blob_pattern_<UID>.png  - Pattern image
    blob_pattern_<UID>.json - Metadata and blob coordinates
"""

using BlobBoards
using BlobBoards: PatternBundle, Size2, px
using VisualGeometryCore: ustrip

# =============================================================================
# Simple pattern generation
# =============================================================================

"""
    generate_pattern(width=800, height=600; dir=".")

Generate a blob pattern from pixel dimensions. Uses default configuration.

# Arguments
- `width`: Pattern width in pixels
- `height`: Pattern height in pixels
- `dir`: Output directory
"""
function generate_pattern(width::Int=800, height::Int=600; dir::String=".")
    println("Generating $(width)x$(height) blob pattern...")
    println("  Output directory: $dir")

    pattern, pattern_blobs, config = blob_pattern(width, height;
                                                  cutoff=2.0, dir=dir)

    num_blobs = length(pattern_blobs)
    println("\nGenerated $(width)x$(height) pattern with $num_blobs blobs")
    println("Files saved to: $dir")

    return pattern, pattern_blobs, config
end

# =============================================================================
# Explicit-config pattern generation
# =============================================================================

"""
    generate_pattern_bundle(pattern_size=Size2(width=600, height=400);
                            blob_sizes=(12px, nothing), seed=0x1234, dir=".")

Generate a pattern with explicit configuration control and save as a bundle.

Returns the PatternBundle for further inspection or use.

# Arguments
- `pattern_size`: Pattern dimensions as `Size2`
- `blob_sizes`: `(min, max)` disc diameter range in logical units (e.g., `(12px, nothing)`;
  `nothing` = default max, or a fraction such as `0.2short_side`)
- `seed`: RNG seed for deterministic placement
- `dir`: Output directory
"""
function generate_pattern_bundle(pattern_size::Size2=Size2(width=600, height=400);
                                 blob_sizes=(12px, nothing), seed::Integer=0x1234,
                                 dir::String=".")
    println("Generating pattern bundle...")
    println("  Size: $(pattern_size.width) x $(pattern_size.height)")
    println("  Blob sizes: $blob_sizes")
    println("  Seed: $(repr(seed))")
    println("  Output directory: $dir")

    # Generate pattern and blobs with explicit config
    pattern, pattern_blobs, config = blob_pattern(pattern_size;
                                                  blob_sizes=blob_sizes,
                                                  seed=seed, cutoff=2.0)

    # Create bundle and save pattern + JSON sidecar
    bundle = PatternBundle(config, pattern_blobs)
    BlobBoards.save(pattern, bundle; dir=dir, overwrite=true)

    # Report results
    num_blobs = length(pattern_blobs)
    scales = [b.σ for b in pattern_blobs]
    min_s = minimum(scales)
    max_s = maximum(scales)

    println("\nSaved pattern bundle:")
    println("  Blobs: $num_blobs")
    println("  Scales: $(round(ustrip(min_s), digits=2))-$(round(ustrip(max_s), digits=2)) px")
    println("  Files saved to: $dir")

    return bundle
end

# =============================================================================
# CLI entry point
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    width  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 800
    height = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 600
    generate_pattern(width, height)
end
