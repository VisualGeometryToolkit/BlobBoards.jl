# Pattern Generation

This guide covers the details of blob pattern generation.

## Pattern Basics

A blob pattern is a collection of Gaussian blobs placed on a white background without overlap. The blobs are distributed across multiple scales (sizes) following a logarithmic distribution.

```julia
using BlobBoards

# Basic pattern generation
pattern, blobs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing))
```

## Scale Distribution

`blob_sizes = (min, max)` sets the range of blob sizes, as the **diameter** of
the rendered disc (`2·cutoff·σ`):

```julia
pattern, blobs, config = blob_pattern(800, 600;
    blob_sizes = (12px, nothing), # Smallest disc 12 px across; default max
    scales_per_octave = 3,        # Rungs per doubling of size
)

# Discs from 12 px to 120 px across
blob_pattern(800, 600; blob_sizes = (12px, 120px))

# A max as a fraction of the pattern's short side
blob_pattern(800, 600; blob_sizes = (12px, 0.2short_side))
```

For `blob_pattern` the sizes are logical (px; a bare number is px), stored as σ
in px (`config.min_scale`, `config.max_scale`). For `blob_board` they are
physical (mm, inch), stored as σ in points (a board is laid out on a physical
grid; see [Authoring a Board](boards.md)).
A pattern's default max (`nothing`) is a disc 20% of the short side across; its
default min, `2·cutoff·2px` (8 px at the default cutoff), is also its floor
(σ ≥ 2 px). A board's `blob_sizes` has no default, and neither end may be
`nothing`.

### Understanding Scales

A blob is a Gaussian of standard deviation σ, rendered as a disc of radius
`cutoff · σ`; `blob_sizes` give that disc's diameter, `2 · cutoff · σ`, and the
config stores σ.

`cutoff` is the Gaussian truncation radius in units of σ. **Use `cutoff = 2`**:
the default (`DEFAULT_CUTOFF`), the cutoff the shipped boards were printed with,
and the radius the detector's NMS and the adaptation default to. Leave it
unset rather than spelling another value; a wider cutoff only spends board area
on the Gaussian's near-white tail.

### The Scale Ladder

The blob scales form a geometric ladder from the min to the max, **both exact**:
```
k  = round(scales_per_octave · log2(max / min))      # number of steps
σᵢ = min · (max / min)^(i / k),   i = 0 … k
```
so the steps are evenly spaced in log at about `1/scales_per_octave` octave. When
the max lies exactly on the `2^(1/scales_per_octave)` ladder the steps are exactly
`min · 2^(i / scales_per_octave)` (every v1 board is such a case). When
`max = nothing`, the default caps the largest disc DIAMETER at 20% of the
pattern's short side, i.e. a σ of `0.2 · min(width, height) / (2 · cutoff)`. The
ladder itself is `calc_blob_scales(min_scale, max_scale; scales_per_octave = 3)` (unitless px).

## Placement Algorithm

Blobs are placed using rejection sampling, collisions checked in per-level uniform grids (cell 2·cutoff·σ):

1. Larger scales are placed first (greedy packing)
2. Each blob attempts random placement up to `max_trials` times
3. Collision is checked using `cutoff` radius
4. If `raster_aligned=true`, centers snap to pixel grid

```julia
pattern, blobs, config = blob_pattern(800, 600;
    blob_sizes = (12px, nothing),
    max_trials = 10000,       # Attempts per blob
    α = 1.2,                  # Scale sampling bias (higher = more small blobs)
    cutoff = 2.0,             # Gaussian truncation radius in σ (default; recommended)
    raster_aligned = true,    # Snap centers to pixel centers (default)
)
```

## Rendering

Blobs are rendered directly at target resolution: the per-blob normalization
pins the truncated Gaussian's boundary to background level (a continuous
profile), and with `raster_aligned = true` (the default) every center sits on
a pixel center, so point sampling is symmetric per blob — no oversampling
step is needed or supported.

## Pixel Types

Control the output pixel format with the leading type argument — the pixel type
is positional, as types are in Julia; `pixel_type = …` as a keyword is an
`ArgumentError` (a `PatternConfig` still records it as a field):

```julia
using Colors: Gray
using FixedPointNumbers: N0f8, N0f16

# 8-bit grayscale (default, matches PDF limitations)
pattern, blobs, config = blob_pattern(Gray{N0f8}, Size2(width=800, height=600);
    blob_sizes = (12px, nothing)
)

# 16-bit grayscale (higher precision)
pattern, blobs, config = blob_pattern(Gray{N0f16}, Size2(width=800, height=600);
    blob_sizes = (12px, nothing)
)
```

## Reproducibility

Patterns are fully deterministic when seeded:

```julia
# Same seed = identical pattern
p1, b1, _ = blob_pattern(800, 600; blob_sizes=(12px, nothing), seed=0x1234)
p2, b2, _ = blob_pattern(800, 600; blob_sizes=(12px, nothing), seed=0x1234)
# p1 == p2 and b1 == b2

# From saved configuration
bundle, image = BlobBoards.load("blob_pattern.json")
config = pattern_config(bundle)
# Regenerate identical pattern from config
```

## Blob Polarity

By default, blobs are dark on a light background. Use the `polarity` parameter for light blobs on dark:

```julia
# Dark blobs on white background (default)
pattern, blobs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing), polarity=:dark)

# Light blobs on dark background
pattern, blobs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing), polarity=:light)
```

## Blob Coordinates

Blobs are stored in COLMAP convention internally (first pixel center at 0.5, 0.5):

```julia
bundle, _ = BlobBoards.load("blob_pattern.json")

# Get blobs in COLMAP convention
b = blobs(bundle)

# Convert to other conventions: a map on plain pixel coordinates
using BlobBoards: EuclideanMap, PixelFrame, centroid, ustrip     # not exported
centres_opencv = [EuclideanMap(PixelFrame{:colmap}, PixelFrame{:opencv})(ustrip.(px, centroid(d))) for d in b]
centres_julia = [EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})(ustrip.(px, centroid(d))) for d in b]
```

## Configuration Object

The `PatternConfig` captures all generation parameters:

```julia
config = PatternConfig(
    pattern_resolution = Size2(width=800px, height=600px),
    min_scale = 3.0px,          # stored σ, px
    max_scale = nothing,        # stored σ, px; nothing = default rule
    seed = 0x1234,
    pixel_type = Gray{N0f8},
    scales_per_octave = 3,
    max_trials = 5000,
    α = 1.2,
    cutoff = 2.0,
    raster_aligned = true,
    polarity = :dark,
)

pattern, blobs = blob_pattern(config)
```

## Saving Patterns

```julia
# Save during generation
pattern, blobs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing), dir="./output")

# Or save explicitly
bundle = PatternBundle(config, blobs)
BlobBoards.save(pattern, bundle; dir="./output")
```

Output files:
- `blob_pattern_XXX_XXXXX.png` - Grayscale pattern image
- `blob_pattern_XXX_XXXXX.json` - Full configuration and blob list
