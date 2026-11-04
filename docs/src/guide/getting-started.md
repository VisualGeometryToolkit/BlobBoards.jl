# Getting Started

This guide walks through the basics of generating blob patterns and calibration boards.

## Installation

Julia 1.12+ is the only requirement; BlobBoards needs no external tools and no
pixi (pixi is only for the [Python interface](python.md)). The examples below
also import a few names from VisualGeometryToolkitCore, the stack package that
owns them:

```julia-repl
pkg> add BlobBoards VisualGeometryToolkitCore
```

Until the General registrations of BlobBoards and LocalOptimizationDoneRight
merge, add [CauRegistry](https://github.com/prittjam/CauRegistry.jl) first; the
rest of the stack comes from General:

```julia-repl
pkg> registry add https://github.com/prittjam/CauRegistry.jl
```

To work on a checkout instead:

```bash
git clone https://github.com/VisualGeometryToolkit/BlobBoards.jl
cd BlobBoards.jl
julia --project=. -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl"); Pkg.instantiate()'
```

The `julia` blocks below run as written, in order, in a fresh environment with
the install line above; the package's test suite runs them. Every file goes to
a temporary directory.

## Author your first board

```julia
using BlobBoards
dir = mktempdir()
bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 0xBEEF, dir)
```

This writes `blob_board_<uid>.pdf` (print it at 100%) and its JSON sidecar.
`blob_sizes` is required: `suggest_blob_sizes(600dpi, "A4")` gives the
smallest disc that prints well on a 600 dpi printer (6 mm across) and the
largest for the default A4 board (190 mm wide inside its margins): 20% of the
board's shorter side, at most 64 mm. The board itself has no pixel density.
The [Authoring a Board](boards.md) guide covers the environment, sizes, frames
and rulers, output types and printing; [`blob_board`](@ref) lists every
keyword.

## Detect boards and estimate pose

[`find_boards`](@ref) finds known boards in an image and returns each one's
homography or pose. See [Detection and Pose](detection.md).

## Generating Patterns Only

For training data or custom layouts, generate just the pattern:

```julia
using BlobBoards

# Generate a 800x600 pixel pattern
pattern, pattern_blobs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing), seed=0x1234)

# Save to disk
pattern_dir = mktempdir()
pattern, pattern_blobs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing), dir=pattern_dir)
```

## Extracting Blob Coordinates

### From Board Files

```julia
board_json = only(filter(endswith(".json"), readdir(dir; join = true)))   # the board authored above
bundle, image = BlobBoards.load(board_json)

# Physical coordinates (mm) relative to pattern origin
b = blobs(bundle, :pattern; units=mm)

# Physical coordinates relative to board origin (includes frame)
b = blobs(bundle, :board; units=mm)

# Pixel coordinates at a density (COLMAP convention, convert as needed)
b = blobs(bundle, :pattern; density = 300dpi)
```

### From Pattern Files

```julia
pattern_json = only(filter(endswith(".json"), readdir(pattern_dir; join = true)))
bundle, _ = BlobBoards.load(pattern_json)

# Pixel coordinates (COLMAP convention)
b = blobs(bundle)
```

### World Coordinates for 3D Calibration

```julia
bundle, image = BlobBoards.load(board_json)

# Get blobs in world frame (centered, y-up, unitless mm)
world = blobs(bundle, :world)

# Lift to 3D points on z=0 plane
using VisualGeometryToolkitCore: Point3
X = [Point3(b.center..., 0.0) for b in world]
```

## Loading Saved Boards

```julia
# Load from any paired file (JSON, PNG, or PDF)
board_pdf = only(filter(endswith(".pdf"), readdir(dir; join = true)))
bundle, image = BlobBoards.load(board_json)
bundle, image = BlobBoards.load(board_pdf)

# Access configuration
config = pattern_config(bundle)
board_cfg = board_config(bundle)
```

## Available Paper Sizes

Standard sizes include:

| ISO | ANSI | Common Names |
|-----|------|--------------|
| A0-A5 | ANSI A-E1 | Letter, Legal |
| | | Tabloid, Ledger |

```julia
# List all available sizes
keys(BlobBoards.PRINT_SIZES)
```

## Next Steps

- [Patterns Guide](patterns.md) - Detailed pattern generation options
- [Authoring a Board](boards.md) - Board layout, output types and printing
- [Detection and Pose](detection.md) - `find_boards`, `track_boards` and the GPU environment
- [Coordinates Guide](coordinates.md) - Coordinate system details
- [API Reference](../api/public.md) - Complete function reference
