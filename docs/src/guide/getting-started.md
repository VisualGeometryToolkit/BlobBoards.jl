# Getting Started

This guide walks through the basics of generating blob patterns and calibration boards.

## Installation

Julia 1.12+ is the only requirement; board authoring needs no external tools and no
pixi (pixi is only for the [Python interface](python.md)). Everything comes
from General.

Every example on this page authors boards and needs BlobBoards alone:

```julia-repl
pkg> add BlobBoards
```

Finding boards, on the CPU or a GPU, tracking, plotting, training and the
Python package each have their own install line: see
[Install for your task](install.md). Each `julia` block below is a complete
script that runs as written in a fresh environment with that line, and writes
its files to a temporary directory; the package's test suite runs each of them.

## Author your first board

```julia
# Install row: author boards
using BlobBoards
dir = mktempdir()
bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 0xBEEF, dir)
count(endswith(".pdf"), readdir(dir)) == 1    # true
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
homography or pose: [Find boards on the CPU](install.md#Find-boards-on-the-CPU)
is a complete example, and [Detection and Pose](detection.md) the guide.

## Generating Patterns Only

For training data or custom layouts, generate just the pattern:

```julia
# Install row: author boards
using BlobBoards

# An 800 × 600 pixel pattern, in memory
pattern, pattern_blobs, config = blob_pattern(800, 600; blob_sizes = (12px, nothing), seed = 0x1234)

# The same, saved to disk: the PNG and its JSON sidecar
pattern_dir = mktempdir()
pattern, pattern_blobs, config = blob_pattern(800, 600; blob_sizes = (12px, nothing), seed = 0x1234, dir = pattern_dir)
size(pattern) == (600, 800)    # true
```

## Extracting Blob Coordinates

### From Board Files

```julia
# Install row: author boards
using BlobBoards
dir = mktempdir()
blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 0xBEEF, dir)
board_json = only(filter(endswith(".json"), readdir(dir; join = true)))
bundle, image = BlobBoards.load(board_json)

# Physical coordinates (mm) relative to pattern origin
b = blobs(bundle, :pattern; units = mm)

# Physical coordinates relative to board origin (includes frame)
b = blobs(bundle, :board; units = mm)

# Pixel coordinates at a density (COLMAP convention, convert as needed)
b = blobs(bundle, :pattern; density = 300dpi)

# The world frame (centred, y-up, unitless mm), lifted to 3D points on the
# z = 0 plane for calibration
world = blobs(bundle, :world)
X = [(x, y, 0.0) for (x, y) in (b.center for b in world)]
length(X) == length(b)    # true
```

### From Pattern Files

```julia
# Install row: author boards
using BlobBoards
pattern_dir = mktempdir()
blob_pattern(800, 600; blob_sizes = (12px, nothing), seed = 0x1234, dir = pattern_dir)
pattern_json = only(filter(endswith(".json"), readdir(pattern_dir; join = true)))
bundle, _ = BlobBoards.load(pattern_json)

# Pixel coordinates (COLMAP convention)
b = blobs(bundle)
!isempty(b)    # true
```

## Loading Saved Boards

```julia
# Install row: author boards
using BlobBoards
dir = mktempdir()
blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 0xBEEF, dir)

# Load from any paired file (JSON, PNG, or PDF)
board_json = only(filter(endswith(".json"), readdir(dir; join = true)))
board_pdf = only(filter(endswith(".pdf"), readdir(dir; join = true)))
bundle, image = BlobBoards.load(board_json)
bundle, image = BlobBoards.load(board_pdf)

# Access configuration
config = pattern_config(bundle)
board_cfg = board_config(bundle)
config.seed == 0xBEEF    # true
```

## Available Paper Sizes

Standard sizes include:

| ISO | ANSI | Common Names |
|-----|------|--------------|
| A0-A5 | ANSI A-E1 | Letter, Legal |
| | | Tabloid, Ledger |

```julia
# Install row: author boards
using BlobBoards
# List all available sizes
"A4" in keys(BlobBoards.PRINT_SIZES)    # true
```

## Next Steps

- [Install for your task](install.md) - The install line and an example for each task
- [Patterns Guide](patterns.md) - Detailed pattern generation options
- [Authoring a Board](boards.md) - Board layout, output types and printing
- [Detection and Pose](detection.md) - `find_boards`, `track_boards` and the GPU environment
- [Coordinates Guide](coordinates.md) - Coordinate system details
- [API Reference](../api/public.md) - Complete function reference
