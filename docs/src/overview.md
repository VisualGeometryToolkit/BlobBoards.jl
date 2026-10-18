# Overview

Generate deterministic multi-scale blob patterns and printable calibration boards for computer vision.

## Overview

BlobBoards.jl creates calibration targets with blob patterns that are:

- **Deterministic**: Bit-exact reproducible via seeded configuration
- **Multi-scale**: Logarithmically distributed blob sizes for scale-invariant detection
- **Print-ready**: PDF output with rulers, QR codes, and metadata
- **Configurable**: Physical units (mm, inch) with run-time unit checking

## Quick Start

```julia
using BlobBoards      # re-exports the units: mm, inch, px, dpi, …

# Generate a printable A4 calibration board
bundle = blob_board("A4"; blob_sizes=(4.0mm, 0.2short_side), seed=0x1234, dir="./output")

# Generate a pattern for training/simulation (core package only)
pattern, discs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing), seed=0x1234)

# Extract blobs from saved files
bundle, _ = BlobBoards.load("blob_board.json")
b = blobs(bundle, :pattern; units=mm)
```

## Key Concepts

### Patterns vs Boards

- **Pattern** (`blob_pattern`): A blob pattern image in pixel/logical units. Used for training, simulation, or embedding in custom layouts.
- **Board** (`blob_board`): A complete calibration board with borders, rulers, QR code, and UID markers. Ready for printing.

### Unit System

Units are checked at run time via [FlexUnits.jl](https://github.com/Deduction42/FlexUnits.jl)
with VisualGeometryToolkitCore's pixel dimension (quantities are `VgcQuantity`).
BlobBoards re-exports `mm`, `cm`, `inch`, `px`, `dpi`, and friends, so
`using BlobBoards` is all you need. Strip a quantity in a named unit,
`ustrip(mm, q)`: lengths are held in metres.

| Unit Type | Examples | Description |
|-----------|----------|-------------|
| Logical | `px` | Pixels (dimensionless) |
| Physical | `mm`, `inch`, `cm` | Real-world lengths |
| Density | `dpi` | Dots per inch |

**Critical distinction**:
- `blob_pattern()`: `blob_sizes` are disc diameters in **logical units** (px, e.g. `(12px, nothing)`)
- `blob_board()`: `blob_sizes` are printed disc diameters in **physical units** (mm, inch, e.g. `(4.0mm, 38mm)`);
  required, both ends: `suggest_blob_sizes(600dpi, "A4")` gives the smallest disc that prints well at a printer's
  density and the largest for the board (20% of its shorter side, at most 64 mm)
- A pattern stores σ in px, a board σ in points — a board is laid out on a fixed
  1/7200-inch grid and has no pixel density (`min_scale`/`max_scale`;
  a disc's diameter is `2·cutoff·σ`). A density belongs to a raster output
  only: `output = Tiff(600dpi)` or `Png(300dpi)`

### Coordinate Specifications

BlobBoards uses three types of coordinate specifications:

| Type | Values | Purpose |
|------|--------|---------|
| **Image Origin** | `:colmap`, `:opencv`, `:julia` | Pixel coordinate convention |
| **Reference Frame** | `:pattern`, `:board`, `:canvas` | Spatial origin on boards |
| **Coordinate System** | `:pixel`, `:physical`, `:world` | Unit system for transforms |

See the [Coordinates Guide](guide/coordinates.md) for detailed documentation.

## Features

**Authoring**
- Deterministic patterns from seeded configurations
- Physical units via FlexUnits (mm, inch, dpi, etc.)
- Printable boards (PDF) or training images (PNG) with ruler or blank frames
- JSON serialization for exact reproduction
- Blob extraction with multiple coordinate conventions
- World coordinates for 3D calibration workflows

**Detection and pose**
- [`find_boards`](@ref) — detect blobs, fit elliptical frames, canonicalize,
  describe, match against a board gallery, and estimate pose (P3P or homography)
- Many boards located per image, each accepted on spatial consistency rather
  than quad segmentation, so partial occlusion is survivable
- Backend-generic hot path (KernelAbstractions): CPU, CUDA, or any KA backend
- [`track_boards`](@ref) for temporally warm-started video tracking

**Tooling**
- Training pipeline: detect, track, and extract canonicalized patches from video
- `.tracks` files are plain HDF5
- Visualization helpers for Makie backends

## Installation

Julia 1.12+ is the only requirement:
```bash
git clone https://github.com/VisualGeometryToolkit/BlobBoards.jl
cd BlobBoards.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

See the [Getting Started](guide/getting-started.md) guide for detailed instructions.

## Contents

```@contents
Pages = [
    "guide/reference.md",
    "guide/getting-started.md",
    "guide/patterns.md",
    "guide/boards.md",
    "guide/coordinates.md",
    "guide/detection.md",
    "guide/board-tracks.md",
    "guide/python.md",
    "api/public.md",
    "api/types.md",
    "api/patterns.md",
    "api/boards.md",
    "api/io.md",
]
Depth = 2
```

## Index

```@index
```

## Role in the VisualGeometryToolkit stack

BlobBoards provides **fiducial blob-board targets** for the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack:
deterministic pattern generation, printable boards, detection, and
calibration-grade correspondence ground truth. It consumes
[VisualGeometryToolkitCore.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore)
(geometry, units, blob detection) and
[VisualGeometryToolkitEstimation.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitEstimation)
(robust correspondence/pose estimation).

All toolkit documentation is aggregated at the portal:
<https://visualgeometrytoolkit.github.io>.
