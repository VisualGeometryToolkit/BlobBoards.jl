# Public API

```@meta
CurrentModule = BlobBoards
```

This page lists the public surface of BlobBoards.jl. Exported names are
listed bare; `public` (supported, unexported) names are listed qualified and
are reached as `BlobBoards.name`.

## Pattern Generation

```@docs
blob_pattern
short_side
BlobBoards.ShortSideFraction
BlobBoards.calc_blob_scales
```

## Board Generation

```@docs
blob_board
suggest_blob_sizes
BlobBoards.BoardOutput
Pdf
Tiff
Png
BlobBoards.render_board
BlobBoards.rebuild_board
BlobBoards.save_canvas_texture
BlobBoards.board_code
BlobBoards.parse_board_code
```

## Blob and Feature Extraction

```@docs
BlobBoards.blobs
```

## Detection and Pose

`find_boards` is the entry point: it detects blobs, fits elliptical frames,
canonicalizes patches, describes them, matches against a board gallery, and
estimates pose. The camera lives inside `config.geometry` — `PoseGeometry`
for the calibrated P3P path, `HomographyGeometry` for the uncalibrated one —
so a camera/geometry mismatch is unrepresentable rather than a runtime error.

Adaptation is **backend-generic** (KernelAbstractions): `device = KA.CPU()`
runs the threaded host implementation, `CUDABackend()` (or any KA GPU
backend) the device kernels. Results agree to within the pipeline's own
run-to-run jitter; CUDA additionally provides cuBLAS/libdevice fast paths.

```@docs
find_boards
track_boards
FindBoardsConfig
PoseGeometry
HomographyGeometry
RadialHomographyGeometry
BlobBoards.prepare_image_features
load_board
BlobBoards.load_bundle
```

### The staged gallery

```@docs
BlobBoards.GalleryIndex
```

### Pipeline stages

Each stage is usable on its own when you want to inspect or replace part of
the chain.

```@docs
BlobBoards.detect_blobs
DetectConfig
BlobBoards.adapt
BlobBoards.AdaptConfig
BlobBoards.canonicalize
BlobBoards.describe
```

### Descriptors and the board gallery

`BoardGallery` caches per-board features on disk. It stores the descriptor
model's hash, so switching networks invalidates the cache and it rebuilds.

```@docs
DescriptorNetwork
BoardGallery
BlobBoards.annotate_blobs
```

### Non-maximum suppression

```@docs
MxmNms
EllipseNms
```

## Board Geometry

Use `Rect(bundle, reference_frame)` to get bounding rectangles in different coordinate systems:

```julia
using GeometryBasics: Rect

# Pattern coords (y-down, origin at pattern top-left)
rect = Rect(bundle, :pattern)  # Pattern area only
rect = Rect(bundle, :board)    # Full board (includes frame)
rect = Rect(bundle, :canvas)   # Full canvas (includes margins)

# World coords (y-up, centered at origin)
rect = Rect(bundle, :world)    # For 3D visualization
```

The `:world` frame returns a centered rect with unitless mm values, suitable for 3D visualization
where the board origin is at the centroid.

## Coordinate Transforms

BlobBoards extends `CoordinateTransformations` constructors:

- `Translation(bc::BoardConfig, from, to)` — reference-frame translations (`:pattern`/`:board`/`:canvas`)
- `AffineMap(bundle::BoardBundle, from, to; image_origin, density)` — coordinate-system transforms (`:pixel`/`:physical`/`:world`)

For image-origin conventions (`:colmap`/`:julia`/`:opencv`), use
`EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})` (from VisualGeometryToolkitCore;
not exported by BlobBoards, so `using BlobBoards: EuclideanMap, PixelFrame`). It
acts on plain pixel coordinates: `ustrip.(px, centroid(d))`.

## Unit Conversion Helpers

```@docs
px_per_mm
mm_per_px
```

## Training Pipeline

```@docs
BlobBoards.build_tracks_for_sequences
BlobBoards.build_absolute_tracks
BlobBoards.save_tracks
BlobBoards.plot_tracks
```

## File I/O

```@docs
BlobBoards.load
BlobBoards.save
BlobBoards.canonical_config_json
```

### `.tracks` files

`.tracks` is plain HDF5 and is **not** registered with FileIO — the
`format"BlobBoardTracks"` hooks were removed in 2026-08 (the format had in fact
never been registered, so `FileIO.load("x.tracks")` never worked). Write them
with [`build_tracks_for_sequences`](@ref); read them with HDF5.jl, with h5py, or
with the Python helper `blobboards.load_tracks`. The layout is documented in
`examples/training/TRACKS_FORMAT.md`.

## Units

The following units are re-exported for convenience:

| Unit | Description |
|------|-------------|
| `px` | Pixels (logical) |
| `mm` | Millimeters |
| `cm` | Centimeters |
| `m` | Meters |
| `inch` | Inches |
| `ft` | Feet |
| `dpi` | Dots per inch |
| `pt` | Points (1/72 inch) |
| `μm` | Micrometers |

## Types

The main types used in BlobBoards are:

- `Size2` - Width/height dimensions (from VisualGeometryToolkitCore)
- `Disc` - Isotropic blob/feature type (from VisualGeometryToolkitCore)
- `PatternConfig` - Pattern generation configuration
- `BoardConfig` - Board layout configuration
- `PatternBundle` - Pattern with metadata (what `blob_pattern(…; dir)` writes as JSON, and `BlobBoards.load` returns)
- `BoardBundle` - Board with metadata (returned by `blob_board`)

## Media enumeration

```@docs
BlobBoards.walk_media
BlobBoards.media_resolution
BlobBoards.group_by_resolution
BlobBoards.is_video
BlobBoards.example_data
```
