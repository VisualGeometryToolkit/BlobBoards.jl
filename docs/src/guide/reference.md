# Quick Reference

This page provides a comprehensive overview of the main entry points and all their parameters.

## Entry Points

BlobBoards has two main entry points:

| Function | Purpose | Output |
|----------|---------|--------|
| `blob_pattern` | Generate blob patterns for training/simulation | `(pattern, blobs, config)` |
| `blob_board` | Generate printable calibration boards | `BoardBundle` |

## `blob_pattern` - Pattern Generation

Generate multi-scale blob patterns in pixel/logical units.

### Signatures

```julia
# From dimensions
blob_pattern(width, height; kwargs...)
blob_pattern(Size2(width=W, height=H); kwargs...)

# With explicit pixel type
blob_pattern(Gray{N0f8}, Size2(width=W, height=H); kwargs...)

# From config object
blob_pattern(config::PatternConfig; dir=nothing)
```

### All Keyword Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `blob_sizes` | Tuple | `(8px, nothing)` | `(min, max)` disc **diameters** (2·cutoff·σ) in **logical units** (px; a bare number is px). `max = nothing` → largest disc = 20% of the short side; `0.2short_side` → a fraction of the short side. Default min = 2·cutoff·2px, also the floor |
| `seed` | Integer | random | Random seed for reproducibility (UInt64) |
| `scales_per_octave` | Int | `3` | Number of scales per doubling of size |
| `α` | Float64 | `1.2` | Scale sampling bias (higher = more small blobs) |
| `cutoff` | Float64 | `2.0` | Gaussian truncation radius in σ — use the default 2 |
| `max_trials` | Int | `5000` | Max placement attempts per blob |
| `raster_aligned` | Bool | `true` | Snap blob centers to pixel centers |
| `polarity` | Symbol | `:dark` | Blob polarity (`:dark` or `:light`) |
| `dir` | String | `nothing` | Output directory (saves PNG + JSON if provided) |

### Stored Configuration (`PatternConfig`)

The returned `PatternConfig` (and the `pattern_config` block of a v2 JSON file)
stores the blob sizes as σ in logical units (a disc's diameter is `2·cutoff·σ`):

| Field | Type | Description |
|-------|------|-------------|
| `min_scale` | LogicalCount | Smallest σ (px) — the bottom rung of the scale ladder |
| `max_scale` | LogicalCount | Largest σ (px) — the top of the scale ladder, exactly as requested |

`PatternConfig(; pattern_resolution, min_scale, max_scale, …)` takes these
storage-level σ values directly (`max_scale = nothing` applies the default rule).

### Examples

```julia
using BlobBoards
using VisualGeometryCore: px

# Basic usage
pattern, blobs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing), seed=0x1234)

# Denser scale sampling
pattern, blobs, config = blob_pattern(800, 600;
    blob_sizes = (12px, nothing),
    scales_per_octave = 4,
    seed = 0x1234
)

# Save to disk
pattern, blobs, config = blob_pattern(800, 600;
    blob_sizes = (12px, nothing),
    dir = "./output"
)
```

## `blob_board` - Board Generation

Generate printable calibration boards with borders, rulers, and metadata.

### Signatures

```julia
# From paper size name (blob_sizes is required)
blob_board(paper_size::String; blob_sizes, kwargs...)

# From explicit dimensions
blob_board(width, height; blob_sizes, kwargs...)
blob_board(Size2(width=W, height=H); blob_sizes, kwargs...)

# From JSON (reconstruction, as printed: `output` and `id_label` default to
# the JSON's `render` block; a keyword overrides it)
BlobBoards.rebuild_board(json_or_path; dir=nothing, output, id_label)
```

### Keyword Arguments

Every keyword — output (`dir`, `output`, `id_label`),
pattern (`blob_sizes`, `seed`, `polarity`, …), board and frame (`margin` or `board_size`,
`corner_size`, `ruler_width`, `frame_padding`) and ruler ticks — with its default
and valid values is listed in the [`blob_board`](@ref) reference.

A board has no pixel density. `blob_sizes = (min, max)` is required:
[`suggest_blob_sizes`](@ref)`(density, extent)` gives the smallest disc that
prints well at a printer's density (3 mm at 1200 dpi, 6 mm at 600, 12 mm at
300) and the largest for the board (20% of its shorter side, at most 64 mm).
The max must be given — a length, or `0.2short_side` (that fraction of
the pattern's short side, no cap); `nothing` is an `ArgumentError`.

### Output Types

| `output` | Written as | Options |
|----------|------------|---------|
| `Pdf()` (default) | the PDF/X-4 print master; no density | `shading_function = :calculator` (or `:sampled`), `icc_profile = :linear` (or a grey ICC file) |
| `Tiff(density)` | the sheet converted at `density`, 8-bit grey TIFF | `icc_profile = :linear` |
| `Png(density)` | the sheet converted at `density`, grey PNG | — |

A raster's `density` is a whole number of pixels per inch, at which the
board's smallest blob must be at least 2 pixels per σ.

### Examples

```julia
using BlobBoards      # re-exports the units: mm, inch, px, dpi, …

# Basic A4 board, sized for a 600 dpi printer
bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), dir="./output")

# Custom size with parameters
bundle = blob_board(200mm, 280mm;
    blob_sizes = (3.2mm, 0.2short_side),
    ruler_width = 25mm,
    seed = 0x1234,
    dir = "./output"
)

# Corner cells only (QR, uid), with padding; blobs fill the rest
bundle = blob_board("A4";
    blob_sizes = (4.0mm, 0.2short_side),
    ruler_width = 0mm,
    frame_padding = 2mm,
    dir = "./output"
)

# No marks at all: blobs fill the whole board
bundle = blob_board("A4";
    blob_sizes = (4.0mm, 0.2short_side),
    corner_size = 0mm, ruler_width = 0mm,
    dir = "./output"
)

# A3 print master with an explicit ICC profile
bundle = blob_board("A3";
    blob_sizes = suggest_blob_sizes(1200dpi, "A3"),
    output = Pdf(icc_profile = :linear),
    dir = "./output"
)

# PNG for digital use, 300 pixels per inch
bundle = blob_board("A4";
    blob_sizes = (12mm, 0.2short_side),
    output = Png(300dpi),
    dir = "./output"
)
```

## Available Paper Sizes

| Category | Sizes |
|----------|-------|
| ISO A-series | `A0`, `A1`, `A2`, `A3`, `A4`, `A5` |
| ANSI | `ANSI A`, `ANSI B`, `ANSI C`, `ANSI D`, `ANSI E`, `ANSI E1` |
| Common | `Letter`, `US Letter`, `Legal`, `Tabloid`, `Ledger` |

## Blob and Saddle Point Extraction

### `blobs` - Unified Extraction

Extract blobs with specified reference frame and units:

```julia
bundle, image = load("blob_board.json")

# Pixel units (default) in pattern frame, at the density you name (a v3 board's own by default)
b = blobs(bundle, :pattern; density = 300dpi)

# Physical units
b = blobs(bundle, :pattern; units=mm)
b = blobs(bundle, :canvas; units=inch)

# From PatternBundle (no reference frame needed)
b = blobs(pattern_bundle)
```

### World Coordinates (Boards Only)

Extract blobs in 3D calibration coordinates:

```julia
b = blobs(bundle, :world)
# Returns: Vector{Disc} with centered, y-up coordinates in unitless mm
```

## File I/O

### Loading

```julia
bundle, image = load("blob_board.json")   # From JSON
bundle, image = load("blob_board.pdf")    # From PDF
bundle, image = load("blob_pattern.png")  # From PNG
```

### Saving

```julia
# Boards: written by blob_board when dir is given, or re-rendered from a stored bundle
bundle = blob_board("A4"; blob_sizes=suggest_blob_sizes(600dpi, "A4"), dir="./output")   # Pdf() by default
BlobBoards.render_board(bundle; dir="./output")                       # Pdf() by default
BlobBoards.render_board(bundle; dir="./output", output=Png(300dpi))

# Patterns (automatic during blob_pattern if dir provided)
BlobBoards.save(pattern, bundle; dir="./output")
```

## Training Pipeline

### `build_tracks_for_sequences` - Track Generation

Detect the boards in every media item under a directory and accumulate their
patch tracks into one `.tracks` file:

```julia
out = build_tracks_for_sequences(board_dir, model_path, video_dir;
    n_frames = 100,            # or stride=5
    output_dir = "./output",   # writes <output_name>_<patch_type>_s<scale>.tracks
    iou_threshold = 0.6,
    patch_type = :logpolar,
    scale = 32,
    blur_sigma_max = 0,        # augmentations; 0 disables
)
```

See [Board Training Tracks](board-tracks.md) for the full guide. The older
`build_tracks`, which chained correspondences from relative `.npy` homographies,
was removed in 2026-08.

### Loading Tracks

`.tracks` is plain HDF5 (see `examples/training/TRACKS_FORMAT.md`); read it with
HDF5.jl, h5py, or the Python helper `blobboards.load_tracks`.

## Critical Unit Distinction

**Pattern functions** take `blob_sizes` in **logical units** (px):
```julia
blob_pattern(800, 600; blob_sizes=(12px, nothing))  # px = pixels
```

**Board functions** take `blob_sizes` in **physical units** (mm, inch):
```julia
blob_board("A4"; blob_sizes=(4.0mm, 0.2short_side))  # mm = millimeters
```

A pattern stores σ in px, a board σ in points (`min_scale`/`max_scale`): a board is laid out on a fixed 1/7200-inch grid and has no pixel density.

## The Board JSON

Every board is saved as a print file plus a JSON sidecar, `blob_board_<uid>_<hash>.json`:

```
{
  "schema": "blobboards/board/v4",
  "preamble": {
    "board_config":   { canvas_size, margin, layout_grid: { value: 0.01, unit: "pt" },
                        corner_size, ruler_width, frame_padding,
                        ruler_spec: { major_tick: { spacing, width },
                                      minor_tick: { spacing, width } } },
    "pattern_config": { min_scale, max_scale,   (pt)
                        seed, scales_per_octave, max_trials,
                        alpha, cutoff, polarity }   (no pattern size: it is the board's;
                                                     no pixel_type or raster_aligned: a v4 board has neither)
  },
  "blobs":         { "unit": "pt", "centers": […], "sigmas": […] },
  "hashes":        { "config": "<sha256 of the schema, the canonical preamble and the blobs>" },
  "render":        { "format": "pdf", "icc_profile": "linear",
                     "id_label": true, "shading_function": "calculator",
                     "density": null }
}
```

For a raster, `render` records `"format": "tiff"` or `"png"` and
`"density": { "value": 600, "unit": "dpi" }` (a PNG's `icc_profile` is `null`).
A custom ICC profile is recorded with its bytes — `{ "name": "press.icc",
"sha256": …, "data": "<base64>" }` — so a rebuild needs no original file.

Lengths are `{ "value": …, "unit": "mm" | "pt" | "dpi" }`; a tick's `width` is
its full drawn length across the frame band. Blob centres are 0.01 pt grid
values, the very numbers the print master holds.

- **Identity** is `hashes.config`, the SHA-256 of the canonical `preamble`; the
  file name and the printed uid are its first characters. It is stored, never
  recomputed on load.
- **Blob sizes** are stored only as σ in points (`min_scale`, `max_scale`);
  `blob_sizes` and its units are an input convenience and are not recorded, so
  every spelling of one board mints one identity. A board has no density; a
  raster output's lives in `render`, outside the identity.
- **`render`** records how the print file was made. It is not part of the
  identity: the same board with or without its id label, as a PDF or a raster,
  has one hash.
  `rebuild_board` and the gallery's regeneration render as recorded; a file
  without the block renders as PDF/X-4 with no label.
- **Older schemas** still load. v1 files (with `max_diameter_fraction`) have the
  fraction converted to the largest σ it produced, so v1 boards re-place
  bit-identically. v1 and v2 rulers carry a `ruler_spec.width` and half-length
  tick widths; the reader drops the one and doubles the others, which is the
  ruler those boards were drawn with. Their stored `hashes.config` stays their
  identity. v3 files (pixels at `board_config.print_density`, blobs in px) keep
  that layout: they load, re-render and rebuild on their own pixel grid exactly
  as printed, and re-save as v3.
