# BlobBoards.jl

[![CI](https://github.com/VisualGeometryToolkit/BlobBoards.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VisualGeometryToolkit/BlobBoards.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-blobboards.github.io-blue.svg)](https://blobboards.github.io)

## Abstract

**[BlobBoards: Robust Markers for Accurate Pose](https://arxiv.org/abs/2608.28830)** — James Pritts, Till Sittart, Hendrik Sauer, Silja Janßen, Felix Seegräber, David Nakath and Kevin Köser. *BMVC 2026*. [Project page](https://blobboards.github.io).

We propose BlobBoards, a fiducial marker system comprising a dense, multi-scale
field of Gaussian blobs and a feature-based pipeline for joint detection,
identification, and pose estimation. Each board is registered from hundreds of
blob features whose dense spatial coverage constrains pose, while multiple
scales preserve detectability across large changes in focal length, distance,
and obliquity. Learned local descriptors are matched to the reference pattern
and spatially verified, so the correspondences determine pose and certify
identity. Against motion-capture ground truth, BlobBoards achieve median
translation errors of 3.6-5.0 mm, reducing AprilTag's median translation error
by 89% on small boards and 70% on large ones. They also produce far fewer
large-rotation failures than state-of-the-art tag systems. BlobBoards achieve
the highest detection rate, 80% versus 74% for AprilTag and 58% for ArUco, with
the largest margin on the smallest markers. Under 50% occlusion, they still
detect 69% of boards with essentially unchanged median translation error, while
AprilTag and ArUco detect none. In experiments BlobBoards give state-of-the-art
detection rate, pose accuracy and occlusion robustness.

## Overview

Generate deterministic, multi-scale blob patterns and printable calibration boards for computer vision tasks (feature evaluation, camera calibration). Patterns are rendered pixel-perfect from seeded configurations; boards include rulers, QR metadata, and a compact UID. All outputs are reproducible with JSON sidecars.

## Features

**Authoring**
- Deterministic patterns from seeded `PatternConfig`
- Physical and logical units via FlexUnits (`mm`, `inch`, `px`, `dpi`), re-exported from VisualGeometryToolkitCore
- Printable boards with rulers, QR code metadata, and UID
- JSON serialization for round-trip equivalence

**Detection and pose**
- `find_boards` — the detection-to-pose pipeline: detect blobs, fit elliptical
  frames, canonicalize patches, describe, match against a board gallery, and
  estimate pose by P3P or homography
- Many boards located per image, each verified by spatial consistency *and*
  by the appearance of its own inliers, rather than by quad segmentation, so
  a board survives partial occlusion without accepting look-alikes
- GPU-resident hot path (KernelAbstractions detection, CUDA adapt and
  canonicalize, ONNX descriptors)
- Temporal tracking across frames via `track_boards`

**Training and tooling**
- Training pipeline: detect, track, and extract canonicalized patches from video
- `.tracks` training files (plain HDF5)
- Makie visualization (any backend); pipeline-stage capture for offline inspection

## Quick Start

Patterns and boards both come from the core package:

```julia
julia --project=.
using Pkg; Pkg.instantiate()

using BlobBoards      # re-exports the units: mm, inch, px, dpi, …

# Generate pattern (pixels)
blob_pattern(800, 600; blob_sizes=(12px, nothing), seed=0x1234, dir="./out")

# Generate a PDF calibration board, its smallest disc sized for a 600 dpi printer
blob_board("A4"; blob_sizes=suggest_blob_sizes(600dpi, "A4"), seed=0xBEEF, dir="./out")

# Generate a PNG board at 300 pixels per inch (for training data)
blob_board("A4"; blob_sizes=(4.0mm, 0.2short_side), seed=0xBEEF, dir="./out", output=Png(300dpi))
```

A board has no pixel density: the PDF prints on any printer. `blob_sizes` is
required, and a density belongs only to a raster output (`Tiff(600dpi)`,
`Png(300dpi)`).

The units (`mm`, `inch`, `px`, `dpi`) are
FlexUnits units (VisualGeometryToolkitCore's registry, with a pixel dimension)
re-exported by BlobBoards; do not import them from another units package.

## Blob Extraction

```julia
# Load board and extract blobs
bundle, img = BlobBoards.load("board.json")

# Get blobs in pixel units (default), at the density of the image they are in
# (a pixel-grid board's own by default; a point-grid board's layout has no density)
blobs_px = blobs(bundle, :pattern; density = 300dpi)   # Pattern coordinate frame
blobs_px = blobs(bundle, :board; density = 300dpi)     # Board coordinate frame (the same as :pattern on a board made today)

# Get blobs in physical units
blobs_mm = blobs(bundle, :pattern; units=mm)    # Physical coordinates in mm

# Convert between image origin conventions: a map on plain pixel coordinates
using BlobBoards: EuclideanMap, PixelFrame, centroid, ustrip
to_julia = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
centres_julia = [to_julia(ustrip.(px, centroid(d))) for d in blobs_px]
```

Image origin conventions:
- `:julia`: pixel center at (1, 1) - Julia 1-based indexing
- `:colmap`: pixel center at (0.5, 0.5) - COLMAP convention
- `:opencv`: pixel center at (0, 0) - OpenCV convention

## Detection and Pose

`find_boards` is the detection entry point. It runs the whole chain —
undistort, detect blobs, fit elliptical frames (`adapt`), canonicalize
patches, describe, match against the gallery, estimate the pose, gate each
candidate on inlier count *and* on whether those inliers look like the board
they claim, then spatially arbitrate between competing boards — and returns
every board it located in the image.

```julia
using BlobBoards, CUDA
import cuDNN                                   # enables the ONNX CUDA provider
using CUDA: CUDABackend
using VisualGeometryToolkitEstimation: RansacConfig
using VisualGeometryToolkitCore: CameraModel
import VisualGeometryToolkitIO as VgtkIO

camera  = VgtkIO.load(CameraModel, "cameras.json")  # canonical vgio/cameras/v3 store
network = DescriptorNetwork("descriptor.onnx") # patch size comes from the graph

# The gallery caches per-board features. Keyed on the network hash, the
# network-declared patch size, the sampling extent and each board's own
# cutoff — so a wrong-network entry can never be returned.
gallery = BoardGallery("board_features.jld2", network)
boards  = Dict(uid => feats for (uid, feats) in
               [insert!(gallery, path, network) for path in board_jsons])
bundles = Dict(let (b, _, ch) = BlobBoards.load_bundle(p); uid_from_hash(ch) => b end
               for p in board_jsons)

# Usually you load the config rather than build it: FindBoardsConfig(path; camera).
# `polarity` and `cutoff` are NOT set here — they are read from each board's
# bundle — and `patch_size` comes from the network.
config = FindBoardsConfig(
    geometry      = PoseGeometry(camera; min_inliers = 8),
    detect        = DetectConfig(peak_threshold = 0.01,
                                 nms = EllipseNms(cutoff = 2.0f0)),
    ransac_config = RansacConfig(min_trials = 100, max_trials = 10_000))

accepted = find_boards(boards, network, gray_image;
                       board_bundles = bundles, config, device = CUDABackend())

for (uid, pose, _, _) in accepted
    @show uid pose.R pose.t                    # board → camera
end
```

Two things about the API are deliberate and worth knowing:

- **The camera lives inside `config.geometry`**, not as a separate argument.
  `PoseGeometry(camera)` selects the calibrated P3P path; `HomographyGeometry()`
  is the uncalibrated one. Encoding it there makes a camera/geometry mismatch
  unrepresentable rather than a runtime error.
- **The pipeline is backend-generic** (KernelAbstractions). `device` selects
  the backend for detection, adaptation, canonicalization, and matching:
  `KA.CPU()` (threaded host implementation, the default), `CUDABackend()`,
  or any other KA GPU backend. CUDA is optional — loading `CUDA` adds
  cuBLAS/libdevice fast paths but no new capability.

There are batch and pre-extracted entry points as well —
`find_boards(boards, network, frames::Vector)` amortises gallery indexing across
a sequence, and `find_boards(boards, image_features)` skips extraction when you
already have features. For video, `track_boards` warm-starts each frame from the last.

## Examples

Examples are available in both **Julia** and **Python**.

### Julia Examples

```bash
# Every example script runs from examples/, the env that carries what they
# import (the checkout, CUDA + cuDNN, GLMakie + CairoMakie). Instantiate once:
julia --project=examples -e 'using Pkg; Pkg.instantiate()'

# Board / pattern generation (board.jl also runs from the package env, --project=.)
julia --project=examples examples/generate/board.jl
julia --project=examples examples/generate/pattern.jl

# Detection pipeline — each runs BOTH the uncalibrated (homography) and
# calibrated (P3P) variants on example data fetched from the BlobBoards Zenodo record
# through the package's artifact catalog (no dataset setup needed)
julia --project=examples examples/pipeline/find_boards.jl
julia --project=examples examples/pipeline/track_boards.jl
julia --project=examples examples/pipeline/scan_media_directory.jl <media-dir>
```

### Python Examples

```bash
# With pixi (recommended)
pixi run -e py python python/examples/boards/generate_a4_board.py
pixi run -e py python python/examples/find_boards.py

# Or with manual setup
python python/examples/boards/generate_a4_board.py
```

See [examples/](examples/) for Julia scripts and [python/examples/](python/examples/) for Python scripts.

## API

### Pattern Generation

```julia
blob_pattern(width, height; kwargs...)
blob_pattern(Size2(width, height); kwargs...)
blob_pattern(pixel_type, Size2(width, height); kwargs...)   # e.g. Gray{N0f16}; the type is positional, never a keyword
```

Key options:
- `blob_sizes`: `(min, max)` disc diameters (2·cutoff·σ) in logical units (default
  `(8px, nothing)`; a bare number is px; `max = nothing` is the default — largest disc
  20% of the pattern's short side — or a fraction such as `0.2short_side`). Stored as
  σ in px (`min_scale`/`max_scale`)
- `seed`: RNG seed for determinism
- `scales_per_octave`: Scales per octave (default 3)
- `cutoff`: Gaussian truncation radius in σ (default 2.0 — use it)
- `α`: Scale sampling bias (default 1.2)

### Board Generation

```julia
blob_board(paper_size; blob_sizes, kwargs...)
blob_board(width, height; blob_sizes, kwargs...)
BlobBoards.rebuild_board(json_or_path; dir=nothing)  # Reconstruct from its JSON, as printed (its `render` block)
```

Key options:
- `blob_sizes` (required): `(min, max)` printed disc diameters in physical units
  (e.g. `(1.0mm, 12mm)`, `(0.04inch, 0.2short_side)`), stored as σ in points.
  `suggest_blob_sizes(density, extent)` gives the smallest disc that prints well
  at a printer's density (3 mm at 1200 dpi, ∝ 1/dpi: 6 mm at 600, 12 mm at 300)
  and the largest for the board (20% of its shorter side, at most 64 mm); `extent`
  is a paper name, a `Size2` or a `(width, height)` tuple.
  The max must be given: a length or a `short_side` fraction. A board is laid
  out on a fixed 1/7200-inch grid and has no pixel density
- `output`: `Pdf()` (default; options `shading_function`, `icc_profile`, `bleed`),
  `Tiff(density; icc_profile)` or `Png(density)` —
  a raster names its own density, e.g. `Tiff(600dpi)`
- `id_label`: print the board's identity under it in 6 pt (default `true`)
- `corner_size`, `ruler_width` (10 mm each; `0mm` turns that mark off), `frame_padding`,
  `major_tick_width`, `minor_tick_width`: the frame's marks
- `margin`: the white margin around the centred board, a `Size2` (`width` left and right,
  `height` above and below; default 10 mm each, zero for full bleed); the board is the
  sheet less twice the margin.
  Or `board_size` instead (not both): the margin is then half the sheet's excess

Paper sizes: `"A4"`, `"A5"`, `"Letter"`, or custom `Size2(width, height)`

## Outputs

- **PDF**: the print master — always PDF/X-4 (OutputIntent, TrimBox at the
  finished sheet, optional print-shop bleed `Pdf(bleed = 2mm)`, "print at actual size"), each blob a vector radial shading; BlobBoards
  writes the file itself, with no external tools
- **TIFF**: the same sheet converted at the TIFF's density, an LZW raster with resolution tags and ICC profile
- **PNG**: the same conversion as a grey PNG, for training datasets
- **JSON**: Full configuration and blob coordinates (board schema v5, every quantity a number in the file's declared unit; see the reference guide)

## Cross-validation (the paper projects)

The tunings of `bmvc2026/` (the detector/estimator factorial,
`configs/detection_thresholds_cv.toml`) are `[tuning.<name>]` specs in the one schema
VisualGeometryToolkitEval's `Tuning` module owns and documents (`Tuning.tuning_spec`). Their runs persist the resolved spec and its hash; tuning artifacts written
before 2026-10-01 are refused and must be regenerated.

## Installation

### As a dependency

BlobBoards, LocalOptimizationDoneRight and PoseLib are in CauRegistry; the rest
of its stack (VisualGeometryToolkitCore, VisualGeometryToolkitIO,
VisualGeometryToolkitFeatures, VisualGeometryToolkitEstimation,
RANSACScoringDoneRight) is in General. Add CauRegistry once (Julia 1.12+):

```julia
using Pkg
Pkg.Registry.add("General")
Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl")   # once
Pkg.add("BlobBoards")
```

Everything else resolves automatically. No pixi or Python is needed: pixi is
only for the Python interface.

### For development

Julia 1.12+ is the only requirement (install via
[juliaup](https://github.com/JuliaLang/juliaup) or
[download](https://julialang.org/downloads/)); BlobBoards needs no external tools. A checkout needs CauRegistry too, for
LocalOptimizationDoneRight and PoseLib; adding it again is harmless.

```bash
git clone https://github.com/VisualGeometryToolkit/BlobBoards.jl
cd BlobBoards.jl
julia --project=. -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl"); Pkg.instantiate()'
```

[pixi](https://pixi.sh) is used only for the Python interface: `pixi run -e py …`
provides Python, numpy, pint and juliacall with the `blobboards` package
installed.

## Building Documentation

```bash
# Install docs dependencies and build
julia --startup-file=no --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=docs docs/make.jl

# View locally
open docs/build/index.html
```

## Python Interface

A Python package with structured return types is available via `juliacall`.

```bash
# With pixi (recommended): the `py` environment has blobboards installed
pixi run -e py python

# Or manual install
cd python && pip install -e .
```

```python
from blobboards import blob_pattern, blob_board, short_side, suggest_blob_sizes

# Generate pattern - returns PatternResult dataclass
result = blob_pattern(800, 600, blob_sizes=(12, None), seed=0x1234)
result.pattern      # (600, 800) NumPy array
result.blobs        # (N, 3) array of [x, y, σ]
result.config.seed  # 0x1234

# Generate board - returns BoardResult dataclass
result = blob_board("A4", blob_sizes=(4.0, 38.0), seed=0xBEEF, dir="./out")   # blob_sizes (mm) is required, both ends
result.blobs              # (N, 3) array of [x, y, σ]
result.preamble.board_config.ruler_width_mm  # 10.0
result.config_hash        # uid = first 3 hex

# Physical parameters accept Pint quantities
import pint
ureg = pint.UnitRegistry()
result = blob_board("A4", blob_sizes=(6.0*ureg.mm, 1.2*ureg.inch), ruler_width=0.5*ureg.inch)

# The max as a fraction of the pattern's short side
result = blob_board("A4", blob_sizes=(1.0, 0.2 * short_side))

# The smallest disc that prints well at a printer's density (dpi) and the largest
# for the board (mm: 20% of its shorter side, at most 64 mm); a raster names its own density
result = blob_board("A4", blob_sizes=suggest_blob_sizes(600, "A4"), dir="./out", format="png", density=300)
```

See [python/README.md](python/README.md) for full API reference.

## Training Pipeline

Build tracks of canonicalized blob patches from videos and photos of the boards,
for training local feature descriptors.

```julia
using BlobBoards

out = BlobBoards.build_tracks_for_sequences(
    "/data/boards",                     # board .json files
    "/data/models/blob_matcher.onnx",   # descriptor network
    "/data/videos";                     # videos and/or photos
    output_dir = "/data/out",
    output_name = "tracks",
)
# => "/data/out/tracks_logpolar_s32.tracks": one patch type at one scale
```

A `.tracks` file is plain HDF5 (`examples/training/TRACKS_FORMAT.md`): read it
with HDF5.jl, h5py, or `blobboards.load_tracks` from Python. See
[Board Tracks](https://blobboards.github.io/guide/board-tracks/) for the options.

See `examples/training/` for the command-line builder and its SLURM drivers.

## Reproducibility

- All generation is seeded; JSON captures full configuration
- Golden SHA-256 tests guard bit-exact output
- Same seed produces identical output (Julia and Python)

## Development

The repository ships git hooks in `.githooks/` that keep your local
`Manifest.toml` resolvable: after a pull, checkout or rebase that changes any
workspace member's `Project.toml`, they re-resolve the workspace (resolve only; precompilation
waits for the next `using`). They are opt-in, once per clone:

```bash
git config core.hooksPath .githooks
```

## Contributors

BlobBoards is developed by James Pritts, with contributions from Hendrik
Sauer and Till Sittart.

## License

MIT
