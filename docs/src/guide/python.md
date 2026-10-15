# Python Interface

```@meta
CurrentModule = BlobBoards
```

The `blobboards` Python package wraps this Julia package through `juliacall`.
It has one function per Julia entry point, with the same names, keywords and
semantics, and with NumPy arrays and dataclasses in place of Julia types. Each
Python function's docstring (`help(blobboards.blob_board)`) is its full
reference: parameters, defaults, units, return fields, errors and an example.
This page covers installation and tasks.

## Installation

### With pixi (recommended)

```bash
pixi run -e py python   # the `py` environment: Python, numpy, pint, juliacall, the blobboards package (Julia is a prerequisite)
```

### Manual setup

**Requirements:**
- Python 3.8+
- Julia 1.12+ (juliacall uses it)

```bash
cd python
pip install -e .        # juliacall, numpy; `pip install pint matplotlib` for units and plots
```

The package runs Julia in the checkout's `authoring/` environment, which adds
PythonCall (the Python wrappers' Julia side) to the checkout, plus CUDA and
cuDNN, which are loaded only when detection first runs. The package activates
it on first use; the first call can take minutes on a fresh machine.

## Functions

| Python | Julia | What it does |
|--------|-------|--------------|
| `blob_board(paper, blob_sizes=…, **kw)` | [`blob_board`](@ref) | a printable board (PDF/X-4, or TIFF/PNG at a `density`, + JSON) |
| `suggest_blob_sizes(dpi, extent)` | [`suggest_blob_sizes`](@ref) | a board's `blob_sizes` (mm): the smallest disc that prints well at a printer's density, and 20% of the board's shorter side, at most 64 mm |
| `blob_pattern(width, height, **kw)` | [`blob_pattern`](@ref) | a blob pattern in pixels |
| `load_board(json_path, density=None)` | `load_board` / `BlobBoards.load_bundle` | a stored board, under its stored identity |
| `render_board(json_path, dir=…, **kw)` | `BlobBoards.render_board` | re-render a stored board without re-placing it |
| `blobs(json_path, …)` | [`BlobBoards.blobs`](@ref) | coordinates in a frame (`pattern`, `board`, `canvas`, `world`) and unit |
| `find_boards(board_paths, model_path, img, …)` | [`find_boards`](@ref) | boards in an image: homography or pose per board |
| `track_boards(board_paths, model_path, frames, …)` | [`track_boards`](@ref) | boards through a video or a frame sequence |
| `device(requested="auto")` | `default_device` | where detection runs: `"cuda"` or `"cpu"` |
| `example_data(name)` | `BlobBoards.example_data` | the bundled example boards, still and clip |
| `descriptor_network_path(config=None)` | `descriptor_network_path` | the descriptor network a run configuration names |
| `PRINT_SIZES` | [`BlobBoards.PRINT_SIZES`](@ref) | paper names → `(width, height)` in mm |
| `short_side`, `ShortSideFraction` | `short_side`, `BlobBoards.ShortSideFraction` | a blob size as a fraction of the pattern's short side |
| `BoardResult`, `BoardPreamble`, `BoardConfig`, `PatternResult`, `PatternConfig` | `BoardBundle`, `BoardPreamble`, [`BoardConfig`](@ref), …, [`PatternConfig`](@ref) | results and configurations |

Two Julia options have no Python form: the radially distorted geometry (`RadialHomographyGeometry`)
and the leading pixel-type argument of `blob_board` / `blob_pattern` (patterns
and boards are 8-bit grey from Python).

## Quick start

```python
from blobboards import blob_pattern, blob_board, short_side

pattern = blob_pattern(800, 600, blob_sizes=(12, None), seed=0x1234)   # discs from 12 px across
print(pattern.pattern.shape, pattern.blobs.shape)                     # (600, 800) (N, 3): x, y, σ

board = blob_board("A4", blob_sizes=(4.0, 0.2 * short_side), seed=0x1234,          # discs from 4 mm across
                   dir="./output")                                    # the PDF/X-4 print master + JSON
print(board.uid, board.blobs.shape)
```

`blob_sizes` are the smallest and largest **disc diameters**: mm for boards
(bare numbers; Pint quantities in any length unit work too), px for patterns.
For a board it is required, and both ends are stated (`None` is an error):

`suggest_blob_sizes(dpi, extent)` gives both ends, in mm:

- the smallest disc is a choice about the printer: the diameter that prints
  well at a printer's density — 3 mm at 1200 dpi, 6 mm at 600, 12 mm at 300;
- the largest is 20% of the board's shorter side, at most 64 mm. `extent` is a
  paper name (the default board on it, the paper less 10 mm on every side) or
  the board's `(width, height)` in mm or Pint lengths.

Either end may also be given directly; the max as `0.2 * short_side` is exactly
that fraction of the pattern's short side, with no cap.

```python
from blobboards import suggest_blob_sizes

# a 600 dpi printer and the default A4 board: (6.0, 38.0)
board = blob_board("A4", blob_sizes=suggest_blob_sizes(600, "A4"), seed=0x1234)
```

Board-generation keywords mirror Julia's (`alpha` is Julia's `α`). A keyword
that is not a parameter (`pdf_standard`, `print_density`, `blob_measure`,
`frame_style`, `frame_width`, `raster_aligned` on boards) raises `TypeError`.

### Output

Julia names a board's output with a value (`Pdf()`, `Tiff(600dpi)`,
`Png(300dpi)`); Python names it with keywords:

| Keyword | Values | Applies to |
|---------|--------|------------|
| `format` | `"pdf"` (default: PDF/X-4), `"tiff"`, `"png"` | |
| `density` | dots per inch (a number or a Pint quantity); required for `"tiff"` and `"png"` | the raster's pixels; for a PDF, only the returned pixel blobs |
| `icc_profile` | `"linear"` (default) or the path of a grey ICC profile | PDF, TIFF |
| `shading_function` | `"calculator"` (default) or `"sampled"` | PDF |

```python
import tempfile
out = tempfile.mkdtemp()
blob_board("A5", blob_sizes=(12, 0.2 * short_side), seed=1, dir=out, format="png", density=300)
```

A raster must give the board's smallest blob at least 2 pixels per σ. The
returned `blobs` are pixels at `density` when given, else at the board's own
raster density — the whole-number dpi at which its smallest disc is at least
16 pixels across — reported as `board.density_dpi`.

## Finding boards

`find_boards(board_paths, model_path, img, *, gallery_path, camera_json=None, device="auto")`
returns one dict per verified board:

- `uid`
- `model`: a 3×3 homography from the board's world plane (mm) to pixels, or
  `{"R", "t"}`, the board → camera pose, when `camera_json` is given
- `quality`, `n_inliers`, `n_csponds`, `weights` (a NumPy array)
- `device` and `execution_provider`: where it ran, `"cuda"` or `"cpu"`

The network, the gallery features (cached on disk at `gallery_path`) and the
bundles are built once per process and reused.

### GPU or CPU

`device="auto"` (the default) runs on the GPU when a usable NVIDIA GPU is
present. Otherwise it runs on the CPU and logs a warning on the `blobboards`
logger.

- `device="cuda"` raises `RuntimeError` when there is no usable GPU.
- `device="cpu"` always uses the CPU.

The detection stages and the descriptor network (ONNX Runtime's CUDA or CPU
provider) run on the same device. CUDA is loaded only when detection first asks
for a device, so `blob_board` never pays for it. To see where detection will
run without running it:

```python
import blobboards
blobboards.device()          # "cuda" or "cpu", as "auto" resolves on this machine
```

The GPU requirements are those of the Julia side (see
[Detection and Pose](detection.md#Requirements)):

- an NVIDIA GPU of compute capability 7.5 or newer
- a driver for CUDA 12 or newer
- Linux or Windows

CUDA, cuDNN and the CUDA 12.9 runtime come as Julia artifacts with the
package's Julia environment, so no CUDA toolkit is needed. Choose a GPU with
`CUDA_VISIBLE_DEVICES` before starting Python; `CUDA_VISIBLE_DEVICES=""` forces
the CPU.

On a B200 with the example still (12 MP, 50 boards), a warm call took 1.8–1.9 s
per image on the GPU and 16–17 s on the CPU. The first call also builds the
gallery: about 2 minutes, or seconds once its caches are warm.

```python
import os
import numpy as np
from blobboards import find_boards, example_data, descriptor_network_path

boards = example_data("boards")
board_paths = sorted(os.path.join(boards, f) for f in os.listdir(boards) if f.endswith(".json"))
model_path = descriptor_network_path()
# img: a grey image, (H, W) uint8 or float in [0, 1]
found = find_boards(board_paths, model_path, img, gallery_path="gallery.jld2")
for b in found:
    print(b["uid"], b["n_inliers"], b["device"], b["execution_provider"])
```

### Checking the result visually

Each homography maps its board's world plane (mm) to image pixels, so the
board's blobs can be drawn straight onto the image with matplotlib:

```python
import matplotlib.pyplot as plt
from blobboards import blobs

uid_to_json = {os.path.basename(p)[11:14]: p for p in board_paths}   # blob_board_<uid>_….json
fig, ax = plt.subplots(figsize=(14, 9.5))
ax.imshow(img, cmap="gray")
for b in found:
    X = blobs(uid_to_json[b["uid"]], reference_frame="world", units="mm")   # (N, 3): x, y, σ
    p = np.c_[X[:, :2], np.ones(len(X))] @ b["model"].T                    # homogeneous pixels
    ax.plot(p[:, 0] / p[:, 2], p[:, 1] / p[:, 2], ".", ms=2)
    ax.annotate(b["uid"], p[0, :2] / p[0, 2], color="yellow")
fig.savefig("found_boards.png", dpi=100)
```

### Video

`track_boards` takes a video path, or any sequence of grey frames. It returns,
per board uid, one dict per frame the board was found in: `frame`, `model`,
`quality`, `n_inliers` and `source` (`"track"`, `"init"` or `"detect"`).

```python
from blobboards import track_boards

frames = [img, img]                       # or a video path, e.g. in example_data("video")
tracks = track_boards(board_paths, model_path, frames, gallery_path="gallery.jld2")
for uid, dets in tracks.items():
    print(uid, len(dets), "frames")
```

### Troubleshooting

- `could not load library ".../lib/julia/sys.so"` at import: juliacall was given
  juliaup's launcher instead of a Julia binary. Point it at the real one:
  `export PYTHON_JULIAPKG_EXE=$(julia -e 'print(joinpath(Sys.BINDIR, "julia"))')`.
- Unless you set them yourself, the package sets
  `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` and `PYTHON_JULIACALL_THREADS=auto`
  before juliacall is first imported. If Python handles Julia's signals, GPU
  work crashes the process. Import `blobboards` before anything else that
  imports `juliacall`.

## Stored boards

A board's JSON sidecar is its definition. `load_board` reads it, in every
schema BlobBoards has written, under the stored identity. `render_board` draws
it again from its recorded blobs:

```python
import tempfile
from blobboards import load_board, render_board

board = load_board(board_paths[0])
dpi = board.density_dpi      # the density of board.blobs' pixels
print(board.uid, dpi)
render_board(board_paths[0], dir=tempfile.mkdtemp(), format="png", density=dpi)
```

`load_board(path, density=300)` returns the pixel blobs at another density.
The default is the density of the board's image: a raster beside the JSON at
its recorded density, else the board's raster density (smallest disc at least
16 px across). A board stored in schema v3 was laid out on its printer's
pixels and takes only its own density. `render_board` writes a PDF by default;
`format="tiff"` or `"png"` needs `density`.

## Coordinates

```python
from blobboards import blob_board, blobs

out = tempfile.mkdtemp()
blob_board("A5", blob_sizes=(12, 0.2 * short_side), seed=1, dir=out)
json_path = [os.path.join(out, f) for f in os.listdir(out) if f.endswith(".json")][0]

b = blobs(json_path, density=300)                       # px at 300 dpi, pattern frame, COLMAP
b_mm = blobs(json_path, units="mm", reference_frame="board")  # mm, board frame
w = blobs(json_path, units="mm", reference_frame="world")     # the board plane: centred, y up
b_cv = blobs(json_path, image_origin="opencv", density=300)   # for NumPy/OpenCV sampling
```

A board's layout has no density (it is physical; see the Julia board guide), so
pixel coordinates need the density they are at: on a v4 board, `units="px"`
(the default) without `density` raises an error. A board stored in schema
v3 was laid out on its printer's pixels, and its own density is the default.

| `image_origin` | First pixel centre | Use |
|----------------|--------------------|-----|
| `"colmap"` | (0.5, 0.5) | COLMAP/SfM, geometry (the default for coordinates) |
| `"opencv"` | (0, 0) | NumPy/OpenCV image sampling (the default for detection results) |
| `"julia"` | (1, 1) | Julia/MATLAB |

| `reference_frame` | Origin | Description |
|-------------------|--------|-------------|
| `"pattern"` | Pattern top-left | Excludes the frame (default) |
| `"board"` | Board top-left | Includes the frame |
| `"canvas"` | Page top-left | Includes the margins |
| `"world"` | Board centre, y up, mm | What a `find_boards` homography maps to pixels |

## Pint units

Lengths can be Pint quantities in any unit; bare numbers are mm for boards and
px for patterns:

```python
import pint
from blobboards import short_side
ureg = pint.UnitRegistry()

board = blob_board("A4", blob_sizes=(6.0 * ureg.mm, 0.2 * short_side),
                   ruler_width=0.5 * ureg.inch, frame_padding=2 * ureg.mm)
board = blob_board("A4", blob_sizes=(4.0, 0.2 * short_side))   # largest disc 20% of the short side
```

## Blob detection only

`detect_blobs` exposes the detector alone (advanced API; see its docstring):

```python
from blobboards import detect_blobs

result = detect_blobs(np.random.rand(240, 320))
print(len(result), result.to_array().shape)    # (N, 3): x, y, σ
```

## Training tracks

`.tracks` files are written from Julia by
[`build_tracks_for_sequences`](@ref); there is no Python builder. Reading them
needs no Julia (h5py only): `load_tracks(path)` returns a `TrackData`, and
`plot_tracks(data, max_tracks=10)` plots the longest tracks.
