# BlobBoards Python Package

Python interface to BlobBoards.jl: make multi-scale blob calibration boards,
print them, and find them in images and video. One function per Julia entry
point, with the same names, keywords and semantics; NumPy arrays and dataclasses
in place of Julia types. Each function's docstring (`help(blobboards.find_boards)`)
is its full reference. The documentation's Python guide
(`docs/src/guide/python.md`) covers tasks, GPU selection and troubleshooting.

## Installation

```bash
pixi run -e py python                          # recommended: Python, numpy, pint, juliacall, blobboards (Julia is a prerequisite)
# or
cd python && pip install -e .                    # juliacall, numpy (+ pint, matplotlib for units, plots)
```

The package runs Julia in the checkout's `authoring/` environment (PythonCall,
and CUDA/cuDNN for GPU detection, loaded only when detection first runs). If import fails with `could not load library
".../lib/julia/sys.so"`, juliacall was given juliaup's launcher:
`export PYTHON_JULIAPKG_EXE=$(julia -e 'print(joinpath(Sys.BINDIR, "julia"))')`. Setting juliacall's own
`PYTHON_JULIACALL_EXE` to that binary works too, together with
`PYTHON_JULIACALL_PROJECT=<checkout>/authoring` (juliacall requires both). The
Julia side's PythonCall is pinned to the lock's juliacall release
(`authoring/Project.toml`): a mismatch fails with "PythonCall.jl did not start
properly".

## Quick start

```python
from blobboards import blob_pattern, blob_board, short_side

pattern = blob_pattern(800, 600, blob_sizes=(12, None), seed=0x1234)   # discs from 12 px across
print(pattern.pattern.shape, pattern.blobs.shape)                     # (600, 800) (N, 3): x, y, σ

board = blob_board("A4", blob_sizes=(4.0, 0.2 * short_side), seed=0x1234,          # discs from 4 mm across
                   dir="./output")                                    # the PDF/X-4 print master + JSON
print(board.uid, board.blobs.shape)
```

A board has no pixel density: the PDF prints on any printer. `blob_sizes` is
required and states both ends: `suggest_blob_sizes(density, extent)` gives the
smallest disc (mm) that prints well at a printer's density and the largest for
the board (20% of its shorter side, at most 64 mm), e.g.
`blob_sizes=suggest_blob_sizes(600, "A4")`. The max may instead be
`0.2 * short_side` (that fraction of the pattern's short side, no cap); `None`
is an error. A raster output names its own density: `format="png", density=300`.

```python
import os
import numpy as np
from PIL import Image
import blobboards
from blobboards import find_boards, example_data, descriptor_network_path

boards = example_data("boards")
board_paths = sorted(os.path.join(boards, f) for f in os.listdir(boards) if f.endswith(".json"))
img = np.asarray(Image.open(os.path.join(example_data("still"), "DSC02847.JPG")).convert("L"))

print(blobboards.device())                       # "cuda" when a usable GPU is present, else "cpu"
found = find_boards(board_paths, descriptor_network_path(), img, gallery_path="gallery.jld2")
for b in found:
    print(b["uid"], b["n_inliers"], b["execution_provider"])   # model: a 3×3 homography, mm → px
```

## Functions

| Python | Julia | What it does |
|--------|-------|--------------|
| `blob_board(paper, blob_sizes=…, **kw)` | `blob_board` | a printable board (PDF/X-4, or TIFF/PNG at a `density`, + JSON) |
| `suggest_blob_sizes(density, extent)` | `suggest_blob_sizes` | a board's `blob_sizes` (mm) for a printer's density and the board's extent (a paper name or `(width, height)`) |
| `blob_pattern(width, height, **kw)` | `blob_pattern` | a blob pattern in pixels |
| `load_board(json_path, density=None)` | `load_board` / `load_bundle` | a stored board, under its stored identity |
| `render_board(json_path, dir=…, **kw)` | `render_board` | re-render a stored board without re-placing it (`format`, `density`, …) |
| `blobs(json_path, …)` | `blobs` | coordinates in a frame (`pattern`, `board`, `canvas`, `world`) and unit |
| `find_boards(board_paths, model_path, img, …)` | `find_boards` | boards in an image: homography or pose per board |
| `track_boards(board_paths, model_path, frames, …)` | `track_boards` | boards through a video or a frame sequence |
| `device(requested="auto")` | `default_device` | where detection runs: `"cuda"` or `"cpu"` |
| `example_data(name)` | `example_data` | the bundled example boards, still and clip |
| `descriptor_network_path(config=None)` | `descriptor_network_path` | the descriptor network a run configuration names |
| `PRINT_SIZES` | `PRINT_SIZES` | paper names → `(width, height)` in mm |
| `short_side`, `ShortSideFraction` | `short_side`, `ShortSideFraction` | a blob size as a fraction of the pattern's short side |
| `BoardResult`, `BoardPreamble`, `BoardConfig`, `PatternResult`, `PatternConfig` | `BoardBundle`, `BoardPreamble`, `BoardConfig`, …, `PatternConfig` | results and configurations |

`blob_sizes` are disc **diameters**: mm for boards (bare numbers; Pint
quantities in any length unit), px for patterns. A board's returned pixel
blobs are at `density` when given, else at the board's own raster density
(smallest disc at least 16 px across; `density_dpi` on the result).
Detection defaults to the GPU
when one is usable (`device="auto"`) and falls back to the CPU with a warning;
each result records its `device` and `execution_provider`. Advanced, unstable
API: `detect_blobs` (the detector alone), `load_tracks` / `plot_tracks`
(training tracks written from Julia; pure Python, h5py).

## Examples

```bash
pixi run -e py python python/examples/find_boards.py        # example still → found_boards.png
pixi run -e py python python/examples/boards/generate_a4_board.py
pixi run -e py python python/examples/blob_extraction.py
```
