# Install for your task

BlobBoards installs with one `pkg> add` line per task. Board authoring and
tracking need BlobBoards alone; the CPU example also imports one name from
VisualGeometryToolkitCore, the stack package that owns it. GPU finding, plotting,
training and the Python interface are package extensions, which load only
once their trigger packages are installed and loaded. Each line below names
exactly the packages its task needs, and the example under it loads exactly
those. Calling an extension's function without its packages raises an error
that names the line to run.

BlobBoards requires Julia 1.12 or later. Until the General registrations of
BlobBoards and LocalOptimizationDoneRight merge, add
[CauRegistry](https://github.com/prittjam/CauRegistry.jl) first; the rest of
the stack comes from General:

```julia-repl
pkg> registry add https://github.com/prittjam/CauRegistry.jl
```

| Task | Install | Example |
|------|---------|---------|
| author boards (pattern, PDF, PNG) | `pkg> add BlobBoards` | [Author boards](#Author-boards) |
| find boards on the CPU | `pkg> add BlobBoards VisualGeometryToolkitCore` | [Find boards on the CPU](#Find-boards-on-the-CPU) |
| find boards on the GPU | `pkg> add BlobBoards CUDA cuDNN VisualGeometryToolkitCore`, then once `using CUDA; CUDA.set_runtime_version!(v"12.9")` and restart Julia | [Find boards on the GPU](#Find-boards-on-the-GPU) |
| track boards in a video | `pkg> add BlobBoards` | [Track boards in a video](#Track-boards-in-a-video) |
| plot or visualize | `pkg> add BlobBoards CairoMakie` | [Plot or visualize](#Plot-or-visualize) |
| train the descriptor network | `pkg> add BlobBoards Lux Optimisers Enzyme Reactant NNlib TensorBoardLogger Zygote` | [Train the descriptor network](#Train-the-descriptor-network) |
| use from Python | in a clone: `pixi run -e py python` (the package instantiates its own Julia environment) | [Use from Python](#Use-from-Python) |
| work in a clone | `julia --project=<member> -e 'using Pkg; Pkg.instantiate()'` | [Work in a clone](#Work-in-a-clone) |

Every example below is a complete script: it runs as written in a fresh
environment made by its row's install line, and writes its files to a
temporary directory. The package's test suite runs each of them that way,
except the training example, which needs data you record yourself.

## Author boards

`blob_board` writes a printable board, a PDF print master by default, and the
JSON sidecar that reproduces it; `blob_pattern` makes a pixel pattern. See
[Authoring a Board](boards.md) and [Patterns](patterns.md).

```julia
# Install row: author boards
using BlobBoards      # re-exports the units: mm, inch, px, dpi, …

dir = mktempdir()
# a PDF board, its smallest disc sized for a 600 dpi printer
blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 0xBEEF, dir)
# the same layout as a PNG at 300 pixels per inch, for training data
blob_board("A4"; blob_sizes = (4.0mm, 0.2short_side), seed = 0xBEEF, dir, output = Png(300dpi))
# a pixel pattern, its smallest disc 12 px across
pattern, pattern_blobs, config = blob_pattern(800, 600; blob_sizes = (12px, nothing), seed = 0x1234, dir)

count(endswith(".pdf"), readdir(dir)) == 1    # true
```

## Find boards on the CPU

A board rendered as an image, then found in that image with no camera (a
homography per board). The default descriptor network is fetched once from
the BlobBoards Zenodo record on first use. See [Detection and Pose](detection.md).

```julia
# Install row: find boards on the CPU
using BlobBoards
using VisualGeometryToolkitCore: is_map_inlier

# A small board rendered at 100 pixels per inch: the image to search
dir = mktempdir()
blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0xBEEF, dir, output = Png(100dpi))
board_json = only(filter(endswith(".json"), readdir(dir; join = true)))
scene = load_board(board_json)                 # the board's uid and bundle
_, image = BlobBoards.load(board_json)         # the rendered sheet, a grey image

network = DescriptorNetwork(descriptor_network_path(); execution_provider = :cpu)
gallery = BoardGallery(joinpath(dir, "board_features.jld2"), network)
boards  = Dict(insert!(gallery, [board_json], network))   # uid => the board's described blobs
bundles = Dict(scene.uid => scene.bundle)

config   = FindBoardsConfig(geometry = HomographyGeometry())
accepted = find_boards(boards, network, image; board_bundles = bundles, config)

uid, H, quality, csponds = only(accepted)      # the board's uid, its board → image homography
n_inliers = count(is_map_inlier, csponds.weights)
uid == scene.uid                               # true
```

## Find boards on the GPU

The same search with every stage on the GPU: CUDA loads BlobBoards' device
kernels, and CUDA with cuDNN loads ONNX Runtime's CUDA provider for the
descriptor network. That provider takes a CUDA 12 runtime, older than CUDA.jl's
default, so pin it once per environment after the install line:

```julia-repl
julia> using CUDA; CUDA.set_runtime_version!(v"12.9")   # then restart Julia
```

See [The GPU environment](@ref gpu-env) for the requirements.

```julia
# Install row: find boards on the GPU
using BlobBoards, CUDA, cuDNN
using VisualGeometryToolkitCore: is_map_inlier

dir = mktempdir()
blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0xBEEF, dir, output = Png(100dpi))
board_json = only(filter(endswith(".json"), readdir(dir; join = true)))
scene = load_board(board_json)
_, image = BlobBoards.load(board_json)

network = DescriptorNetwork(descriptor_network_path())   # ONNX Runtime's CUDA provider
gallery = BoardGallery(joinpath(dir, "board_features.jld2"), network)
boards  = Dict(insert!(gallery, [board_json], network))
bundles = Dict(scene.uid => scene.bundle)

config   = FindBoardsConfig(geometry = HomographyGeometry())
accepted = find_boards(boards, network, image; board_bundles = bundles, config, device = CUDABackend())

uid, H, quality, csponds = only(accepted)
n_inliers = count(is_map_inlier, csponds.weights)
uid == scene.uid                               # true
```

## Track boards in a video

`track_boards` carries each board from frame to frame and runs the full search
only on keyframes. It takes a video file (`track_boards(boards, network,
"clip.mp4"; board_bundles)`) or any sequence of grey frames, as here: a short
synthetic clip of a rendered board sliding across the frame.

```julia
# Install row: track boards in a video
using BlobBoards

dir = mktempdir()
blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0xBEEF, dir, output = Png(100dpi))
board_json = only(filter(endswith(".json"), readdir(dir; join = true)))
scene = load_board(board_json)
_, sheet = BlobBoards.load(board_json)

# Four frames: the sheet on a white background, 6 px further right in each
h, w = size(sheet)
clip = map(0:6:18) do dx
    frame = fill(one(eltype(sheet)), h + 40, w + 40)
    frame[21:(20 + h), (1 + dx):(dx + w)] .= sheet
    frame
end

network = DescriptorNetwork(descriptor_network_path(); execution_provider = :cpu)
gallery = BoardGallery(joinpath(dir, "board_features.jld2"), network)
boards  = Dict(insert!(gallery, [board_json], network))
bundles = Dict(scene.uid => scene.bundle)
config  = FindBoardsConfig(geometry = HomographyGeometry())

tracks = track_boards(boards, network, clip; board_bundles = bundles, config)
detections = tracks[scene.uid]                 # one BoardDetection per frame the board was found in
[d.source for d in detections]                 # :detect on the first frame, then :track
length(detections) == length(clip)             # true
```

## Plot or visualize

The plots are a Makie extension: any Makie backend loads it. CairoMakie writes
files; GLMakie opens windows.

```julia
# Install row: plot or visualize
using BlobBoards, CairoMakie

dir = mktempdir()
blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0xBEEF, dir, output = Png(100dpi))
board_json = only(filter(endswith(".json"), readdir(dir; join = true)))
scene = load_board(board_json)
_, image = BlobBoards.load(board_json)

network = DescriptorNetwork(descriptor_network_path(); execution_provider = :cpu)
gallery = BoardGallery(joinpath(dir, "board_features.jld2"), network)
boards  = Dict(insert!(gallery, [board_json], network))
bundles = Dict(scene.uid => scene.bundle)
accepted = find_boards(boards, network, image; board_bundles = bundles,
                       config = FindBoardsConfig(geometry = HomographyGeometry()))

# Each board found, outlined over the image with its inlier blobs
fig = BlobBoards.plot_boards(nothing, accepted, bundles, image)
figure_path = joinpath(dir, "found_boards.png")
save(figure_path, fig)
isfile(figure_path)                            # true
```

## Train the descriptor network

Training is an extension that loads with the training stack. It reads
annotation files built from your own recordings of printed boards
([Board Training Tracks](board-tracks.md)), so the example is an illustration.

```julia
# Illustrative: needs annotation files from your own recordings of printed boards, and hours on a GPU
# Install row: train the descriptor network
using BlobBoards, Lux, Optimisers, Enzyme, Reactant, NNlib, TensorBoardLogger, Zygote

checkpoints = mktempdir()
result = BlobBoards.train_descriptor("annotations/train.h5", "annotations/val.h5"; output_dir = checkpoints)
# checkpoints/best_model.jld2: the network with the lowest false-positive rate at 95% recall
```

## Use from Python

The `blobboards` Python package runs in a clone of the repository. Its pixi
environment provides Python, NumPy, pint, matplotlib and juliacall, and the
package instantiates its own Julia environment (the clone's `authoring/`) on
first import:

```bash
# Install row: use from Python
git clone https://github.com/VisualGeometryToolkit/BlobBoards.jl
cd BlobBoards.jl
pixi run -e py python find_board.py
```

where `find_board.py` is the script below. With Julia installed by juliaup,
point juliacall at the Julia binary rather than juliaup's launcher, or the
import fails with `could not load library ".../lib/julia/sys.so"`:
`export PYTHON_JULIAPKG_EXE=$(julia -e 'print(joinpath(Sys.BINDIR, "julia"))')`.
See the [Python Interface](python.md).

```python
# Install row: use from Python
import os
import tempfile

import matplotlib.image
from blobboards import blob_board, descriptor_network_path, find_boards, short_side

directory = tempfile.mkdtemp()
board = blob_board("A5", blob_sizes=(4.0, 0.2 * short_side), seed=0xBEEF,
                   dir=directory, format="png", density=100)
board_json = next(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json"))
board_png = board_json[: -len(".json")] + ".png"
image = matplotlib.image.imread(board_png)            # grey, floats in [0, 1]

found = find_boards([board_json], descriptor_network_path(), image,
                    gallery_path=os.path.join(directory, "board_features.jld2"), device="cpu")
print(found[0]["uid"] == board.uid)                    # True
```

## Work in a clone

The repository is one Pkg workspace with one Manifest, at its root. Each
member is the environment for one kind of work; instantiate the one you need:

| Member | For |
|--------|-----|
| `.` | the package itself: authoring and finding boards on the CPU |
| `gpu` | the GPU (CUDA, cuDNN, the pinned runtime) and the track-building command line tools |
| `examples` | the scripts in `examples/`: the GPU, GLMakie windows and CairoMakie figures |
| `authoring` | the Python package's Julia side (PythonCall), CairoMakie plots, Revise |
| `test` | the test suite |
| `docs` | this documentation |

```bash
# Install row: work in a clone
git clone https://github.com/VisualGeometryToolkit/BlobBoards.jl
cd BlobBoards.jl
git config core.hooksPath .githooks     # re-resolve the Manifest after a pull moves a Project.toml
julia --project=. -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl"); Pkg.instantiate()'
julia --project=. -e 'using BlobBoards; blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), dir = mktempdir())'
```

Instantiate another member the same way (`julia --project=gpu -e 'using Pkg;
Pkg.instantiate()'`) and run its scripts with it: `julia --project=examples
examples/pipeline/find_boards.jl` finds the example boards in the example
still. The test suite runs with `julia --project=. -e 'using Pkg; Pkg.test()'`,
and the documentation builds with
`julia --startup-file=no --project=docs docs/make.jl`.
