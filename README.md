# BlobBoards.jl

[![CI](https://github.com/VisualGeometryToolkit/BlobBoards.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VisualGeometryToolkit/BlobBoards.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-blobboards.github.io-blue.svg)](https://blobboards.github.io)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/BlobBoards.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/VisualGeometryToolkit/BlobBoards.jl)

BlobBoards is a fiducial marker system: printable boards covered in a dense,
multi-scale field of Gaussian blobs, and a pipeline that finds them in images,
identifies each one and estimates its pose from hundreds of blob
correspondences. This package authors the boards (PDF, TIFF or PNG, each with
a JSON sidecar that reproduces it) and finds and tracks them on the CPU or a
GPU. It is the implementation of
[BlobBoards: Robust Markers for Accurate Pose](https://arxiv.org/abs/2608.28830)
(BMVC 2026).

## Installation

BlobBoards requires Julia 1.13 or later.

The example below finds a board on the CPU, and imports one name from
VisualGeometryToolkitCore, the stack package that owns it:

```julia-repl
pkg> add BlobBoards VisualGeometryToolkitCore
```

### Install for your task

GPU finding, plotting, training and the Python interface are package
extensions: they load once their trigger packages are installed and loaded.
Each line names exactly the packages its task needs. Calling an extension's
function without them raises an error naming the line to run.

| Task | Install | Example |
|------|---------|---------|
| author boards (pattern, PDF, PNG) | `pkg> add BlobBoards` | [author boards](https://blobboards.github.io/guide/install/#Author-boards) |
| find boards on the CPU | `pkg> add BlobBoards VisualGeometryToolkitCore` | [below](#example) |
| find boards on the GPU | `pkg> add BlobBoards CUDA cuDNN VisualGeometryToolkitCore`, then once `using CUDA; CUDA.set_runtime_version!(v"12.9")` and restart Julia | [GPU](https://blobboards.github.io/guide/install/#Find-boards-on-the-GPU) |
| track boards in a video | `pkg> add BlobBoards` | [tracking](https://blobboards.github.io/guide/install/#Track-boards-in-a-video) |
| plot or visualize | `pkg> add BlobBoards CairoMakie` | [plots](https://blobboards.github.io/guide/install/#Plot-or-visualize) |
| train the descriptor network | `pkg> add BlobBoards Lux Optimisers Enzyme Reactant NNlib TensorBoardLogger Zygote` | [training](https://blobboards.github.io/guide/install/#Train-the-descriptor-network) |
| use from Python | in a clone: `pixi run -e py python` (the package instantiates its own Julia environment) | [Python](https://blobboards.github.io/guide/install/#Use-from-Python) |
| work in a clone | `julia --project=<member> -e 'using Pkg; Pkg.instantiate()'` | [clone](https://blobboards.github.io/guide/install/#Work-in-a-clone) |

## Example

Render a board as an image, then find it in that image with no camera (one
homography per board). The default descriptor network is fetched once from the
BlobBoards Zenodo record on first use; every file goes to a temporary
directory.

```julia
# Install row: find boards on the CPU
using BlobBoards
using VisualGeometryToolkitCore: is_map_inlier

# A small board rendered at 100 pixels per inch: the image to search
dir = mktempdir()
blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0xBEEF, dir, output = Png(100dpi))
board_json = only(filter(endswith(".json"), readdir(dir; join = true)))
scene = load_board(board_json)                 # the board's uid and bundle
_, image = BlobBoards.load(BlobBoards.BoardBundle, board_json)         # the rendered sheet, a grey image

network = DescriptorNetwork(descriptor_network_path(); execution_provider = :cpu)
gallery = BoardGallery(joinpath(dir, "board_features.jld2"), network)
boards  = Dict(insert!(gallery, [board_json], network))   # uid => the board's described blobs
bundles = Dict(scene.uid => scene.bundle)

config   = FindBoardsConfig(geometry = HomographyGeometry())
accepted = find_boards(boards, network, image; board_bundles = bundles, config)

uid, H, quality, scored = only(accepted)      # the board's uid, its board → image homography
n_inliers = count(is_map_inlier, scored.weights)
uid == scene.uid                               # true
```

[Getting Started](https://blobboards.github.io/guide/getting-started/) authors,
prints and loads a first board.

## Documentation

- [Install for your task](https://blobboards.github.io/guide/install/): one install line and one runnable example per task.
- [Getting Started](https://blobboards.github.io/guide/getting-started/): a first board and its blob coordinates.
- [Authoring a Board](https://blobboards.github.io/guide/boards/) and [Patterns](https://blobboards.github.io/guide/patterns/): sizes, frames, outputs and printing.
- [Detection and Pose](https://blobboards.github.io/guide/detection/): `find_boards`, `track_boards`, calibrated pose and the GPU.
- [Python Interface](https://blobboards.github.io/guide/python/): the `blobboards` Python package.
- [API Reference](https://blobboards.github.io/api/public/).

BlobBoards is part of the [Visual Geometry Toolkit](https://visualgeometrytoolkit.github.io),
Julia packages for robust visual geometry from Kiel University.

## Citing

If you use BlobBoards in your research, please cite:

```bibtex
@inproceedings{pritts2026blobboards,
  author    = {Pritts, James and Sittart, Till and Sauer, Hendrik and
               Jan{\ss}en, Silja and Seegr{\"a}ber, Felix and
               Nakath, David and K{\"o}ser, Kevin},
  title     = {{BlobBoards}: Robust Markers for Accurate Pose},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026},
  eprint    = {2608.28830},
  archivePrefix = {arXiv},
}
```

## License

MIT
