# Detection and Pose

This guide covers finding known boards in images with [`find_boards`](@ref),
which returns each board's homography or its pose relative to the camera. It
also covers tracking boards through video with [`track_boards`](@ref) and
setting up the GPU environment. The full keyword references are the
[`find_boards`](@ref) and [`FindBoardsConfig`](@ref) docstrings; this guide does
not repeat them.

## The pipeline in one call

`find_boards` takes three inputs:

- **a gallery** of board descriptors. Each board's printed blobs are rendered,
  canonicalized and described once, then cached on disk (a [`BoardGallery`](@ref)).
- **a descriptor network** ([`DescriptorNetwork`](@ref)): the ONNX model that
  describes both the gallery and the image.
- **an image**: grey, as an `AbstractMatrix`.

For each image it detects blobs, refines them to ellipses, describes them, and
matches the descriptors against every board. For each candidate board it then
fits the geometry with RANSAC and local optimization and gates the fit on inlier
count and on the inliers' appearance. Each accepted board claims its image
region, so no detection can serve two boards.

## Example: find the boards in a still

The example data ships through the package's artifact catalog: 50 board
definitions (`:boards`), a still image with its camera calibration (`:still`)
and a short clip (`:video`). Each downloads on first use. The descriptor network
named by the default configuration (`configs/default.toml`) is fetched the same
way, from its catalog entry in the BlobBoards Zenodo record.

Run it from the checkout's `gpu/` environment (`julia --project=gpu`, see
[The GPU environment](@ref gpu-env)), which provides every package it loads.
On a machine without an NVIDIA GPU, leave out `using CUDA, cuDNN` and give the
network `execution_provider = :cpu`; nothing else changes.

```julia
using CUDA, cuDNN                      # the GPU path (leave out for the CPU)
using BlobBoards
using FileIO, Colors                   # image loading
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: CameraModel, is_map_inlier

boards_dir = BlobBoards.example_data(:boards)
still = BlobBoards.example_data(:still)

# the network named by configs/default.toml, fetched from the artifact catalog;
# on the CPU: DescriptorNetwork(descriptor_network_path(); execution_provider = :cpu)
network = DescriptorNetwork(descriptor_network_path())

# the gallery: board descriptors, cached on disk (the first build takes a while)
gallery = BoardGallery(joinpath(tempdir(), "example_gallery.jld2"), network)
board_paths = filter(endswith(".json"), readdir(boards_dir; join = true))
board_features = Dict(insert!(gallery, board_paths, network))
board_bundles = Dict(b.uid => b.bundle for b in load_board.(board_paths))

img = Gray.(load(joinpath(still, "DSC02847.JPG")))

# uncalibrated: one homography per board
found = find_boards(board_features, network, img; board_bundles)
for (uid, H, quality, scored) in found
    println(uid, ": ", count(is_map_inlier, scored.weights), " inliers")
end
```

On this still, four boards are found. The pipeline needs the board bundles as
well as their features, because polarity, cutoff and each board's outline come
from the bundle.

The example boards are definitions only: JSON, no PDF or PNG. Building the
gallery needs each board's image, so it re-renders the stored board into a
cache (once per board; nothing beyond `using BlobBoards` is needed).

### The gallery image's density

A board has no pixel density, so the gallery converts it to pixels at the
board's own **raster density**: the whole-number dpi at which the board's
smallest disc is at least 16 pixels across (`min_blob_px` under
`[board_raster]` in `configs/authoring.toml`). That is fine enough to sample
the smallest blob well, and no finer.

| Beside the JSON | Gallery image |
|---|---|
| a PDF | the PDF converted at the raster density |
| nothing | the stored board rendered into a cache as its PDF, then converted at the raster density |
| a PNG | used as it is, at the density its sidecar records |

A board stored in schema v3 is laid out on its printer's pixels and is always
converted at that density. [`load_board`](@ref) reports it in its
`density` field, and takes `density` as a keyword to override the rule:

```julia
board = load_board("blob_board_<uid>.json")                             # board.density: the rule above
board = load_board("blob_board_<uid>.json"; density = 300dpi)    # the image, and canvas_frames, at 300 dpi
```

The override applies to a v4 board with a PDF or no file beside its JSON; a
stored raster and a v3 board each have one density, and another is an
`ArgumentError`.

## Calibrated pose or homography

The camera lives in the geometry configuration, `config.geometry`. That one
choice selects the estimator:

| `config.geometry` | Needs | `model` in each result |
|---|---|---|
| [`HomographyGeometry`](@ref)`()` (default) | nothing | homography: board `:world` plane (mm, z = 0) → image pixels |
| [`PoseGeometry`](@ref)`(camera)` | a calibrated `CameraModel` | board → camera pose: `model.R`, `model.t` (mm) |
| [`RadialHomographyGeometry`](@ref)`((width, height))` | the image size | homography plus a one-parameter division-model distortion, for an unknown camera whose lens bends lines |

```julia
camera = VgtkIO.load(CameraModel, joinpath(still, "cameras.json"))
config = FindBoardsConfig(geometry = PoseGeometry(camera))
posed = find_boards(board_features, network, img; board_bundles, config)

uid, pose, _, scored = first(posed)
pose.R, pose.t          # rotation and translation (mm), board world frame → camera frame
```

The pose path undistorts nothing by default. It detects on the raw image and
verifies through the full camera model (`PoseGeometry(camera; undistortion =
:none)`), so the inlier gate is in raw pixels everywhere on a strongly
distorted lens.

## Reading the results

Each result is `(uid, model, quality, scored)`:

- `uid`: the board's 3-character uid. It is printed in the board's corners and
  is a key of `board_bundles`. `blobs(board_bundles[uid], :world)` gives the
  board's points.
- `model`: the homography or pose (see the table above).
- `quality`: the fit's RANSAC score.
- `scored`: the board-blob → image-detection correspondences, with posterior
  weights in `scored.weights`. `count(is_map_inlier, scored.weights)` is the
  inlier count.

## Checking the result visually

Each board's model maps its `:world` plane into the image, so drawing the board
through it shows at a glance whether the fit is right. `board!` draws a board's
outline and blobs through any map, on any Makie backend:

```julia
using CairoMakie                      # or GLMakie, to open a window instead
using VisualGeometryToolkitCore: imshow, HomographyMap
using CoordinateTransformations: PerspectiveMap
using StaticArrays: SMatrix
using GeometryBasics: Rect

fig = Figure(size = (1400, 950))
ax, _ = imshow(fig[1, 1], img)
for (uid, H, quality, scored) in found
    bundle = board_bundles[uid]
    proj = PerspectiveMap() ∘ HomographyMap(SMatrix{3, 3, Float64}(H))   # board world (mm) → pixels
    board!(ax, proj, Rect(bundle, :world), blobs(bundle, :world); cutoff = cutoff(bundle))
end
CairoMakie.save("found_boards.png", fig)   # with GLMakie: display(fig)
```

`examples/pipeline/find_boards.jl` does the same for both the homography and
the pose path, with the inlier correspondences drawn. With a display it opens
GLMakie windows; on a headless machine it saves the figures as PNGs.

## Configuration

[`FindBoardsConfig`](@ref) holds every pipeline setting: detection
([`DetectConfig`](@ref)), adaptation, geometry, RANSAC bounds, the matching and
appearance-verification thresholds, and the tracker. Its defaults come from the
package's `configs/default.toml`. To record a run's configuration, keep it in a
TOML file and load it with the camera as data:

```julia
config = FindBoardsConfig("configs/default.toml")                      # homography
config = FindBoardsConfig("my_run.toml"; camera)                       # [find_boards.geometry] type = "pose"
```

To change one setting, rebuild the immutable config with `Accessors.@set`, as
shown in the [`FindBoardsConfig`](@ref) docstring.

## Video: `track_boards`

For video, [`track_boards`](@ref) carries each board from frame to frame:
motion-guided association, then a per-frame fit over the associated blobs. Full
discovery (the `find_boards` path) runs only on keyframes. It takes a video path
or any sequence of grey frames:

```julia
video = BlobBoards.example_data(:video)
clip = joinpath(video, "blobboards_handheld.mov")
tracks = track_boards(board_features, network, clip; board_bundles)
# Dict(uid => Vector{BoardDetection}), one detection per frame the board was found in
```

Unrelated stills belong in `find_boards` with a vector of frames instead. That
runs each frame independently and still stages the gallery only once.

## [The GPU environment](@id gpu-env)

Every stage of the pipeline is written against KernelAbstractions. The
`device` keyword picks the backend: `KA.CPU()` runs the threaded host code, and
`CUDABackend()` runs the device kernels. The default, `default_device()`,
picks the GPU when CUDA is loaded and functional, and the CPU otherwise. The
descriptor network runs in ONNX Runtime, on its CUDA execution provider or its
CPU one.

CUDA and cuDNN are **weak** dependencies of BlobBoards, and a package's own
project cannot load its weak dependencies. The checkout ships a companion environment, `gpu/`, that dev's the checkout and adds:

| Package | Activates |
|---------|-----------|
| CUDA | `BlobBoardsCUDAExt` (the device kernels; `default_device()` becomes the GPU) |
| cuDNN | ONNX Runtime's CUDA execution provider (needs CUDA **and** cuDNN loaded) |
| CUDA_Runtime_jll, pinned to 12.9 | ONNX Runtime's CUDA provider requires a 12.x runtime |
| GLMakie | interactive inspection: overlay windows (`examples/pipeline/find_boards.jl`) |
| FileIO, ImageIO, Colors, CairoMakie, GeometryBasics, … | what the example and training scripts load |

```bash
julia --project=gpu -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=gpu -t auto                             # GPU REPL
```

On a machine without a display (a compute node, SSH without X forwarding),
GLMakie cannot precompile, because its OpenGL setup needs a display, and
`Pkg.instantiate` reports it as one errored dependency. That is harmless:
nothing that runs headless loads GLMakie.

pixi adds only the Python interface (`pixi run -e py …`); CUDA comes from
Julia's own artifacts, so plain `julia --project=gpu` is all the GPU path needs.

### Requirements

- **An NVIDIA GPU of compute capability 7.5 (Turing) or newer**, the minimum
  of CUDA.jl 5.11, the version `gpu/` resolves to.
- **An NVIDIA driver for CUDA 12 or newer.** CUDA.jl requires it, and ONNX
  Runtime's CUDA provider (ONNX Runtime 1.20.1) needs a 12.x CUDA *runtime*,
  hence the 12.9 pin below. On some systems CUDA.jl can load a newer driver
  library for the session (CUDA forward compatibility); `CUDA.versioninfo()`
  says when it does.
- **Linux or Windows on x86-64.** These are the platforms CUDA.jl supports and
  for which ONNX Runtime.jl ships a GPU build. macOS has no CUDA: there the
  pipeline runs on the CPU only.
- **No CUDA toolkit to install.** CUDA.jl downloads the CUDA runtime, cuBLAS and
  cuDNN as Julia artifacts, and ONNXRunTime.jl its own GPU build of ONNX
  Runtime, so only the driver must be on the system.
- **GPU memory: about 6 GiB for a 12 MP image** (`gpu/` was checked on a
  180 GiB B200, where more fit). The descriptor network's ONNX Runtime arena
  takes up to 2 GiB (`DescriptorNetwork`'s default `max_batch = 64`). Detection
  keeps one image pyramid per tilt of the affine-simulation schedule on the
  GPU, about 2.8 GiB each at 12 MP, so it needs at least one, and it holds as
  many of the 25 as fit in 80% of the free memory. A bigger card is therefore
  faster, not required. The pyramids scale with the image's pixel count.
- **Measured on that B200**, for the example still (12 MP, 50 boards), after
  warm-up: `find_boards` took 1.3 s per image on the GPU against 16 s on the
  CPU (224 threads). Both found the same four boards, with poses agreeing to
  3·10⁻⁵ mm.

### Without a checkout

A registry install (`Pkg.add("BlobBoards")`) has no `gpu/` environment. Make
your own project with the triggers and the runtime pin. The pin must be in
*your* project, either as a `[preferences]` table in its `Project.toml` or as a
`LocalPreferences.toml` beside it:

```toml
# Project.toml of your environment
[deps]
BlobBoards = "93a5632c-6c4c-4c98-b3b2-221d1d9ee9e0"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[preferences.CUDA_Runtime_jll]
version = "12.9"
```

Or interactively:

```julia
using Pkg
Pkg.add(["BlobBoards", "CUDA", "cuDNN", "CUDA_Runtime_jll"])
using CUDA
CUDA.set_runtime_version!(v"12.9")   # writes the pin to LocalPreferences.toml; restart Julia
```

After the restart, the check below must report `:cuda`. Add FileIO, Colors
and ImageIO to load images.

The pin matters because the newest CUDA runtime CUDA.jl would otherwise pick
can be 13.x, and ONNX Runtime 1.20's CUDA provider accepts only 12.x, so
`DescriptorNetwork` would refuse to start on the GPU.

### From Python

The `blobboards` Python package runs Julia in the checkout's `authoring/`
environment, which lists CUDA, cuDNN and the pinned CUDA 12.9 runtime. Its
`find_boards` runs on the GPU when one is usable (`device="auto"`, the default)
and on the CPU otherwise, with a warning; each result records its `device` and
`execution_provider`, and `blobboards.device()` reports the choice up front.
See [GPU or CPU](python.md#GPU-or-CPU) in the Python guide.

### Making sure the GPU is used

Two things must both be true. Check them before trusting a timing:

```julia
using BlobBoards, CUDA, cuDNN
CUDA.functional()                     # true: a usable GPU and driver
BlobBoards.default_device()           # CUDABackend(): the kernels run on the GPU
network = DescriptorNetwork(descriptor_network_path())
network.session.execution_provider    # :cuda: the network runs on the GPU
```

By default `DescriptorNetwork` **throws** if the CUDA execution provider cannot
start. The CPU provider is about 34× slower on this network, so the constructor
refuses rather than report a misleading speed. The usual cause is `using CUDA`
without `cuDNN`; a full card at construction is another. To choose the CPU
deliberately, use `DescriptorNetwork(path; execution_provider = :cpu)`.
`allow_cpu_fallback = true` restores the old behaviour: warn and continue on the
CPU. With that option, `network.session.execution_provider` is the only reliable
way to see which provider is in use.

### CPU only

Without CUDA loaded, `default_device()` is the CPU and every stage runs there;
only the network needs `execution_provider = :cpu`, as in the example above. It
needs no `gpu/` environment, just patience. On the example still (12 MP, 50
boards), the CPU path took about 3 minutes to build the gallery once and 16 s
per image after warm-up (224 threads).
