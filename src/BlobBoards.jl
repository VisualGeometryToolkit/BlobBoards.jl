"""
    BlobBoards

Generate calibration boards with blob patterns for computer vision applications.

# Quick Start

```julia
using BlobBoards   # units (mm, px, dpi, …) are re-exported

# Generate a PDF board with standard paper size
bundle = blob_board("A4"; blob_sizes=(4.0mm, 0.2short_side), cutoff=2.0, seed=0x1234, dir="./output")

# Generate a pattern with logical dimensions
pattern, blobs, config = blob_pattern(800, 600; blob_sizes=(12px, nothing), cutoff=2.0, seed=0x1234)

# Extract blobs from saved files
bundle, _ = load("blob_board.json")
b = blobs(bundle, :pattern; units=mm)  # Physical units
b = blobs(bundle, :canvas; density = 300dpi)   # Pixel units, at a density
```

# Key Interface Notes

- **`blob_board`**: `blob_sizes = (min, max)` are printed disc **diameters** in **physical units**, both required (e.g., `suggest_blob_sizes(600dpi, "A4")`, or `(6mm, 38mm)`); stored as σ in points
- **`blob_pattern`**: `blob_sizes = (min, max)` are disc **diameters** in **logical units** (e.g., `(12px, nothing)`)
- **`blobs(bundle, reference_frame; units=px, density)`**: Extract blobs with specified frame and units

# Coordinate Specifications

BlobBoards uses three types of coordinate specifications:

- **Image origins** (`:colmap`, `:opencv`, `:julia`): Pixel coordinate conventions
- **Reference frames** (`:pattern`, `:board`, `:canvas`): Spatial origins on boards
- **Coordinate systems** (`:pixel`, `:physical`, `:world`): Unit systems for transforms

Blobs are stored internally in `:colmap` convention. Use
`EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})` to convert.

# Features

- Deterministic patterns from seeded configurations
- Physical units via FlexUnits (mm, inch, dpi, etc.)
- Printable boards (PDF) or training images (PNG) with ruler or blank frames
- JSON serialization for exact reproduction
- Blob extraction with multiple coordinate conventions and reference frames
- World coordinates for 3D pose workflows
- Visualization helpers for Makie backends
- Bit-exact reproducibility across Julia and Python interfaces
"""
module BlobBoards

import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO

# =============================================================================
# Dependencies
# =============================================================================

# Core Julia packages
using LinearAlgebra: LinearAlgebra, dot, norm, normalize
using Statistics: Statistics
# SpecialFunctions: loggamma consumed directly by the adaptation host path
# (src/adapt_ka.jl, which is where the `using` lives); the device kernels use
# their own Lanczos/libdevice lgamma
using Random: Random, MersenneTwister, RandomDevice, Xoshiro
import Artifacts   # descriptor-network catalog lookups (config_toml.jl)
import Pkg         # Pkg.Artifacts.ensure_artifact_installed — lazy artifact fetch
import VisualGeometryToolkitIO   # append_jsonl — run logs / telemetry (one JSONL writer)
using Scratch: @get_scratch!   # regenerated board-render cache (io.jl)
using Base: ImmutableDict

# Data structures and arrays
using StaticArrays: StaticArrays, @SMatrix, FieldVector, SMatrix, SVector
using StructArrays: StructArrays, StructArray

# Geometry and spatial indexing
using GeometryBasics:
    GeometryBasics, Circle, Point2, Rect, Rect2, origin, widths
using SpatialIndexing: SpatialIndexing   # RTree, SpatialElem, Rect, insert!, id
using VisualGeometryToolkitCore.SpatialQueries: foreach_intersecting, nearest_neighbor!, NearestNeighborState, bulk_load!

# Units and physical quantities: FlexUnits with VisualGeometryToolkitCore's pixel
# dimension (values in base units — name the unit at every `ustrip`)
using FlexUnits: Quantity, Units, dimension
using VisualGeometryToolkitCore:
    VisualGeometryToolkitCore, # for qualified access (e.g., VisualGeometryToolkitCore.estimate)
    px,
    dpi,
    pt,
    m, mm, cm, inch, ft, μm,
    ustrip,
    filter_kwargs,
    validate_dir,
    Disc, # Isotropic feature frame type
    project, # camera-aware world→image projection (verify.jl)
    EuclideanMap, PixelFrame,     # pixel-origin frame maps (0.7 successor to Translation(:sym,:sym))
    is_map_inlier,                # the posterior > ½ MAP inlier decision (every `weights` read)
    PlanarTarget                  # a board as a target model (board_target, pipeline.jl)

using VisualGeometryToolkitCore: Size2, parse_unit, strip_or_bare, unit_string, base_quantity

# Image processing and visualization
using Colors: Colors, Gray
import ConstructionBase   # setproperties: rewrite one config field, no restating
using FixedPointNumbers: Normed, N0f16, N0f8
import KernelAbstractions as KA   # KA devices are the backend dispatch tokens

# Serialization and I/O
using JSON: JSON             # every JSON file (src/json.jl: the writer, byte-pinned)
using Base.ScopedValues: ScopedValue, @with   # the unit declaration a file is read under; identity vs file lowering (json.jl)
using StructUtils: StructUtils   # `structlike(T) = false`: `JSON.parse(str, T)` lifts the whole object
using Preferences: @load_preference
using SHA: SHA, sha256
using Base64: base64encode, base64decode   # an ICC profile embedded in a sidecar
using FileIO: FileIO
using PNGFiles: PNGFiles      # a board's and a pattern's PNGs, read and written without a FileIO backend lookup
using Libtiff_jll: libtiff    # the PDF rasterizer's page (and a `Tiff` output)

using Rotations: Rotations
using CoordinateTransformations: CoordinateTransformations,
    AffineMap, LinearMap, Translation

# =============================================================================
# Constants
# =============================================================================

# Default pixel type of a generated pattern (`blob_pattern`); a board has none
# (its blobs are vector shadings, its TIFF/PNG 8-bit conversions of the PDF).
# N0f8 = 8-bit normalized fixed-point, 8 bits per channel
const DEFAULT_PIXEL_TYPE = Gray{N0f8}

# Outer radius (in σ) of the log-polar canonical patch: the descriptor's
# radial range is `inner_extent/outer_extent` (src/canonicalize.jl). Gallery and
# query patches MUST be sampled at the same value or their descriptors are not
# comparable — silently. Until the 2026-09-07 audit it was spelled `96.0` at
# five sites and settable on the gallery side only; this constant is the one
# authority: `BoardGallery` no longer takes it, and `describe_frames` /
# `extract_board_features` default their keyword to it.
"""
    DEFAULT_OUTER_EXTENT

The outer radius, in σ, of the log-polar canonical patch a descriptor is
computed on (the radial range is `inner_extent/outer_extent`, see
`canonicalize`). Gallery and query patches must use the same value or their
descriptors are not comparable, so this constant is the one authority:
`BoardGallery` records it, and `describe_frames` / `extract_board_features`
default to it.
"""
const DEFAULT_OUTER_EXTENT = 96.0

# NOTE: the descriptor checkpoint is deliberately NOT named here.
#
# This file used to carry `DEFAULT_DESCRIPTOR_MODEL`, plus a
# `descriptor_network_relpath()` reading a LocalPreferences key and an env var.
# That made a LIBRARY assert the filename of one lab's checkpoint: anyone
# installing BlobBoards got a default pointing at a file they had never had,
# and — because two drivers bypassed it with their own `BB_ONNX` — two paper
# artifacts silently ran a different network from the paper's main sweep.
#
# Which network a run uses is a property of the EXPERIMENT, not of the
# package and not of the machine. It is named in the run's TOML config and
# resolved by `descriptor_network_path(config)` (src/config_toml.jl). There is
# no default: a run that does not say which network it used cannot be
# reproduced.

# =============================================================================
# Data Paths - Resolve via LocalPreferences.toml (ENV override supported)
# =============================================================================

"""
    data_dir() -> String

Return the root data directory for BlobBoards test data and training videos.

Set in `LocalPreferences.toml`:
```toml
[BlobBoards]
data_dir = "/path/to/blobboards-data"
```

Override with `ENV["BLOBBOARDS_DATA"]` if needed.
"""
function data_dir()
    path = get(ENV, "BLOBBOARDS_DATA", nothing)
    if path === nothing
        path = @load_preference("data_dir", nothing)
    end
    path === nothing && error(
        """No data directory configured. Add to LocalPreferences.toml:

        [BlobBoards]
        data_dir = "/path/to/data"

        Or set ENV[\"BLOBBOARDS_DATA\"]."""
    )
    isdir(path) || error("BlobBoards data_dir does not exist: $path")
    return String(path)
end

# =============================================================================
# Module Structure - Julia-idiomatic flat organization
# =============================================================================

# Core includes in dependency order (flat, consolidated)
include("config_defaults.jl") # configs/default.toml -> the keyword defaults of every config struct below
include("json.jl")         # JSON field access and the one JSON writer
include("types.jl")        # Structs and their JSON lowering/lifting
include("coordinates.jl")  # Coordinate maps, reference frames, unit conversion
include("pattern.jl")      # Pattern subsystem (placement, rendering, API)
include("board_output.jl") # Pdf / Tiff / Png — what a board is written as
include("io.jl")           # File I/O: JSON sidecars + loading (figure export in authoring/)
include("migrate_boards.jl")   # the retired {value, unit} schemas: vgio_migrate's migrate_schema methods for boards and patterns
include("board_code.jl")   # board_code / parse_board_code — the definition a board's QR carries
include("authoring_stubs.jl")  # blob_board/rebuild_board/render_board docs (authoring/); board! etc. (MakieExt)
include("media.jl")        # media enumeration: walk/probe/group by resolution
include("device.jl")        # default_device() — GPU when one is functional
include("detect_config.jl")   # DetectConfig + Mxm/EllipseNms types
include("detect.jl")          # Blob detection pipeline (VGF KernelAbstractions ASIFT)
include("canonicalize.jl") # Canonical patch extraction
include("descriptor.jl")   # ONNX descriptor model wrapper
include("adapt_utils.jl")   # Shared adapt constants, helpers, result type
include("adapt.jl")          # Unified adapt interface (backend-generic via adapt_ka.jl)
include("adapt_ka.jl")       # KernelAbstractions adaptation — CPU/CUDA/Metal/ROCm
include("inspect.jl")        # Pipeline stage capture (viewer in BlobBoardsMakieExt)
include("api.jl")          # Public exports (keep as single surface)
include("pipeline.jl")
include("config_toml.jl")   # FindBoardsConfig from TOML — the run's single source of truth
include("verify.jl")       # board_quad + appearance verification (mandatory gate)
include("gallery.jl")      # BoardGallery — persistent board descriptor store
include("track_trace.jl")  # TrackFrameRecord/TrackTrace — capture/replay records
include("track.jl")        # track_boards — temporal board tracking
include("annotate.jl")     # annotate_blobs — GT-labeled patches from board detection
include("training.jl") # TrainingData struct and builders from annotations
include("authoring/Authoring.jl")   # board authoring: blob_board, rebuild_board, render_board, the PDF/X-4 writer
"""
    CONTRACT_GENERICS

BlobBoards' declared extension points: the functions another stack package may
add methods to (VisualGeometryToolkitCore's `contract_generic_violations`
checks it). CaliBlobs adds a `GeometryConfig` arm of its own and its run
configuration through them:

- [`geometry_tag`](@ref) and [`geometry_from_toml`](@ref): the TOML tag of a
  `GeometryConfig` defined outside this package, and its reader;
- [`config_dict`](@ref) and [`write_config`](@ref): a run configuration's TOML
  table, and its writer, for a configuration type defined outside this package.

(BlobBoards also adds methods of VisualGeometryToolkitIO's `migrate_schema`, the
extension point IO declares for the owner of a schema: src/migrate_boards.jl.)
"""
const CONTRACT_GENERICS = (geometry_tag, geometry_from_toml, config_dict, write_config)

# Training pipeline (Lux + Optimisers + Enzyme + Reactant + NNlib + cuDNN)
# lives in ext/BlobBoardsTrainingExt.jl — loads only when all those trigger
# packages are imported.

# =============================================================================
# FileIO Format Registration
# =============================================================================

function __init__()
end

end # module BlobBoards
