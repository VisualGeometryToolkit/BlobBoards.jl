# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Setup (with pixi - recommended)
pixi install
pixi run julia --project=.

# Setup (standard Julia)
julia --project=. -e 'using Pkg; Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl"); Pkg.instantiate()'

# Once per clone: let git regenerate the Manifest whenever a pull, checkout or
# rebase moves Project.toml (compat bumps from a release cut elsewhere).
# Without this every such pull leaves the Manifest illegal. See .githooks/.
git config core.hooksPath .githooks

# Development mode with auto-reload
julia --project=. dev.jl

# Run tests
julia --project=. dev.jl test

# Clean precompilation cache
julia --project=. dev.jl clean

# Run examples
julia --project=. examples/generate/board.jl

# Anything that needs the GPU (headless-safe: no Makie) — use gpu/, not .
julia --project=gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=gpu -t auto examples/training/build_tracks.jl VIDEOS BOARDS MODEL.onnx

# The example scripts (GPU + GLMakie/CairoMakie) — use examples/
julia --project=examples examples/pipeline/find_boards.jl

# Board authoring is part of the package (no CairoMakie): --project=. works
julia --project=. examples/generate/board.jl
# authoring/ = the checkout + PythonCall (the Python package runs here) + CairoMakie/Revise
julia --project=authoring -e 'using Pkg; Pkg.instantiate()'
```

### One workspace, one Manifest

The repository is one Pkg workspace (root `[workspace]`: `test`, `docs`, `gpu`,
`examples`, `authoring`, `bmvc2026`). Each member keeps its own `Project.toml`
— its own deps, and `[sources] BlobBoards = {path = ".."}` — but all of them
resolve together into the ONE `Manifest.toml` at the root, which is not
committed (no member has a Manifest; `.gitignore` ignores them all). The test
deps live in `test/Project.toml` (no `[extras]`/`[targets]`), and `Pkg.test()`
runs that member without a sandbox, against the shared Manifest. After a pull
that moves any member's `Project.toml`, the `.githooks` re-resolve the root.

### `--project=.` has no CUDA — use `--project=gpu`

CUDA and cuDNN are `[weakdeps]` (they must be, or `BlobBoardsCUDAExt` could not
be an extension), and Julia refuses to load a package's own weakdeps from that
package's project env — `julia --project=. -e 'using CUDA'` is an
`ArgumentError`. The failure is quiet rather than fatal: `default_device()`
stays `KA.CPU()`, ONNXRunTime's CUDAExt never registers, and `DescriptorNetwork`
logs *"ONNX CUDA execution provider unavailable; falling back to CPU EP"* before
running the descriptor on the CPU.

Board authoring used to bite the same way (an extension triggered by
`using CairoMakie`); since 2026-10-14 it is the submodule `BlobBoards.Authoring`
(src/authoring/) and needs nothing extra — the sheet font is vendored in
assets/fonts, libtiff/zlib/FreeType are ordinary deps. `authoring/Project.toml`
remains the companion env for what `--project=.` cannot load: PythonCall (the
Python package activates it), CairoMakie for the Makie plots, Revise; it
resolves with the rest of the workspace.

`gpu/Project.toml` is the GPU env (renamed from `dev/` 2026-09-07; `dev/` is now
tooling only — `dev.jl` baselines, converters, archived logs, no Project.toml):
it dev's the checkout and lists CUDA, cuDNN
and CUDA_Runtime_jll as ordinary deps, and carries the runtime pin itself in a
`[preferences.CUDA_Runtime_jll]` block (12.9, the SAME pin as the root
Project.toml — ORT's CUDA provider takes 12.x only). The pin lives in `Project.toml` rather than a `LocalPreferences.toml`
because `.gitignore` excludes the latter. Use `--project=.` for CPU-only work;
`test/Project.toml` carries CUDA + cuDNN (and the same pin) of its own.

`examples/training/build_tracks.jl` (the `.tracks` builder CLI and its SLURM
drivers) runs under this env. `examples/Project.toml` is the desktop superset
(gpu + GLMakie + CairoMakie + every package the example scripts import) for
`examples/pipeline/*.jl` and `examples/generate/*.jl`; GLMakie cannot
precompile headless, which is why the two envs are separate. The paper-side annotation and training scripts
live in `bmvc2026/scripts/training/` — see that directory's README for the
annotation-vs-training-loop environment split.

## Architecture

BlobBoards generates deterministic multi-scale blob patterns and printable calibration boards for computer vision. All outputs are bit-exact reproducible via seeded configuration.

### Module Structure (Flat)

```
src/                        (include order)
├── BlobBoards.jl        # Entry point, includes all files in order
├── config_defaults.jl   # Defaults read from configs/default.toml (run) and configs/authoring.toml (boards, patterns), before every config struct
├── json.jl              # JSON field access and the one (byte-pinned) JSON writer
├── types.jl             # PatternConfig, BoardConfig, PatternBundle, BoardBundle
├── coordinates.jl       # Coordinate maps, reference frames, unit conversion
├── pattern.jl           # blob_pattern() — level-grid collision placement, blob rendering
├── board_output.jl      # Pdf / Tiff / Png — what a board is written as (the writer dispatches on it)
├── io.jl                # JSON serialization, board image loading (render cache, raster density)
├── board_code.jl        # board_code / parse_board_code — the board definition its QR carries
├── authoring_stubs.jl   # Stubs for BlobBoards.Authoring (blob_board, rebuild_board, render_board)
├── media.jl             # Media enumeration — directory tree → ordered frame work list
├── device.jl            # re-exports VisualGeometryToolkitCore.default_device (resolver lives upstream since 2026-09-07)
├── detect_config.jl     # DetectConfig + NMS strategies
├── detect.jl            # Blob detection pipeline (detect_blobs, ASIFT tilts)
├── canonicalize.jl      # Canonical patch extraction (canonicalize)
├── descriptor.jl        # Descriptor model abstract type + ONNX backend (Lux backend in ext/BlobBoardsLuxExt.jl)
├── adapt_utils.jl       # Shared adapt types + site accessors
├── adapt.jl             # Unified adapt interface (backend-generic via adapt_ka.jl)
├── adapt_ka.jl          # KernelAbstractions adaptation — CPU/CUDA/Metal/ROCm
├── inspect.jl           # Pipeline stage capture (trace_pipeline/save_trace)
├── api.jl               # Public exports (single surface)
├── pipeline.jl          # find_boards/find_board — agenda-driven multi-board pipeline; PoseGeometry.undistortion (:none raw detection verified through the FULL camera, the default | :global image warp + pinhole twin)
├── config_toml.jl       # FindBoardsConfig ↔ TOML — the single source of truth for a run
├── verify.jl            # Descriptor + spatial verification (board_quad)
├── gallery.jl           # BoardGallery — cached per-board feature extraction
├── track_trace.jl       # TrackTrace — capture/replay records for track_boards
├── track.jl             # Board tracking across frames
├── annotate.jl          # annotate_blobs — GT annotation for training data
├── training.jl          # .tracks builder — track generation from video + GT homographies
└── authoring/           # BlobBoards.Authoring: board, sheet, glyphs, vector_blobs, pdf, io_save, tiff — layout, frame marks, the PDF/X-4 writer, TIFF (libtiff)/PNG conversions; qrencode/ (vendored)

ext/
├── BlobBoardsPythonExt.jl    # *_python wrappers — fires on `using PythonCall`
├── BlobBoardsLuxExt.jl       # Lux descriptor inference (Lux + NNlib + cuDNN)
├── BlobBoardsTrainingExt.jl  # Training loop (adds Reactant + Enzyme + Optimisers + Zygote + TensorBoardLogger)
├── BlobBoardsCUDAExt.jl      # CUDA adapt/canonicalize/warp kernels + ORT device tensors
└── BlobBoardsMakieExt.jl     # Makie recipes: board overlay, inspection grid, TrackTrace viewers
```

`BlobBoardsLuxExt` is the inference-only Lux backend; loads with just
`using BlobBoards, Lux, NNlib, cuDNN`. `BlobBoardsTrainingExt` (Lux +
Optimisers + Enzyme + Reactant + NNlib + TensorBoardLogger + Zygote — NO
cuDNN) is independent of it: both include `ext/training/model.jl`, so the
architecture is one source file compiled in two extension modules. It used
to reach into LuxExt via `get_extension` behind a "LuxExt must load first"
assertion, on the false claim that its trigger set was a superset — and since
`Pkg.precompile` compiles an extension with exactly its own triggers,
TrainingExt could never precompile (found 2026-09-07 updating bb-slurm-env).
The split predates a cuDNN/Reactant load conflict (libReactantExtra.so
symbol mismatch — Reactant bundles its own cuDNN); on 2026-09-07 cuDNN 1.4.7
and Reactant 0.2.264 loaded together in one process without error, so the
conflict is historical, but the lean inference trigger set is still the
reason to keep the extensions separate.

The Python interface lives in `ext/BlobBoardsPythonExt.jl` (since 0.1.7).
Inside `pixi run -e py julia ...`, `JULIA_PYTHONCALL_EXE` (set by
`pixi.toml`'s `[feature.py.activation.env]`) points PythonCall at pixi's
Python — numpy + pint preinstalled. The DEFAULT pixi env carries
nothing: under it (`JULIA_PYTHONCALL_EXE` unset) `test/runtests.jl` SKIPS the
Python suite with a warning, so a green run there says nothing about the
Python interface — always test with `pixi run -e py`. Outside pixi, BB's
pure-Julia core works but the Python wrappers will fail at
`pyimport("numpy")` unless the user provides a numpy-equipped Python. There is no `CondaPkg.toml`; pixi is
authoritative.

### Key Type Hierarchy

- **Config objects** → **Bundles** → **Files**
  - `PatternConfig` + blobs → `PatternBundle` → `.json` + `.png`
  - `BoardConfig` + `PatternConfig` → `BoardPreamble` → `BoardBundle` → `.json` + `.pdf`

### Adaptation Strategies

`adapt(detections, img; config=AdaptConfig(), device=CUDABackend())` fits
elliptical frames to raw blob detections. **Adaptation is backend-generic**
(src/adapt_ka.jl, KernelAbstractions): `KA.CPU()` runs a per-blob threaded
host implementation of the same math; GPU backends run `cpu=false` KA
kernels. CUDA keeps two bit-exactness hooks in BlobBoardsCUDAExt — the
cuBLAS batched 6×6 LU (its getrf `info` decides which near-degenerate blobs
are invalidated) and the `__nv_lgamma` libdevice call; other backends get a
per-blob register LU and a pure-Julia Lanczos lgamma. Adapt output was
ALREADY nondeterministic run-to-run on CUDA (atomic pixel-slot assignment),
so cross-backend parity is frame-level with tolerances, never bitwise.

The per-blob solver is FIXED: the Student-t / uniform mixture IRLS with the
sort-and-sweep posterior partition, whitened normal equations. It used to be
selectable via `AdaptConfig.solve` (whitened_linear / linear / hq); that field
is gone. Two of the three were inert — `_adapt` dispatched on
`AdaptConfig{T}` alone, so `linear` never reached a kernel — and the
HQ-Welsch arm (`src/adapt_hq.jl`, since deleted) measured worse than the
mixture.

Adaptation precision: accumulation and the 6×6 solve are `Float64` wherever
the backend supports it (`KA.supports_float64`), `Float32` on Metal — a
distinct numeric regime; per-pixel staging is pinned Float32.
(`AdaptConfig.store` was DELETED 2026-09 — inert since the KA port.)

Seeding uses `Paraboloid3D(disc, z_max)` + `coeffs(p)`.

Whitened radius is called `r_whitened`.

### Descriptor inference — use the `_cuda` ONNX variant

ONNXRuntime's CUDA EP has no kernel for `Pad` in `wrap`/`edge` mode. The
as-exported log-polar network uses circular padding, so all ten `Pad` nodes
fell back to the CPU EP, cutting the graph into CPU islands with 16 `Memcpy`
nodes — a flat ~0.85 ms/patch that batching could not improve, and 93% of
frame time. `bmvc2026/scripts/onnx_depad.py` rewrites those `Pad`s as
`Slice`+`Concat` (CUDA-supported, semantically identical); descriptors are
bit-identical and the graph places entirely on the GPU. The current network
of record is `models/track_logpolar_circ_s96_iter4.onnx` (RealFourthIteration,
2026-08-10): it exports with ZERO Pad nodes so it needs no depad, runs at the
same µs/patch, and is measurably stronger on distant small boards than the
older depadded `_cuda` export. See
`bmvc2026/_research/findings/descriptor_inference_speedup.md`.

Consequence for `max_batch`: with the graph GPU-resident, **bigger batches are
slower**. The ORT arena shares a 24 GB card with the ~13.5 GiB detector plan,
and past ~64 the allocator, not the convolutions, sets the pace (64 →
~65 µs/patch, 256 → ~80, 512 → OOM). Both defaults are 64. Batching is
numerically exact here — each patch is normalized independently and the export
folded away every BatchNorm — so it is purely a throughput knob.

### Student-t mixture — the sweep is on-device now

The adaptation's sort-and-sweep posterior seed no longer calls RVG's
host-side `sweep!`/`ScoringWorkspace` (those imports were removed): each
blob's sweep runs inside the per-blob device kernel (bitonic sort of r² in
shared memory → outlier_halfwidth median → marginal sweep, `_sort_sweep_ka!`).
The device formula is pinned to RANSACScoringDoneRight's Jeffreys
`_sweep_score` algebra — `S = lgamma(j/2) − (j/2)·log(π·RSS) − (n−k)·log_two_outlier_halfwidth`
is RSDR's score at `alpha0 = beta0 = 0`. RVG/RSDR ship no generic device
sweep (RSDR's KA files are eval-harness drivers), so this hand-rolled kernel
is the sole device implementation and is intentional.

The CPU `posterior_adapt.jl` implementation that extended RVG's `irls!` hooks
was removed: that driver is now VGC's `block_coordinate_descent!`.
The KA adapt path provides its own per-blob CPU implementation of the same
math (Threads.@threads, plain sort — no bitonic needed).

## Unit System

Units are FlexUnits with VisualGeometryToolkitCore's pixel dimension, checked at run
time (quantities are FlexUnits `Quantity`s; BlobBoards re-exports `mm`, `inch`, `px`,
`dpi`, …). Lengths are held in metres, so strip in a named unit
(`ustrip(mm, q)`, `ustrip(px, q)`); a bare `ustrip` of a length is metres. The
project was ported off Unitful and DynamicQuantities — never import units
from another package:

- **Logical units**: `px` (pixels)
- **Physical units**: `mm`, `inch`, `cm`
- **Density**: `dpi` (dots per inch)

Critical distinction:
- `blob_pattern()`: `blob_sizes` are disc **diameters** (2·cutoff·σ) in **logical** units (e.g., `(12px, nothing)`)
- `blob_board()`: `blob_sizes` are printed disc **diameters** in **physical** units, BOTH ends REQUIRED (no default, and no `nothing` max): `suggest_blob_sizes(600dpi, "A4")` — `suggest_blob_sizes(density, extent)` returns `(smallest, largest)`: the disc that prints well on a printer (3 mm × 1200/dpi), and 20% of the board's shorter side, at most 64 mm (`extent`: a paper name — the default board on it, less 10 mm margins — a `Size2` or a `(w, h)` tuple); a plain length or `0.2short_side` (of the PATTERN's short side, no cap) is taken as given; stored as σ in **points** (`min_scale`/`max_scale`). A board (identity schema v4: the point grid) is laid out on a fixed 1/7200-inch grid (`PointGrid`, 0.01 pt — sized for 6 significant digits below 9999.99 pt, the precision of the Ghostscript writer it was introduced with) and has NO density: `blob_board("A4"; blob_sizes, dir)` takes none, and the PDF mentions none. A density belongs to a raster OUTPUT only — `output = Pdf()` (default) | `Tiff(600dpi)` | `Png(300dpi)` (src/board_output.jl; `render_board(bundle; dir, output)`), which must give the smallest blob ≥ 2 px per σ — and to pixel coordinates you ask for (`blobs(…; density)`, `AffineMap(…; density)` — every pixel density in the API is called `density`). The detector's image of a v4 board is rendered at the board's own raster density (smallest disc 16 px across, `[board_raster]` in configs/authoring.toml; `load_board(path; density)` overrides). Placement keeps SHAPES apart in one per-level grid of `Union{Circle, Rect2}`: each blob's footprint `Circle(disc, cutoff)` and the keep-out `Rect2`s (`_keepouts`) as the first level, checked first (A/B 2026-10-14: 0.83–0.91× a circles-only grid on ruler boards) — the frame's marks are switched by their sizes — `corner_size` (corner cells: QR, uid; blobs keep `CORNER_CLEARANCE` off them) and `ruler_width` (ruler bands), 10 mm each, a zero length turning one off (no `frame_style`/`frame_width` on a v4 board); the circle test was measured bit-identical to the old σ form on every stored board. Stored v3 boards keep `PixelGrid(density)` (JSON key still `print_density`), render exactly as printed (`test/fixtures/v3_render`) and rasterize at that density only.

## Coordinate Systems

### Image Origin Conventions
- `:julia` - pixel center at (1, 1)
- `:colmap` - pixel center at (0.5, 0.5)
- `:opencv` - pixel center at (0, 0)

### Reference Frames (for boards)
- `:pattern` - origin at pattern top-left (excludes borders; on a v4 board the pattern is the whole board, so it coincides with `:board`)
- `:board` - origin at board top-left (includes borders, negative coords possible)
- `:canvas` - origin at page top-left (includes margins)

Both `image_origin` and `reference_frame` are **positional arguments with no defaults** - users must explicitly choose.

## Serialization

JSON.jl 1.x: each serialized type has `JSON.lower(::JSON.JSONStyle, x)` /
`JSON.lift(::Type{T}, obj)` and `StructUtils.structlike(::Type{T}) = false`
(src/types.jl; VisualGeometryToolkitFeatures does the same for `Size2` and
quantities). Configuration captures everything needed for bit-exact
reproduction. Board identities are SHA-256 over a canonical encoding
independent of any JSON library (`canonical_config_json`, VisualGeometryToolkitIO's
`canonical_json`).

Every JSON file is written by ONE writer, `_write_json` (src/json.jl), which
reproduces byte for byte what the retired JSON3 writer (`JSON3.pretty`) wrote,
quirks included (integral floats printed as integers except late in an all-float
array; two-line empty containers). `test/suites/identity_pins.jl` pins board
uids, the annotation config hash, the render-cache key and the saved bytes of
patterns, point-grid boards and the stored pixel-grid fixtures; a serializer
change that moves a uid or an identity is a bug, never a re-pin.

### Board and pattern files: board v5, pattern v3

A board file is `blobboards/board/v5` and a pattern file `blobboards/pattern/v3`
(2026-10-27): every quantity a NUMBER in the unit the file's `units` block
declares (VisualGeometryToolkitIO's `UnitContract`: `world` mm, `density` dpi,
the length role `pattern` pt on a point-grid board, pixels undeclared). Nothing
reads the earlier `{value, unit}` schemas (board v1–v4, pattern v1–v2): they are
MIGRATED once by `VgtkIO.vgio_migrate(path)`, through the `migrate_schema`
methods of src/migrate_boards.jl, which hold the retired reader (the one place
that parses a `{value, unit}` object). A migration keeps the stored
`hashes.config`. IO's command knows these migrations only once BlobBoards is
loaded: `julia --project=. -e 'using BlobBoards; import VisualGeometryToolkitIO as
VgtkIO; include(joinpath(pkgdir(VgtkIO), "bin", "vgio_migrate.jl"))' tree --apply DIR`.

Two forms of the same lowering, never mixed (src/json.jl): a FILE holds numbers,
and a board's IDENTITY is minted over a canonical tree in which each quantity is
the object `{value, unit}` (`_IDENTITY`, set by `_plain`) — frozen, because it is
what every printed uid was minted from. The file format version is not in the
identity: `IDENTITY_SCHEMA` ("blobboards/board/v4", frozen) is the tree's schema
field and the source of the QR code's `BB4` tag; `BOARD_SCHEMA` is the file's.
A change to the file format therefore moves no uid and no printed QR code; a
change to what a configuration places is a new IDENTITY schema and a new tag.
The memory ↔ number conversion is ONE pair, `_file_number`/`_file_quantity`
(src/json.jl), for every file number including the point grid's pt values and
the blob columns, and it is TEMPORARY: it is `ustrip(unit, q)` and
`number * unit` because IO's decimal-exact `write_quantity`/`read_quantity` is
not exact for a quantity built as `v * unit` (a board is built from literals
such as `9.0mm`; IO would write `9.000000000000002`). When IO's writer writes `v`
back for `v * unit`, the pair is replaced by IO's under the file's
`UnitContract` and nothing else changes. The file declares its unit once: the
`units` block (blobs carry none).

## GLMakie off the main thread — thread 1

GLFW requires thread 1 for GLMakie module init AND every `display(fig)`.
A REPL backend that evaluates on a worker thread (agent or editor tooling,
`Threads.@spawn`-ed evals) cannot display — a worker-thread `display` kills the
session, usually silently.

**Compute in the worker; plot in a plain REPL.** `trace_pipeline` +
`save_trace` (src/inspect.jl) capture pipeline stages on the GPU;
`load_trace` + `inspect_pipeline` render them in a plain `julia -i` process.
If you must plot from such a worker, route through
`REPL.eval_on_backend(Meta.parseall(code), Base.active_repl.backendref)`.

**Never call `GLMakie.closeall()` from such a worker** — window teardown
kills the session even when `display` via the backend worked. Open a new
`GLMakie.Screen` per figure and close windows by hand instead.

Black (unrendered) windows are separate: use
`GLMakie.activate!(render_on_demand = false)`.

## Makie Image Display

When displaying images with `image!` in Makie, always use:
```julia
ax = Axis(fig[1,1], aspect=DataAspect(), yreversed=true)
image!(ax, rotr90(img))  # or transpose(img)
```

- **`rotr90(img)`** or **`transpose(img)`**: Required because Julia arrays are column-major
- **`yreversed=true`**: Image convention has y=0 at top, Makie has y=0 at bottom
- **`DataAspect()`**: Preserves pixel aspect ratio

For SpecApi scenes with vector graphics in image coordinates, use `yreversed=true` in scenekw:
```julia
scenekw=(; camera=campixel!, yreversed=true)
```

## External Dependencies

- **None for the print master**: `src/authoring/pdf.jl` writes the PDF/X-4 master directly (objects, Flate via Zlib_jll, the text set in the embedded sheet font, TeX Gyre Heros — searchable; no image); Ghostscript and qpdf were removed 2026-10-13, with `pdf_standard` `:x3`/`:x1a`/`:none`. An X-4 master is also PDF/A-2b, so veraPDF (`flatpak run --command=verapdf org.verapdf.veraPDF --flavour 2b`; no PDF/X profile) validates it; the suite runs it when installed
- **Poppler** (`Poppler_jll`): `pdftocairo` converts the master to its TIFF/PNG and gallery rasters
- **GLMakie/CairoMakie**: Visualization and figure export
- **QR encoding**: the board QR corner uses the QR encoder of QRCoders.jl 1.4.5, vendored in `src/authoring/qrencode/` (declared in `VENDORED.toml`; no QRCoders dependency)
- **SpatialIndexing** (General 0.1.6): the detection R-tree; its allocation-free queries (`foreach_intersecting`, `nearest_neighbor!`) and in-place bulk load (`bulk_load!`) come from `VisualGeometryToolkitCore.SpatialQueries`

## Code Conventions

- Functions: `snake_case` (e.g., `blob_pattern`, `logical_blobs`)
- Types: `CamelCase` (e.g., `PatternConfig`, `BoardBundle`)
- **Acronyms in types**: treat as words, not all-caps (e.g., `IrlsLo` not `IRLSLO`, `SvdWorkspace` not `SVDWorkspace`, `GnSolveType` not `GNSolveType`). This prevents unreadable compound acronyms like `IRLSLO`.
- Constants: `UPPERCASE_SNAKE_CASE` (e.g., `PRINT_SIZES`, `DEFAULT_PIXEL_TYPE`)
- Section headers: `# ===...===` dividers

### Argument Order

**Julia's style-guide order**, in this priority:

1. a function argument (so `do` blocks work)
2. an IO stream — an RNG takes this slot
3. the input(s) being mutated
4. a type
5. the other, non-mutated inputs — the model or tool first (below)
6. a key, 7. a value, 8. an `else`, 9. varargs
10. keywords

Consequences:

- **`!` means a Julia OBJECT argument is modified**, and the mutated inputs
  lead (after a function/IO/RNG argument): `_run_matching_phase!(gallery_pool,
  all_accepted, feat_pool)`, `annotate_frame!(refs_inserted, output_dir,
  gallery, network, uid_to_path, frame_path, source_id, img)`. Writing FILES
  alone takes no `!` — Base's `write`, `mkpath`, `rm`, `cp` —
  `append_log(output_dir, meta)`, `check_annotation_hashes(output_dir, m_hash,
  c_hash)`. A path is not a mutated argument.
- **A keyword OUTPUT the caller reads back is RETURNED** (as a field of the
  result the function already returns) **or delivered through a callback**,
  never a vector/object keyword the function fills in. When returning would
  make the return type depend on a keyword, use a sink function:
  `find_board(…; on_guided_read = j -> push!(deps, j))`,
  `build_absolute_tracks(…; on_view_stats = stats -> append!(collected, stats))`,
  `track_boards(…; on_frame = f -> push!(trace, f))` (a `TrackFrame` carries
  the record, the pixels and the configuration of record). A callback is not
  mutation — the caller's closure decides what to do. A SCRATCH keyword whose
  contents do not matter afterwards is exempt (Base: `sort(v; scratch)`).
- **Every type argument is positional and leads the non-mutated inputs; never
  a keyword**: `blob_pattern(Gray{N0f16}, size)`,
  `undistort!(dst, Float32, camera, src)`, `relabel(Src, Dst, x)`,
  `VisualGeometryToolkitIO.save(PixelFrame{:julia}, path, obs)`. Supplying `pixel_type =`
  to a generation entry point is an `ArgumentError` naming the positional form.
  A config STRUCT keeps its type-valued field (`PatternConfig(; pixel_type)`) —
  that is data, not an argument.
- **An RNG leads** (slot 2).
- **Among the non-mutated inputs, the model or tool leads** (the stack's
  `conventions/signatures.md`: the model or shape first, then the data it
  acts on or is measured against, then parameters):
  - a given model, in task verbs too:
    `find_board(camera, board_features, image_features)`,
    `board_quad(camera, model, bundle)`, `symmetric_iou_match(H_ij, frames_i,
    frames_j)`, `trace_pipeline(cam, img)`;
  - a required tool; a defaulted policy stays last: the gallery and the network
    before the images — `find_boards(board_feature_dict, descriptor_network,
    img)`, `track_boards(…, video)`,
    `annotate_blobs(…, img)`, `prepare_image_features(network, img)`,
    `extract_board_features(model, board)`, `describe_frames(model, frames,
    img)`, `load_board_gallery(descriptor_model_path, board_dir)`,
    `build_tracks_for_sequences(board_dir, descriptor_model_path, video_dir)`.
    Several leading tools keep their relative order (gallery, then network);
  - shapes lead the image they are sampled from — and the image a plot draws
    on comes last: `adapt(detections, img)`, `canonicalize(frames, img)`,
    `plot_boards(camera, results, board_bundles, img)`,
    `plot_board_poses(camera, poses, board_bundles, img)`.
- A method on a function owned elsewhere keeps the owner's order
  (`Base.insert!(gallery, path, model)`, `Base.get!(f, cache, key)`); a writer
  whose arguments do not fit the owner's shape gets its own name
  (`save_tracks(path, tracks)`, not a `save` method).

**A reorder or rename must make the old call fail LOUDLY and add no
ambiguity.** An untyped argument lets a call in the retired order bind with
every value in the wrong slot, so type a slot with the narrowest type every
caller satisfies (`AbstractDict` for a gallery, `AbstractMatrix` for an image,
`AbstractVector` for detections and frames, `CameraModel` for a camera, `Set`
for a mutated uid set) and check no ambiguity appears among same-named
methods. When every argument is a string, check what each must be before any
work and throw an `ArgumentError` naming the order (the descriptor model is a
FILE: `_check_model_file`). A removed keyword is already loud (unsupported
keyword); a renamed function leaves the old name undefined.
`test/suites/argument_order.jl` pins the retired forms with `applicable`,
`hasmethod` (with keyword names) and `isdefined`.

### Stack Conventions

The conventions every stack package follows — naming and the estimator verbs,
signatures, data arguments, coordinate frames and normalization ("frame in =
frame out"), pixel origins, bearings, Jacobians, units and the stack
contracts — live ONCE, in VisualGeometryToolkitCore's `conventions/` folder
(`conventions/README.md` indexes it). Read them before changing code here.
This file carries no copy of them, only how they land in this package. Where
the folder is:

- in a checkout of the stack's monorepo: `../VisualGeometryToolkitCore/conventions/`
  (the root CLAUDE.md imports it);
- in VisualGeometryToolkitCore's own repository: `conventions/`;
- from Julia, the installed copy: `VisualGeometryToolkitCore.conventions_dir()`, i.e.
  `joinpath(pkgdir(VisualGeometryToolkitCore), "conventions")`.

The agent-docs test checks that this file points there and holds no copy.

### Coordinate Frames and Normalization

The rule is the stack's (`conventions/frames.md`); how it lands here:

**In this package:**

Task verbs forward COLMAP pixels and pixel thresholds to VisualGeometryToolkitEstimation
untouched and read results back in pixels. BlobBoards builds no normalization.
`image_size` and `center` are sensor data (the distortion centre), never
described as a normalization. The Python bindings state the pixel convention of
everything they return.

### Correspondence Nomenclature

**A feature correspondence is a `cspond`** — one word, in identifiers,
comments and docstrings alike. `pairs` is BANNED for them: it says nothing
(a `(uid, model)` tuple, an `(R, t)` pose, `grid_size = [gw, gh]` and
`Base.pairs` are all "pairs"), and it let the same object be called three
different things across files.

| term | what it is |
|------|------------|
| `matches` | the RAW descriptor matches, BEFORE verification (VisualGeometryToolkitFeatures `match`) |
| `csponds` | feature correspondences: the verified, scored, weighted set the estimators consume |

The distinction is a real pipeline stage, so both names stay — but nothing
else may name either of them.

Singular is `cspond`. "pair" survives ONLY where the thing genuinely is an
arbitrary 2-tuple and is not a correspondence — `(R, t)`, `grid_size`, a
MetaGraphsNext edge label, `Base.pairs` — and as a VERB ("a false board's
inliers pair blobs with unrelated detections").

## Git Commit Conventions

- **NEVER** add "Co-Authored-By: Claude" or similar AI attribution to commit messages
- **NEVER** add "🤖 Generated with Claude Code" or similar footers
- Write clean, concise commit messages focused on the change itself
