# =============================================================================
# Public API — the two entry points and their supporting vocabulary
# =============================================================================
#
# BlobBoards exposes TWO workflows to end users:
#
#   GENERATION:  blob_board / blob_pattern  →  bundle JSON + printable output
#   DETECTION:   find_boards               →  per-board (uid, model, quality,
#                                              csponds)  # weights in csponds.weights
#
# Everything else — the pipeline stages (detect_blobs, adapt, canonicalize,
# describe), verification internals, annotation/training tooling, and ruler
# plumbing — is intentionally NOT exported. It remains accessible as
# `BlobBoards.name` and is declared `public` below where it forms a
# supported-but-advanced surface.

# -----------------------------------------------------------------------------
# 1. GENERATION — boards and patterns
# -----------------------------------------------------------------------------
export blob_board, blob_pattern            # THE generation entry points
export short_side                          # `0.2short_side`: a blob size relative to the pattern's short side
export suggest_blob_sizes                                # a board's `blob_sizes` for a printer's density and the board's extent
export Pdf, Tiff, Png                      # what a board is written as: `blob_board(…; output)`, `render_board(bundle; output)`
public BoardOutput
public ShortSideFraction                  # what `x * short_side` builds
public calc_blob_scales                   # the σ ladder from min_scale to max_scale (both exact)
public board_code, parse_board_code       # a board's definition as its QR code's text, and back (checked)
public render_board                       # re-render a stored board (its recorded blobs) without placement
public rebuild_board                      # rebuild a stored board from its JSON, as printed
export PatternConfig, BoardConfig          # Seeded configs (bit-exact reproduction)
public save                                # Write a pattern bundle's JSON + PNG (`BlobBoards.save`: Makie exports a `save` too)
export Size2                               # Pattern/board dimensions
export blobs                               # Bundle accessor: blobs(bundle, frame; units=px)
export cutoff, pattern_config, board_config
export hash_config, uid_from_hash  # config → SHA-256 hash / 3-char UID
export px_per_mm, mm_per_px                # Print density ↔ scale conversion
export mm, cm, m, inch, ft, dpi, px, pt, μm  # Units (VGC's FlexUnits constants, re-exported)
# Pixel types — the leading positional argument of `blob_pattern(T, size)`,
# never a keyword (a board has none) — come from Colors/FixedPointNumbers
# directly (`using Colors: Gray`); the re-exports were deleted in the 0.23 API round.
export DEFAULT_CUTOFF
public DEFAULT_OUTER_EXTENT               # log-polar outer sampling radius (σ); gallery and query share it

# -----------------------------------------------------------------------------
# 2. DETECTION — finding boards in images
# -----------------------------------------------------------------------------
export find_boards                         # THE detection entry point
export FindBoardsConfig                    # All pipeline knobs in one struct
export GeometryConfig, HomographyGeometry, PoseGeometry, RadialHomographyGeometry  # config.geometry selects H vs P3P vs radial (H, k)
public geometry_from_toml, geometry_tag    # the TOML tag of a GeometryConfig defined outside this package
public CONTRACT_GENERICS                  # the functions another stack package may add methods to (geometry_tag, geometry_from_toml, config_dict, write_config)
export DetectConfig, MxmNms, EllipseNms
export AbstractDescriptorNetwork, DescriptorNetwork, LuxDescriptorNetwork
export BoardGallery                        # Cached per-board feature extraction
export load_board                          # Board JSON loading (cameras: VisualGeometryToolkitIO.load)
export board_points, board_library, board_resolver, board_target   # the design resolver: a board uid → its nominal plane points
export extract_board_features, extract_image_features  # Pre-extraction for multi-frame reuse
export Disc, EllipticFrame, BlobDetection  # Result/geometry types
export track_boards                        # video: temporally warm-started tracking
# Media enumeration (src/media.jl). The detector caches one scale-space plan
# per resolution (bounded LRU), so walk, probe, and process like-sized media
# together — see group_by_resolution's docstring for the ordering contract.
public walk_media, media_resolution, group_by_resolution, is_video

# Annotation-driver seam — the build_annotations drivers' entry points
# (de-underscored in the 0.23 API round).
public annotate_frame!, append_log, check_annotation_hashes   # the last two only write files: no `!`
public existing_ref_uids, is_source_processed, image_path, config_hash
public example_data            # example-data artifacts (:boards, :still, :video)
export TrackTrace                          # capture: track_boards(...; on_frame = f -> push!(trace, f)) → save_trace/load_trace
public TrackFrame                          # what track_boards' on_frame sink receives (record + image + configuration of record)
export save_trace, load_trace              # trace bundles (TrackTrace dirs; PipelineTrace files)
public PipelineTrace, PipelineCapture, trace_pipeline, inspect_pipeline   # pipeline-stage capture (+ its Attributed provenance) + viewer (src/inspect.jl)
export data_dir                            # LocalPreferences-backed data root
export write_config, read_run_config, config_dict, descriptor_network_path
export default_config_path
export default_device

# -----------------------------------------------------------------------------
# 3. VISUALIZATION — Makie recipes (methods land via the Makie/GLMakie exts)
# -----------------------------------------------------------------------------
# `board` / `board!` are 2D recipes (any Makie backend). The LScene-based
# `board3d`/`board3d!`/`BoardMesh` cluster (BlobBoardsGLMakieExt) was deleted in
# the 2026-09-07 audit: zero users in-repo or downstream since the 0.23 round.
export board, board!

"Plot a grid of accepted find_boards results. Requires a Makie backend."
function plot_boards end

"Overlay (uid, pose) pairs on one image axis — for poses reloaded from disk. Requires a Makie backend."
function plot_board_poses end

# --- the composable painter layer (Makie's `f!` convention: the caller owns
# the axis). Every `plot_*` wrapper above is a Figure plus one of these, so a
# caller can put two overlays in one figure without reaching into internals.
"Paint boards — accepted `find_boards` results or `(uid, model)` pairs — onto an existing axis. Requires a Makie backend."
function board_overlay! end

"Paint a result set's inlier ellipses onto an existing axis, batched into one `ellipses!` call. Requires a Makie backend."
function inliers! end

"Paint one `find_boards` result into a figure position: image, title, board overlay. Requires a Makie backend."
function board_panel! end

"Compare labelled `find_boards` result sets side by side, one panel each, on ONE shared palette. Requires a Makie backend."
function plot_comparison end

"Render one frame of a TrackTrace (detections, predictions, associations). Requires a Makie backend."
function plot_tracks end

"Scrub through a TrackTrace interactively. Requires GLMakie."
function replay end

# -----------------------------------------------------------------------------
# 4. ADVANCED / SUPPORTED-UNEXPORTED — use as `BlobBoards.name`
# -----------------------------------------------------------------------------
# Pipeline stages (find_boards composes these; useful standalone for research)
public detect_blobs, adapt, filter_blobs, canonicalize, describe
public AdaptConfig, TrackConfig
public Telemetry, with_tag           # experiment instrumentation sinks for find_boards (pipeline.jl)
public prepare_image_features       # detect-once seam (sweeps, annotate, track)
public find_board                          # single-board match+estimate
public match_boards                        # one image's features against the board gallery, global mutual-NN, keyed per board
public guided_search, detection_rtree      # a board's printed blobs transferred through a map and associated by IoU; the detection index it queries
public fit_rng                             # one board's RANSAC stream (config.ransac_seed ⊻ uid)
public BoardDetection                      # one board in one frame: track_boards' record
public GalleryIndex                        # the staged board gallery; the type of every `gpu_index` keyword
public board_palette                       # one stable color per board id (sorted) — every overlay uses it
public board_quad, radial_outline            # model → image board outline (verify.jl); the Makie overlay consumes them
public network_hash, patch_size, reserve_workspace!
public plot_boards               # grid of find_boards results (BlobBoardsMakieExt)
public plot_board_poses          # saved-pose overlay on one axis (BlobBoardsMakieExt)
public board_overlay!, inliers!, board_panel!   # the composable painters the wrappers are built from
public plot_comparison           # labelled result sets side by side on a shared palette
public plot_tracks, replay       # TrackTrace viewers (BlobBoardsMakieExt)
# Annotation / training-data tooling
public annotate_blobs, build_annotations, split_training_data
public load_annotations, load_cspond_sets, plot_annotations, summarize_annotations
# Bundle I/O internals
public load                                # BlobBoards.load — avoids FileIO.load clash
public load_bundle
public rasterize_pdf                       # a PDF page as pixels (pdftocairo): how every raster of a board is made
public write_tiff                          # the TIFF writer (libtiff, LZW) behind a `Tiff` output
public canonical_config_json               # the byte string identities are minted from
public PatternBundle, BoardBundle, BoardPreamble
public PRINT_SIZES
# Board authoring internals (consumed by BlobBoards.Authoring)
public RulerOrientation, hruler, vruler, RulerTick, RulerSpec
# Training pipeline — `public`, never exported. The two workflows named at the
# top of this file are the whole export surface; training tooling is reached as
# `BlobBoards.name`, or imported by name the way the other advanced entry
# points are (`using BlobBoards: build_tracks_for_sequences`).
public build_tracks_for_sequences, build_absolute_tracks
public merge_track_files, load_board_gallery, collect_media, save_board_anchors
public save_tracks                         # one sequence into a `.tracks` file (the default sink's writer)
public symmetric_iou_match
# Chart methods land via BlobBoardsMakieExt; `save_view_angle_figure` needs a
# backend that writes files (CairoMakie for headless runs).
public view_angle_profile, print_view_angle_profile
public view_angle_figure, save_view_angle_figure
# Result element types: `symmetric_iou_match -> Vector{TrainingPair}` and
# `view_angle_profile -> Vector{ViewAngleStat}`, same treatment as PatternBundle.
public TrainingPair, ViewAngleStat
# `.tracks` sequences are written by `save_tracks(path, tracks)` — its own name,
# not a method of `save`, whose other methods take the thing saved first.
# `.tracks` files are NOT wired into FileIO: `format"BlobBoardTracks"` was never
# registered (`__init__` is empty), so the FileIO.load/save hooks that claimed to
# read them went with `build_tracks` (2026-08-19). Read a `.tracks` file with
# h5py or HDF5.jl, or with the Python `blobboards.load_tracks`.

# =============================================================================
# Python Interface — provided by BlobBoardsPythonExt
# =============================================================================
# These are stubs (function declarations with no methods). Methods are
# defined in `ext/BlobBoardsPythonExt.jl` and become callable when the user
# loads PythonCall. Inside `pixi run`, JULIA_PYTHONCALL_EXE points PythonCall
# at pixi's Python (numpy + pint preinstalled); outside pixi the extension
# still loads but pyimport("numpy") will fail unless the user provides a
# Python with numpy themselves.

"""
    blob_pattern_python(width, height; blob_sizes, seed, cutoff, dir, …)

The Python package's `blobboards.blob_pattern`: [`blob_pattern`](@ref) for arguments as Python
objects — Python numbers or pint lengths (bare numbers: pixels) — returning Python dicts and numpy arrays, with every pixel
coordinate in the stated `image_origin` convention. A stub: its method is
defined by `BlobBoardsPythonExt`, which loads with PythonCall (the Python
package's Julia side); call it from Python, see docs/src/guide/python.md.
"""
function blob_pattern_python end
"""
    blob_board_python(canvas_size; blob_sizes, seed, margin, board_size, …, dir, format, density, …)

The Python package's `blobboards.blob_board`: [`blob_board`](@ref) for arguments as Python
objects — a paper name, or Python numbers / pint lengths (bare numbers: mm) — returning Python dicts and numpy arrays, with every pixel
coordinate in the stated `image_origin` convention. A stub: its method is
defined by `BlobBoardsPythonExt`, which loads with PythonCall (the Python
package's Julia side); call it from Python, see docs/src/guide/python.md.
"""
function blob_board_python end
"""
    blobs_python(json_path; units = "px", reference_frame = "pattern", image_origin = "colmap", density = nothing)

The Python package's `blobboards.blobs`: [`blobs`](@ref) for arguments as Python
objects — a pattern or board JSON path — returning Python dicts and numpy arrays, with every pixel
coordinate in the stated `image_origin` convention. A stub: its method is
defined by `BlobBoardsPythonExt`, which loads with PythonCall (the Python
package's Julia side); call it from Python, see docs/src/guide/python.md.
"""
function blobs_python end
"""
    detect_blobs_python(img; peak_threshold = 0.01, edge_threshold = 5.0, polarity = "dark", image_origin = "opencv")

The Python package's `blobboards.detect_blobs`: [`detect_blobs`](@ref) for arguments as Python
objects — a 2-D numpy image — returning Python dicts and numpy arrays, with every pixel
coordinate in the stated `image_origin` convention. A stub: its method is
defined by `BlobBoardsPythonExt`, which loads with PythonCall (the Python
package's Julia side); call it from Python, see docs/src/guide/python.md.
"""
function detect_blobs_python end
"""
    find_boards_python(board_paths, model_path, img; …)

The Python package's `blobboards.find_boards`: [`find_boards`](@ref) for arguments as Python
objects — board JSON paths, the descriptor ONNX path and a numpy image — returning Python dicts and numpy arrays, with every pixel
coordinate in the stated `image_origin` convention. A stub: its method is
defined by `BlobBoardsPythonExt`, which loads with PythonCall (the Python
package's Julia side); call it from Python, see docs/src/guide/python.md.
"""
function find_boards_python end
"""
    load_board_python(json_path; image_origin = "colmap", density = nothing)

The Python package's `blobboards.load_board`: [`load_board`](@ref) for arguments as Python
objects — a board JSON path — returning Python dicts and numpy arrays, with every pixel
coordinate in the stated `image_origin` convention. A stub: its method is
defined by `BlobBoardsPythonExt`, which loads with PythonCall (the Python
package's Julia side); call it from Python, see docs/src/guide/python.md.
"""
function load_board_python end
"""
    render_board_python(json_path; dir, format, density, icc_profile, id_label, shading_function, image_origin)

The Python package's `blobboards.render_board`: [`render_board`](@ref) for arguments as Python
objects — a board JSON path — returning Python dicts and numpy arrays, with every pixel
coordinate in the stated `image_origin` convention. A stub: its method is
defined by `BlobBoardsPythonExt`, which loads with PythonCall (the Python
package's Julia side); call it from Python, see docs/src/guide/python.md.
"""
function render_board_python end
"""
    track_boards_python(board_paths, model_path, frames; gallery_path, camera_json, device, image_origin)

The Python package's `blobboards.track_boards`: [`track_boards`](@ref) for arguments as Python
objects — board JSON paths, the descriptor ONNX path and numpy frames — returning Python dicts and numpy arrays, with every pixel
coordinate in the stated `image_origin` convention. A stub: its method is
defined by `BlobBoardsPythonExt`, which loads with PythonCall (the Python
package's Julia side); call it from Python, see docs/src/guide/python.md.
"""
function track_boards_python end

export blob_pattern_python, blob_board_python, blobs_python
export detect_blobs_python, find_boards_python
public load_board_python, render_board_python, track_boards_python
