# Changelog

## 0.7.0 (unreleased)

On the VisualGeometryToolkit wave-10 stack: VisualGeometryToolkitCore 0.10, IO 0.10, Features 0.9,
Estimation 0.11 (bmvc2026 also Eval 0.8), with LocalOptimizationDoneRight 0.7. Breaking:

- **The stack's terminology** (Core 0.10: detection, correspondence, observation; `check_naming` refuses the old
  words). No alias for any old name:
  - `load_cspond_sets` → `load_correspondence_sets`; it returns Features' `CorrespondenceSet` (was `CspondSet`):
    per board blob, its reference patch and its observations in the camera images (a track).
  - `find_boards`' fourth element is Core's `Correspondences` set (`ScoredCorrespondence` rows, was
    `Csponds`/`ScoredCspond`); its content is unchanged.
  - Python: a board's dictionary key `n_csponds` → `n_correspondences` (`find_boards`, `BlobBoards.find_boards`
    through PythonCall); the `blobboards` Python package is 0.2.0.
  - "measurement" is out of every name and of the prose (observation, or the experiment it meant); the
    stack's observation-noise convention is `conventions/observation-noise.md`.
- **A pose model projects through its posed camera** (`PosedCamera(model(camera), pose)`, Core's `plane_map`) on
  both pose arms. Under `undistortion = :global` the guided association now applies the same cheirality test as
  under `:none` (Core's `transfer_ellipse(::PosedCamera, _)`): a board blob behind the camera transfers nothing,
  where the bare plane homography imaged it as a (spurious) ellipse. In front of the camera the transfer is
  bit-identical.
- **`BoardDetection` → `BoardObservation`** (no alias): one board observed in one frame, its fitted model with
  quality and inlier count; `track_boards` returns each board's track, `Dict(uid => Vector{BoardObservation})`.
- **`adapt` fits a blob with 10–19 usable samples isotropically** (one curvature, θ₄ = θ₆, θ₅ = 0) instead of refusing
  it: the 4-parameter path solved θ₁…θ₄ against a shape gate that read θ₅ = θ₆ = 0, so every such blob came back
  `:not_peak` at its seed (`InitialFailure`). Fewer than 10 samples stays invalid. Small, faint and clipped blobs are
  now adapted; adapted frames and `count_rejected` move accordingly.
- **`find_boards`' `max_detections` is `detect_blobs`' cap** (Features' `spread_strongest`: about that many of the
  strongest, spread over the image, applied per polarity group), not the top-N by |response|. With a finite cap
  (the default is unbounded) the kept detections, and so the boards found and `build_tracks`' training data, change.
- bmvc2026: the motion-capture markers are written as `TargetObservations{WorldFrame}` (IO's `observations` kind;
  `TargetPoints3D` and `vgio/target_points3d` are retired by Core 0.10 and IO 0.10); its config of record's
  `formats = "io-0.10"`, so the run key moves and the records are regenerated.

Not breaking:

- CompatHelper runs on the root project only and no longer auto-merges: Pkg's writer deleted the workspace members'
  comments (authoring/'s header and the PythonCall pin's reason, restored here). Every member's `BlobBoards` compat
  is stated (`"0.7"`); test/'s CUDA compat is the root's (`"5.11.0"`; the workspace resolves by the root).
- test/runtests.jl takes item-name patterns (`julia --project=test test/runtests.jl <regex>…`,
  `Pkg.test(; test_args = […])`), and sizes its worker pool from `RETESTITEMS_NWORKERS` /
  `RETESTITEMS_NWORKER_THREADS` when set.

## 0.6.2 (unreleased)

A patch: requires Julia 1.13 (`julia = "1.13"` in the `[compat]` of the package and of every
workspace member: test, docs, gpu, examples, authoring and bmvc2026). CI runs Julia 1.13 only. No API change.

- Compat floors raised to the versions the release's gated environment resolves (latest only), in the package and
  every workspace member; the Downgrade workflow (the suite at the old floors) is deleted.

## 0.6.1 (2026-11-13)

A patch: no API change, the same results. On VisualGeometryToolkitEstimation 0.10 (`[compat]`
VisualGeometryToolkitEstimation "0.10" in the package, test and bmvc2026 environments). Estimation 0.10.0 retired the image-space refine of
a free SplineLens camera's controls (`refine_camera_and_poses` on a SplineLens camera now takes `fixed = (:controls,)`
only); BlobBoards never called that route, so nothing here changes.

## 0.6.0 (2026-11-12)

On the VisualGeometryToolkit 0.9 stack, from General: VisualGeometryToolkitCore 0.9.3, IO and
Estimation 0.9, Features 0.8 (bmvc2026 also Eval 0.7), with LocalOptimizationDoneRight 0.6
(its 0.9 port). Breaking:

- **The pose arm takes a `BoundedCamera`** (the stack's rule: pixel operations
  take a BoundedCamera). `PoseGeometry(camera)`, `FindBoardsConfig(path; camera)`,
  `board_quad`, `trace_pipeline` and the Makie overlays take a `BoundedCamera`;
  a bare `CameraModel` is refused with an `ArgumentError` naming
  `BoundedCamera(camera, Size2(width, height))` and
  `VisualGeometryToolkitIO.load(BoundedCamera, path)` (a camera store records
  the extent). The estimators are handed the model; nothing else moves.

- **`read_run_config` (and `FindBoardsConfig(path)`) admit a top-level
  `[inputs]` table**, unchecked like `[experiment]` and `[gallery]`: the paper
  projects' `[inputs.<id>]` statement of what a run reads (tier, path,
  provenance, sha256; VisualGeometryToolkitEval's convention).

- **The geometry's strategy is `local_optimization_strategy`** (the stack's 0.9
  rename: a `find_*` takes `local_optimization_strategy`, a `refine_*` and a
  `fit_*` take `optimization_strategy`). The `[find_boards.geometry]` key, the
  `HomographyGeometry`/`PoseGeometry`/`RadialHomographyGeometry` keyword and
  field, and the public reader (`geometry_local_optimization_strategy`, was
  `geometry_local_optimization`) are renamed. The old key is refused with an
  `ArgumentError` naming the new one; there is no alias. The bmvc2026 configs of
  record use the new key, so their run keys move. A Welsch-HQ table's width is
  `efficiency_constant` (was `scale_multiple`; Welsch's value is unchanged).

- **`AdaptConfig`'s `max_irls_iter` default is 100** (was 20; `rtol` stays
  1e-6). At 20 most blobs were still moving when the budget ran out (21 of 30
  on the suite's synthetic field) and were returned as if converged; at 100
  they converge. Adapted frames move accordingly; the bmvc2026 sets of record
  take 100 too.
- **`adapt` records every blob's outcome.** `BlobDetection`'s metadata is
  `(; id, retcode, stop_rule, rejection)`, in
  VisualGeometryToolkitCore's termination record (`retcode::ReturnCode.T`,
  `stop_rule::StopRule`; a blob's convergence is `successful_retcode(retcode)`,
  §14, not a field of its own): converged at `rtol` (`Success`, `IncrementSize`: VisualGeometryToolkitEstimation's `irls_termination`, the reweighted fits' one record),
  capped at `max_irls_iter` (`MaxIters`, `IterationCap`: unconverged), or
  rejected by the shape gate (`Failure`, or `InitialFailure` at the first step
  from the seed; `ShapeGate`; `rejection` is `:not_peak` or `:aspect_ratio`). Capped and
  rejected blobs are still returned, flagged; `count_unconverged(frames)` and
  `count_rejected(frames[, rejection])` (public) count them. The frames
  themselves, and which blobs are returned, are unchanged. Code that builds a
  `BlobDetection` itself supplies the outcome fields.

- **`image_origin` is `pixel_frame`**, the stack's name for the pixel convention
  (`:colmap`, `:opencv`, `:julia`), everywhere: `AffineMap(…; pixel_frame)`, the
  `*_python` functions and the Python package's keywords (`pixel_frame="opencv"`).
  No alias.
- **`detect_blobs(…; max_detections)` keeps about that many of the strongest
  detections spread over the image** (VisualGeometryToolkitFeatures'
  `spread_strongest`, which replaces its `strongest`), ordered by response. The
  default (`"unbounded"`) is unchanged.
- **`load` and `save` are VisualGeometryToolkitIO's functions** (the stack's one file reader's
  and writer's names); BlobBoards adds the methods of its own types and owns no
  `load`/`save` binding. Reading takes the type first, as IO's `load` does:
  `BlobBoards.load(path)` is now `BlobBoards.load(BlobBoards.BoardBundle, path)`
  (or `PatternBundle`, or `AbstractBundle` for either; a file holding the other
  kind is an `ArgumentError`). `BlobBoards.save(pattern, bundle; dir)` is unchanged
  in form, a method of IO's `save`. `AbstractBundle` is public. No other BlobBoards
  function was named `load` or `save`, so nothing is renamed.
- **`config_dict` is VisualGeometryToolkitEstimation's generic** (one of its
  CONTRACT_GENERICS); BlobBoards adds its methods. `BlobBoards.config_dict` is the
  same function, and a downstream configuration adds its method as before;
  `BlobBoards.CONTRACT_GENERICS` no longer lists it.
- **New public helpers for a downstream geometry arm** (CaliBlobs'):
  `board_fit_keywords`, `board_endpoints`, `distortion_center`,
  `load_descriptor_network`, `geometry_local_optimization_strategy`.
- **One adaptation implementation on every backend**: the per-blob
  KernelAbstractions kernels run on `KA.CPU()` (one work item per blob); the
  threaded host twin is deleted. CPU frames move by at most ~2e-7 px.
- **The adaptation computes the same numbers on every backend**: its ν step is
  VisualGeometryToolkitCore's (`nu_point_terms`, `next_nu_from_sums`: Newton only
  where the objective is concave), and one lgamma (Core's `lgamma_lanczos`) and
  one 6×6 solve run on every backend (CUDA's cuBLAS LU and `__nv_lgamma`, and the
  CPU's SpecialFunctions `loggamma`, are gone; SpecialFunctions is no longer a
  dependency). Frames change: blobs whose ν Newton step went downhill move, so the
  CPU and CUDA frames now agree to ~1e-8 px (they differed by up to 0.3 px).
- **RANSACScoringDoneRight is no longer a dependency.** The Student-t + uniform
  scorer and its per-partition score `partition_score` are VisualGeometryToolkitCore's;
  every fit builds the scorer from its table through VisualGeometryToolkitEstimation's
  `scoring_from_toml` (`outlier_halfwidth` the one number it states, the priors the
  stack's configuration: Jeffreys and flat, as before). The scores are unchanged.
- The fallback of every function whose methods live in a package extension is
  VisualGeometryToolkitCore's `require_extension` (Core 0.9.3; BlobBoards' private copy is
  deleted): the same install-line error without the extension, the same `MethodError`
  with it loaded.
- The docstring gates are VisualGeometryToolkitCore's `test_docs`; the suite runs
  Core's `check_duplication` (its open findings listed, each an item of
  `.release/duplication-checklist.md`).

Known issue (bmvc2026, not the package): `bmvc2026/scripts/make_param_variations.jl`
does not run on this release. It builds each panel's `PatternConfig` from a v1
`pattern_config` table through `JSON.parse(…, PatternConfig)`, which BlobBoards no
longer reads (a v1 table's `max_diameter_fraction` is converted by the board-file
migration only). The fix (a public v1 pattern-config reader, or each panel's
`max_scale` stated in configs/figures/param_variations.toml) is for the next release;
the paper's `blob_variations_squared` panels are the tracked ones meanwhile.

## 0.5.1 (2026-11-09)

On the VisualGeometryToolkit 0.8 stack, from General: VisualGeometryToolkitCore, IO and
Estimation 0.8, Features 0.6.1, RANSACScoringDoneRight 0.6 and LocalOptimizationDoneRight
0.5.1 (bmvc2026 also Eval 0.6.1). The radial-homography arm runs on Core 0.8's
`RadialHomography{T, 1}` unchanged. Nothing else changes.

## 0.5.0 (2026-11-09)

On the 0.7 series of the VisualGeometryToolkit stack, every dependency from
General, and every JSON file through VisualGeometryToolkitIO's file boundary.

- **Board files `blobboards/board/v6`, pattern files `blobboards/pattern/v4`.**
  The records are board v5's and pattern v3's, key for key and number for
  number; the file is written by VisualGeometryToolkitIO's `write_record`
  (JSON.jl's printing: an integral float keeps its `.0`) instead of the
  JSON3.pretty clone BlobBoards carried, so the bytes differ. Every earlier
  file still loads: board v5 and pattern v3 through a step that reads the
  record with the current reader, board v1–v4 and pattern v1–v2 through their
  steps as before, in
  memory, the stored identity kept; `VisualGeometryToolkitIO.vgio_migrate`
  rewrites one once. **No board identity moves**: every uid, `hashes.config`
  and QR code is the one it was (checked on 386 stored boards and patterns,
  the published records' included).
- **Track-trace manifests `blobboards/track_trace/v2`** (`trace.json`), the
  same keys under a `schema` string, written by `write_record`; a v1 manifest
  (`"schema_version": 1`) still loads.
- **Removed: BlobBoards' JSON hooks on JSON.jl's own style.** Its types lower
  and lift on VisualGeometryToolkitIO's `VgioStyle` only, so loading BlobBoards
  changes no other package's JSON. `JSON.json(bundle)` and `JSON.parse(text,
  BoardBundle)` no longer give a board file or a board: use `save_json` and
  `load_bundle`. The StructUtils dependency is dropped.
- **A board's lengths round-trip exactly.** A board file declares as its
  `world` unit the first of millimetres, inches and metres in which every
  length it holds (a PDF's bleed included) reads back bit for bit. Every board
  built in millimetres, and every stored or published one, stays millimetres,
  byte for byte. A board built in inches (`8.5inch`, a `0.33inch` margin) is
  written in inches, and one mixing values neither unit holds in metres; before,
  about one inch-built length in forty came back one ulp off. The values in
  memory are the ones built, so no identity and no QR code moves. The reader has
  always honoured the declared `world`: no schema change.
- A migrated board v5 file keeps the unit it declares; its render numbers are
  carried as it states them (a raster's `52.4` dpi stays `52.4`; a length only
  takes the writer's Float64 form, so `"bleed": 0` is `0.0`), so a migrated
  file is byte for byte what a fresh write of the board gives.
- **Removed: the `pretty_json` keyword** of `save_json` and `save`: every file
  is written in the one vgio file shape.
- Every board and pattern file, and a JSON text given to `rebuild_board`, is
  read by VisualGeometryToolkitIO's `read_record` (`read_record(io)`, new, for
  the text), so the Python wrappers now read every earlier schema too.
- Core 0.6: `shape_matrix` is `shape_factor` (the `.tracks` builder's affine
  shapes and overlays; values unchanged).

### On the 0.7 stack

Breaking: BlobBoards 0.5.0 requires the VisualGeometryToolkit 0.7 series and
LocalOptimizationDoneRight / RANSACScoringDoneRight 0.5.

- The 0.7 renames: a paraboloid's shape is `shape_matrix` (M), and the
  branch log's inter-branch gap reads `mahalanobis_distances!`.
- **Removed: `px_per_mm` and `mm_per_px`.** A density's conversions are
  FlexUnits' own: `ustrip(px / mm, density)` and `inv(density)` (the stack has
  no density-taking conversion verb). Values unchanged.
- **Removed: `inliers!`'s `threshold` keyword.** An inlier is the stack's MAP
  rule, `is_map_inlier` of its posterior weight (the same decision as the
  former default, 0.5).
- What BlobBoards had copied is the stack's, every number unchanged (measured
  on the example data: the 4057 detections of the still, the 50 boards rebuilt
  from their definitions, the gallery's features, and `find_boards` under the
  homography, pose and radial arms, models and weights bit for bit):
  - the adapt E-step is Core's `StudentTMixture{T}` (built in the kernel's
    float type with the backend's lgamma), and its apply-time shape gate Core's
    closed-form `is_peak` / `is_valid_shape`;
  - blob placement's collision tests are Core's `intersects` (Circle–Circle
    in its norm form, Circle–Rect); no example board's placement moves;
  - the detection cap is VisualGeometryToolkitFeatures' `strongest` (the capped
    set is then ordered by response, as before), and the supersampled patch's
    downfilter its `box_downsample`;
  - the radial arm's distortion and undistortion are Core's
    `distortion_map(::RadialHomography)` and its inverse;
  - the config tags are `TagRegistry`s (`NMS_TAGS`, `GEOMETRY_TAGS`), written
    through `tag_of`; an unknown tag's error is the registry's, naming the
    known ones;
  - the migration check is VisualGeometryToolkitIO's `same_value(…;
    length_ulps = 0)`; every SHA-256 is Core's `sha256_hex` (the SHA dependency
    is dropped); a QR length's unit is spelled by `unit_string`; the point
    grid's density is the stack's `px/pt`;
  - the `.tracks` builder's view angle and overlays use Core's
    `PerspectiveMap() ∘ HomographyMap(H)` and its `transform_deriv` (the view
    angle moves by rounding only, at most 1.3e-15 rad).
- The adapt ν M-step keeps its own Newton step: Core's `next_nu_from_sums`
  adds an ascent safeguard that moves results (on the example still, inlier
  counts 177/39 → 174/37 under the homography arm), which awaits a decision.

## 0.4.1

Installing for a task no longer means adding packages by hand.

- **Install for your task.** The README and the new documentation page
  *Install for your task* give one `pkg> add` line per task: authoring boards,
  finding boards on the CPU, finding boards on the GPU (with the CUDA 12 runtime
  pin ONNX Runtime's CUDA provider needs), tracking boards in a video, plotting,
  training the descriptor network, using the Python package, and working in a
  clone. Each line names exactly the packages its task needs, including the
  trigger packages of the extension that provides it.
- **One runnable example per task.** Each is a complete script that starts by
  naming its install row, loads exactly that row's packages and writes its files
  to a temporary directory. The test suite builds a fresh environment from each
  row's install line and runs that row's examples in it; the GPU row runs where
  a CUDA device is functional and records a skip otherwise, and the Python row
  runs under the pixi `py` environment. The training example is illustrative
  only: it needs annotation files from the reader's own recordings.
- **A clear error when an extension is missing.** Calling a function whose
  methods live in an extension (the Makie plots and overlays, `board`/`board!`,
  `inspect_pipeline`, the `*_python` functions, `LuxDescriptorNetwork`,
  `train_descriptor`) without the extension's packages loaded raises an error
  that names the exact `pkg> add` line, instead of a `MethodError`.
  With the extension loaded, a call no method fits still raises the
  `MethodError`; `applicable` and `hasmethod` on these functions are now true
  for any arguments (the fallback), so a check of an argument order calls the
  function and expects the `MethodError` instead.
  `DescriptorNetwork`'s refusal to start without the CUDA provider names the GPU
  row's line and the runtime pin.
- **`BlobBoards.train_descriptor`** (public) is the training loop of
  BlobBoardsTrainingExt, reached without `Base.get_extension`.
- The README holds the purpose, installation, one example, the documentation
  links and the citation. The reference material it carried is in the
  documentation (the contributors on its home page), and the paper project's
  cross-validation note in `bmvc2026/README.md`. A coverage badge is added.
- **Documentation gates.** The docs build checks that every public docstring
  is in the manual, runs the doctests and the link check, and tolerates no
  warning. A new API page, *Extensions, Tooling and Internals*, holds the 95
  public names that were missing from the manual; the vendored QR encoder is the
  one exempt module. A `docs` test item finds public names with no docstring,
  docstrings detached from their definitions and signature lines no method
  accepts. `BoardConfig`, `TrainingPair` and `ViewAngleStat` gained docstrings,
  `BoardConfig`'s was reattached, and the docstrings Documenter flagged now
  open with a one-sentence summary.
- Every dependency is in General: CI, CompatHelper, the Downgrade and docs
  workflows, the install instructions and the Python interface's first-import
  bootstrap add General only (CauRegistry is no longer needed).
