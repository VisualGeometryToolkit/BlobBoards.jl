# BlobBoards: duplicate removals for the 0.9 cascade leg

Owner decision 12 (2026-11-11) and the owner's ruling for the BlobBoards and CaliBlobs ports: every duplicate goes;
common code is hoisted, without exception. Sources: `cascade-duplicate-removals.md` ("BlobBoards ↔ CaliBlobs"),
every BlobBoards item of `duplication-audit.md` (2026-11-10), and the stack's `test_duplication` run on the package
(`VGTK_STACK_PATHS` = vgtk-09, rsdr-09, lodr-09, cb-09). An item is ticked with the commit that closes it; an open
item says what it waits for. The release leg refuses while any `- [ ]` line is left.

## BlobBoards ↔ CaliBlobs (cascade list)

- [x] the shared helpers → one owner: BlobBoards (audit 6). BlobBoards' side: `board_fit_keywords`, `board_endpoints`,
  `distortion_center`, `load_descriptor_network` and `geometry_local_optimization` are public (48a592f; renamed
  `geometry_local_optimization_strategy` in the 0.9 strategy rename);
  CaliBlobs deleted its copies in its own leg (cb-09 `caliblobs-09`, 639abd1).
- [x] BlobBoards' CPU twin of its GPU kernels → one KernelAbstractions implementation (audit 7): `_adapt_cpu`,
  `_adapt_cpu_one!`, `_cpu_residuals!`, `_cpu_weights!` deleted; the per-blob kernels run on `KA.CPU()` with one
  work item per blob (0e14141). CPU frames vs the deleted twin's: max 2.2e-7 px centroid; GPU frames vs the old
  kernels': max 2.4e-8 px.

## Further BlobBoards items of the audit and of `test_duplication`

- [x] `test/suites/docs_gates.jl` (audit 1; a whole-file copy of LODR's) → Core's `test_docs` (48a592f)
- [x] the private upload helper (audit 17), and canonicalize's two hand-written `allocate` + `copyto!` uploads → Core's `ka_upload` (0e14141, 48a592f)
- [x] `config_dict`, owned by both BlobBoards and VisualGeometryToolkitEstimation → Estimation's generic (one of its
  CONTRACT_GENERICS); BlobBoards adds its methods (48a592f)
- [x] the private extent wrapper (Estimation's `_with_extent`, restated; retired in 0.9) → deleted, `BoundedProblem` at its one call (48a592f)
- [x] homonyms of an upstream name for a different concept, renamed to what they are: `_reads_back` (IO's is the
  pure-bending rule) → `_holds_exactly`; `_plain` (the identity lowering, not IO's Dict conversion) →
  `_identity_lowering`; `_RETIRED_KEYS`/`_check_keys` (BlobBoards' own retired run keys) → `_RETIRED_RUN_KEYS`/
  `_check_run_keys` (48a592f)
- [x] bmvc2026's `cached` (audit 6: a one-line wrapper of Eval's `cached_result`, copied in icra2026) → deleted,
  `cached_result` called (f172702)
- [x] bmvc2026's `\newcommand` writers (make_detection_stats, flip_table, make_hand_eye_table, make_supp_tables,
  plots/error_distributions) → Eval's `PaperNumbers`/`add_number!`/`write_numbers`, through one `figure_numbers`
  (f172702). Not run here (the paper's inputs); a regeneration reviews them (REPRODUCE.md).

- [x] bmvc2026's Slurm annotation drivers (`test_duplication`: `_init_worker!`, `_process_source` restated between
  driver.jl and driver_scm.jl) → one file both include, annotate_worker.jl (d84baf3)

## Hoists into the stack (the coordinator implements; BlobBoards then switches)

- [x] `_sweep_score` → RANSACScoringDoneRight's `partition_score` (audit 7; `test_duplication`: a name RSDR owned):
  RSDR made its per-partition score public and device-callable (rsdr-09 6ddd808); the sweep kernel calls it at the
  Jeffreys prior (f0fe0d8). CPU frames unchanged against the earlier port (max 2.2e-7 px vs the old twin, as
  before).
- [x] the package defaults reader (`test_duplication`: `_dflt` and `default_config_path` owned by Estimation,
  `_TOML_DEFAULTS` by LODR; audit 20): not the same concept but each package's own data, read through Core's one
  `parse_defaults` (BlobBoards now parses its two files with it) and `toml_default` — the idiom `parse_defaults`'
  docstring states. DUPLICATION.toml entries with the proof, per the owner's ruling (eef3144).
- [x] hoist `_Record` (an exact copy) and `_snap_integer` → VisualGeometryToolkitIO, public (IO src/style.jl:93, — CLOSED: VisualGeometryToolkitIO `Record`, `snap_integer` (with its Integer and Rational methods) public (monorepo e5d4d274); the copies deleted, the lifts read `VgtkIO.Record` (cd4ff03)
  src/core_json.jl:74). Proposed: `Record = Union{JSON.Object, NamedTuple}` and `snap_integer(v)` with BlobBoards'
  `Integer` and `Rational` methods beside IO's float one. Callers: BlobBoards src/json.jl, src/types.jl,
  src/migrate_boards.jl. The switch: `using VisualGeometryToolkitIO: Record as _Record, snap_integer as _snap_integer`.
- [x] hoist the README install-row runner (`test/suites/readme_rows.jl`'s `examples`, `loaded_packages` and the — CLOSED: VisualGeometryToolkitCore's README toolkit in its Test extension (`readme_examples`, `install_table`, `install_packages`, `loaded_packages`, `readme_environment`, `run_readme_child`, `test_readme_row`; monorepo 1d58e428, a8da29d8); test/suites/readme_rows.jl keeps the pages, the rows and `run_row` on it (cd4ff03)
  per-row fresh-environment runner; audit 6; CaliBlobs' test/readme_rows.jl restates it) → VisualGeometryToolkitCore's
  Test extension, beside `test_docs`: `test_readme_rows(M; pages, rows, row_lines)`. Callers: BlobBoards
  test/suites/readme_rows.jl, CaliBlobs test/readme_rows.jl.
- [x] hoist `.githooks/sync-manifest.jl` (a byte-identical copy of RSDR's; audit 20) → one copy owned by the stack's — CLOSED: one copy, VisualGeometryToolkitCore's `githooks/` with `sync_manifest`/`install_githooks` (monorepo, in ae154f63), not cau-tooling vendoring: the stack package owns it and every repository installs it; .githooks/ deleted here, CLAUDE.md and the install guide name `install_githooks` (cd4ff03)
  release tooling (cau-tooling), vendored into each repo through `VENDORED.toml`. Callers: BlobBoards, RSDR.
- [x] hoist `linear_to_srgb` → VisualGeometryToolkitFeatures (audit 16; `test_duplication` pattern `srgb_transfer`): — CLOSED: Features' `linear_to_srgb` (plane-maps, in the 0.9 branch); docs/make_hero.jl calls it, Features in docs/Project.toml (cd4ff03)
  it is on plane-maps (7a50364d), not yet in the 0.9 branch. Caller: docs/make_hero.jl:42 `srgb`, which also needs
  Features in docs/Project.toml.
- [x] hoist the paper-support copies shared with CaliBlobs' icra2026 (audit 6) → VisualGeometryToolkitEval: — CLOSED: VisualGeometryToolkitEval `Experiments.assert_active_project`, `save_hand_eye`/`load_hand_eye`/`hand_eye_mount_file`, `Render.station_cells` (monorepo 950b6b41); `_diag` is not hoisted (a reflective field probe, against the accessors rule): the fit's `metadata.σ_rot`/`σ_trans` are read directly (cd4ff03)
  `_assert_project` (bmvc2026/src/data.jl:21), `_diag` (bmvc2026/src/calibration.jl:165), the station cells
  (bmvc2026/scripts/plots/error_distributions.jl `_cells`) and `load_hand_eye` (bmvc2026/src/pose_error.jl). The
  paper tree leaves the package repository in 0.5.1 (owner decision 11).

- [x] the ONNX descriptor's own session handling (src/descriptor.jl: `ONNXRunTime.load_inference` with its CUDA arena cap and CPU fallback) → VisualGeometryToolkitFeatures' ONNX runtime (src/onnx_runtime.jl: the execution-provider resolution, the sm_100-capable CUDA provider, the capped arena). — CLOSED: Features' `onnx_session(path; execution_provider, gpu_mem_limit)` is public (monorepo f0342189); DescriptorNetwork calls it with `gpu_mem_limit = _arena_cap(max_batch)`, BlobBoards' `load_inference` handling deleted, its `allow_cpu_fallback` kept around an explicit `:cuda`. On a B200 the CUDA session runs (Features' sm_100 build): annotate smoke 57/57 in a session, its four `cudaErrorNoKernelImageForDevice` errors gone; the device handoff's descriptors equal the host handoff's on the CUDA EP exactly (closed by 9536308)

## CPU/GPU twins (the owner's one-kernel rule: a GPU branch's CPU branch is the same KernelAbstractions kernel on the CPU backend)

Source: cpu-gpu-twin-audit.md ("BlobBoards"), hoists-09.md ("the one-kernel gate"); Core's `check_duplication`
`:backend_branch` / `:twin_name` rules (vgtk-hoists d3a02dbf, 9b14e0cb).

- [x] B3: BlobBoardsCUDAExt's `_canon_describe(::CUDABackend)` and `_canonicalize_cuda_device` (the inline warp builder and their own driver) → the generic warp builder (`_region_bm`) in `_patch_cube` (Core's one batched KA warp, one Float32 cube on every backend: `_canonicalize`'s `device isa KA.CPU` branch gone with it) and the generic chunked `_canon_describe`; the extension keeps only the ONNX device-tensor handoff (`_describe_patches` of a CUDA cube). Patches (logpolar/cartesian, supersample 1/2, CPU and CUDA) and descriptors (CPU EP, CPU and CUDA devices) bit-identical to the old code; the device handoff equals the host handoff on the CUDA EP exactly (closed by f037213)
- [x] the duplicate uploaders `_upload_f32` / `_gpu_upload_f32` (and the `_undistort_adapt` methods they served; its one call gave it `nothing`) → Core's `on_device`/`ka_upload` (`_device_image`) (closed by f037213)
- [x] B4: `_nu_point_terms` / `_next_nu` (drifted: no concave/ascent safeguard) → Core's `nu_point_terms` / `next_nu_from_sums` from the kernel. Results CHANGE (numbers in the port report): on three synthetic fields 12/63, 0/25, 42/150 frames move by more than 1e-3 px; the median centroid error against the rendered truth falls 4–7× (e.g. 1.1e-4 → 2.8e-5 px), CPU ≡ GPU tightens from 0.23–0.29 px to ≤ 6e-8 px (closed by 6706f1d)
- [x] the host-side Float32 conversion before an upload (`_device_image`, the adaptation's image intake) → Core's `on_device`/`ka_upload`, which converts an 8-bit image on the device (monorepo 49f96521): the image crosses as its N0f8 plane; bit for bit (CPU; the B200's uploaded images and patches/descriptors of fixed frames), a 50 Mpx upload to a B200 439 → 23 ms (closed by bb9f89a)
- [x] gate hit `_execution_provider` (training.jl): a provider choice, no second computation → DUPLICATION.toml `[[backend]]` with the reason (closed by 71151ab)
- [x] gate hit `detect_cpu`/`detect_gpu` (bmvc2026/_research): two different detectors (BlobBoards' and the VKS C library), not one on two backends → named for what they are, `detect_blobboards`/`detect_vks` (closed by 71151ab)
- [x] B1: the cuBLAS batched-LU override `_solve_6x6_batched!(::CUDABackend)` vs `_lu6_kernel!` → `_lu6_kernel!` on every backend, the override and the hook deleted (owner ruling, 2026-11-11: unify) (closed by 508508c; the new CUDA baseline 8ad05ba)
- [x] B2: `_device_lgamma`'s three implementations (SpecialFunctions on the CPU, `lgamma_lanczos`, CUDA's `__nv_lgamma`) → Core's `lgamma_lanczos` on every backend, `_device_lgamma` and the SpecialFunctions dependency deleted (owner ruling, 2026-11-11: unify). CPU frames bit for bit unchanged; CUDA frames vs the old CUDA code max 2.2e-12 px, inside the old code's run-to-run spread; the adapt parity item now holds CPU ≡ CUDA under 1e-6 px with no validity disagreement (measured 8.3e-9 px) (closed by 508508c; the new CUDA baseline 8ad05ba)

## Robust schema

- [x] the adapt GPU kernel's robust IRLS (`adapt_ka.jl:469-503`, `adapt.jl:84-86`: the private half-width √median(r²), ν ≥ 2 and
  its floors) → DEFERRED by the owner (ruling 3, 2026-11-11): the kernel and its parameters stay as they are for 0.9; not a blocker
  of this leg (robust-schema audit, borderline 5)

### The Student-t keyword (owner ruling 2026-11-11: "there is no threshold for the Student-t, only the outlier half-width")

Every scorer takes its own keyword: `StudentTUniformScoring` its `outlier_halfwidth` (and its priors), never `threshold`; the
threshold scorers (RANSAC, MSAC, the loss-width families) `threshold`, never `outlier_halfwidth`. No number may cross between
the two meanings. Audit: studentt-keyword-overload.md.

The package's keys, keywords and constructions already take a half-width (`[find_boards.geometry] outlier_halfwidth`,
`_scoring(P, outlier_halfwidth)`, the Python `outlier_halfwidth`); the overloads left were prose, and one reuse of the number.

- [x] bmvc2026/papers/bmvc_supp.tex:279 the settings table's row "inlier half-width $a$" → "outlier half-width $a$" (this commit)
- [x] bmvc2026/configs/bmvc2026_260527.toml:158, bmvc2026_260527_noflip.toml:161 "Reprojection inlier half-width" → "Reprojection outlier half-width" (comments: the run keys hash the parsed tables, unchanged) (this commit)
- [x] src/pipeline.jl:1655 (PoseGeometry's docstring) "the same inlier band as the homography path" → "the homography path's" outlier half-width (this commit)
- [x] ext/BlobBoardsPythonExt.jl:575 and python/blobboards/__init__.py:1318 "`min_inliers`, `outlier_halfwidth`: geometry gates" → the inlier-count gate and the scorer's outlier half-width, not a gate (this commit)
- [x] test/suites/config_toml.jl "the Student-t path: outlier_halfwidth, never threshold": a geometry table with `threshold` is refused on every arm, the table's `outlier_halfwidth` reaches the scorer exactly (`===`), the scorer refuses `threshold`, and `_find_model` refuses a `threshold` beside its scorer (this commit)
- [x] src/pipeline.jl:430 `_scoring` and test/suites/raw_pose_arm.jl:88: `StudentTUniformScoring(P, a)` (RANSACScoringDoneRight's constructor, its priors defaulted in code) → VisualGeometryToolkitEstimation's `scoring_from_toml` of the table `{ type = "student_t_uniform", outlier_halfwidth = a }`, its priors the stack's `[scoring.student_t_uniform]` (Jeffreys, flat: the defaults it had). The scorer moved to VisualGeometryToolkitCore with both priors required (monorepo c7b8106e, 74db20d2), so the bare call no longer builds; the half-width is the one number the table states (this commit)
- [x] src/adapt_ka.jl:104/483 `partition_score` (the adapt kernel's per-partition score) is VisualGeometryToolkitCore's (the Student-t hoist; the same formula); RANSACScoringDoneRight is no longer a dependency of BlobBoards or its tests (this commit)
- [x] src/pipeline.jl `find_board`'s guided final fit and `guided_search`: the Student-t's `outlier_halfwidth` was handed to `guided_search` as `max_px`, a nearest-centre association radius (a gate). By the owner's ruling (2026-11-11) ellipse IoU (≥ `min_iou`, at the blob's rendered boundary) is the ONLY acceptance criterion, as an overlap query: a projected blob's candidates are the detections whose ellipses' bounding boxes intersect the projected ellipse's box (`detection_rtree(det_ells)`, per board cutoff, per round against the claim state), and it takes the candidate of highest IoU; one-to-one, best overlap first, RANSAC's `keep` first. `max_px`, the half-width hand-off, the nearest-centre step (`_rtree_nearest`) and `find_board`'s `guided_detc` are deleted, no key replaces them. The read dependencies are every candidate of the query, recorded before the gate (was: the one nearest centre within the radius). Results move: measured in studentt-keyword-overload.md (this commit)


### The robust-estimation contract (robust-contract.md; port-bb-contract.md)

The stack's one contract for robust stages (verbs, the scale hand-off, weights, termination) lands in BlobBoards as follows.
Each item names the contract step it waits for.

- [x] `[sources]` of every workspace member (root, test, docs, gpu, examples, authoring, bmvc2026) → the integration branch
  robust-core, through a PINNED snapshot worktree `vgtk-pin-<sha>` (never the integrator's live worktree, which merges
  move under running suites); now `vgtk-pin-01e54733`, with LocalOptimizationDoneRight at `lodr-pin-613e235` (temporary, like the hoists pointer it replaces)
- [x] `decision_boundary` (Core 648cd520): BlobBoards binds none of the three retired names (Core's RETIRED_NAMES); the one left was
  prose, configs/default.toml's note on the Student-t boundary → Core's `decision_boundary` (this commit)
- [x] R12 (the Student-t scorer has no threshold): test/suites/config_toml.jl's two refusal tests named the refused word beside the
  scorer → the same refusals, the word spelt apart from the scorer's name and tag (this commit)
- [x] §14, the owner's rulings for BlobBoards: the adaptation IRLS keeps a per-blob state (live, converged at `rtol`, refused by
  the shape gate as not a peak / beyond `max_aspect_ratio`) and returns each blob's outcome in `BlobDetection`'s metadata —
  `converged`, `retcode` (`:Success`, `:MaxIters`, `:Failure`, `:InitialFailure`), `stop_rule` (`:ParameterChange`,
  `:IterationCap`, `:ShapeGate`), `rejection` (`:not_peak`, `:aspect_ratio`, `:none`) — BB-local Symbols one-to-one with
  Core's `ReturnCode`/`StopRule` until s14-core lands them. Capped blobs are unconverged (`count_unconverged`), refused ones
  rejected (`count_rejected[, reason]`); a `@debug` line per call counts each. Validity and frames unchanged. Fail-first test
  `test/suites/adapt_converged.jl`; the parity item checks the flag CPU ≡ GPU (408de3d, this commit)
- [x] the outcome Symbols are Core's `ReturnCode`/`StopRule` (`ShapeGate` added by s14-core; robust-core c29e0373); `rejection`, the gate's reason, stays a BlobBoards Symbol (this commit)
- [x] refine_* with `robust =`: none in BlobBoards (scanned); `refine_homography`/`refine_radial_homography` (src/pipeline.jl
  `_refine`) take their strategy as a keyword already; after the 0.9 rename it is `optimization_strategy` (this commit)
- [x] B1 (robust-core 17ab6cc2): a `refine_*` runs no scorer. BlobBoards' `_refine` (the guided refit and the pose-ambiguity
  LRT) needs the scorer's quality and consensus set, so the plain and radial homography arms call `locally_optimize` on the
  same (bounded) problem, as the pose arm always did. Until B1 `refine_homography`/`refine_radial_homography` were exactly
  that call, so no number moves (by construction: the same call on the same problem). Audit P11
- [x] A3, the width multiple is `efficiency_constant` (the width mode `EfficiencyConstant`; old names in Core's RETIRED_NAMES): four files, Welsch only (no
  Geman–McClure or BarronHQ value in BlobBoards, so no number moves) (this commit)
- [x] the strategy rename (owner, 0.9): the geometry's strategy key/keyword/field `local_optimization_strategy`, the public
  reader `geometry_local_optimization_strategy`, Estimation's `optimization_strategy_from_toml`; the two refine calls take
  `optimization_strategy`; the old key is refused naming the new one (`_RENAMED_RUN_KEYS`, fail-first test) (this commit)
- [x] B4 (robust-core ee5de64f): the find_* result's mask is Core's `inlier_mask(r)` (src/pipeline.jl's branch telemetry,
  test/suites/raw_pose_arm.jl, bmvc2026/_research/fit_homography.jl), its scale `noise_scale(r)`: the branch telemetry
  read `inc.scale`, a field RansacMetadata never had, so a run with a branch log threw there (fixed, this commit). The
  other weights consumers already read `weights` with `is_map_inlier`, and the persisted `n_inliers` keys are unchanged
- [x] refine_homography's `RefineMetadata`: BlobBoards calls no `refine_*` since B1 (above), so it reads none
- [x] §14 on BlobBoards' find/LO reads: "a model exists" is Core's `found_model(r)` (src/pipeline.jl ×4, src/track.jl, the raw pose arm test), never the now strict `converged` (this commit)

## Public names an upstream package owns: an API decision

- [x] `load`/`save`, names VisualGeometryToolkitIO owns → IO's functions (owner decision, Option A): BlobBoards owns
  no `load`/`save` binding; its reader is the method `load(T::Type{<:AbstractBundle}, path)` (was `load(path)`) and
  its writer the method `save(pattern, bundle::PatternBundle; dir)` (f0fe0d8). Until the hoist below lands,
  the two methods are the contract_generics gate's only findings (report mode; the suite pins exactly them).
- [x] hoist: IO lists load/save in CONTRACT_GENERICS. — CLOSED: IO `CONTRACT_GENERICS = (migrate_schema, load, save)` (monorepo 5aff63ca); the gate is a plain `@test isempty(cv)` (cd4ff03)
- [x] `board`/`board!` (BlobBoards' Makie recipe), a name Core's Cameras owns (`board(t::PlanarTarget)`): not the
  same concept (a recipe that draws a board vs the accessor of a target's physical board); a DUPLICATION.toml entry
  with the proof, per the owner's ruling (eef3144).

## test_duplication

The final run on the package is in the port report (`/home/jpr/logs/dgx-agent/cascade-09-BlobBoards.md`); every
finding left is tied to an open item above.
