# BlobBoards — BMVC 2026 subproject

Motion-capture pose-error evaluation for the absolute-pose experiments
(`data_root()/absolute_pose_experiments/260527`). **The dataset holds inputs
only** (captures, the motion tracker's raw exports, the cameras of record,
board definitions, the synthetic occlusion images, the networks); every
result — the motion-capture records, the detections of all three marker
systems, the hand-eye fits, the pose-error cells — is regenerated from them by
`scripts/regenerate_paper.jl` into keyed run directories under
`results_root()` (`src/runs.jl`; the stages and their configs of record are
tabled in REPRODUCE.md).

A DrWatson project nested inside the BlobBoards package repo. The package
(`../src`, `../test`, `../ext`) stays a pure registered library; everything
paper-related lives here and never enters its dependency graph.

    bmvc2026/
    |-- Project.toml        this project (BlobBoards dev'd via ..; + DrWatson)
    |-- REPRODUCE.md        every paper asset: label -> asset -> generator -> data
    |-- test/               the reproducible-figures check (`make check` in papers/)
    |-- src/                BlobBoardsEval - reusable evaluation machinery
    |-- scripts/            the drivers; plots/, training/(+slurm)
    |-- configs/            bmvc2026_260527.toml (the method), detection_cv.toml
    |-- _research/          probes, one-offs, findings/ write-ups
    |-- data/
    |   |-- exp_raw ->      symlink; IMMUTABLE inputs (images, boards, cameras,
    |   |                   poses/motion_tracker = the raw ground truth)
    |   `-- exp_pro ->      symlink; ADD-ONLY results: every run, keyed
    |                       (src/runs.jl), and the gallery caches
    |-- plots/              intermediate LaTeX/PGF output
    `-- papers/             manuscript source (the paper itself; `make paper`,
                            `make figures`) — DrWatson's papersdir()

`bmvc2026/Project.toml` is the ONLY project file in the subtree. Training
needs Reactant, which cannot coexist with the cuDNN the evaluation path needs
(Reactant bundles its own; that is why BlobBoardsLuxExt and
BlobBoardsTrainingExt are separate extensions), so `scripts/training/` ought
to have its own environment — but it does not have one yet. Supply `--project`
explicitly; the drivers' `@quickactivate` then finds this same project (it
activates the nearest Project.toml above the script, which is this one).

## DrWatson: what is used, what is extended

This is a DrWatson project in layout and habit, but several load-bearing
mechanisms are project-local EXTENSIONS of DrWatson ideas past what its
primitives cover. The map, so nobody hunts for a DrWatson feature that does
not exist here — or reinvents one that does:

| Mechanism | Origin | Where |
|---|---|---|
| `data/exp_raw` / `exp_pro` / `plots/` / `_research/` layout, `projectdir()`, `plotsdir()` | DrWatson | everywhere |
| Tuning SEARCH: the joint coarse-fine search over the spec's axes | VisualGeometryEval `coarse_fine_search` | `detection_cv.jl` |
| Grid APPLICATION: cell → config | custom — `Accessors.@set` over the deserialized `FindBoardsConfig` | `detection_cv.jl` |
| Single-file derived caches (`cached`) | VisualGeometryEval `cached_result`: DrWatson's `produce_or_load(filename = hash)` file name, `wload`/`tagsave`, but a save error propagates (produce_or_load swallows it, saving_files.jl:107-116) | `src/cache.jl` |
| Code provenance: gitcommit + dirty-tree patch | VisualGeometryEval `write_provenance`/`read_provenance`: DrWatson `tag!(storepatch = true)` as one flat TOML record | `src/cache.jl`, `src/sweep.jl` |
| Run keys: the SHA-256 `run_key` of a stage's config of record (+ the network's sha256 for a sweep, the upstream run keys for a consumer) | VisualGeometryEval `run_key` | `src/runs.jl` |
| Keyed run directories `<stage>-<key>/`, found through the config of record | DrWatson `produce_or_load` idea, for a directory | `src/runs.jl`, every stage |
| Effective-config sidecar `_config.toml` | custom (`BlobBoards.write_config`) | `src/sweep.jl` |
| Immutable-data distribution + root ladder | Julia **Artifacts** (Pkg stdlib) through VisualGeometryEval `ArtifactView`/`data_root` | `Artifacts.toml`, `src/data.jl` |
| Board-feature cache | custom (VGF FileCache; self-keyed on network sha + cutoff + patch geometry) | `BoardGallery` |

Nearest DrWatson neighbours, deliberately NOT used (checked against
`names(DrWatson)` — do not re-litigate without new information):

- `safesave` versions colliding filenames by SUFFIX (`_#2`); it prevents
  overwrite but keys nothing — a re-run duplicates instead of cache-hitting,
  and no suffix says which config made it. The keyed run dirs subsume it.
- `struct2dict` is a shallow fieldnames→Dict; `BlobBoards.config_dict` is a
  recursive walk that handles nested configs, geometry type tags, and units —
  that is why the effective-config machinery is package-side, not DrWatson.
- `savename` filters values by TYPE and silently drops nested tables — the
  exact silent-stale failure keying exists to prevent (see `src/cache.jl`).
- `collect_results` exists (behind DrWatson's DataFrames extension) and could
  replace hand-rolled result aggregation if DataFrames ever enters this env.
- ~~`papersdir()`~~ — CONFORMANT since 2026-08-18: the manuscript dir was
  renamed `tex/` → `papers/`, so `papersdir()` now resolves to it. (It moved
  in on 2026-08-14 from an Overleaf-synced repo, which is retired; `papers/`
  is the authority, and `make figures` there regenerates every
  script-produced figure/table from this project's own drivers rather than
  having them hand-carried.)

The two big custom pieces are the same DrWatson idea scaled past its
primitives:

- **`bb-<key>/` is `produce_or_load` for a directory.** A sweep's output
  is a pose file and an overlay PER IMAGE — `produce_or_load` caches one
  file, not a tree, so every stage implements the identical contract by hand:
  key = `run_key(config of record ⊕ what decides the result and is not in
  it)`, add-only paths, skip-if-complete, and a consumer derives the same
  directory from the same config (`src/runs.jl`). CODE is deliberately NOT keyed — it is RESTORABLE:
  `tag!` records the commit and the dirty-tree patch in every sidecar, so
  any run regenerates under its exact code state, and `_is_complete` warns
  when skipping work recorded under a different commit (`BB_FORCE=1`
  redoes it). Commit-keying would orphan 45-minute sweeps on every
  unrelated commit; a manual version constant would never be bumped; a
  source-content hash was judged too heavyweight. `filename = hash` (never
  `savename`) for the same reason `src/cache.jl` documents: `savename`
  silently drops nested tables, and these configs are almost entirely
  nested.
- **Selection is a config, not a promote.** Which run the paper reads is
  named by the evaluation's config of record (`configs/figures/evaluation.toml`
  `[runs]`): the config IS the selection, versioned in git, and its key IS
  the directory. Nothing is copied into the dataset.

## Provenance

Two sidecars beside every run, answering different questions:

  `_config.toml`      WHAT TO RUN — the EFFECTIVE configuration (for a sweep,
                      the FindBoardsConfig, reloadable with
                      `FindBoardsConfig(path; camera)`). The authored config
                      is INTENT (sparse keys resolve to whatever the library
                      default is at load time); the sidecar is the FACT,
                      frozen beside the artifact it produced.
  `_provenance.toml`  WHAT ACTUALLY RAN — DrWatson's tagged dict as ONE flat
                      TOML table: gitcommit, gitpatch (the diff when the tree
                      is dirty), driver, timestamp, hostname, julia, the run
                      key and the inputs. VisualGeometryEval.Experiments
                      `write_provenance` / `write_run_records`.

Every result stage writes both (the sweeps per subset, the other stages at
the run root); the pose-error cells carry the run keys of their sources
inside each cell, and every figure generator writes its record under
`plotsdir("records")`.

`configs/bmvc2026_260527.toml` IS the method. `detection_cv.toml` declares only
the axes searched over it and how to score them - it holds no knob the config
already carries. Its `[tuning.detection]` is a `joint_coarse_fine` tuning spec
(the half-width, peak-threshold and similarity axes, the invariance mAA
criterion, and its cross-validation: the registered split
`blobboards_cross_session` of VisualGeometryEval `configs/splits.toml` — the
session it tunes on and the one it reports — and the frame sampling) read by
VisualGeometryEval's one `[tuning]` parser — the schema is documented once, in
VisualGeometryEval `configs/tuning.toml`; `[inputs]` holds the run config,
the session's camera and the invariance counts. The run's records carry the
RESOLVED spec, and its directory is keyed by it.

## Data root

Data is distributed via the Julia Artifacts system (`Artifacts.toml` here;
Zenodo-hosted tiers, lazy per-tier download). The resolution ladder is the
stack's one implementation (VisualGeometryEval.Experiments), bound in
`src/data.jl`:

1. `ENV["CAU_DATA_ROOT"]` — explicit override;
2. `datadir("exp_raw")` — the per-machine link, resolved;
3. `BlobBoards.data_dir()` (`BLOBBOARDS_DATA` / `LocalPreferences.toml`) —
   the author's primary **writable** copy;
4. composed artifact view (`output/data_view`; it moves to
   `datadir("exp_raw")` at the next full regeneration) — consumers; read-only,
   downloads only the tiers a driver needs.

Expected download time: Zenodo served ~0.1–0.5 MB/s per single stream (measured 2026-10-09), so a multi-GB tier takes hours.

The data policy (VisualGeometryEval `data_policy_violations`): D1 no machine paths in
code or launchers; D2 nothing writes under `data_root()`;
D3 output directories are keyed (`<stage>-<run_key>`, `<prefix>_<hash>`).
`test/paper_project_tests.jl` checks it (VisualGeometryEval PaperProjects).

The descriptor's frozen training set ships as two further root-mounted tiers,
`bb_descriptor_training_metadata` (boards, bootstrap network chain, recipe
README) and `bb_descriptor_training_sources` (the 634 captures); see
`descriptor_training/README.md` in the dataset and the `training` tier of
`scripts/make_zenodo_artifacts.jl`.

**Nothing writes into the dataset.** `data_root()` is the input set; every
result goes to `results_root()` (`bmvc2026/data/exp_pro`, `CAU_RESULTS_ROOT`
to override) and figures to `plotsdir()` (`bmvc2026/plots`).
`test/data_tests.jl` enforces it: every run resolver points into
`results_root()`, no driver joins a `data_root()` path with a result kind.

## Run versioning

Every result — BlobBoards and the AprilTag/ArUco baselines alike — is a run:

    exp_pro/convert_formats-<key>/   the motion-capture records (stage 0)
    exp_pro/bb-<key>/                a BlobBoards sweep (poses/, val/)
    exp_pro/occlusion-<key>/         the occlusion sweep (occlusion/<LL>/poses, val)
    exp_pro/baselines-<key>/         AprilTag + ArUco (exports/, poses/, val/, occlusion/)
    exp_pro/hand_eye-<key>/          the hand-eye records
    exp_pro/pose_results-<key>/      the pose-error cells

`<key>` is the SHA-256 `run_key` of the stage's config of record plus what
decides the result and is not in it (the network's content for a sweep, the
upstream run keys for a consumer). A changed config is a NEW run rather than
an overwrite, and a stage skips a run that is already complete. Keying the
path is what makes skip-if-complete safe: with unkeyed paths the skip is
silently wrong the moment the config changes.

## Environment

`bmvc2026/scripts/` is its own Julia project with BlobBoards (and the local Cau
dev chain) as path dependencies:

```sh
julia --project=bmvc2026 bmvc2026/scripts/regenerate_paper.jl     # every result + the paper's assets
julia --project=bmvc2026 bmvc2026/scripts/make_pose_results.jl     # just the result cells (after the detections)
julia --project=bmvc2026 bmvc2026/scripts/plots/absolute_pose.jl   # working figures
```

(or `cd bmvc2026/papers && make figures`). The AprilTag/ArUco baselines run
in the pixi environment `baselines` of `bmvc2026/pixi.toml` (pinned to the
versions the external evaluation code ran with); `detect_baselines.jl`
invokes it.

**The training track is currently unrunnable from anything checked in.** This
README used to claim `scripts/training/` has its own Project.toml; it does
not — `bmvc2026/Project.toml` is the only one in the subtree, and it declares
neither Reactant nor Enzyme; no training script uses `@quickactivate`. The
Reactant/cuDNN conflict that motivates a separate environment is real (it is
why BlobBoardsLuxExt and BlobBoardsTrainingExt are separate extensions), but
the environment itself was never created.

## Pipeline

| Stage | File | Output |
|---|---|---|
| Rig join | `src/rig.jl` | mocap ↔ image pairing, cam-to-board poses, tripod-spot segmentation, signed rig angles |
| Recovery | `src/recovery.jl` | evidence-gated static-camera substitution for camera-untracked calibration frames |
| Error metrics | `src/pose_error.jl` | relative-to-base rotation error (calibration-free) + robot-world hand-eye absolute rotation/translation errors |
| Hand-eye | `make_hand_eye.jl` | `hand_eye-<key>/hand_eye_<type>.json` (vgio transform records + fit covariance) |
| Error table | `src/error_table.jl` (`evaluate_dataset`), `make_pose_results.jl` | one DrWatson cell per (system, level) under `pose_results-<key>/`; read as one long-form DataFrame with `pose_results(eval_sources(cfg))` |
| Figures | `plots/absolute_pose.jl` | error vs rig angle/distance, detection rate, occlusion sweep |

Hand-eye calibration (the fixed camera-rig→camera and marker→board-rig
transforms) is estimated from the `calibration_*` subsets via
RobustVisualGeometry's `fit_robot_world_hand_eye` (staged linear Kabsch estimator,
RVG ≥ 0.14) and applied to the evaluation sweep. See
`bmvc2026/_research/findings/hand_eye.md` for the current calibrations.

### Conventions worth knowing

- The evaluation reads VisualGeometryIo kinds only, every one a result of a
  keyed run: detections `<run>/poses/<system>/target_poses.json`
  (vgio/target_poses/v1) + `telemetry.jsonl` (the sweeps write them; the
  baseline stage converts the baselines' per-image exports); the motion-capture
  records, written from the raw exports by `scripts/convert_formats.jl` (config
  of record `configs/convert_formats.toml`): the motion tracker's
  `poses/motion_tracker/<subset>/markers.json` (vgio/target_points3d/v1, the
  14 markers — the source measurements), `board_rig.json`/`camera_rig.json`
  (vgio/trajectory/v3, the vendor's solve; recovered camera frames flagged)
  and `constellation_<body>.json` (vgio/rig/v4 of one PointTarget, estimated);
  the camera is the session's camera of record, an input —
  `calibration/<camera>.json` (vgio/cameras/v3 with its calibration block;
  `camera_of_record`, src/data.jl), the native port of the upstream
  calibration export by `scripts/import_cameras.jl`; the images tier's own
  camera files are superseded and unused; hand-eye records
  vgio/transforms/v3 + vgio/rig/v4 (each printed board a pose-only
  PlanarTarget named by its id, `BlobBoards.board_resolver` resolves its
  points), with the fit's covariances. Every kind is the direct serialization
  of a VisualGeometryCore value; rotations are stored, and read back, as unit
  quaternions. The raw per-frame exports stay the source of record.
- The raw mocap export's `rot_mat` is **column-major**. The original
  Python pipeline read it row-major — invisible in distance norms and
  relative-angle magnitudes (transpose-invariant, so the shipped CSV rig
  columns still agree) but fatal for any frame-dependent computation:
  hand-eye self-residuals are 0.4°/3.7 mm column-major vs 14.6°/150 mm
  row-major. The trajectory files declare their rotation encoding.
- `error_angles` is the conjugation-invariant relative metric; a
  calibration-free relative *translation* metric does not exist (marker
  lever-arm swing is indistinguishable from detector error), so
  `error_distances` is the hand-eye absolute translation error.
- The shipped CSVs' error columns came from a pandas notebook that was never
  delivered; the definitions here are canonical replacements (rig columns
  reproduce the shipped files: distances to ~0.02 mm, angle bins 55/57).

## Legacy tree — removed 2026-08-11

The BMVC 2026 extraction/RMSE harness — `types.jl`, `config.jl`,
`dataset.jl`, `utils.jl`, `pose_extractor.jl`, `convert_legacy.jl`,
`generators/occlusion_generator*.jl`,
`plots/{spatial_accuracy,occlusion,helpers,reader}.jl` — was deleted;
recover any of it from git history at this commit. It produced the on-disk
poses and manufactured the occlusion image tiers; the tiers are frozen
immutable inputs now (the published occlusion numbers, including the
AprilTag/ArUco baselines, were computed on them), so the generators'
remaining role was provenance: the occlusion images are published as inputs
with that provenance (BlobBoards-Evaluation 88bca9f
`evaluation_pipe/occlusion_generator.py`, in the inputs tier's code archive).
The mocap-based generator `make_occlusion.jl` was retired with
`export_reconstruction.jl` and `hand_eye_reprojection.jl` (2026-10-15); its
levels never fed a published number. The reprojection-RMSE track and
its ground-truth labeling GUI remain in the student's Python repo
(`BlobBoards-Evaluation-main.zip`).
