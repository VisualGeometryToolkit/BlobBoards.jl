# BlobBoards: duplicate removals for the 0.9 cascade leg

Owner decision 12 (2026-11-11) and the owner's ruling for the BlobBoards and CaliBlobs ports: every duplicate goes;
common code is hoisted, without exception. Sources: `cascade-duplicate-removals.md` ("BlobBoards ↔ CaliBlobs"),
every BlobBoards item of `duplication-audit.md` (2026-11-10), and the stack's `test_duplication` run on the package
(`VGTK_STACK_PATHS` = vgtk-09, rsdr-09, lodr-09, cb-09). An item is ticked with the commit that closes it; an open
item says what it waits for. The release leg refuses while any `- [ ]` line is left.

## BlobBoards ↔ CaliBlobs (cascade list)

- [x] the shared helpers → one owner: BlobBoards (audit 6). BlobBoards' side: `board_fit_keywords`, `board_endpoints`,
  `distortion_center`, `load_descriptor_network` and `geometry_local_optimization` are public (48a592f);
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
  Jeffreys prior (COMMIT_LOADSAVE). CPU frames unchanged against the earlier port (max 2.2e-7 px vs the old twin, as
  before).
- [x] the package defaults reader (`test_duplication`: `_dflt` and `default_config_path` owned by Estimation,
  `_TOML_DEFAULTS` by LODR; audit 20): not the same concept but each package's own data, read through Core's one
  `parse_defaults` (BlobBoards now parses its two files with it) and `toml_default` — the idiom `parse_defaults`'
  docstring states. DUPLICATION.toml entries with the proof, per the owner's ruling (eef3144).
- [ ] hoist `_Record` (an exact copy) and `_snap_integer` → VisualGeometryToolkitIO, public (IO src/style.jl:93,
  src/core_json.jl:74). Proposed: `Record = Union{JSON.Object, NamedTuple}` and `snap_integer(v)` with BlobBoards'
  `Integer` and `Rational` methods beside IO's float one. Callers: BlobBoards src/json.jl, src/types.jl,
  src/migrate_boards.jl. The switch: `using VisualGeometryToolkitIO: Record as _Record, snap_integer as _snap_integer`.
- [ ] hoist the README install-row runner (`test/suites/readme_rows.jl`'s `examples`, `loaded_packages` and the
  per-row fresh-environment runner; audit 6; CaliBlobs' test/readme_rows.jl restates it) → VisualGeometryToolkitCore's
  Test extension, beside `test_docs`: `test_readme_rows(M; pages, rows, row_lines)`. Callers: BlobBoards
  test/suites/readme_rows.jl, CaliBlobs test/readme_rows.jl.
- [ ] hoist `.githooks/sync-manifest.jl` (a byte-identical copy of RSDR's; audit 20) → one copy owned by the stack's
  release tooling (cau-tooling), vendored into each repo through `VENDORED.toml`. Callers: BlobBoards, RSDR.
- [ ] hoist `linear_to_srgb` → VisualGeometryToolkitFeatures (audit 16; `test_duplication` pattern `srgb_transfer`):
  it is on plane-maps (7a50364d), not yet in the 0.9 branch. Caller: docs/make_hero.jl:42 `srgb`, which also needs
  Features in docs/Project.toml.
- [ ] hoist the paper-support copies shared with CaliBlobs' icra2026 (audit 6) → VisualGeometryToolkitEval:
  `_assert_project` (bmvc2026/src/data.jl:21), `_diag` (bmvc2026/src/calibration.jl:165), the station cells
  (bmvc2026/scripts/plots/error_distributions.jl `_cells`) and `load_hand_eye` (bmvc2026/src/pose_error.jl). The
  paper tree leaves the package repository in 0.5.1 (owner decision 11).

## Public names an upstream package owns: an API decision

- [x] `load`/`save`, names VisualGeometryToolkitIO owns → IO's functions (owner decision, Option A): BlobBoards owns
  no `load`/`save` binding; its reader is the method `load(T::Type{<:AbstractBundle}, path)` (was `load(path)`) and
  its writer the method `save(pattern, bundle::PatternBundle; dir)` (COMMIT_LOADSAVE). Until the hoist below lands,
  the two methods are the contract_generics gate's only findings (report mode; the suite pins exactly them).
- [ ] hoist: IO lists load/save in CONTRACT_GENERICS.
- [x] `board`/`board!` (BlobBoards' Makie recipe), a name Core's Cameras owns (`board(t::PlanarTarget)`): not the
  same concept (a recipe that draws a board vs the accessor of a target's physical board); a DUPLICATION.toml entry
  with the proof, per the owner's ruling (eef3144).

## test_duplication

The final run on the package is in the port report (`/home/jpr/logs/dgx-agent/cascade-09-BlobBoards.md`); every
finding left is tied to an open item above.
