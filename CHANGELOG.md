# Changelog

## 0.5.0 (Unreleased)

On the 0.6 series of the VisualGeometryToolkit stack, and every JSON file
through VisualGeometryToolkitIO's file boundary.

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
- CompatHelper adds CauRegistry, where LocalOptimizationDoneRight is until its
  General registration merges.
