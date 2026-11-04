# Changelog

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
