# BlobBoards Examples

Minimal demos of the two supported workflows:

## Board generation
- `generate/board.jl` — printable calibration board (PDF, or TIFF/PNG at a
  density, + JSON), standard paper sizes, polarity, frame styles; the smallest
  disc is sized for the printer with `suggest_blob_sizes(density, paper)`
- `generate/pattern.jl` — standalone blob pattern (PNG + JSON)

## Board finding
- `pipeline/find_boards.jl` — end-to-end detection of boards in a photograph
  (requires CUDA; loads a descriptor model + board gallery, displays results)

## Print quality
- `print_quality/print_quality.jl` — how faithfully a printer reproduces the
  blob profile, per blob size, from a scan of a printed test board: `generate`
  the A4 test board (discs 0.4–12 mm), `selfcheck` the measurement's floor on a
  perfect render, `evaluate` a scan (finds the board with `find_boards`, maps
  the known blobs onto the scan, fits out the tone curve; requires CUDA)

Python one-to-one equivalents live in `python/examples/`
(`boards/generate_a4_board.py`, `find_boards.py`, `blob_extraction.py`).

Research, evaluation, and training material (adaptation studies, ASIFT
benchmarks, annotation/SLURM drivers) is in `bmvc2026/_research/` — functional but
unsupported.
