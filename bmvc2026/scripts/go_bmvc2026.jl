#=
go_bmvc2026.jl — BlobBoards absolute-pose estimation for the BMVC 2026
experiments. Runs the calibrated `find_boards` (P3P) path over the dataset's
BlobBoard image sets (inputs) and writes, per subset:

  <run>/poses/<dst>/target_poses.json — the detected poses (+ telemetry.jsonl,
                                        _config.toml, _provenance.toml)
  <run>/val/<dst>/<image>_val.png      — colored-board overlay on the undistorted
                                        color image

where <run> is the run config's keyed directory `…/bmvc2026/bb-<key>` (src/runs.jl
`sweep_dir`: Eval's `keyed_cell_dir`; the key states the network's sha256
through its [inputs.network]): the evaluation finds the
run through the same config, so nothing is promoted or copied anywhere. The
sweep is ADD-ONLY; a complete subset is skipped.

Subsets ([experiment] subsets of the config):
  images/bb              -> <run>/poses/bb,             <run>/val/bb
  images/calibration_bb  -> <run>/poses/calibration/bb, <run>/val/calibration/bb

The camera is the session's camera of record (an input, calibration/<camera>.json).
The sweep itself lives in `src/sweep.jl`, shared with go_bmvc2026_occlusion.jl.

Run:  JULIA_NUM_THREADS=auto julia --project=bmvc2026 bmvc2026/scripts/go_bmvc2026.jl [--run-config=<name under configs/>] [--force]
      [--branch-log=<jsonl>] [--appearance-log=<jsonl>] [--phase2-log=<jsonl>]
(CUDA + ONNXRunTime cuda EP; the ONNX descriptor network the config names.)
=#

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

using BlobBoards
using CairoMakie; CairoMakie.activate!()           # headless render for the val PNGs
using CUDA
import cuDNN
using CUDA: CUDABackend

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
const BBE = Main.BlobBoardsEval
include(joinpath(@__DIR__, "..", "src", "sweep.jl"))
using .BoardSweep: SweepUnit, sweep_context, run_sweep

# =============================================================================
# Configuration
# =============================================================================
# ONE config file is the source of truth for this run: configs/bmvc2026_260527.toml,
# or the run config named by `--run-config=<path>` (the no-disambiguation ablation
# is configs/bmvc2026_260527_noflip.toml). No env knobs: ambient values that
# nothing recorded are how a stale BB_LIMIT=40 silently truncated a 56-image
# sweep while looking perfectly clean.
using VisualGeometryToolkitEval: cli_option
const CONFIG_PATH = BBE.config_path(cli_option(ARGS, "run-config", "bmvc2026_260527.toml"))
const RUN = read_run_config(CONFIG_PATH)
const IMAGES = BBE.input_path(RUN.config_toml, "images")
const CAMERA_JSON = BBE.input_path(RUN.config_toml, "camera")
const GALLERY = BBE.gallery_path(RUN.config_toml)
const BOARDS_DIR = BBE.input_path(RUN.config_toml, "boards")
# Frame cap for smoke runs: the config's [experiment] limit, a stated literal
# (so it is in the run key), NOT an env var: a cap that is not recorded beside
# the output is a lie. `select_frames` in src/sweep.jl STRIDES.
const FRAME_LIMIT = Int(RUN.experiment["limit"])

mkpath(dirname(GALLERY))
ctx = sweep_context(;
    config_path = CONFIG_PATH, run_config = RUN,
    network_path = BBE.network_path(RUN.config_toml),
    gallery_path = GALLERY, boards_dir = BOARDS_DIR,
    camera_json = CAMERA_JSON, device = CUDABackend()
)

# The run's directory: its config of record and network content (src/runs.jl).
const RUN_KEY = BBE.sweep_key(CONFIG_PATH)
const RUN_ROOT = BBE.sweep_dir(CONFIG_PATH)

const ALL_UNITS = [
    SweepUnit(
        "bb",
        joinpath(IMAGES, "bb"),
        joinpath(RUN_ROOT, "poses", "bb"),
        joinpath(RUN_ROOT, "val", "bb")
    ),
    SweepUnit(
        "calibration_bb",
        joinpath(IMAGES, "calibration_bb"),
        joinpath(RUN_ROOT, "poses", "calibration", "bb"),
        joinpath(RUN_ROOT, "val", "calibration", "bb")
    ),
]
const UNITS = filter(u -> u.label in RUN.experiment["subsets"], ALL_UNITS)

# Telemetry sinks — experiment instrumentation, OFF unless an argument names a
# path (never the environment: a side effect no record states). --branch-log
# is an OUTPUT path here and an INPUT path of branch_lrt_accuracy.jl — the same
# name for both ends of the same file. --appearance-log: one JSON line per
# gated candidate (appearance-threshold calibration). --phase2-log: one line
# per frame with the phase-1/phase-2 acceptance counts — the measurement behind
# the keep-or-delete-phase-2 decision. Supplied TO find_boards per frame
# (`with_tag`), never set on the package.
_log_path(k) = (v = cli_option(ARGS, k, ""); isempty(v) ? nothing : abspath(v))
const TELEMETRY = let t = BlobBoards.Telemetry(;
        branch_log = _log_path("branch-log"),
        appearance_log = _log_path("appearance-log"),
        phase2_log = _log_path("phase2-log"),
    )
    all(isnothing, (t.branch_log, t.appearance_log, t.phase2_log)) ? nothing : t
end

run_sweep(
    UNITS, ctx;
    limit = FRAME_LIMIT, telemetry = TELEMETRY,
    driver = "bmvc2026/scripts/go_bmvc2026.jl",
    write_provenance = BBE.write_provenance,
    extra_provenance = Dict("run_key" => RUN_KEY, "config_source" => CONFIG_PATH, "camera" => CAMERA_JSON),
    force = "--force" in ARGS
)

println("\nDone. Run $RUN_KEY -> $RUN_ROOT")
