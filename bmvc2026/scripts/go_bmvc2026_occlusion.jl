#=
go_bmvc2026_occlusion.jl — BlobBoards absolute-pose estimation for the BMVC 2026
*occlusion sweep*. Same calibrated `find_boards` (P3P) path as go_bmvc2026.jl,
run across the synthetic occlusion levels (inputs: `<dataset>/occlusion/<LL>/
images/bb`, made by the external generator BlobBoards-Evaluation 88bca9f
`evaluation_pipe/occlusion_generator.py`; configs/data_artifacts.toml). Per
level writes:

  <run>/occlusion/<LL>/poses/bb/target_poses.json — the detected poses (+ telemetry.jsonl)
  <run>/occlusion/<LL>/val/bb/<image>_val.png      — colored-board overlay

where <run> is the run config's keyed directory `…/bmvc2026/occlusion-<key>`
(src/runs.jl `sweep_dir(config; kind = "occlusion")`, Eval's `keyed_cell_dir`):
the same run config of record as the main sweep, the
levels its `[experiment] occlusion_levels`. Each level is bb-only (no
calibration subset); the camera, boards and gallery are the parent session's.

Run:  JULIA_NUM_THREADS=auto julia --project=bmvc2026 bmvc2026/scripts/go_bmvc2026_occlusion.jl [--run-config=<name under configs/>] [--force]
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
# Configuration — occlusion sweep
# =============================================================================
# ONE config of record, shared with the main sweep (the occlusion curve and the
# headline numbers come from the same estimator, the same network, the same
# gallery).
using VisualGeometryToolkitEval: cli_option
const CONFIG_PATH = BBE.config_path(cli_option(ARGS, "run-config", "bmvc2026_260527.toml"))
const RUN = read_run_config(CONFIG_PATH)
const LEVELS = String.(RUN.experiment["occlusion_levels"])
const CAMERA_JSON = BBE.input_path(RUN.config_toml, "camera")
const GALLERY = BBE.gallery_path(RUN.config_toml)
const BOARDS_DIR = BBE.input_path(RUN.config_toml, "boards")         # shared across levels
const FRAME_LIMIT = Int(RUN.experiment["limit"])

mkpath(dirname(GALLERY))
ctx = sweep_context(;
    config_path = CONFIG_PATH, run_config = RUN,
    network_path = BBE.network_path(RUN.config_toml),
    gallery_path = GALLERY, boards_dir = BOARDS_DIR,
    camera_json = CAMERA_JSON, device = CUDABackend()
)

const RUN_KEY = BBE.sweep_key(CONFIG_PATH)
const RUN_ROOT = BBE.sweep_dir(CONFIG_PATH; kind = "occlusion")

units = [
    SweepUnit(
            "occ$level",
            joinpath(BBE.input_path(RUN.config_toml, "occlusion_$(level)"), "bb"),
            joinpath(RUN_ROOT, "occlusion", level, "poses", "bb"),
            joinpath(RUN_ROOT, "occlusion", level, "val", "bb")
        ) for level in LEVELS
]

run_sweep(
    units, ctx;
    limit = FRAME_LIMIT,
    driver = "bmvc2026/scripts/go_bmvc2026_occlusion.jl",
    write_provenance = BBE.write_provenance,
    extra_provenance = Dict("levels" => join(LEVELS, ","), "run_key" => RUN_KEY, "config_source" => CONFIG_PATH, "camera" => CAMERA_JSON),
    force = "--force" in ARGS
)

println("\nDone. Run $RUN_KEY -> $RUN_ROOT")
