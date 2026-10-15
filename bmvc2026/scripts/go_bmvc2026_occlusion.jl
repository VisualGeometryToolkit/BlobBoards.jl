#=
go_bmvc2026_occlusion.jl — BlobBoards absolute-pose estimation for the BMVC 2026
*occlusion sweep*. Same calibrated `find_boards` (P3P) path as go_bmvc2026.jl,
run across the synthetic occlusion levels (inputs: `<dataset>/occlusion/<LL>/
images/bb`, made by the external generator BlobBoards-Evaluation 88bca9f
`evaluation_pipe/occlusion_generator.py`; configs/data_artifacts.toml). Per
level writes:

  <run>/occlusion/<LL>/poses/bb/target_poses.json — the detected poses (+ telemetry.jsonl)
  <run>/occlusion/<LL>/val/bb/<image>_val.png      — colored-board overlay

where <run> is results_root()/occlusion-<key> (src/runs.jl `sweep_dir(config;
kind = "occlusion")`): the same run config of record as the main sweep, the
levels its `[experiment] occlusion_levels`. Each level is bb-only (no
calibration subset); the camera, boards and gallery are the parent session's.

Run:  JULIA_NUM_THREADS=auto julia --project=bmvc2026 bmvc2026/scripts/go_bmvc2026_occlusion.jl [--run-config=<run toml>]
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
const CONFIG_PATH = let given = [a[(length("--run-config=") + 1):end] for a in ARGS if startswith(a, "--run-config=")]
    isempty(given) ? BBE.config_path("bmvc2026_260527.toml") : abspath(only(given))
end
const RUN = read_run_config(CONFIG_PATH)
const LEVELS = String.(RUN.experiment["occlusion_levels"])
const OCC_TIERS = ["bb_absolute_pose_occlusion_$(l)_260527" for l in LEVELS]
const DATASET = joinpath(BBE.data_root(; tiers = vcat(BBE.EVAL_TIERS, OCC_TIERS)), "absolute_pose_experiments", RUN.experiment["dataset"])
const CAMERA_JSON = BBE.camera_of_record(DATASET)
const GALLERY = BBE.gallery_path(RUN.config_toml)
const BOARDS_DIR = joinpath(DATASET, "boards", "blobboards")         # shared across levels
const FRAME_LIMIT = Int(get(RUN.experiment, "limit", 0))

mkpath(dirname(GALLERY))
ctx = sweep_context(;
    config_path = CONFIG_PATH, run_config = RUN,
    network_path = BBE.network_path(RUN.config_toml),
    gallery_path = GALLERY, boards_dir = BOARDS_DIR,
    camera_json = CAMERA_JSON, device = CUDABackend()
)

const RUN_KEY = BBE.sweep_key(CONFIG_PATH; kind = "occlusion")
const RUN_ROOT = BBE.sweep_dir(CONFIG_PATH; kind = "occlusion")

units = [
    SweepUnit(
            "occ$level",
            joinpath(DATASET, "occlusion", level, "images", "bb"),
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
    force = get(ENV, "BB_FORCE", "") == "1"
)

println("\nDone. Run $RUN_KEY -> $RUN_ROOT")
