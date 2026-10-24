#=
detect_baselines.jl — the AprilTag and ArUco baselines, regenerated from the
inputs: every image subset of the session (the evaluation captures, the
calibration captures, every synthetic occlusion level).

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/detect_baselines.jl [--config=<toml>]

The detectors are BlobBoards-Evaluation 88bca9f's (dt-apriltags, OpenCV ArUco,
per-tag IPPE-square PnP on the undistorted image), ported faithfully to
scripts/baselines/detect_markers.py and run in the pinned pixi environment
`baselines` (bmvc2026/pixi.toml) with the settings of the config of record
(configs/baselines.toml). This stage:

  1. writes the camera of record (calibration/<camera>.json, an input) in
     OpenCV's convention — K with the principal point in OpenCV's pixel frame
     (VisualGeometryToolkitCore's frame map), the distortion vector by name
     (VisualGeometryToolkitIO `opencv_coefficients`) — as `camera_opencv.json`;
  2. runs the driver, which writes one export per image under `exports/` and
     the overlays under `val/` (and `occlusion/<LL>/val/`);
  3. turns each export directory into the evaluation's kind,
     `poses/<system>/target_poses.json` + `telemetry.jsonl`
     (`export_target_poses`; also `poses/calibration/<system>/` and
     `occlusion/<LL>/poses/<system>/`);
  4. writes the run records.

Everything lands in results_root()/baselines-<run key> (src/runs.jl
`baselines_dir`); a run whose records exist is complete and is skipped.
=#
# reproduce: reads configs/baselines.toml

using DrWatson
@quickactivate

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: data_root, figure_config, run_dir, camera_of_record, export_target_poses, EVAL_TIERS
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitIO: opencv_coefficients
using VisualGeometryToolkitCore: BoundedCamera, EuclideanMap, PixelFrame
using VisualGeometryToolkitEval.Experiments: run_key, write_run_records
import JSON

const CFG, CFG_PATH = figure_config("baselines"; default = projectdir("configs", "baselines.toml"))
const KEY = run_key(CFG)
const RUN_ROOT = run_dir("baselines", KEY)
const LEVELS = String.(CFG["occlusion"]["levels"])
const TIERS = vcat(EVAL_TIERS, ["bb_absolute_pose_occlusion_$(l)_260527" for l in LEVELS])
const SESSION = joinpath(data_root(; tiers = TIERS), "absolute_pose_experiments", CFG["experiment"]["dataset"])
const EXPORTS = joinpath(RUN_ROOT, "exports")
const DRIVER = joinpath(@__DIR__, "baselines", "detect_markers.py")
const PIXI = joinpath(dirname(@__DIR__), "pixi.toml")
const OPENCV_KEYS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tilt_x", "tilt_y")

# The camera of record in OpenCV's convention, for the driver.
function write_opencv_camera(path)
    store = VgtkIO.load(Dict{Symbol, BoundedCamera}, camera_of_record(SESSION))
    name, c = only(store)
    bc = c isa BoundedCamera ? c : c.value
    K = bc.camera.intrinsics.K                        # canonical frame (VisualGeometryToolkitIO's CANONICAL_PIXEL_FRAME)
    iszero(K[1, 2]) || error("the camera of record has skew; OpenCV's camera matrix has none")
    shift = EuclideanMap(VgtkIO.CANONICAL_PIXEL_FRAME, PixelFrame{:opencv}).t
    k = opencv_coefficients(bc.camera.lens)
    body = merge(
        (fx = Float64(K[1, 1]), fy = Float64(K[2, 2]), cx = Float64(K[1, 3]) + shift[1], cy = Float64(K[2, 3]) + shift[2]),
        NamedTuple{Tuple(Symbol.(OPENCV_KEYS[1:length(k)]))}(Tuple(k)),
        (image_origin = "opencv", camera = String(name), source = basename(camera_of_record(SESSION))),
    )
    open(io -> JSON.json(io, body; pretty = 4), path, "w")
    return (; name = String(name), K)
end

function main()
    isdir(RUN_ROOT) && error("$(RUN_ROOT) exists without its run records — an interrupted run; move it aside to redo")
    mkpath(RUN_ROOT)
    cam = write_opencv_camera(joinpath(RUN_ROOT, "camera_opencv.json"))
    run(`pixi run --manifest-path $(PIXI) -e baselines python $(DRIVER) --config $(CFG_PATH)
        --camera $(joinpath(RUN_ROOT, "camera_opencv.json")) --session $(SESSION)
        --exports $(EXPORTS) --val $(joinpath(RUN_ROOT, "val"))`)
    prov = (driver = relpath(DRIVER, dirname(dirname(@__DIR__))), run_key = KEY)   # bmvc2026/scripts/baselines/…
    for (system, sub) in ((s, CFG["subsets"][s]) for s in ("apriltags", "aruco"))
        parts = [
            (joinpath("poses", system), joinpath(SESSION, "images", sub["images"])),
            (joinpath("poses", "calibration", system), joinpath(SESSION, "images", sub["calibration"])),
            ((joinpath("occlusion", l, "poses", system), joinpath(SESSION, "occlusion", l, "images", sub["images"])) for l in LEVELS)...,
        ]
        for (rel, images) in parts
            out = export_target_poses(joinpath(EXPORTS, rel), images, joinpath(RUN_ROOT, rel);
                camera_name = cam.name, K = cam.K, provenance = merge(prov, (image_subset = relpath(images, SESSION),)))
            println("$(system): $(rel) -> $(out)")
        end
    end
    write_run_records(RUN_ROOT, CFG; driver = "bmvc2026/scripts/detect_baselines.jl", config_source = CFG_PATH, key = KEY,
        inputs = [SESSION])
    println("baselines -> $(RUN_ROOT)")
    return nothing
end

isfile(joinpath(RUN_ROOT, "_provenance.toml")) ? println("detect_baselines: run $(KEY) complete at $(RUN_ROOT)") : main()
