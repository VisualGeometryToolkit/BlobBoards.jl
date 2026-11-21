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
     (VisualGeometryToolkitIO `opencv_parameters`) — as `camera_opencv.json`;
  2. runs the driver, which writes one export per image under `exports/` and
     the overlays under `val/` (and `occlusion/<LL>/val/`);
  3. turns each export directory into the evaluation's kind,
     `poses/<system>/target_poses.json` + `telemetry.jsonl`
     (`export_target_poses`; also `poses/calibration/<system>/` and
     `occlusion/<LL>/poses/<system>/`);
  4. Eval's `produce` writes the run records.

Everything lands in the product `…/bmvc2026/baselines/<key>` (src/runs.jl
`baselines_product`, Eval's `Product`); a produced run is reused.
=#
# reproduce: reads configs/baselines.toml

using DrWatson
@quickactivate

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: figure_config, baselines_product, input_path, export_target_poses
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitIO: opencv_parameters
using VisualGeometryToolkitCore: BoundedCamera, value_of
using VisualGeometryToolkitEval.Experiments: produce, product_dir
import JSON

const CFG, CFG_PATH = figure_config("baselines"; default = projectdir("configs", "baselines.toml"))
const PRODUCT = baselines_product(CFG_PATH)
const KEY = PRODUCT.key
const RUN_ROOT = product_dir(PRODUCT)
const LEVELS = String.(CFG["occlusion"]["levels"])
const SESSION = input_path(CFG, "session")
const IMAGES = input_path(CFG, "images")
const CAMERA = input_path(CFG, "camera")
const OCCLUSION = Dict(l => input_path(CFG, "occlusion_$(l)") for l in LEVELS)
const EXPORTS = joinpath(RUN_ROOT, "exports")
const DRIVER = joinpath(@__DIR__, "baselines", "detect_markers.py")
const PIXI = joinpath(dirname(@__DIR__), "pixi.toml")
const OPENCV_KEYS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tilt_x", "tilt_y")

# The camera of record in OpenCV's convention, for the driver.
function write_opencv_camera(path)
    store = VgtkIO.load(Dict{Symbol, BoundedCamera}, CAMERA)
    name, c = only(store)
    (; K, dist) = opencv_parameters(value_of(c).camera)   # IO: OpenCV's K (its pixel frame) and dist; a skewed K is refused
    body = merge(
        (fx = K[1, 1], fy = K[2, 2], cx = K[1, 3], cy = K[2, 3]),
        NamedTuple{Tuple(Symbol.(OPENCV_KEYS[1:length(dist)]))}(Tuple(dist)),
        (pixel_frame = "opencv", camera = String(name), source = basename(CAMERA)),
    )
    open(io -> JSON.json(io, body; pretty = 4), path, "w")
    return (; name = String(name), K)
end

function main()
    mkpath(RUN_ROOT)
    cam = write_opencv_camera(joinpath(RUN_ROOT, "camera_opencv.json"))
    run(`pixi run --manifest-path $(PIXI) -e baselines python $(DRIVER) --config $(CFG_PATH)
        --camera $(joinpath(RUN_ROOT, "camera_opencv.json")) --session $(SESSION)
        --exports $(EXPORTS) --val $(joinpath(RUN_ROOT, "val"))`)
    prov = (driver = relpath(DRIVER, dirname(dirname(@__DIR__))), run_key = KEY)   # bmvc2026/scripts/baselines/…
    for (system, sub) in ((s, CFG["subsets"][s]) for s in ("apriltags", "aruco"))
        parts = [
            (joinpath("poses", system), joinpath(IMAGES, sub["images"])),
            (joinpath("poses", "calibration", system), joinpath(IMAGES, sub["calibration"])),
            ((joinpath("occlusion", l, "poses", system), joinpath(OCCLUSION[l], sub["images"])) for l in LEVELS)...,
        ]
        for (rel, images) in parts
            out = export_target_poses(joinpath(EXPORTS, rel), images, joinpath(RUN_ROOT, rel);
                camera_name = cam.name, K = cam.K, provenance = merge(prov, (image_subset = relpath(images, SESSION),)))
            println("$(system): $(rel) -> $(out)")
        end
    end
    println("baselines -> $(RUN_ROOT)")
    return nothing
end

produce(_ -> main(), PRODUCT; driver = "bmvc2026/scripts/detect_baselines.jl", config_source = CFG_PATH,
        inputs = [SESSION, IMAGES, CAMERA, (OCCLUSION[l] for l in LEVELS)...])
println("detect_baselines: run $(KEY) at $(RUN_ROOT)")
