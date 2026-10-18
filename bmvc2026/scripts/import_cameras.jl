#=
import_cameras.jl — the ONE-TIME port of each session's upstream intrinsic
calibration into VisualGeometryToolkitIo's native vgio/cameras/v3 (the camera of
record the inputs tier publishes).

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/import_cameras.jl [--config=<toml>]

The upstream export is BlobBoards-Evaluation `calibrate_pinhole`'s
`pinhole_radtan` JSON: `camera_matrix` (OpenCV pixel frame), OpenCV's
fourteen-coefficient `distortion_coefficients`, `reprojection_error` (a MEAN
per-image error), `num_calibration_images`, `chessboard_size`, `square_size`,
`flags`. This script reads it ONCE (src/cameras.jl `upstream_camera`): the
camera is built in VisualGeometryToolkitCore (`CameraModel(K; lens = opencv_lens(k), image_origin =
:opencv)`, which moves the principal point into the canonical frame exactly),
its calibration block is filled from the export, and it is written ONLY through
VisualGeometryToolkitIo's `save`, read back, and checked equal.

Writes, add-only, under results_root()/import_cameras-<run key>/:
    <session>/calibration/<output>     vgio/cameras/v3, the camera of record
    _config.toml, _provenance.toml     the run's records (source paths + sha256)

The outputs are then packed into the inputs tier by make_zenodo_artifacts.jl;
test/cameras_tests.jl holds each to the previous camera of record, K and lens
bit for bit.

# reproduce: reads configs/import_cameras.toml
=#

using DrWatson
@quickactivate

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: run_dir, figure_config, upstream_camera
import VisualGeometryToolkitIo as VGIo
using VisualGeometryToolkitCore: BoundedCamera
using VisualGeometryToolkitEval.Experiments: run_key, write_run_records

const CFG, CFG_PATH = figure_config("import_cameras"; default = projectdir("configs", "import_cameras.toml"))
const KEY = run_key(CFG)
const OUT = run_dir("import_cameras", KEY)
function main()
    written, inputs = String[], String[]
    for entry in CFG["cameras"]
        name, camera, (; src, sha) = upstream_camera(entry, CFG["upstream"])
        out = joinpath(OUT, String(entry["session"]), "calibration", String(entry["output"]))
        mkpath(dirname(out))
        store = Dict(name => camera)
        VGIo.save(out, store; provenance = (converted_from = basename(src), source_sha256 = sha, run_key = KEY))
        VGIo.load(Dict{Symbol, BoundedCamera}, out) == store || error("$(out): the camera does not read back equal")
        push!(written, out); push!(inputs, src)
        println("$(entry["session"]): $(name) -> $(out)")
    end
    write_run_records(OUT, CFG; driver = "bmvc2026/scripts/import_cameras.jl", config_source = CFG_PATH, key = KEY, inputs)
    return nothing
end

isfile(joinpath(OUT, "_provenance.toml")) ? println("import_cameras: run $(KEY) complete at $(OUT)") : main()
