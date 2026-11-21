#=
import_cameras.jl — the ONE-TIME port of each session's upstream intrinsic
calibration into VisualGeometryToolkitIO's native vgio/cameras/v3 (the camera of
record the inputs tier publishes).

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/import_cameras.jl [--config=<toml>]

The upstream export is BlobBoards-Evaluation `calibrate_pinhole`'s
`pinhole_radtan` JSON: `camera_matrix` (OpenCV pixel frame), OpenCV's
fourteen-coefficient `distortion_coefficients`, `reprojection_error` (a MEAN
per-image error), `num_calibration_images`, `chessboard_size`, `square_size`,
`flags`. This script reads it ONCE (src/cameras.jl `upstream_camera`): the
camera is built in VisualGeometryToolkitCore (`CameraModel(K; lens = opencv_lens(k), pixel_frame =
:opencv)`, which moves the principal point into the canonical frame exactly),
its calibration block is filled from the export, and it is written ONLY through
VisualGeometryToolkitIO's `save`, read back, and checked equal.

Writes, add-only, in its product `…/bmvc2026/import_cameras/<key>/` (Eval's `Product`):
    <session>/calibration/<output>     vgio/cameras/v3, the camera of record
    _cameras.toml                      the cameras written (its output)
    _config.toml, _provenance.toml     the run's records (source paths + sha256)

The outputs, placed at <session>/calibration/ of the author's tree, are packed
into the inputs tier by make_zenodo_artifacts.jl (configs/data_artifacts.toml);
test/cameras_tests.jl holds each to the previous camera of record, K and lens
bit for bit.

# reproduce: reads configs/import_cameras.toml
=#

using DrWatson
@quickactivate

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: figure_config, upstream_camera, stage_product
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: BoundedCamera
using VisualGeometryToolkitEval.Experiments: produce, product_dir, Produced
import TOML

const CFG, CFG_PATH = figure_config("import_cameras"; default = projectdir("configs", "import_cameras.toml"))
const PRODUCT = stage_product(CFG, "import_cameras"; output = "_cameras.toml")
const KEY = PRODUCT.key
const OUT = product_dir(PRODUCT)
function main()
    written, inputs = String[], String[]
    for entry in CFG["cameras"]
        name, camera, (; src, sha) = upstream_camera(CFG, entry)
        out = joinpath(OUT, String(entry["session"]), "calibration", String(entry["output"]))
        mkpath(dirname(out))
        store = Dict(name => camera)
        VgtkIO.save(out, store; provenance = (converted_from = basename(src), source_sha256 = sha, run_key = KEY))
        VgtkIO.load(Dict{Symbol, BoundedCamera}, out) == store || error("$(out): the camera does not read back equal")
        push!(written, out); push!(inputs, src)
        println("$(entry["session"]): $(name) -> $(out)")
    end
    open(io -> TOML.print(io, Dict("cameras" => [relpath(w, OUT) for w in written])), joinpath(OUT, "_cameras.toml"), "w")
    return Produced(nothing; inputs)
end

produce(_ -> main(), PRODUCT; driver = "bmvc2026/scripts/import_cameras.jl", config_source = CFG_PATH)
println("import_cameras: run $(KEY) at $(OUT)")
