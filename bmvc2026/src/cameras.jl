# =============================================================================
# The cameras of record from the upstream calibration exports
# =============================================================================
#
# BlobBoards-Evaluation's `calibrate_pinhole` (OpenCV calibrateCamera on a
# chessboard) exported each session's camera as `pinhole_radtan` JSON:
# `camera_matrix` (OpenCV pixel frame), OpenCV's fourteen-coefficient
# `distortion_coefficients`, `reprojection_error` (a MEAN per-image error),
# `num_calibration_images`, `chessboard_size`, `square_size`, `flags`.
# scripts/import_cameras.jl ports each ONCE into the native vgio/cameras/v3
# camera of record through `upstream_camera`; nothing else reads the shape.

using VisualGeometryToolkitIO: opencv_lens
using VisualGeometryToolkitCore: CameraModel, BoundedCamera, Attributed, Size2

_flag_names(flags::Integer, names::AbstractDict) = join(
    [name for (bit, name) in sort!([(parse(Int, k), v) for (k, v) in names]) if flags & bit != 0],
    " | "
)

"""
    upstream_source(entry) -> String

The upstream export of one `[[cameras]]` entry: its `source`, a
data_root()-relative path, resolved through `data_root()`. An absolute
(machine) path is refused.
"""
function upstream_source(entry::AbstractDict)
    s = String(entry["source"])
    isabspath(s) && error("import_cameras: source $(repr(s)) is a machine path; give it relative to data_root()")
    return joinpath(data_root(), s)
end

"""
    upstream_camera(entry, upstream) -> (name, camera::Attributed{BoundedCamera}, facts)

The camera of one `[[cameras]]` entry of configs/import_cameras.toml (its
`[upstream]` table: `upstream`), built from its upstream export: K from
`camera_matrix` in OpenCV's pixel frame, the lens `opencv_lens` of the fitted
coefficients (every coefficient past them must be zero), the calibration block
from the export's own facts.
"""
function upstream_camera(entry::AbstractDict, UP::AbstractDict)
    src = upstream_source(entry)
    sha = _sha256(src)
    sha == entry["sha256"] || error("$(src): sha256 $(sha), the config of record states $(entry["sha256"])")
    d = JSON3.read(read(src, String))
    d.model == "pinhole_radtan" || error("$(src): model $(d.model), expected pinhole_radtan")
    M = d.camera_matrix
    (M[1][2] == 0 && M[2][1] == 0 && M[3][1] == 0 && M[3][2] == 0 && M[3][3] == 1) ||
        error("$(src): camera_matrix is not [fx 0 cx; 0 fy cy; 0 0 1]")
    K = SMatrix{3, 3, Float64}(M[1][1], 0, 0, 0, M[2][2], 0, M[1][3], M[2][3], 1)
    k = Float64.(collect(d.distortion_coefficients))
    n = UP["n_fitted"]
    length(k) == 14 || error("$(src): $(length(k)) distortion coefficients, expected OpenCV's 14")
    all(iszero, k[(n + 1):end]) || error("$(src): coefficients past the fitted $(n) are not zero: $(k[(n + 1):end])")
    model = CameraModel(K; lens = opencv_lens(k[1:n]), image_origin = Symbol(UP["pixel_frame"]))
    cb, sq = collect(d.chessboard_size), d.square_size
    calibration = (
        tool = String(UP["tool"]),
        source = "$(basename(src)) (sha256 $(sha))",
        date = String(entry["date"]),
        n_images = Int(d.num_calibration_images),
        note = "camera $(d.camera_id), lens $(d.lens_id); mean reprojection error $(d.reprojection_error) px " *
            "(calibrate_pinhole's statistic: the per-image mean point error averaged over images — not an RMS); " *
            "chessboard $(cb[1])×$(cb[2]) inner corners, square $(sq) m; flags $(d.flags) = $(_flag_names(d.flags, UP["flags_names"])); " *
            "principal point converted from OpenCV's pixel frame; date = the export file's modification date " *
            "(the export records no fit date)",
    )
    camera = Attributed(BoundedCamera(model, Size2(Int(d.image_width), Int(d.image_height))), (; calibration))
    return Symbol(String(entry["name"])), camera, (; src, sha)
end
