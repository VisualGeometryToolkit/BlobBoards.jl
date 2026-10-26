# =============================================================================
# Per-image detection exports → vgio/target_poses/v1 + telemetry.jsonl
# =============================================================================
#
# The AprilTag/ArUco baselines (scripts/baselines/detect_markers.py, the
# faithful port of BlobBoards-Evaluation's pose_extractor.py) write one JSON per
# image, `{image, board_type, undist_K, elapsed_time_ms, poses: [{board_id,
# rot_mat (3×3 rows), trans (mm), size_mm}]}`. The evaluation reads every
# marker system the same way, one `target_poses.json` per run and subset
# (`load_detected_poses`), so the baseline stage (scripts/detect_baselines.jl)
# turns each export directory into that kind here, ONCE, in its own run
# directory. The per-image files stay beside it as the driver wrote them.

import VisualGeometryToolkitIO as VgtkIO
import VisualGeometryToolkitIO
using VisualGeometryToolkitIO: append_jsonl
using VisualGeometryToolkitCore: TargetPoses, CameraFrame, EuclideanMap, geom
using StaticArrays: SMatrix, SVector
using Rotations: RotMatrix

"""
    export_target_poses(export_dir, image_dir, out_dir; camera_name, K, provenance = (;)) -> String

Read the per-image exports in `export_dir` (one `<stem>.json` per image of
`image_dir`) and write `out_dir/target_poses.json` (every frame's image, the
camera `camera_name` as its sensor, each detected target's pose `target:<id>
→ camera_optical`, mm) and `out_dir/telemetry.jsonl` (one `{image, t,
elapsed_time_ms}` per image). Frames are indexed by the image's position in
the sorted listing of `image_dir`. An export's `undist_K` must be `K` (the
camera of record's, canonical frame) in OpenCV's pixel frame to float32
precision — the camera it says it used; `size_mm` is dropped (a target's size
is its definition's). Rotations read back within 1e-6 (an export's float32
matrix is projected onto SO(3) by the quaternion encoding); translations
exactly. Returns the path of `target_poses.json`.
"""
function export_target_poses(
        export_dir::AbstractString, image_dir::AbstractString, out_dir::AbstractString;
        camera_name::AbstractString, K::AbstractMatrix, provenance::NamedTuple = (;),
    )
    mkpath(out_dir)
    images = sort(filter(_is_image, readdir(image_dir)))
    index = Dict(splitext(f)[1] => (t, f) for (t, f) in enumerate(images))
    Kc = SMatrix{3, 3, Float64}(K)
    frames = Dict{Int, Dict{Symbol, EuclideanMap}}()
    names = Dict{Int, String}()
    rows = Tuple{Int, NamedTuple}[]
    files = sort(filter(f -> endswith(f, ".json") && !startswith(f, "_"), readdir(export_dir)))
    for f in files
        d = JSON.parse(read(joinpath(export_dir, f), String))
        stem = splitext(f)[1]
        haskey(index, stem) || error("$(export_dir)/$(f): no image $(stem).* in $(image_dir)")
        t, img = index[stem]
        String(d.image) == img || error("$(export_dir)/$(f): names image $(d.image), the subset has $(img)")
        Ku = SMatrix{3, 3, Float64}(reduce(hcat, [Float64.(row) for row in d.undist_K])')
        D = Ku - Kc
        (abs(D[1, 3] + 0.5) < 1.0e-3 && abs(D[2, 3] + 0.5) < 1.0e-3 && maximum(abs, D[:, 1:2]) < 1.0e-3) ||
            error("$(export_dir)/$(f): undist_K $(Ku) is not the camera of record's K $(Kc) in OpenCV's pixel frame")
        frames[t] = Dict(
            Symbol(String(p.board_id)) => EuclideanMap(
                RotMatrix{3, Float64}(SMatrix{3, 3, Float64}(reduce(hcat, [Float64.(row) for row in p.rot_mat])')),
                SVector{3, Float64}(p.trans) .* mm; check = false,   # the export's `trans` is millimetres; a file takes lengths as quantities
            ) for p in d.poses
        )
        names[t] = img
        push!(rows, (t, (image = img, t = t, elapsed_time_ms = Float64(d.elapsed_time_ms))))
    end
    length(files) == length(images) || error("$(export_dir): $(length(files)) exports for $(length(images)) images of $(image_dir)")
    tel = joinpath(out_dir, "telemetry.jsonl")
    isfile(tel) && rm(tel)
    for (_, r) in sort(rows; by = first)
        append_jsonl(tel, r)
    end
    tp = TargetPoses{CameraFrame}(frames; images = names, sensors = String(camera_name))
    out = joinpath(out_dir, "target_poses.json")
    VgtkIO.save(
        out, tp;
        provenance = merge(
            (
                source_format = "per-image detection export {image, board_type, undist_K, elapsed_time_ms, " *
                    "poses: [{board_id, rot_mat (3×3 rows), trans (mm), size_mm}]}",
                n_files = length(files),
                undist_K = "the camera of record's K in OpenCV's pixel frame (checked per image)",
                size_mm = "dropped: a target's size is its definition's (boards/<system>/)",
                elapsed_time_ms = "moved to telemetry.jsonl",
            ),
            provenance,
        ),
    )
    back = VgtkIO.load(TargetPoses, out)
    for (t, ps) in frames, (id, X) in ps
        Y = geom(back[t][id])
        Y.t == X.t || error("$(out): frame $(t) target $(id) translation does not read back")
        maximum(abs, SMatrix{3, 3}(Y.R) - SMatrix{3, 3}(X.R)) < 1.0e-6 || error("$(out): frame $(t) target $(id) rotation does not read back")
    end
    return out
end
