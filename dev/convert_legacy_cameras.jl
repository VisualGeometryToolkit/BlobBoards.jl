# Convert legacy OpenCV-style camera JSONs ("camera_matrix" /
# "distortion_coefficients" — no schema tag, no pixel frame) to the canonical
# vgio/cameras store (v2 since VisualGeometryToolkitIo 7675965), IN PLACE, the way
# commit d767827 (2026-08-18) did for the primary dataset copy (to v1, then).
# Older dataset copies can still hold ad-hoc camera files; this converts them
# the same way. (The 2026-09-07 package-cleanup audit found 16 such files under
# one machine's data_dir(), which made the annotate smoke test error there with
# `KeyError: :schema`.)
#
# The frame is the whole point: the legacy intrinsics are expressed in the
# OpenCV pixel frame, so the CameraModel is built through the opencv→colmap
# frame map (VGC converts on the way in, +0.5 on cx/cy). The record carries no
# frame of its own — the model is canonical by construction and the store
# writes that frame (VisualGeometryToolkitIo 0.11 retired the camera record's
# `pixel_chart` keyword: a record labelled :opencv over a canonical K
# double-shifted, the exact mistake the original conversion's field-exact
# verification caught). The store is a name → `BoundedCamera` Dict.
#
# Every converted file is copied to <backup_root>/<same relative path> first,
# and the new store is reloaded and compared field-exact (K and lens params)
# BEFORE the original is replaced. Files that are not pinhole_radtan, or that
# lack image_width/image_height, are listed and left untouched.
#
# Usage (the checkout's own env has VisualGeometryToolkitIo + VisualGeometryToolkitCore):
#
#   cd BlobBoards.jl
#   R=/path/to/20251028_BlobBoards_Office      # the dataset root, data_dir()
#   FILES=$(grep -l '"camera_matrix"' $(find $R -name '*.json' -size -100k) | sed "s@$R/@@")
#   julia --startup-file=no --project=. dev/convert_legacy_cameras.jl \
#       "$R" "$R/_legacy_camera_backup_$(date +%F)" $FILES

import VisualGeometryToolkitIo as VGIo
using VisualGeometryToolkitCore: CameraModel, CameraCalibrationMat, EuclideanMap, PixelFrame, lens_parameters, BoundedCamera, Size2
using CoordinateTransformations: AffineMap
using JSON3

length(ARGS) >= 3 || error("usage: convert_legacy_cameras.jl <root> <backup_root> <relative json paths...>")
root, backup_root = ARGS[1], ARGS[2]
files = ARGS[3:end]

converted = String[]
skipped = String[]
for rel in files
    f = joinpath(root, rel)
    p = JSON3.read(read(f, String))
    legacy = haskey(p, :camera_matrix) && haskey(p, :distortion_coefficients)
    model_tag = String(get(p, :model, "pinhole_radtan"))
    sized = haskey(p, :image_width) && haskey(p, :image_height)
    if !(legacy && model_tag == "pinhole_radtan" && sized)
        why = !legacy ? "not a legacy store" : model_tag != "pinhole_radtan" ? "model = $model_tag" : "no image_width/image_height"
        push!(skipped, "$rel  ($why)")
        continue
    end
    K = CameraCalibrationMat(stack(p.camera_matrix)')          # row-major 3×3 → matrix
    coeffs = collect(Float64, p.distortion_coefficients)       # OpenCV order; VGIo.opencv_lens knows it
    model = CameraModel(
        EuclideanMap(PixelFrame{:opencv}, PixelFrame{:colmap}) ∘ AffineMap(K);
        lens = VGIo.opencv_lens(coeffs)
    )
    name = join(filter(!isempty, [String(get(p, :camera_id, "")), String(get(p, :lens_id, ""))]), " / ")
    isempty(name) && (name = splitext(basename(f))[1])
    store = Dict(Symbol(name) => BoundedCamera(model, Size2(Int(p.image_width), Int(p.image_height))))

    tmp = f * ".v3.tmp"
    VGIo.save(tmp, store)
    back = VGIo.load(CameraModel, tmp)
    back.intrinsics.K == model.intrinsics.K || error("K mismatch after reload: $rel")
    lens_parameters(back.lens) == lens_parameters(model.lens) || error("lens mismatch after reload: $rel")

    b = joinpath(backup_root, rel)
    mkpath(dirname(b))
    cp(f, b; force = true)
    mv(tmp, f; force = true)
    push!(converted, rel)
end

println("converted ", length(converted), " (originals under ", backup_root, "):")
foreach(r -> println("  ", r), converted)
println("skipped ", length(skipped), ":")
foreach(r -> println("  ", r), skipped)
