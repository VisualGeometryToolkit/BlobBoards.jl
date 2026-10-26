#=
import VisualGeometryToolkitCore as VgtkCore
Find Boards on a FISHEYE image — the uncalibrated radial arm, side by side
with the calibrated path.

Sony FX3 + Sigma 8mm fisheye at f/8: 4240x2832 stills from the office
dataset (`data_dir()/sonyfx3_sigma8mm_fisheye_f8/jpeg`). The lens bends
straight lines hard enough that a plain homography leaks inliers toward the
frame edges. Three geometries on the SAME image:

  • HomographyGeometry()               → plain 8-DOF H (baseline)
  • RadialHomographyGeometry((W, H))   → (H, k) about the image middle —
                                         UNCALIBRATED, detector on the raw
                                         fisheye frame
  • PoseGeometry(cam)                  → calibrated P3P through the session's
                                         `intrinsics.json` (Brown–Conrady
                                         fisheye approximation; the image is
                                         undistorted before detection)

The finale: the raw frame next to the image UNDISTORTED THROUGH THE FITTED
(H, k) — `undistort(CameraModel(m), img)`, the fused KernelAbstractions
division warp, no calibration anywhere — and the calibrated Brown–Conrady
undistortion for comparison.

Sanity anchor: to first order the division and BC radial terms agree, so
the fitted pixel-frame k should land near k1/f² ≈ −0.162/1226² ≈
−1.1e-7 px⁻². Caveat spelled out: the calibrated distortion center
(cx, cy) ≈ (1772, 1609) sits ~350 px from the image middle the radial arm
assumes; the uncalibrated fit absorbs most of that offset into H at some
cost in k accuracy. Guided refit and tracking are not wired for the radial
arm (v1); its RANSAC + 9-dof chart-LM fit is the result.

Usage:
  include("examples/pipeline/find_boards_fisheye.jl")
=#

using BlobBoards
using GLMakie; GLMakie.activate!(render_on_demand = false)
using FileIO: load as load_image
using Colors: Gray
using GeometryBasics: Point2, Point2f, Rect, widths
using StaticArrays: SVector, SMatrix
using CUDA          # triggers GPU extensions (CPU works too — see DEVICE)
import cuDNN        # ONNXRunTime :cuda execution provider
using VisualGeometryToolkitCore: is_map_inlier, imshow, matches!, window!, ustrip, centroid,
    HomographyMap, HomographyMat, RadialHomography, radial_transfer,
    PosedCamera, CameraModel, Pinhole, undistort, division_bounded,
    sources, targets
using CoordinateTransformations: PerspectiveMap
import VisualGeometryToolkitIO as VgtkIO   # canonical vgio/cameras/v3 camera stores

# =============================================================================
# Config
# =============================================================================

const CUTOFF = BlobBoards.DEFAULT_CUTOFF
const DEVICE = default_device()   # CUDABackend when CUDA is functional

session_dir = joinpath(data_dir(), "sonyfx3_sigma8mm_fisheye_f8")
image_path = joinpath(session_dir, "jpeg", "DSC03011.JPG")
camera_json = joinpath(session_dir, "intrinsics.json")
boards_dir = BlobBoards.example_data(:boards)
model_path = descriptor_network_path()
gallery_path = joinpath(tempdir(), "blobboards_fisheye_gallery.jld2")

const IMG_W, IMG_H = 4240, 2832
# First-order expectation from the shipped Brown–Conrady calibration.
const K_EXPECTED = -0.162 / 1226.0^2

# =============================================================================
# Visualization — quad via the ONE model→image builder (`board_quad`:
# projective for H, boundary-sampled/curved for the radial model, camera-
# composed for poses); blob centroids via the model-appropriate transfer.
# =============================================================================

_transfer_pt(m::RadialHomography, ::Nothing, p) =
    something(radial_transfer(m, p), Point2(NaN, NaN))
_transfer_pt(H, ::Nothing, p) =
    (PerspectiveMap() ∘ HomographyMap(SMatrix{3, 3, Float64}(H)))(p)
_transfer_pt(pose, cam::CameraModel, p) =
    (PerspectiveMap() ∘ HomographyMap(SMatrix{3, 3, Float64}(HomographyMat(PosedCamera(cam, pose)))))(p)

function plot_results(img, results, board_bundles, cam, label; n = min(10, length(results)))
    top = results[1:n]
    ncols = min(5, n); nrows = cld(n, ncols)
    fig = Makie.Figure(size = (ncols * 500, nrows * 500))
    for (i, (uid, model, quality, csponds)) in enumerate(top)
        r, c = fldmod1(i, ncols)
        n_inl = count(is_map_inlier, csponds.weights)
        ax, _ = imshow(fig[r, c], img; title = "$uid  q=$(round(quality; digits = 1))  n=$n_inl")
        quad = BlobBoards.board_quad(cam, model, board_bundles[uid])
        Makie.poly!(ax, quad; color = :transparent, strokecolor = :cyan, strokewidth = 2)
        proj_pts = Point2f.(_transfer_pt.(Ref(model), Ref(cam), Point2.(centroid.(sources(csponds)))))
        det_pts = Point2f.(centroid.(targets(csponds)))
        matches!(ax, proj_pts, det_pts; weights = csponds.weights)
    end
    return window!(fig, label)
end

# =============================================================================
# Main
# =============================================================================

img = Gray.(load_image(image_path))
cam = VgtkIO.load(CameraModel, camera_json)

# The descriptor's default constructor DEMANDS the CUDA execution provider
# (silent CPU fallback misreports throughput ~34x). On a CPU device, opt
# into the CPU provider explicitly instead.
const _KA = BlobBoards.KernelAbstractions
descriptor_network = DEVICE isa _KA.CPU ?
    DescriptorNetwork(model_path; execution_provider = :cpu) :
    DescriptorNetwork(model_path)
gallery = BoardGallery(gallery_path, descriptor_network)
board_paths = filter(endswith(".json"), readdir(boards_dir; join = true))
board_feature_dict = Dict(
    uid => features for (uid, features) in insert!(gallery, board_paths, descriptor_network; device = DEVICE)
)
board_bundles = Dict(
    let (b, _, ch) = BlobBoards.load_bundle(bp)
            uid_from_hash(ch) => b
    end for bp in board_paths
)
println("$(length(board_paths)) boards, image $(basename(image_path)) ($(IMG_W)x$(IMG_H) fisheye)")

_detect = DetectConfig(
    polarity = :dark,
    peak_threshold = 0.02,
    edge_threshold = 5.0,
    nms = EllipseNms(cutoff = Float32(CUTOFF))
)

screen = nothing
best_radial = nothing
for (label, geometry) in (
        ("Homography (plain H — fisheye baseline)", HomographyGeometry()),
        ("Radial homography (H, k — uncalibrated)", RadialHomographyGeometry((IMG_W, IMG_H))),
        ("P3P Poses (calibrated, intrinsics.json)", PoseGeometry(cam)),
    )
    config = FindBoardsConfig(; geometry, detect = _detect)
    accepted = find_boards(
        board_feature_dict, descriptor_network, img;
        config, device = DEVICE, board_bundles
    )
    calibrated = geometry isa PoseGeometry

    println("\n== $label ==  $(length(accepted)) verified board(s):")
    for (uid, model, quality, csponds) in accepted
        n_inl = count(is_map_inlier, csponds.weights)
        extra = if model isa RadialHomography
            "  k=$(round(model.k; sigdigits = 3)) px⁻²" *
                " (calibration says ≈ $(round(K_EXPECTED; sigdigits = 3)))" *
                "  bounded=$(division_bounded(model, (IMG_W, IMG_H)))"
        elseif calibrated
            "  t=$(round.(model.t; digits = 1))"
        else
            ""
        end
        println("  $uid  inliers=$n_inl$extra")
        model isa RadialHomography && best_radial === nothing && (global best_radial = model)
    end

    if !isempty(accepted)
        # Calibrated poses live in undistorted-image coordinates.
        disp_img = calibrated ? undistort(cam, img) : img
        cam_quad = calibrated ? CameraModel(cam.intrinsics, Pinhole()) : nothing
        global screen = plot_results(disp_img, accepted, board_bundles, cam_quad, label)
    end
end

# The payoff: undistort the fisheye frame through the FITTED (H, k) — no
# calibration used — next to the calibrated Brown–Conrady undistortion.
# Rendered even when `division_bounded` says false: an over-strong fitted k
# (a single board constrains k weakly, and the true distortion center sits
# off the assumed image middle) pushes the PERIPHERY toward the 90° circle
# where the perspective image is unbounded, but the center region remains
# informative — the title carries the flag instead of hiding the panel.
if best_radial !== nothing
    bounded = division_bounded(best_radial, (IMG_W, IMG_H))
    # FULL-extent render: a same-size canvas would crop the expanding
    # periphery away. `undistorted_rect` sizes the output rect from the
    # undistorted frame boundary (clipped at 92% of the 90° circle when
    # unbounded); the rect form of `undistort` renders it, scale folded in.
    ext = VgtkCore.undistorted_rect(best_radial, (IMG_W, IMG_H))
    σ = max(1.0, widths(ext)[1] / 5000)
    canvas_est = undistort(CameraModel(best_radial), img, ext; scale = σ)
    img_cal = undistort(cam, img)
    fig = Makie.Figure(size = (3 * 620, 460))
    imshow(fig[1, 1], img; title = "raw fisheye")
    imshow(
        fig[1, 2], canvas_est;
        title = "FULL undistorted extent via FITTED (H, k)  ↓×$(round(σ; digits = 2))" *
            (bounded ? "" : "  [clipped at 92% of the 90° circle]")
    )
    imshow(fig[1, 3], img_cal; title = "undistorted via intrinsics.json (Brown–Conrady)")
    global screen = window!(fig, "Division-model undistortion from the estimate")
end

abspath(PROGRAM_FILE) == (@__FILE__) && screen !== nothing && wait(screen)