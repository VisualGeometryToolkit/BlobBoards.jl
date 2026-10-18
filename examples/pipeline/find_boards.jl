#=
Find Boards — one entry point, geometry picks the path.

`find_boards` dispatches entirely on `config.geometry`:
  • HomographyGeometry()      → no camera, 2D↔2D homography
  • PoseGeometry(camera)       → calibrated, 3D(z=0)↔2D P3P pose

This script runs BOTH on the same image — one figure per variant — so the
two paths are directly comparable. There is no separate calibrated function
or script: the camera lives in the geometry config.

With a display, each figure opens in a GLMakie window; without one (a headless
server, an SSH session without X forwarding) it is saved as a PNG next to the
working directory instead, through CairoMakie. Both come with the gpu/ env.

Usage:
  julia --project=gpu examples/pipeline/find_boards.jl
  include("examples/pipeline/find_boards.jl")     # from a gpu/ REPL
=#

using BlobBoards
# A window needs a display: on Linux that is an X11 or Wayland session.
const HAS_DISPLAY = !Sys.islinux() || !isempty(get(ENV, "DISPLAY", "")) ||
    !isempty(get(ENV, "WAYLAND_DISPLAY", ""))
if HAS_DISPLAY
    # render_on_demand=false: in a remote-driven REPL session the on-demand renderer gets
    # no damage events and the window stays black; continuous rendering fixes it.
    using GLMakie; GLMakie.activate!(render_on_demand = false)
end
# CairoMakie: the PNG writer when there is no display
import CairoMakie
using CairoMakie: Makie
using FileIO: load as load_image
using Colors: Gray
using GeometryBasics: Point2, Point2f, Rect
using StaticArrays: SVector, SMatrix
using CUDA          # triggers GPU extensions
import cuDNN        # ONNXRunTime :cuda execution provider
using VisualGeometryToolkitCore: is_map_inlier, imshow, matches!, window!, ustrip, centroid,
    HomographyMap, HomographyMat, Camera, undistort, CameraModel, Pinhole,
    sources, targets
using CoordinateTransformations: PerspectiveMap
using CUDA: CUDABackend   # the KA device token
import VisualGeometryToolkitIo as VGIo   # canonical vgio/cameras/v3 camera stores

# =============================================================================
# Config
# =============================================================================

const CUTOFF = BlobBoards.DEFAULT_CUTOFF
const DEVICE = CUDABackend()

# Tilt schedule: the FULL IMAS-25 default (DetectConfig's empty tilts).
# The historical 17-tilt reduction here dated from before the ASIFT driver
# chunked the warp schedule to device memory (VGF asift/driver.jl
# `_chunk_warps`) — all 25 pyramids no longer need to be resident at once,
# so the full schedule runs on any card. Do not reintroduce a reduced ring;
# the smoke suite guards the default (test_annotate_smoke.jl).

boards_dir = BlobBoards.example_data(:boards)
# Network from configs/default.toml — the one place a default entry point
# learns which descriptor to load. Same log-polar _cuda net as the paper.
model_path = descriptor_network_path()
image_path = joinpath(BlobBoards.example_data(:still), "DSC02847.JPG")
gallery_path = joinpath(tempdir(), "blobboards_example_gallery.jld2")
camera_json = joinpath(BlobBoards.example_data(:still), "cameras.json")

# =============================================================================
# Visualization — branch only on geometry; both project world Rect+blobs
# through `PerspectiveMap ∘ HomographyMap` into the image.
# =============================================================================

# Homography path: model IS the world→image H. Pose path: VGC's Camera
# composes pose + intrinsics — same idiom as src/verify.jl, no hand-rolled
# K·[r1|r2|t].
_to_homography(H_mat, ::Nothing) = SMatrix{3, 3, Float64}(H_mat)
_to_homography(pose, cam::CameraModel) =
    SMatrix{3, 3, Float64}(HomographyMat(Camera(cam, pose)))

function plot_results(img, results, board_bundles, cam, label; n = min(10, length(results)))
    top = results[1:n]
    ncols = min(5, n); nrows = cld(n, ncols)
    fig = Makie.Figure(size = (ncols * 500, nrows * 500))
    for (i, (uid, model, quality, csponds)) in enumerate(top)
        r, c = fldmod1(i, ncols)
        n_inl = count(is_map_inlier, csponds.weights)
        ax, _ = imshow(fig[r, c], img; title = "$uid  q=$(round(quality; digits = 1))  n=$n_inl")
        proj = PerspectiveMap() ∘ HomographyMap(_to_homography(model, cam))
        bundle = board_bundles[uid]
        board!(ax, proj, Rect(bundle, :world), blobs(bundle, :world); cutoff = cutoff(bundle))
        proj_pts = Point2f.(proj.(Point2.(centroid.(sources(csponds)))))
        det_pts = Point2f.(centroid.(targets(csponds)))
        matches!(ax, proj_pts, det_pts; weights = csponds.weights)
    end
    # window!(fig, title) creates the titled GLMakie screen AND displays the
    # figure — it is the whole display step; do not display() again on top.
    HAS_DISPLAY && return window!(fig, label)
    png = replace(lowercase(label), r"[^a-z0-9]+" => "_") * ".png"
    CairoMakie.save(png, fig; backend = CairoMakie)
    println("  no display: saved ", abspath(png))
    return nothing
end

# =============================================================================
# Main
# =============================================================================

img = Gray.(load_image(image_path))
cam = VGIo.load(CameraModel, camera_json)

descriptor_network = DescriptorNetwork(model_path)
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
println("$(length(board_paths)) boards")

# The camera lives in the geometry — that single choice selects homography vs
# P3P. Both variants run on the SAME image with the SAME detection config.
_detect = DetectConfig(
    polarity = :dark,
    peak_threshold = 0.02,
    edge_threshold = 5.0,
    nms = EllipseNms(cutoff = Float32(CUTOFF))
)

screen = nothing
for (label, geometry) in (
        ("Homography (uncalibrated)", HomographyGeometry()),
        ("P3P Poses (calibrated)", PoseGeometry(cam)),
    )
    config = FindBoardsConfig(; geometry, detect = _detect)
    # One call — the same entry point for both paths.
    accepted = find_boards(
        board_feature_dict, descriptor_network, img;
        config, device = DEVICE, board_bundles
    )
    calibrated = geometry isa PoseGeometry

    println("\n== $label ==  $(length(accepted)) verified board(s):")
    for (uid, model, quality, csponds) in accepted
        n_inl = count(is_map_inlier, csponds.weights)
        extra = calibrated ? "  t=$(round.(model.t; digits = 1))" : ""
        println("  $uid  inliers=$n_inl$extra")
    end

    if !isempty(accepted)
        # Calibrated poses live in undistorted-image coordinates, so that
        # variant displays the undistorted image; homography uses the raw one.
        disp_img = calibrated ? undistort(cam, img) : img
        cam_pinhole = calibrated ? CameraModel(cam.intrinsics, Pinhole()) : nothing
        global screen = plot_results(disp_img, accepted, board_bundles, cam_pinhole, label)
    end
end
# Block only when run as a script — under include() the windows stay up and
# the caller keeps its REPL.
abspath(PROGRAM_FILE) == (@__FILE__) && screen !== nothing && wait(screen)
