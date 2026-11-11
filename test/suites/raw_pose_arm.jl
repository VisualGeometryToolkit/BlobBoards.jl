# =============================================================================
# The pose arm's `:none` policy: nothing is undistorted anywhere. Detections
# are raw pixels, the estimator verifies through the FULL camera (the residual
# currency is the pixels the detector produced), and guided association
# transfers each blob's conic through the camera — the plane homography
# exactly, the lens's affinity at the transferred centre.
# =============================================================================
using Test, LinearAlgebra, Random
using StaticArrays: SVector, SMatrix
using StructArrays: StructArray
using Rotations: RotMatrix, RotXYZ, rotation_angle
import VisualGeometryToolkitCore
using VisualGeometryToolkitCore: CameraModel, Extrinsics3, LogicalIntrinsics, CameraCalibrationMat, Division, Pinhole, px,
    EuclideanMap, EllipticFrame, Point2, Point3, Circle, Ellipse, PosedCamera, HomographyMap,
    centroid, shape_matrix, project, pinhole, transfer_ellipse, Vec3, boundary_points
using VisualGeometryToolkitEstimation: AbsolutePoseProblem, scoring_from_toml
using LocalOptimizationDoneRight: HQ, tuning_constant
using VisualGeometryToolkitCore: WelschShape
using BlobBoards: PoseGeometry, HomographyGeometry, RadialHomographyGeometry, _estimation_camera,
    _model_to_homography, _guided_transfer, _undistort_detect, _camera, _project_blob_centers

const _CAM = CameraModel(LogicalIntrinsics(CameraCalibrationMat(950.0px, [2120.0px, 1416.0px])), Division(-0.3, -0.08))
const _PIN = pinhole(_CAM)
const _POSE = EuclideanMap(RotMatrix(RotXYZ(0.35, -0.35, 0.2)), Vec3{Float64}(850.0, 250.0, 350.0))   # far off-axis: the board reaches the rim

@testset "PoseGeometry.undistortion" begin
    @test PoseGeometry(_CAM).undistortion === :none                       # raw pixels by default
    @test PoseGeometry(_CAM; undistortion = :global).undistortion === :global
    @test_throws ArgumentError PoseGeometry(_CAM; undistortion = :local)  # retired with the undistorted-plane residual
    @test_throws ArgumentError PoseGeometry(_CAM; undistortion = :resample)
end

@testset "_estimation_camera follows the policy" begin
    @test _estimation_camera(PoseGeometry(_CAM)) === _CAM                             # :none — the FULL camera
    @test _estimation_camera(PoseGeometry(_CAM; undistortion = :global)) == _PIN      # :global — the pinhole twin
    @test _estimation_camera(HomographyGeometry()) === nothing
    g = RadialHomographyGeometry((4240, 2832))
    @test _estimation_camera(g) === g                                                 # the radial token survives
    # and the detector sees the raw image under :none, the warp only under :global
    img = zeros(UInt8, 8, 8)
    @test _undistort_detect(BlobBoards.default_device(), img, PoseGeometry(_CAM)) === img
end

@testset "the transfer object: a homography under :global, the camera under :none" begin
    pose = Extrinsics3(_POSE.R, _POSE.t)
    @test _model_to_homography(pose, _PIN) isa HomographyMap                          # exact plane map
    @test _model_to_homography(pose, _CAM) isa PosedCamera                                 # no homography exists through a lens
    cam = _model_to_homography(pose, _CAM)
    # the guided transfer compares in the RAW image and searches with the same ellipse
    wreg = Circle(Point2(60.0, -40.0), 2.5)
    ep, es = _guided_transfer(cam, wreg)
    @test ep isa Ellipse && es === ep
    # the conic is transferred, not the point: a projective map does not carry
    # a centre to the transferred conic's centre (second order — 4e-3 px here),
    # and the conic centre is what the imaged blob actually has
    exact = SVector{2, Float64}(project(PosedCamera(_CAM, _POSE), Point3{Float64}(60.0, -40.0, 0.0)))
    @test isapprox(SVector{2, Float64}(ep.center), exact; atol = 0.05)
    Σi = inv(SMatrix{2, 2, Float64}(shape_matrix(ep)))
    worst = maximum(boundary_points(wreg, 48)) do p
        q = SVector{2, Float64}(project(PosedCamera(_CAM, _POSE), Point3{Float64}(p[1], p[2], 0.0)))
        d = q - SVector{2, Float64}(ep.center)
        abs(sqrt(d' * Σi * d) - 1.0)
    end
    @test worst < 0.02
    # a board behind the camera transfers nothing
    behind = _model_to_homography(pose, _CAM)
    @test _guided_transfer(PosedCamera(_CAM, EuclideanMap(RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64})), Vec3{Float64}(0.0, 0.0, -500.0))), wreg) === nothing
end

@testset "the pose fit itself: raw pixels through the lens, rim included" begin
    rng = MersenneTwister(5)
    c = SVector(2120.0, 1416.0)
    src = Point3{Float64}[]; dst = Point2{Float64}[]
    for X in -160.0:20.0:160.0, Y in -120.0:20.0:120.0
        u = project(_CAM, _POSE.R * SVector(X, Y, 0.0) + _POSE.t)
        all(isfinite, u) || continue
        push!(src, Point3{Float64}(X, Y, 0.0))
        push!(dst, Point2{Float64}(u[1] + 0.3randn(rng), u[2] + 0.3randn(rng)))
    end
    radius = [norm(SVector{2, Float64}(d) - c) for d in dst]
    @test count(>(1100), radius) >= 15                                   # the board really reaches the rim
    # The recipe this test pinned, stated (RVG's own default is now MSAC at a
    # threshold with least squares on the inliers): the Student-t + uniform
    # marginal at a 5 px half-width and the cross-validated Welsch-HQ lift.
    est = BlobBoards.VisualGeometryToolkitEstimation.find_absolute_pose(
        _CAM, src .=> dst; rng = MersenneTwister(1),
        scoring = scoring_from_toml(Dict{String, Any}("type" => "student_t_uniform", "outlier_halfwidth" => 5.0); problem = AbsolutePoseProblem),
        local_optimization = HQ{WelschShape}(; scale_multiple = tuning_constant(:absolute_pose, "welsch").scale_multiple)
    )
    @test est.converged   # a failed search is a :no_model result, not `nothing` (Estimation 0.5)
    if est.converged      # a failed search has no estimate to read
        @test rad2deg(rotation_angle(RotMatrix{3}(est.estimate.R' * _POSE.R))) < 0.05
        @test norm(est.estimate.t - _POSE.t) < 1.0
        @test count(est.inlier_mask .& (radius .> 1100)) == count(radius .> 1100)   # the rim is kept
    end
end

# track_boards' bookkeeping projects every blob centre through the transfer
# `_model_to_homography` hands back — the `PosedCamera` itself on the full-camera
# pose arm. Through a pinhole that is the plane homography's projection; through
# the lens, the camera's own projection of the z = 0 lift.
@testset "blob centres through the full camera (track_boards, :none)" begin
    pose = Extrinsics3(_POSE.R, _POSE.t)
    wd = [EllipticFrame(Point2(x, y), 2.0) for x in -60.0:30.0:60.0 for y in -40.0:40.0:40.0]
    through_cam = _project_blob_centers(_model_to_homography(pose, _CAM), wd)
    @test through_cam isa Vector{Point2{Float64}} && length(through_cam) == length(wd)
    for (p, d) in zip(through_cam, wd)
        c = centroid(d)
        @test SVector{2, Float64}(p) ≈ SVector{2, Float64}(project(PosedCamera(_CAM, _POSE), Point3{Float64}(c[1], c[2], 0.0)))
    end
    # a pinhole camera: the PosedCamera method agrees with the homography method
    via_H = _project_blob_centers(_model_to_homography(pose, _PIN), wd)
    via_cam = _project_blob_centers(PosedCamera(_PIN, _POSE), wd)
    @test all(SVector{2, Float64}(a) ≈ SVector{2, Float64}(b) for (a, b) in zip(via_H, via_cam))
end

# Guided association is ellipse overlap alone (the owner's ruling, 2026-11-11): a projected blob's candidates
# are the detections whose ellipses' boxes intersect its own projected ellipse's box, it takes the candidate
# of highest exact IoU when that IoU is ≥ min_iou, and there is no radius. Every candidate is a read
# dependency (recorded before the IoU gate), accepted or not.
@testset "guided association: ellipse IoU only, no radius" begin
    H = HomographyMap(VisualGeometryToolkitCore.HomographyMat(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1)))
    wregions = [Circle(Point2(100.0, 100.0), 6.0), Circle(Point2(200.0, 100.0), 6.0), Circle(Point2(300.0, 100.0), 6.0)]
    E(x, y, r) = Ellipse(Point2(x, y), r, r, 0.0)
    det_ells = [E(103.0, 100.0, 6.0),     # blob 1: 3 px off centre, IoU 0.52 — accepted, far past any sub-pixel radius
                E(200.0, 100.0, 1.5),     # blob 2: the nearest centre (0 px), IoU 0.06 — refused
                E(301.0, 100.0, 4.0),     # blob 3: the nearer centre, IoU 0.44
                E(298.5, 100.0, 6.0)]     # blob 3: the farther centre, IoU 0.73 — the one taken
    reads = Int32[]
    c = BlobBoards._guided_candidates(BlobBoards.detection_rtree(det_ells), H, wregions, det_ells, 0.5, j -> push!(reads, j))
    @test [(wi, di) for (_, wi, di) in c] == [(1, 1), (3, 4)]
    @test all(v -> v[1] >= 0.5, c)
    @test sort(reads) == Int32[1, 2, 3, 4]                       # every candidate read, the refused ones too
    # no radius anywhere in the API: guided_search takes none, and find_board supplies none
    @test !any(m -> :max_px in Base.kwarg_decl(m), methods(BlobBoards.guided_search))
    @test !any(m -> :guided_detc in Base.kwarg_decl(m), methods(BlobBoards.find_board))
end
