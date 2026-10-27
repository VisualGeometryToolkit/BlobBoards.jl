# =============================================================================
# The evaluation's file formats (scripts/convert_formats.jl, src/rig.jl,
# src/pose_error.jl): the rig sequence from vgio/trajectory/v3, the detected
# poses from vgio/target_poses/v1, and the hand-eye record — whose writer
# states the fit's uncertainty on every pose it publishes.
# =============================================================================

using Test
using DrWatson: projectdir
if !isdefined(Main, :BlobBoardsEval)
    include(projectdir("src", "BlobBoardsEval.jl"))
end
using .BlobBoardsEval
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: Trajectory, TargetPoses, PosedTarget, PlanarTarget, is_resolved, model, with_unit, mm
using VisualGeometryToolkitCore: EuclideanMap, BodyFrame, WorldFrame, CameraFrame, TargetFrame, ViewFrame, Uncertain, CovMat,
    geom, relabel, unlabel, pose, ids, Rig
import Rotations
using Rotations: RotMatrix, RotXYZ, rotation_angle
using StaticArrays: SVector, SMatrix
using LinearAlgebra: norm, I
using Statistics: cov
import JSON

const BBE = BlobBoardsEval

_map(a, b, c, t) = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(a, b, c)), SVector{3, Float64}(t); check = false)

# A synthetic calibration: the rigs in the world, the camera mount X, the board
# mount Y_m, and the detections they imply (T_det = X ∘ T_cb ∘ Y_m).
function synthetic_capture(dir; n = 12)
    X = _map(0.3, -0.2, 0.1, [10.0, -40.0, 60.0])                 # camera rig → optical
    Ys = Dict("b$(m)" => _map(0.05m, 0.1, -0.02m, [30.0m, -15.0, 5.0m]) for m in 1:3)
    Twc = Dict(t => _map(0.01t, 0.0, 0.0, [2000.0, 0.0, 1300.0]) for t in 1:n)
    Twb = Dict(t => _map(0.2 * sin(t), 0.35 * cos(1.3t), 0.15t, [1000.0 + 20t, 30.0 * cos(t), 1400.0]) for t in 1:n)
    images = Dict(t => "DSC$(lpad(t, 4, '0')).JPG" for t in 1:n)
    mt = mkpath(joinpath(dir, "poses", "motion_tracker", "calibration_bb"))
    for (body, P) in ((:board_rig, Twb), (:camera_rig, Twc))
        VgtkIO.save(joinpath(mt, "$(body).json"), with_unit(Trajectory(BodyFrame{body}, WorldFrame, P; images), mm);
            provenance = (recovery = (status = "not needed",),))
    end
    # detections with a small deterministic error, so the fit has a residual
    # (and so a covariance) to report
    jitter(t, m) = _map(1.0e-3 * sin(t + m), 1.0e-3 * cos(2t), 1.0e-3 * sin(3m), [0.1 * sin(t), 0.1 * cos(m), 0.1 * sin(t * m)])
    det = Dict(
        t => Dict(Symbol(k) => X ∘ inv(Twc[t]) ∘ Twb[t] ∘ Y ∘ jitter(t, i) for (i, (k, Y)) in enumerate(sort!(collect(Ys); by = first))) for t in 1:n
    )
    pd = mkpath(joinpath(dir, "poses", "calibration", "bb"))
    VgtkIO.save(joinpath(pd, "target_poses.json"), with_unit(TargetPoses{CameraFrame}(det; images, sensors = "cam"), mm))
    return (; X, Ys, Twc, Twb, images)
end

@testset "evaluation formats" begin
    mktempdir() do dir
        cap = synthetic_capture(dir)

        @testset "rig sequence from the trajectories" begin
            frames = BBE.rig_sequence(joinpath(dir, "poses", "motion_tracker"), "calibration_bb")
            @test length(frames) == 12 && all(f -> f.T_cb !== nothing && !f.recovered, frames)
            f = frames[3]
            @test f.image == cap.images[3] && f.t == 3
            @test f.T_cb isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:board_rig}, BodyFrame{:camera_rig}}
            T = unlabel(inv(cap.Twc[3]) ∘ cap.Twb[3])
            @test maximum(abs, SMatrix{3, 3}(f.T_cb.R) - SMatrix{3, 3}(T.R)) < 1.0e-14 && norm(f.T_cb.t - T.t) < 1.0e-9
            @test BBE.static_recovery_accepted(joinpath(dir, "poses", "motion_tracker"), "calibration_bb")
        end

        @testset "detected poses from target_poses" begin
            det = BBE.load_detected_poses(joinpath(dir, "poses", "calibration", "bb"))
            @test sort(collect(keys(det))) == sort(collect(values(cap.images)))
            @test [p.id for p in det[cap.images[1]]] == ["b1", "b2", "b3"]
        end

        @testset "the hand-eye writer states the fit's uncertainty" begin
            frames = BBE.rig_sequence(joinpath(dir, "poses", "motion_tracker"), "calibration_bb")
            det = Dict(splitext(k)[1] => v for (k, v) in BBE.load_detected_poses(joinpath(dir, "poses", "calibration", "bb")))
            he = BBE.hand_eye_calibrate(frames, det; min_rel_angle = deg2rad(5))
            file = joinpath(dir, "hand_eye_bb.json")
            BBE.save_hand_eye(file, he)
            # every pose of the record is Uncertain: X in the transforms file, each Y_m in the mount
            d = JSON.parse(read(file, String))
            @test d.schema == "vgio/transforms/v3" && only(d.transforms).from == "body:camera_rig" && haskey(only(d.transforms), :covariance)
            m = JSON.parse(read(joinpath(dir, "hand_eye_bb_board_rig.json"), String))
            @test m.schema == "vgio/rig/v5" && m.frame == "body:board_rig" && all(t -> haskey(t.pose, :covariance), m.targets)
            @test all(t -> t.target.type == "planar" && t.target.reference == t.id, m.targets)   # pose-only boards, named by id
            X, mount = BBE.load_hand_eye(file)
            @test X isa Uncertain && geom(X) isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:camera_rig}, ViewFrame{:optical}}
            @test mount isa Rig{BodyFrame{:board_rig}} && all(id -> pose(mount, id) isa Uncertain, ids(mount))
            @test all(id -> mount[id] isa PosedTarget && model(mount[id]) == PlanarTarget(id) && !is_resolved(model(mount[id])), ids(mount))
            @test rotation_angle(RotMatrix{3}(SMatrix{3, 3}(geom(X).R)' * SMatrix{3, 3}(cap.X.R))) < 1.0e-2
            @test all(isfinite, cov(X)) && !iszero(cov(X))
            # the covariance of X is the fit's Y_rvg block pushed through the inverse
            Σ = he.fit.metadata.cov[1:6, 1:6]
            @test cov(X) ≈ BBE.inv_pose_covariance(he.fit.estimate.Y, (Σ + Σ') / 2)
            # an estimate whose covariance is unbounded is not published
            bad = (; X = he.X, Y = he.Y, fit = (; metadata = (; cov = fill(Inf, size(he.fit.metadata.cov))), estimate = he.fit.estimate))
            @test_throws ErrorException BBE.save_hand_eye(joinpath(dir, "bad.json"), bad)
        end

        @testset "inv_pose_covariance: the chart's first-order law" begin
            # perturb Y on the chart (R ↦ exp(δω)·R, t ↦ t + δt), compare the
            # inverse's perturbation with J·δ
            Y = _map(0.4, -0.3, 0.2, [100.0, -50.0, 300.0])
            J = BBE.inv_pose_covariance(Y, Matrix{Float64}(I, 6, 6))     # J·Jᵀ
            exp_so3(w) = (θ = norm(w); θ == 0 ? SMatrix{3, 3, Float64}(I) : SMatrix{3, 3, Float64}(RotMatrix(Rotations.RotationVec(w...))))
            ε = 1.0e-6
            Jfd = zeros(6, 6)
            for k in 1:6
                δ = zeros(6); δ[k] = ε
                R = exp_so3(SVector{3}(δ[1:3])) * SMatrix{3, 3}(Y.R)
                t = Y.t + SVector{3}(δ[4:6])
                Xi_R, Xi_t = R', -R' * t
                X0_R, X0_t = SMatrix{3, 3}(Y.R)', -SMatrix{3, 3}(Y.R)' * Y.t
                dR = Xi_R * X0_R'
                Jfd[1:3, k] = SVector(dR[3, 2] - dR[2, 3], dR[1, 3] - dR[3, 1], dR[2, 1] - dR[1, 2]) / (2ε)
                Jfd[4:6, k] = (Xi_t - X0_t) / ε
            end
            @test J ≈ Jfd * Jfd' rtol = 1.0e-4
        end
    end
end
