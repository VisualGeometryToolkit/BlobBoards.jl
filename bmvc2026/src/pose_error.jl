# =============================================================================
# Pose error of detected board poses against the motion-capture rig.
#
# Two metrics, both per (image, marker):
#
# 1. Relative-to-base (headline, calibration-free). The fixed transforms
#    printed-board→board-rig (per marker) and camera-optical→camera-rig are
#    unknown; they CONJUGATE relative motions, so the conjugation-invariant
#    comparison is between the rotation-angle scalars of the relative motions:
#        error_angle = | θ(R_det_base' R_det_i) − θ(R_rig_base' R_rig_i) |
#    There is no analogous conjugation-invariant translation comparison — a
#    marker's camera distance changes with its lever arm about the rig's
#    rotation axis, which the rig-center distance does not see — so
#    translation error is reported only by the hand-eye absolute metric.
#
# 2. Absolute via hand-eye (secondary). Model: T_det(i,m) = X · T_cb_rig(i) · Y_m
#    with X = camera-rig→camera-optical and Y_m = printed-marker-m→board-rig
#    fixed. X is solved marker-independently from relative motions
#    (B_ij = X A_ij X⁻¹, Kabsch on rotation axes), each Y_m by chordal rotation
#    averaging, translations by linear least squares. Errors are then
#    angle/‖Δt‖ between the predicted and detected absolute poses.
# =============================================================================

import VisualGeometryToolkitIO as VgtkIO
import VisualGeometryToolkitEval.Experiments: save_hand_eye   # the hand-eye records' writer, a fit's method added below
using VisualGeometryToolkitEval.Experiments: load_hand_eye, hand_eye_mount_file
using Rotations: RotMatrix, rotation_angle
using BlobBoards: load_bundle, uid_from_hash, Rect
using VisualGeometryToolkitCore: ustrip, mm, parse_unit

# A board JSON's `{value, unit}` field as a quantity, its unit read through
# VisualGeometryToolkitCore's one guarded parse.
_quantity(x) = Float64(x.value) * parse_unit(String(x.unit))

"""
    load_detected_poses(poses_dir) -> Dict{String, Vector{NamedTuple}}

The detected board poses of one detection run: `poses_dir/target_poses.json`
(`vgio/target_poses/v1`, one file per run — every frame's image, its camera
record and each detected target's pose `target:<id> → camera_optical`, mm).
Keys are the image file names; each entry carries `(id, R, t)` with `R`
mapping printed-marker coordinates into camera coordinates (object→camera, as
returned by solvePnP and by `find_boards`' `PoseGeometry`), targets in id
order. A target's size is its definition's ([`marker_sizes`](@ref)), not the
detection's.
"""
function load_detected_poses(poses_dir::AbstractString)
    tp = VgtkIO.load(TargetPoses, joinpath(poses_dir, "target_poses.json"))
    out = Dict{String, Vector{@NamedTuple{id::String, R::SMatrix{3, 3, Float64, 9}, t::SVector{3, Float64}}}}()
    for (t, ps) in tp
        out[tp.images[t]] = [
            (id = String(id), R = SMatrix{3, 3, Float64}(geom(ps[id]).R), t = SVector{3, Float64}(ustrip.(mm, geom(ps[id]).t)))
                for id in sort!(collect(keys(ps)))
        ]
    end
    return out
end

"""
    marker_sizes(dataset_dir, system) -> Dict{String, Float64}

Each printed target's size (mm) by id, from its DEFINITION in
`<dataset>/boards/<system>/` — `system` the boards subdirectory (`apriltags`,
`aruco`, `blobboards`): a BlobBoards board's world extent (the largest side of
`Rect(bundle, :world)`, as the sweep measures it), a tag board's detectable
pattern (`detectable_pattern_resolution / print_density`, in mm through
VisualGeometryToolkitCore's units).
The sizes only RANK the targets into scale classes (`_scale_labels`).
"""
function marker_sizes(dataset_dir::AbstractString, system::AbstractString)
    dir = joinpath(dataset_dir, "boards", system)
    out = Dict{String, Float64}()
    for f in sort(filter(endswith(".json"), readdir(dir; join = true)))
        if system == "blobboards"
            b, _, ch = load_bundle(f)
            out[String(uid_from_hash(ch))] = ustrip(maximum(Rect(b, :world).widths))
        else
            m = match(r"^board_[A-Z]+_(\d+)\.json$", basename(f))
            m === nothing && continue
            pre = JSON.parse(read(f, String)).preamble
            # the detectable pattern (a pixel count) at the print density (a
            # pixel density) is a length: VisualGeometryToolkitCore's units convert it
            out[String(m[1])] = ustrip(mm, _quantity(pre.pattern_config.detectable_pattern_resolution) / _quantity(pre.board_config.print_density))
        end
    end
    return out
end

# --- relative-to-base ---------------------------------------------------------

"""
    relative_errors(frames, detected) -> Dict{Tuple{String,String}, Float64}

Per (stem, marker-id): calibration-free relative rotation-angle error against
the rig (see file header), radians. Markers missing from their spot's base
image get no entry.
"""
function relative_errors(frames::Vector{RigFrame}, detected::Dict)
    # base detections per spot: marker id => R
    base_det = Dict{Int, Dict{String, SMatrix{3, 3, Float64, 9}}}()
    base_rig = Dict{Int, RigFrame}()
    for fr in frames
        (fr.is_base && haskey(detected, _stem(fr.image))) || continue
        base_rig[fr.spot] = fr
        base_det[fr.spot] = Dict(p.id => p.R for p in detected[_stem(fr.image)])
    end

    out = Dict{Tuple{String, String}, Float64}()
    for fr in frames
        fr.T_cb === nothing && continue
        haskey(base_det, fr.spot) || continue
        bd, br = base_det[fr.spot], base_rig[fr.spot]
        θ_rig = rotation_angle(RotMatrix{3}(br.T_cb.R' * fr.T_cb.R))
        for p in get(detected, _stem(fr.image), ())
            haskey(bd, p.id) || continue
            θ_det = rotation_angle(RotMatrix{3}(bd[p.id]' * p.R))
            out[(_stem(fr.image), p.id)] = abs(θ_det - θ_rig)
        end
    end
    return out
end

# --- hand-eye (robot-world AX = YB, multi-marker) ------------------------------

"""
    hand_eye_calibrate(frames, detected; min_rel_angle=deg2rad(5)) ->
        (; X, Y, fit) with X::EuclideanMap, Y::Dict{String,EuclideanMap},
        fit::FitResult{<:RobotWorldHandEyeCalibration, <:RobotWorldHandEyeMetadata}

Solve `T_det(i,m) = X · T_rig(i) · Y_m` on a (calibration) sequence via
VisualGeometryToolkitEstimation's `fit_robot_world_hand_eye` — the staged linear robot-world
estimator (Kabsch → chordal means → joint translation LSQ → covariance;
RVG ≥ 0.14). The NLLS refinement was removed upstream: on this dataset the
pose-residual MLE absorbed an unmodeled systematic and extrapolated worse
than the linear solution.

Letter mapping onto RVG's `A_i X_m = Y B_i^m` with W := the camera rig:
`A_i = T_cb(i)` (board rig in camera rig, mocap), `B_i^m = T_det(i,m)`
(marker in camera optical, PnP), so `X = inv(Y_rvg)` (camera_rig →
camera_optical) and `Y_m = X_rvg[m]` (printed marker m → board rig).

`fit.metadata` carries the calibration covariance (rad, mm tangent
coordinates), variance factor, and the min-eigenpair observability
diagnostics — persist them with the calibration, never discard.
"""
function hand_eye_calibrate(
        frames::Vector{RigFrame}, detected::Dict;
        min_rel_angle::Float64 = deg2rad(5)
    )
    seq = [
        (fr, detected[_stem(fr.image)])
            for fr in frames if fr.T_cb !== nothing && haskey(detected, _stem(fr.image))
    ]
    isempty(seq) && error("hand-eye: no usable frames")

    # RVG's estimator is label-agnostic and composes internally — strip the
    # frame labels at its boundary, re-mint them on the results. Its input is
    # keyed by marker (the correspondence standard): per marker, every frame
    # that saw it as `T_cb(i) => the marker's detected pose`, in frame order.
    raw = Dict{String, Vector{Any}}()
    for (fr, ps) in seq, p in ps
        push!(
            get!(() -> Any[], raw, String(p.id)),
            unlabel(fr.T_cb) => EuclideanMap(RotMatrix{3, Float64}(p.R), p.t; check = false),
        )
    end
    markers = Dict(id => identity.(v) for (id, v) in raw)

    fit = fit_robot_world_hand_eye(markers; min_rel_angle)
    X = relabel(CameraRigFrame, ViewFrame{:optical}, inv(fit.estimate.Y))
    Y = Dict(id => fit.estimate.X[id] for id in fit.estimate.ids)
    return (; X, Y, fit)
end

"""
    absolute_errors(frames, detected, X, targets) -> Dict{Tuple{String,String}, NamedTuple}

Absolute pose error per (image, marker) after hand-eye calibration: the
predicted pose `X ∘ T_cb(i) ∘ pose(targets, m)` vs the detected pose —
rotation discrepancy angle and translation distance. `targets` is the
marker-mount `Rig{BoardRigFrame}` of targets; the prediction chain is
frame-checked end to end (`TargetFrame{m} → BoardRigFrame →
CameraRigFrame → ViewFrame{:optical}`).
"""
function absolute_errors(
        frames::Vector{RigFrame}, detected::Dict,
        X, targets::Rig
    )
    out = Dict{Tuple{String, String}, @NamedTuple{error_angle::Float64, error_distance::Float64}}()
    for fr in frames
        fr.T_cb === nothing && continue
        for p in get(detected, _stem(fr.image), ())
            haskey(targets, Symbol(p.id)) || continue
            T̂ = X ∘ fr.T_cb ∘ pose(targets, Symbol(p.id))
            out[(_stem(fr.image), p.id)] = (
                error_angle = rotation_angle(RotMatrix{3}(T̂.R' * p.R)),
                error_distance = norm(T̂.t - p.t),
            )
        end
    end
    return out
end

# =============================================================================
# The hand-eye record: X (vgio/transforms/v3) and the marker mount (vgio/rig/v6)
# =============================================================================
#
# ONE writer and ONE reader of a hand-eye calibration, shared by every producer
# (src/calibration.jl's cache, make_hand_eye.jl and export_reconstruction.jl):
# `<file>` holds X, `body:camera_rig → camera_optical`;
# `<file stem>_board_rig.json` holds the mount, one target per printed board,
# `target:<id> → body:board_rig`. An estimate is written with its covariance
# (VisualGeometryToolkitCore conventions/persisted-geometry.md): `save_hand_eye(file,
# he)` of a fit always writes `Uncertain` poses.

using VisualGeometryToolkitCore: Uncertain, CovMat, geom, BodyFrame, CameraFrame

"""
    save_hand_eye(file, he; provenance = (;)) -> (file, mount)

A hand-eye FIT (`hand_eye_calibrate`'s `(; X, Y, fit)`) as VisualGeometryToolkitEval's
two hand-eye records (`save_hand_eye(file, X, Y)`, read back by its
`load_hand_eye`), with the fit's covariances ([`uncertain_hand_eye`](@ref)),
never without.
"""
save_hand_eye(file::AbstractString, he::NamedTuple; provenance::NamedTuple = (;)) =
    save_hand_eye(file, uncertain_hand_eye(he)...; provenance)   # VisualGeometryToolkitEval's two records

# The geometric target rig of an uncertain one (the chain is composed on values).
_plain_targets(r::Rig) = Rig{BoardRigFrame}(Dict(id => PosedTarget(model(r[id]), geom(pose(r[id]))) for id in ids(r)))

# Block `b` (0-based) of a joint covariance over 6-blocks, symmetrized exactly.
_cov_block(C::AbstractMatrix, b::Integer) = (B = SMatrix{6, 6, Float64, 36}(C[(6b + 1):(6b + 6), (6b + 1):(6b + 6)]); (B + B') / 2)

"""
    uncertain_hand_eye(he) -> (X, Y)

A fit's poses with their covariances (RVG's joint covariance over
`[Y_rvg | X_rvg(ids…)]`, each block `(δω, δt)` in the `RigidExpManifold6`
chart): `Y_m = X_rvg[m]` takes its own block; `X = inv(Y_rvg)` takes Y_rvg's
block pushed through the inverse (VisualGeometryToolkitCore's `inv` of an uncertain pose). A fit whose
covariance is unbounded (an unobservable direction) is refused: an estimate is
not published without its uncertainty.
"""
function uncertain_hand_eye(he)
    C = he.fit.metadata.cov
    all(isfinite, C) || error("hand-eye: the fit's covariance is unbounded (an unobservable direction) — not publishing it")
    X = Uncertain(he.X, cov(inv(Uncertain(he.fit.estimate.Y, CovMat(_cov_block(C, 0))))))   # Core's uncertain inverse; X keeps its frames
    Y = Dict(id => Uncertain(he.Y[id], CovMat(_cov_block(C, b))) for (b, id) in enumerate(he.fit.estimate.ids))
    return X, Y
end

