# =============================================================================
# Motion-tracker (mocap) rig: the image join, cam-to-board poses, per-spot base
# frames, signed rig angles.
#
# The rigs' poses are `vgio/trajectory/v3` files, one per tracked body and
# subset (<records>/poses/motion_tracker/<subset>/{board_rig,camera_rig}.json): the
# body's pose in the tracker's world per frame index `t`, typed
# `body:<id> → world`, each frame joined to its image by name. An instant the
# tracker did not see is absent; a pose substituted rather than measured is
# flagged `recovered` (the static-camera recovery of the calibration subsets,
# recovery.jl, with its evidence in the file's provenance). Those files are
# DERIVED from the tracker's raw marker measurements
# (poses/motion_tracker/<subset>/markers.json, `vgio/target_points3d/v1`)
# by the tracker vendor's solver, and scripts/convert_formats.jl wrote them
# from the vendor's per-frame export, which stays the source of record.
#
# The vendor export stored each rotation as a flat COLUMN-major 3×3, which the
# original Python pipeline read row-major — invisible to distance norms and
# relative-rotation angles (both transpose-invariant) but fatal to anything
# frame-dependent (hand-eye self-residuals 0.4°/3.7 mm column-major vs
# 14.6°/150 mm row-major). The trajectory files declare their rotation
# encoding, so that class of error has nowhere left to live.
# =============================================================================

# The bodies of the hand-eye chain (VGC's `BodyFrame`): with these labels
# `X ∘ T_cb ∘ Y_m` is junction-checked by dispatch — `Y_m : TargetFrame{m} →
# BoardRigFrame` (a target-rig pose), `T_cb : BoardRigFrame → CameraRigFrame`
# (mocap, per frame), `X : CameraRigFrame → ViewFrame{:optical}` (the
# hand-eye); a mislabeled junction is a MethodError, not a wrong pose.
import VisualGeometryToolkitIo as VgtkIo
using VisualGeometryToolkitCore: BodyFrame, WorldFrame, geom, Trajectory, TargetPoses
using VisualGeometryToolkitIo: VisualGeometryToolkitIo

const CameraRigFrame = BodyFrame{:camera_rig}
const BoardRigFrame = BodyFrame{:board_rig}

const RESET_DIST_MM = 700.0   # distance jump that starts a new tripod spot

# The eval set's narrow image predicate lives in eval_images.jl (shared with
# BoardSweep by include).

"""
One image's worth of rig state. `t` is the frame index the per-frame files
share; `T_cb` maps board-rig coordinates into camera-rig coordinates (same
object→camera direction as the detected poses), `nothing` when either rig was
untracked; `recovered` says the camera rig's pose was substituted, not
measured. `spot` counts tripod placements (base resets); `rig_angle` is the
signed relative rotation from the spot's base frame (radians).
"""
struct RigFrame
    image::String
    t::Int
    T_cb::Union{Nothing, EuclideanMap{3}}            # nothing => a rig untracked
    recovered::Bool
    dist::Float64                                    # ‖t_cb‖ mm (NaN if untracked)
    spot::Int
    is_base::Bool
    rig_angle::Float64                               # rad, NaN if untracked
end

# One body's trajectory of a subset, checked to be that body in the tracker's world.
function _body_trajectory(dir::AbstractString, body::Symbol)
    tr = VgtkIo.load(Trajectory, joinpath(dir, "$(body).json"))
    tr isa Trajectory{BodyFrame{body}, WorldFrame} ||
        error("$(dir)/$(body).json: expected the trajectory body:$(body) → world, got $(typeof(tr).parameters[1]) → $(typeof(tr).parameters[2])")
    return tr
end

# Signed relative rotation angle from the base frame. Sign convention: the rig
# rotates about the (near-vertical) axis; the sign is taken from the +y
# component of the rotation axis in the camera-rig frame. This is a canonical
# definition — it reproduces the shipped CSV signs on 53/56 frames and the 10°
# rig_angles_bin on 56/57 (the discrepancies are axis-noise at near-zero
# angles).
function _signed_rel_angle(R_base, R)
    aa = AngleAxis(RotMatrix{3}(R_base' * R))
    s = aa.axis_y == 0 ? 1.0 : sign(aa.axis_y)
    return s * aa.theta
end

"""
    rig_sequence(mocap_dir, subset) -> Vector{RigFrame}

The rig sequence of `<mocap_dir>/<subset>` (`mocap_dir`: the motion-capture
records' `poses/motion_tracker`, [`mocap_dir`](@ref)): every frame index either
rig's trajectory holds, in order, with its image (the trajectories' image
join), the cam-to-board pose `T_cb = inv(T_wc) ∘ T_wb` where both rigs have a
pose (a recovered camera pose included, and flagged), the distance, the spot
segmentation (`RESET_DIST_MM` jumps) and the signed rig angle relative to
each spot's base frame.
"""
function rig_sequence(mocap_dir::AbstractString, subset::AbstractString)
    dir = joinpath(mocap_dir, subset)
    board, camera = _body_trajectory(dir, :board_rig), _body_trajectory(dir, :camera_rig)
    ts = sort!(collect(union(keys(board.poses), keys(camera.poses))))
    frames = Vector{RigFrame}(undef, length(ts))
    base_R = nothing
    last_dist = 0.0
    spot = 0
    for (i, t) in enumerate(ts)
        img = get(board.images, t, get(camera.images, t, nothing))
        img === nothing && error("$(dir): frame $(t) names no image")
        if !(haskey(board.poses, t) && haskey(camera.poses, t))
            frames[i] = RigFrame(img, t, nothing, false, NaN, spot, false, NaN)
            continue
        end
        # BoardRigFrame → WorldFrame, then WorldFrame → CameraRigFrame: labeled end to end
        T = inv(camera.poses[t]) ∘ board.poses[t]
        dist = norm(T.t)
        is_base = base_R === nothing || abs(last_dist - dist) > RESET_DIST_MM
        if is_base
            base_R = T.R
            spot += 1
        end
        angle = _signed_rel_angle(base_R, T.R)
        frames[i] = RigFrame(img, t, T, t in camera.recovered, dist, spot, is_base, angle)
        last_dist = dist
    end
    return frames
end
