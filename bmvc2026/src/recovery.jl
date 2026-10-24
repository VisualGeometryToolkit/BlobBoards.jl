# =============================================================================
# Mocap recovery: static-camera substitution for untracked camera-rig frames
# =============================================================================
#
# The 260527 calibration_bb session lost camera-rig tracking for 37 of 41
# captures (setup error, fixed mid-session) while the hand-held BOARD rig
# tracked throughout. The camera sat on a tripod: when its pose is provably
# constant, the missing measurement is recoverable by substitution — which
# took the hand-eye calibration from 4 usable frames (yaw ⊕ t unobservable,
# 7/21/36 mm range-proportional bias) to 41 (2.8/4.2/6.7 mm; 2026-08-06).
#
# Recovery is EVIDENCE-GATED, never assumed:
#   1. mocap constancy — the camera-rig pose across all tracked frames stays
#      within (t_tol_mm, ang_tol_deg) of the chordal mean;
#   2. image constancy — background strips (top rows + side columns; the
#      hand-held board is center-frame) are pixel-static across the WHOLE
#      session including the untracked era (max abs gray difference below
#      bg_tol; a one-pixel camera shift on textured background reads ≥ 5×).
# Both must succeed or the function throws.
#
# Recovery is a PRODUCER step: scripts/convert_formats.jl runs it once, while
# writing the camera-rig trajectory of each calibration subset from the
# tracker's export, and the trajectory carries its result — every substituted
# pose flagged `recovered`, the gate outcome and its evidence in the file's
# provenance (`recovery`). The evaluation is a consumer: it reads the
# trajectory (`rig_sequence`) and never re-runs the gates, so a reader of the
# published record gets the same frames as the producer did.

using VisualGeometryToolkitCore: chordal_mean, EuclideanMap
using Rotations: RotMatrix, rotation_angle
using Colors: Gray
import FileIO

"""
    static_camera_recovery(camera, board_tracked, images, img_dir;
                           t_tol_mm = 1.0, ang_tol_deg = 0.5, bg_tol = 0.15)
        -> nothing | (; pose, untracked, evidence)

The static-camera substitution for one calibration subset (see the file
header). `camera[i]` is frame `i`'s camera-rig pose in the tracker's world
(`nothing` when untracked), `board_tracked[i]` whether its board rig tracked,
`images[i]` its image in `img_dir`. Returns `nothing` when no frame needs
recovery (every board-tracked frame has its camera), and otherwise the
camera's constant pose (the chordal mean of the tracked poses), the frames it
substitutes into (board tracked, camera not), and the gates' evidence. Throws
when either gate fails — the camera moved, so there is nothing to recover.
"""
function static_camera_recovery(
        camera::AbstractVector, board_tracked::AbstractVector{Bool}, images::AbstractVector{<:AbstractString},
        img_dir::AbstractString; t_tol_mm::Real = 1.0, ang_tol_deg::Real = 0.5, bg_tol::Real = 0.15
    )
    cam_ok = [c !== nothing for c in camera]
    tracked = findall(cam_ok)
    untracked = findall(board_tracked .& .!cam_ok)
    isempty(untracked) && return nothing
    length(tracked) >= 2 || error("only $(length(tracked)) camera-tracked frame(s) — cannot establish constancy")

    Rc = [SMatrix{3, 3, Float64}(camera[i].R) for i in tracked]
    tc = [SVector{3, Float64}(camera[i].t) for i in tracked]

    # Gate 1: mocap constancy about the chordal mean
    R_mean = chordal_mean(Rc)
    t_mean = sum(tc) / length(tc)
    drift_t = maximum(norm(t - t_mean) for t in tc)
    drift_a = maximum(rad2deg(rotation_angle(RotMatrix{3}(R' * R_mean))) for R in Rc)
    (drift_t <= t_tol_mm && drift_a <= ang_tol_deg) || error(
        "camera rig NOT static per mocap (drift $(round(drift_t; digits = 2)) mm, " *
            "$(round(drift_a; digits = 3))°) — refusing recovery"
    )

    # Gate 2: image-background constancy across the whole session
    imgA = FileIO.load(joinpath(img_dir, images[first(untracked)]))
    imgB = FileIO.load(joinpath(img_dir, images[last(tracked)]))
    gray(x) = Float64.(Gray.(x))   # Gray/RGB tolerant magnitude
    a, b = gray(imgA), gray(imgB)
    bg = max(
        maximum(abs.(a[1:min(200, end), :] .- b[1:min(200, end), :])),
        maximum(abs.(a[:, 1:min(150, end)] .- b[:, 1:min(150, end)]))
    )
    bg <= bg_tol || error(
        "image background differs ($(round(bg; digits = 3)) gray levels) " *
            "between untracked and tracked eras — camera may have moved; refusing recovery"
    )
    evidence = (
        n_tracked = length(tracked),
        mocap_drift_mm = round(drift_t; digits = 3),
        mocap_drift_deg = round(drift_a; digits = 4),
        background_max_absdiff = round(bg; digits = 4),
        tolerances = (t_tol_mm = Float64(t_tol_mm), ang_tol_deg = Float64(ang_tol_deg), bg_tol = Float64(bg_tol)),
    )
    return (; pose = EuclideanMap(RotMatrix{3, Float64}(R_mean), t_mean; check = false), untracked, evidence)
end

# The outcome the converter recorded in a camera-rig trajectory's provenance.
const RECOVERY_STATUSES = ("not needed", "applied", "refused")

"""
    static_recovery_accepted(mocap_dir, subset) -> Bool

Whether the static-camera recovery gates of `subset` held (`"applied"`, or
`"not needed"` when no frame lacked its camera) rather than refused, as
recorded in the provenance of its camera-rig trajectory.
"""
function static_recovery_accepted(mocap_dir::AbstractString, subset::AbstractString)
    f = joinpath(mocap_dir, subset, "camera_rig.json")
    rec = get(JSON.parse(read(f, String)), :provenance, nothing)
    status = rec === nothing ? nothing : get(get(rec, :recovery, Dict()), :status, nothing)
    status in RECOVERY_STATUSES || error("$(f): provenance.recovery.status is $(repr(status)), not one of $(RECOVERY_STATUSES)")
    return status != "refused"
end
