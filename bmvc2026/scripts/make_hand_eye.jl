#=
make_hand_eye.jl — hand-eye (robot-world) calibrations for the marker boards.

For each marker system, fit the robot-world calibration T_det(i,m) = X · T_cb(i)
· Y_m on its calibration subset (the motion-capture records'
calibration_<abbr> + the system's detections on the calibration captures) —
camera-untracked calibration frames carry the evidence-gated static-camera
substitution (flagged `recovered` in the camera-rig trajectory) — and write
the result, every pose with the fit's covariance:

  results_root()/hand_eye-<key>/hand_eye_<name>.json   (src/runs.jl `derived_dir`:
                                       keyed on the run keys of its sources)
    vgio/transforms/v3   X: body:camera_rig → camera_optical (+ diagnostics in provenance)
  …/hand_eye_<name>_board_rig.json
    vgio/rig/v4          Y_m: target:<uid> → body:board_rig, one per marker

The estimator is RVG's `fit_robot_world_hand_eye` staged linear solve (Kabsch on
pooled relative-motion axes, chordal marker means, joint translation LSQ).
Nothing is re-detected.

Usage:
  julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/make_hand_eye.jl [--config=<evaluation toml>]
=#
# reproduce: reads configs/figures/evaluation.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)


using VisualGeometryToolkitCore: pose, ids
using Rotations: RotMatrix
using StaticArrays

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval

const EVAL, _ = figure_config("evaluation"; default = projectdir("configs", "figures", "evaluation.toml"))
const SRC = eval_sources(EVAL)
# The hand-eye records are DERIVED from the detections and the motion-capture
# records, so they are keyed on those runs (`hand_eye_dir`), never written into
# the input tree. The fit itself is cached and keyed by `hand_eye`
# (src/calibration.jl); this is its published, human-readable form.
results_dir = hand_eye_dir(SRC)
mkpath(results_dir)

_stem(f) = splitext(basename(f))[1]

println("sources: $(SRC.keys)")
for bt in BlobBoardsEval.BOARD_TYPES
    calib_poses = poses_dir(SRC, bt.name; calibration = true)

    # The calibration rig sequence carries the static-camera recovery as
    # `recovered` frames of the camera-rig trajectory (src/recovery.jl; the
    # converter ran the evidence gates and recorded their outcome).
    frames = rig_sequence(SRC.mocap, "calibration_" * bt.abbr)
    n_used = count(f -> f.T_cb !== nothing, frames)
    n_tracked = count(f -> f.T_cb !== nothing && !f.recovered, frames)
    detected = Dict(_stem(i) => p for (i, p) in load_detected_poses(calib_poses))

    # The ONE hand-eye authority (src/calibration.jl): cached, keyed on the
    # frames and markers that determine the fit, and tagged with the commit.
    # This was an independent refit — the project had FOUR of them.
    he = hand_eye(
        frames, detected;
        dataset = SRC.dataset, subset = "calibration_" * bt.abbr
    )
    diag = he.diagnostics

    # The published records: the cache's pair (X with its covariance, the
    # marker mount with theirs) written under the published name by the one
    # hand-eye writer, the run's facts in the transforms file's provenance.
    out = joinpath(results_dir, "hand_eye_$(bt.name).json")
    X, targets = he.uncertain
    save_hand_eye(
        out, X, Dict(String(id) => pose(targets[id]) for id in ids(targets));
        provenance = (
            method = "VisualGeometryToolkitEstimation fit_robot_world_hand_eye (staged linear: " *
                "Kabsch + chordal means + joint translation LSQ; RVG >= 0.14); covariance first order (the fit's)",
            calibration_subset = "calibration_" * bt.abbr,
            n_frames_tracked = n_tracked,
            n_frames_used = n_used,
            n_frames_recovered = n_used - n_tracked,
            recovery = "camera-untracked frames recovered by evidence-gated " *
                "static-camera substitution (see the camera-rig trajectory's provenance)",
            n_markers = diag.n_markers,
            sigma_rot_rad = round(diag.sigma_rot_rad; digits = 6),
            sigma_trans_mm = round(diag.sigma_trans_mm; digits = 4),
            gitcommit = diag.gitcommit,
        )
    )
    println(
        "$(bt.name): $(n_used) frames ($(n_used - n_tracked) recovered), " *
            "$(diag.n_markers) markers, " *
            "σ_rot=$(round(rad2deg(diag.sigma_rot_rad); digits = 3))°, " *
            "σ_trans=$(round(diag.sigma_trans_mm; digits = 2))mm -> $out"
    )
end
