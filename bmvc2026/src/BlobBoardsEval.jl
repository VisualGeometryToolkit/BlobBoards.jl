#=
BlobBoardsEval — motion-capture pose-error evaluation for the absolute-pose
experiments (the inputs under data_root()/absolute_pose_experiments, every
result under results_root(): src/runs.jl).

Consumes the detected board poses (<run>/poses/{apriltags,aruco,bb}/target_poses.json,
vgio/target_poses/v1) and the motion-tracker records
(<records>/poses/motion_tracker/<subset>/{board_rig,camera_rig}.json, vgio/trajectory/v3); produces
per-(image, marker) pose-error tables as DrWatson result cells
(`write_pose_results`, collected with `pose_results`) and plots.
Nothing here re-runs detection.
=#
module BlobBoardsEval

using JSON
using DataFrames
using Glob
using LinearAlgebra
using Rotations
using StaticArrays
using Statistics
using VisualGeometryToolkitCore: EuclideanMap, Frame, ViewFrame, TargetFrame, Rig, PosedTarget, PlanarTarget,
    relabel, unlabel, pose, model, ids, ustrip, with_unit, mm
using VisualGeometryToolkitEstimation: fit_robot_world_hand_eye
using BlobBoards

include("data.jl")
# Where every result lives: the configs of record → run keys → results_root() dirs.
include("runs.jl")
include("paper_assets.jl")
include("eval_images.jl")
include("rig.jl")
include("pose_error.jl")
include("error_table.jl")
include("recovery.jl")
# The provenance record (write_provenance / read_provenance) is the stack's
# one implementation, VisualGeometryToolkitEval.Experiments; cache.jl wraps its keyed
# result cache.
using VisualGeometryToolkitEval.Experiments: write_provenance, read_provenance
include("cache.jl")
# After cache.jl (uses `cached`) and error_table.jl (uses `evaluate_dataset`).
include("results.jl")
# after cache.jl and pose_error.jl: the one hand-eye authority
include("calibration.jl")
# Where the tuned detection operating point lives (scripts/detection_cv.jl writes it).
include("operating_points.jl")
# Per-image detection exports (the AprilTag/ArUco baselines) → vgio/target_poses/v1.
include("detections.jl")
# The one-time port of the upstream calibration exports (scripts/import_cameras.jl).
include("cameras.jl")

export RigFrame, rig_sequence, load_detected_poses, marker_sizes, detected_marker_sizes,
    CameraRigFrame, BoardRigFrame,
    # hand_eye_calibrate is deliberately NOT exported: it is the primitive,
    # and `hand_eye` (calibration.jl) is the one authority that calls it.
    # Exporting it is how this project ended up with four independent fits.
    pose_error_table, evaluate_dataset,
    static_camera_recovery, static_recovery_accepted,
    data_root, results_root, paper_images_dir, paper_figures_dir, figure_config, write_figure_records, EVAL_TIERS,
    camera_of_record,
    config_path, load_config, run_dir, records_key, records_dir, mocap_dir, network_path, gallery_path,
    sweep_key, sweep_dir, baselines_key, baselines_dir, PoseRun, Sources, eval_sources, poses_dir, val_dir, derived_dir, hand_eye_dir,
    write_provenance, read_provenance,
    cached, hand_eye, save_hand_eye, load_hand_eye, uncertain_hand_eye, inv_pose_covariance,
    write_pose_results, pose_results,
    export_target_poses, upstream_camera

end # module
