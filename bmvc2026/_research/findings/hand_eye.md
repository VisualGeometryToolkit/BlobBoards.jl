# Hand-eye (robot-world) calibrations — dataset 260527

Driver: `bmvc2026/scripts/make_hand_eye.jl` (BB_SET=260527, 2026-08-07; bb
detections regenerated the same day with the log-polar s96 descriptor
network — see `bmvc2026/scripts/go_bmvc2026.jl`).
Data: `absolute_pose_experiments/260527/results/hand_eye_{bb,apriltags,aruco}.json`.

One robot-world fit per marker system, `T_det(i,m) = X · T_cb(i) · Y_m`,
solved on the system's calibration subset by VisualGeometryToolkitEstimation's
`fit_robot_world_hand_eye` (staged linear estimator: Kabsch on pooled relative-motion
axes → chordal marker means → joint translation least squares; RVG ≥ 0.14).
Camera-untracked calibration frames are first recovered by the
evidence-gated static-camera substitution (mocap constancy ≤ 1 mm / 0.5°
AND image-background constancy), which matters only for the blob-board
subset (4 tracked → 41 usable frames).

## Fit summaries

| System | Frames (recovered) | Markers | s² | σ_rot | σ_trans | min eig JᵀJ |
|---|---|---|---|---|---|---|
| blob boards | 41 (37) | 30 | 1.026 | 0.27° | 0.68 mm | 5.41 |
| AprilTags   | 22 (0)  | 30 | 1.050 | 0.43° | 2.54 mm | 0.21 |
| ArUco       | 15 (0)  | 30 | 1.075 | 1.89° | 2.81 mm | 0.21 |

σ values are the profiled per-block noise scales of the decoupled pose
residual at the estimate; s² ≈ 1 confirms the profiling. The min-eigenvalue
column is the observability diagnostic — the 25× gap between blob boards
and the tag systems reflects the recovered 41-frame geometry vs the smaller
single-session tag subsets (a collapsed value would indicate an
unconstrained gauge direction, the 4-frame pathology this diagnostic
exists to expose).

## Notes

- The estimator is deliberately linear. A pose-residual NLLS refinement
  (MLE under an isotropic, homoscedastic pose-noise model) was implemented,
  validated on synthetics, and REMOVED (RVG 0.14): on this dataset it
  reduced in-sample cost by the amount its own quadratic model predicts
  (ΔC ≈ 283 ≈ 17σ) yet extrapolated worse across the evaluation range
  (3.3/6.3/10.4 vs 2.8/4.2/6.7 mm common-mode bands) — model
  misspecification absorbing an unmodeled systematic along a weakly
  constrained direction. See RVG `docs/robot_world_hand_eye_objective.pdf`.
- The sound upgrade path is measurement-level: minimize reprojection error
  of the extracted image features (blob centroids, tag corners) over
  (X, Y₁…Y_M), initialized at this linear solution. Features are not
  stored in the dataset but are recomputable: the board-side gallery
  (`260527/board_features*.jld2`) and the matcher checkpoint ship with the
  archive; tag corners re-detect deterministically.
- Each JSON carries the full 6×6 marginal covariance of X in (rad, mm)
  tangent coordinates plus provenance in `meta` — persist, never discard.
