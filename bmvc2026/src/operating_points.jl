# =============================================================================
# Operating points — where the tuned detection operating point lives
# =============================================================================
#
# The detection operating point — the pose arm's outlier half-width, the
# detector's peak threshold and the descriptor similarity gate — is selected
# JOINTLY by the cross-validation of scripts/detection_cv.jl and stored in its
# tuning artifact (VisualGeometryToolkitEval `save_coarse_fine_artifact`) in the run
# directory (Eval's `keyed_cell_dir`) keyed by the run key of
# configs/detection_cv.toml with its run config's folded in. A consumer reads
# a tuned value through VisualGeometryToolkitEval's one gated reader,
# `deployed_setting(path)`, never from a literal; test/operating_points_tests.jl
# holds the configs of record to the artifacts they were adopted from
# (`ADOPTED_OPERATING_POINTS`).
#
# Self-contained (its own imports), so the test reads it without loading the
# detection stack.
# =============================================================================

import TOML
using DrWatson: projectdir
using VisualGeometryToolkitEval.Experiments: results_root, run_key, keyed_cell_dir

"""
    EXPERIMENT

The experiment every bmvc2026 run directory is filed under (Eval's
`keyed_cell_dir(cfg, EXPERIMENT, kind)`). Here because this file is
self-contained: src/runs.jl reads the same binding through BlobBoardsEval.
"""
const EXPERIMENT = "bmvc2026"

"""
    detection_cv_config(cfg) -> Dict

What the joint detection CV is keyed on: `cfg` (the parsed
configs/detection_cv.toml) with the run key of the run config of record it
searches around (`[runs] bb`) folded in as `sources`.
"""
detection_cv_config(cfg::AbstractDict) = merge(cfg, Dict{String, Any}("sources" =>
    Dict{String, Any}("bb" => run_key(TOML.parsefile(projectdir("configs", cfg["runs"]["bb"]))))))

"""
    detection_cv_dir(cfg) -> String

The run directory of the joint detection CV under the config `cfg`:
`keyed_cell_dir(detection_cv_config(cfg), EXPERIMENT, "detection_cv")` — keyed,
so a changed config (or run config) is a new run.
"""
detection_cv_dir(cfg::AbstractDict) = keyed_cell_dir(detection_cv_config(cfg), EXPERIMENT, "detection_cv")

"The tuning artifact inside a joint detection CV run directory."
detection_cv_artifact(dir::AbstractString) = joinpath(dir, "detection.jld2")

"""
    ADOPTED_OPERATING_POINTS

Every tuned value the run configs of record carry: its key path in a run
config, the tuning artifact it is deployed from (`results_root()`-relative)
and the artifact's axis. All three are the record of configs/detection_cv.toml
under its run key of the 0.8 layout, `f5a385376eda453e`
(`results_root()/tuning/detection_cv-<key>`): the joint search kept its
declared incumbent, the configs' operating point
(scripts/detection_cv_replay.jl). The 0.9 re-key (`keyed_cell_dir`, the
[experiment] layout, the run config's key folded in) names a new directory
for the config; the adopted record stays where it was made.
"""
const ADOPTED_OPERATING_POINTS = [
    (key = k, axis = a, artifact = joinpath("tuning", "detection_cv-f5a385376eda453e", "detection.jld2"))
    for (k, a) in ((("find_boards", "geometry", "outlier_halfwidth"), :outlier_halfwidth),
                   (("find_boards", "detect", "peak_threshold"), :peak_threshold),
                   (("find_boards", "min_similarity"), :min_similarity))
]
