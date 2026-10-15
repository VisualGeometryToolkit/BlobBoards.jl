# =============================================================================
# Operating points — where the tuned detection operating point lives
# =============================================================================
#
# The detection operating point — the pose arm's outlier half-width, the
# detector's peak threshold and the descriptor similarity gate — is selected
# JOINTLY by the cross-validation of scripts/detection_cv.jl and stored in its
# tuning artifact (VisualGeometryEval `save_coarse_fine_artifact`) in the run
# directory keyed by the run key of configs/detection_cv.toml. A consumer reads
# a tuned value through VisualGeometryEval's one gated reader,
# `deployed_setting(path)`, never from a literal; test/operating_points_tests.jl
# holds the configs of record to the artifacts they were adopted from
# (`ADOPTED_OPERATING_POINTS`).
#
# Self-contained (its own imports), so the test reads it without loading the
# detection stack.
# =============================================================================

import VisualGeometryEval
using VisualGeometryEval.Experiments: results_root

"""
    detection_cv_dir(cfg) -> String

The run directory of the joint detection CV under the config `cfg` (the
parsed configs/detection_cv.toml): `results_root()/tuning/detection_cv-<run_key(cfg)>`
— keyed, so a changed config is a new run.
"""
detection_cv_dir(cfg::AbstractDict) =
    joinpath(results_root(), "tuning", "detection_cv-" * VisualGeometryEval.Experiments.run_key(cfg))

"The tuning artifact inside a joint detection CV run directory."
detection_cv_artifact(dir::AbstractString) = joinpath(dir, "detection.jld2")

"""
    ADOPTED_OPERATING_POINTS

Every tuned value the run configs of record carry: its key path in a run
config, the tuning artifact it is deployed from (`results_root()`-relative)
and the artifact's axis. All three are the record of configs/detection_cv.toml
(run key `f5a385376eda453e`): the joint search kept its declared incumbent,
the configs' operating point (scripts/detection_cv_replay.jl).
"""
const ADOPTED_OPERATING_POINTS = [
    (key = k, axis = a, artifact = joinpath("tuning", "detection_cv-f5a385376eda453e", "detection.jld2"))
    for (k, a) in ((("find_boards", "geometry", "outlier_halfwidth"), :outlier_halfwidth),
                   (("find_boards", "detect", "peak_threshold"), :peak_threshold),
                   (("find_boards", "min_similarity"), :min_similarity))
]
