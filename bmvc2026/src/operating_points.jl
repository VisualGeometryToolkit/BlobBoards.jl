# =============================================================================
# Operating points — where the tuned detection operating point lives
# =============================================================================
#
# The detection operating point — the pose arm's outlier half-width, the
# detector's peak threshold and the descriptor similarity gate — is selected
# JOINTLY by the cross-validation of scripts/detection_cv.jl and stored in its
# tuning artifact (VisualGeometryToolkitEval `save_coarse_fine_artifact`) in its
# product's directory (Eval's `Product`), keyed by configs/detection_cv.toml
# with its run config's key folded in. A consumer reads
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
using VisualGeometryToolkitEval.Experiments: results_root, run_key, Product, product_dir

"""
    EXPERIMENT

The experiment every bmvc2026 product is filed under, below a config's
`[experiment] layout` ([`stage_product`](@ref)). Here because this file is
self-contained: src/runs.jl reads the same binding through BlobBoardsEval.
"""
const EXPERIMENT = "bmvc2026"

"""
    stage_product(cfg, name; output, upstream = nothing) -> Product

The keyed product `name` of the config of record `cfg` (Eval's `Product`):
filed under `<[experiment] layout…>/bmvc2026`, keyed by `cfg` itself, the code
it runs and the `upstream` products or keys it reads; `output` is the file (or
directory) its driver writes.
"""
stage_product(cfg::AbstractDict, name::AbstractString; output::AbstractString, upstream = nothing) =
    Product(join([String.(cfg["experiment"]["layout"]); EXPERIMENT], "/"), name, cfg; output, upstream)

"""
    config_path(name) -> String

A config of record by name: `configs/<name>` of the project (an absolute path
is returned as is).
"""
config_path(name::AbstractString) = isabspath(name) ? String(name) : projectdir("configs", name)

"""
    load_config(name) -> Dict

The parsed config of record `configs/<name>`.
"""
load_config(name::AbstractString) = TOML.parsefile(config_path(name))

"""
    detection_cv_upstream(cfg) -> Dict

What the joint detection CV folds into its key beside `cfg` (the parsed
configs/detection_cv.toml): the run key of the run config of record it
searches around (`[runs] bb`), Eval's `run_key(…; upstream)`.
"""
detection_cv_upstream(cfg::AbstractDict) =
    Dict{String, String}("bb" => run_key(load_config(cfg["runs"]["bb"])))

"""
    detection_cv_product(cfg) -> Product

The product of the joint detection CV under the config `cfg`, its run
config's key folded in (`detection_cv_upstream`): a changed config (or run
config) is a new product. Its output is the tuning artifact `detection.jld2`.
"""
detection_cv_product(cfg::AbstractDict) =
    stage_product(cfg, "detection_cv"; output = "detection.jld2", upstream = detection_cv_upstream(cfg))

"The directory of the joint detection CV's product."
detection_cv_dir(cfg::AbstractDict) = product_dir(detection_cv_product(cfg))

"The tuning artifact inside a joint detection CV run directory."
detection_cv_artifact(dir::AbstractString) = joinpath(dir, "detection.jld2")

"The supplementary table of the axes `ax` (one axis: its curve; two: their surface) inside a joint detection CV run directory."
detection_cv_table(dir::AbstractString, ax) = joinpath(dir, "table_" * join(ax, "__") * ".tex")

"""
    ADOPTED_OPERATING_POINTS

Every tuned value the run configs of record carry: its key path in a run
config, the tuning artifact it is deployed from (`results_root()`-relative)
and the artifact's axis. All three are the record of configs/detection_cv.toml
under its run key of the 0.8 layout, `f5a385376eda453e`
(`results_root()/tuning/detection_cv-<key>`): the joint search kept its
declared incumbent, the configs' operating point
(scripts/detection_cv_replay.jl). The products of Eval 0.9 on (the
[experiment] layout, the run config's key and the code folded in) name a new
directory for the config; the adopted record stays where it was made.
"""
const ADOPTED_OPERATING_POINTS = [
    (key = k, axis = a, artifact = joinpath("tuning", "detection_cv-f5a385376eda453e", "detection.jld2"))
    for (k, a) in ((("find_boards", "geometry", "outlier_halfwidth"), :outlier_halfwidth),
                   (("find_boards", "detect", "peak_threshold"), :peak_threshold),
                   (("find_boards", "min_similarity"), :min_similarity))
]
