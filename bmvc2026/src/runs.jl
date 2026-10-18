# =============================================================================
# The runs — where every result lives, derived from the configs of record.
# =============================================================================
#
# ONLY INPUTS are published (data_root(): images, raw motion capture, the
# cameras of record, board definitions, the synthetic occlusion images, the
# descriptor networks). EVERY result — the motion-capture records, the
# detections of all three marker systems, the hand-eye fits, the pose-error
# cells — is produced by a codified stage into a keyed directory under
# results_root(), and found again the DrWatson way: the stage's config of
# record → its run key → its directory. Nothing is promoted, copied into the
# dataset or named by hand; a consumer that names an upstream config derives
# the same directory the producer wrote, so a changed config is a new
# directory and a stale result has no name to be read under.
#
#   stage                       config of record                 directory
#   motion-capture records      configs/convert_formats.toml     convert_formats-<key>/
#   BlobBoards detections       configs/bmvc2026_260527*.toml    bb-<key>/       (+ the run's subsets)
#   BlobBoards, occlusion       the same config                  occlusion-<key>/<LL>/
#   AprilTag/ArUco baselines    configs/baselines.toml           baselines-<key>/
#   hand-eye records            the evaluation's [runs]          hand_eye-<key>/
#   pose-error cells            the evaluation's [runs]          pose_results-<key>/
#
# A key is VisualGeometryToolkitEval's SHA-256 `run_key` of the config of record
# together with what decides the result and is not in it: the descriptor
# network's content (sha256) for a sweep, the upstream run keys for a
# consumer. Code is provenance (DrWatson `tag!` in every run's records), not
# key — see the run records.

import TOML
import SHA
using DrWatson: projectdir
using VisualGeometryToolkitEval.Experiments: run_key as config_run_key

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
    run_dir(generator, key) -> String

THE layout of a run: `results_root()/<generator>-<key>`.
"""
run_dir(generator::AbstractString, key::AbstractString) = joinpath(results_root(), string(generator, "-", key))

_sha256(path) = bytes2hex(open(SHA.sha256, path))

# --- motion-capture records ---------------------------------------------------

"The run key of the motion-capture records (scripts/convert_formats.jl)."
records_key(name::AbstractString = "convert_formats.toml") = config_run_key(load_config(name))

"""
    records_dir(dataset; config = "convert_formats.toml") -> String

The session directory of the motion-capture records of `dataset`:
`convert_formats-<key>/absolute_pose_experiments/<dataset>`.
"""
records_dir(dataset::AbstractString; config::AbstractString = "convert_formats.toml") =
    joinpath(run_dir("convert_formats", records_key(config)), "absolute_pose_experiments", dataset)

"The motion-tracker directory of a session's records (`poses/motion_tracker`)."
mocap_dir(dataset::AbstractString; config::AbstractString = "convert_formats.toml") =
    joinpath(records_dir(dataset; config), "poses", "motion_tracker")

# --- BlobBoards sweeps --------------------------------------------------------

"""
    network_path(run_config) -> String

The descriptor network a run config names (`[descriptor] model`,
data_root()-relative: an input of the assets tier).
"""
network_path(rc::AbstractDict) = joinpath(data_root(; tiers = ["bb_assets"]), String(rc["descriptor"]["model"]))

"""
    gallery_path(run_config) -> String

Where a sweep keeps its gallery: a CACHE of (board set × network), under
`results_root()/galleries/` — never in the input tree, and not in any key (its
own cache key carries the network, patch and each board's cutoff).
"""
gallery_path(rc::AbstractDict) = joinpath(results_root(), "galleries", basename(String(rc["gallery"]["path"])))

"""
    sweep_key(config; kind = "bb") -> String

The run key of a BlobBoards sweep: the run config of record, the content of
the network it names, and the sweep `kind` (`"bb"` — the run's image subsets —
or `"occlusion"` — the occlusion levels).
"""
function sweep_key(config::AbstractString; kind::AbstractString = "bb")
    kind in ("bb", "occlusion") || error("sweep kind $(repr(kind)): bb or occlusion")
    rc = load_config(config)
    return config_run_key(Dict{String, Any}("config" => rc, "network_sha256" => _sha256(network_path(rc)), "kind" => kind))
end

"The directory of a BlobBoards sweep (`bb-<key>` or `occlusion-<key>`)."
sweep_dir(config::AbstractString; kind::AbstractString = "bb") = run_dir(kind, sweep_key(config; kind))

# --- AprilTag/ArUco baselines -------------------------------------------------

"The run key of the AprilTag/ArUco baselines (scripts/detect_baselines.jl)."
baselines_key(config::AbstractString = "baselines.toml") = config_run_key(load_config(config))

"The directory of the AprilTag/ArUco baselines."
baselines_dir(config::AbstractString = "baselines.toml") = run_dir("baselines", baselines_key(config))

# --- the evaluation's sources -------------------------------------------------

"""
    PoseRun(main, occlusion, name)

Where one series' detections live: `main/poses/<name>` (and
`main/poses/calibration/<name>`), and per occlusion level
`occlusion/occlusion/<LL>/poses/<name>` (`occlusion === nothing`: the series
has no occlusion sweep).
"""
struct PoseRun
    main::String
    occlusion::Union{Nothing, String}
    name::String
end

"""
    Sources

Everything the evaluation of one dataset reads, resolved from its config of
record: the INPUTS (`inputs`, data_root()'s session directory: images, boards),
the motion-capture records (`mocap`), the detections of each series
(`poses[series]`) and the run keys that identify them all (`keys`).
"""
struct Sources
    dataset::String
    inputs::String
    mocap::String
    poses::Dict{String, PoseRun}
    keys::Dict{String, String}
end

"""
    eval_sources(cfg; tiers) -> Sources

The sources of an evaluation config (configs/figures/evaluation.toml): its
`[experiment] dataset` and its `[runs]` table, which names the config of
record of each upstream stage —

    [runs]
    records = "convert_formats.toml"           # motion-capture records
    bb = "bmvc2026_260527.toml"                # BlobBoards (+ its occlusion sweep)
    bb_noflip = "bmvc2026_260527_noflip.toml"  # the no-disambiguation ablation
    baselines = "baselines.toml"               # AprilTag + ArUco

The series are `apriltags`, `aruco` (both the baselines run), `bb` and
`bb_noflip`.
"""
function eval_sources(cfg::AbstractDict; tiers = EVAL_TIERS)
    ds = String(cfg["experiment"]["dataset"])
    r = cfg["runs"]
    bb, nf, bl = String(r["bb"]), String(r["bb_noflip"]), String(r["baselines"])
    bl_dir = baselines_dir(bl)
    poses = Dict(
        "apriltags" => PoseRun(bl_dir, bl_dir, "apriltags"),
        "aruco" => PoseRun(bl_dir, bl_dir, "aruco"),
        "bb" => PoseRun(sweep_dir(bb), sweep_dir(bb; kind = "occlusion"), "bb"),
        "bb_noflip" => PoseRun(sweep_dir(nf), nothing, "bb"),
    )
    keys = Dict(
        "records" => records_key(String(r["records"])), "baselines" => baselines_key(bl),
        "bb" => sweep_key(bb), "occlusion" => sweep_key(bb; kind = "occlusion"), "bb_noflip" => sweep_key(nf),
    )
    return Sources(ds, joinpath(data_root(; tiers), "absolute_pose_experiments", ds),
        mocap_dir(ds; config = String(r["records"])), poses, keys)
end

"""
    poses_dir(src, series; level = nothing, calibration = false) -> String

The directory holding `target_poses.json` of one series: its main run, its
calibration subset, or one occlusion level.
"""
function poses_dir(src::Sources, series::AbstractString; level = nothing, calibration::Bool = false)
    p = src.poses[series]
    level === nothing && return calibration ? joinpath(p.main, "poses", "calibration", p.name) : joinpath(p.main, "poses", p.name)
    calibration && error("an occlusion level has no calibration subset")
    p.occlusion === nothing && error("series $(series) has no occlusion sweep")
    return joinpath(p.occlusion, "occlusion", String(level), "poses", p.name)
end

"""
    val_dir(src, series) -> String

The detection overlays (`<image>_val.png`) of a series' main run.
"""
val_dir(src::Sources, series::AbstractString) = joinpath(src.poses[series].main, "val", src.poses[series].name)

"""
    derived_dir(src, generator, series_keys...) -> String

The run directory of a stage derived from the evaluation's sources (the
hand-eye records, the pose-error cells): keyed on the run keys it reads.
"""
derived_dir(src::Sources, generator::AbstractString, which = sort!(collect(keys(src.keys)))) =
    run_dir(generator, config_run_key(Dict{String, Any}("dataset" => src.dataset, (k => src.keys[k] for k in which)...)))

"""
    hand_eye_dir(src) -> String

The hand-eye records of the evaluation (scripts/make_hand_eye.jl):
`hand_eye-<key>/`, keyed on the motion-capture records and the detections of
the calibration subsets they are fitted on.
"""
hand_eye_dir(src::Sources) = derived_dir(src, "hand_eye", ["records", "baselines", "bb"])
