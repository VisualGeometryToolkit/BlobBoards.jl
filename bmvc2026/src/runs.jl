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
#   stage                       config of record                 product (its name; output)
#   motion-capture records      configs/convert_formats.toml     convert_formats  (_report.json)
#   BlobBoards detections       configs/bmvc2026_260527*.toml    bb               (poses/: the run's subsets)
#   BlobBoards, occlusion       the same config                  occlusion        (occlusion/<LL>/)
#   AprilTag/ArUco baselines    configs/baselines.toml           baselines        (poses/)
#   camera import               configs/import_cameras.toml      import_cameras   (_cameras.toml)
#   hand-eye records            the evaluation's [runs]          hand_eye         (hand_eye_<series>.json)
#   pose-error cells            the evaluation's [runs]          pose_results     (one per system and level)
#
# Every directory is a VisualGeometryToolkitEval `Product` of the config of
# record (`stage_product`): `results_root()/<layout…>/bmvc2026/<name>/<key>`,
# DrWatson's `datadir("exp_pro", experiment, name, key)`, the layout the
# config's `[experiment] layout` literal. A key is Eval's `run_key` of the
# config of record, which states every input it reads as `[inputs.<id>]`
# (tier, path, provenance, sha256, checked by Eval's `input_path` — the
# descriptor network's content is in the key that way), with the code trees it
# runs folded in; a derived stage keys the evaluation's [experiment] layout
# with the upstream products it reads (their keys and their outputs' content
# hashes, `upstream`). A product is produced once (`produce`: its
# `_provenance.toml`, written last, is the completion mark) and reused while
# `produced`.

import TOML
using DrWatson: projectdir
# `input_path(cfg, id)`: Eval's one reader of an `[inputs.<id>]` (its tier's
# root, its sha256 checked); re-exported by BlobBoardsEval for the drivers.
using VisualGeometryToolkitEval.Experiments: run_key, input_path, Product, product_dir, product_output, produce, produced, load_output

# `EXPERIMENT`, the name every run directory is filed under, is
# src/operating_points.jl's (a self-contained file; BlobBoardsEval includes it), and so are
# `config_path`/`load_config`, the config of record by name, and `stage_product`.

# --- motion-capture records ---------------------------------------------------

"The product of the motion-capture records (scripts/convert_formats.jl) of the config of record `config`."
records_product(config::AbstractString = "convert_formats.toml") =
    stage_product(load_config(config), "convert_formats"; output = "_report.json")

"""
    records_dir(dataset; config = "convert_formats.toml") -> String

The session directory of the motion-capture records of `dataset`:
`<convert_formats product>/absolute_pose_experiments/<dataset>`.
"""
records_dir(dataset::AbstractString; config::AbstractString = "convert_formats.toml") =
    joinpath(product_dir(records_product(config)), "absolute_pose_experiments", dataset)

"The motion-tracker directory of a session's records (`poses/motion_tracker`)."
mocap_dir(dataset::AbstractString; config::AbstractString = "convert_formats.toml") =
    joinpath(records_dir(dataset; config), "poses", "motion_tracker")

# --- BlobBoards sweeps --------------------------------------------------------

"""
    gallery_path(run_config) -> String

Where a sweep keeps its gallery: a CACHE of (board set × network), under
`results_root()/galleries/` — never in the input tree, and not in any key (its
own cache key carries the network, patch and each board's cutoff).
"""
gallery_path(rc::AbstractDict) = joinpath(results_root(), "galleries", basename(String(rc["gallery"]["path"])))

"""
    sweep_product(config; kind = "bb") -> Product

The product of a BlobBoards sweep of the run config `config`: `bb` (the run's
image subsets, under `poses/` and `val/`) or `occlusion` (the occlusion levels,
under `occlusion/<LL>/`).
"""
function sweep_product(config::AbstractString; kind::AbstractString = "bb")
    kind in ("bb", "occlusion") || error("sweep kind $(repr(kind)): bb or occlusion")
    return stage_product(load_config(config), kind; output = kind == "bb" ? "poses" : "occlusion")
end

"The directory of a BlobBoards sweep (its product's)."
sweep_dir(config::AbstractString; kind::AbstractString = "bb") = product_dir(sweep_product(config; kind))

# --- AprilTag/ArUco baselines -------------------------------------------------

"The product of the AprilTag/ArUco baselines (scripts/detect_baselines.jl)."
baselines_product(config::AbstractString = "baselines.toml") = stage_product(load_config(config), "baselines"; output = "poses")

"The directory of the AprilTag/ArUco baselines."
baselines_dir(config::AbstractString = "baselines.toml") = product_dir(baselines_product(config))

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
record: its `[experiment]` layout (`experiment`: its `layout`, and `dataset`), the
INPUTS (`inputs`, data_root()'s session directory: images, boards), the
motion-capture records (`mocap`), the detections of each series
(`poses[series]`), the upstream products that hold them all (`products`) and
the occlusion levels of the BlobBoards sweep (`levels`, its run config's
`[experiment] occlusion_levels`).
"""
struct Sources
    experiment::Dict{String, Any}
    dataset::String
    inputs::String
    mocap::String
    poses::Dict{String, PoseRun}
    products::Dict{String, Product}
    levels::Vector{String}
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
    products = Dict{String, Product}(
        "records" => records_product(String(r["records"])), "baselines" => baselines_product(bl),
        "bb" => sweep_product(bb), "occlusion" => sweep_product(bb; kind = "occlusion"), "bb_noflip" => sweep_product(nf),
    )
    bl_dir = product_dir(products["baselines"])
    poses = Dict(
        "apriltags" => PoseRun(bl_dir, bl_dir, "apriltags"),
        "aruco" => PoseRun(bl_dir, bl_dir, "aruco"),
        "bb" => PoseRun(product_dir(products["bb"]), product_dir(products["occlusion"]), "bb"),
        "bb_noflip" => PoseRun(product_dir(products["bb_noflip"]), nothing, "bb"),
    )
    # the [experiment] keys a derived product is filed and keyed by
    layout = Dict{String, Any}(k => v for (k, v) in cfg["experiment"] if k in ("dataset", "layout"))
    return Sources(layout, ds, joinpath(data_root(; tiers), "absolute_pose_experiments", ds),
        joinpath(product_dir(products["records"]), "absolute_pose_experiments", ds, "poses", "motion_tracker"), poses, products,
        String.(load_config(bb)["experiment"]["occlusion_levels"]))
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
    derived_product(src, name, which; output, record = Dict()) -> Product

The product of a stage derived from the evaluation's sources (the hand-eye
records, a pose-error cell): Eval's `Product` over the evaluation's
[experiment] layout (and `record`, the cell's own keys), the upstream products
`which` folded in (`upstream`: their keys and their outputs' content hashes).
"""
derived_product(src::Sources, name::AbstractString, which; output::AbstractString,
                record::AbstractDict = Dict{String, Any}()) =
    stage_product(merge(Dict{String, Any}("experiment" => copy(src.experiment)), record), name; output,
        upstream = Dict{String, Product}(k => src.products[k] for k in which))

"""
    hand_eye_product(src) -> Product

The hand-eye records of the evaluation (scripts/make_hand_eye.jl), keyed on
the motion-capture records and the detections of the calibration subsets they
are fitted on; one `hand_eye_<series>.json` per series.
"""
hand_eye_product(src::Sources) = derived_product(src, "hand_eye", ["records", "baselines", "bb"]; output = "hand_eye_bb.json")

"The directory of the evaluation's hand-eye records."
hand_eye_dir(src::Sources) = product_dir(hand_eye_product(src))
