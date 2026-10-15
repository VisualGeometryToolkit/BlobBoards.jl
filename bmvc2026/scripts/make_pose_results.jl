#=
make_pose_results.jl — motion-capture pose-error results for one absolute-pose
dataset and its occlusion levels.

`evaluate_dataset` joins the detections of the three marker systems (their
keyed runs) with the motion-capture records through the hand-eye chain;
nothing is re-detected. Each (system, level) becomes ONE cell written through
DrWatson's `produce_or_load` into the keyed `pose_results-<key>/` of the
evaluation's sources (src/results.jl), carrying its parameters, the upstream
run keys and a `tag!` git record inside the file. Analysis then goes through
`pose_results(src)`, which `collect_results` into a single long-form
DataFrame — one row per observation, with `system` and `level` as ordinary
columns.

Usage:
  julia --project=bmvc2026 bmvc2026/scripts/make_pose_results.jl [--config=<evaluation toml>]

Env knob:
  BB_FORCE   recompute cells that already exist (config-keyed caching cannot
             see a CODE change — same reason the sweeps need it)
=#
# reproduce: reads configs/figures/evaluation.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
using .BlobBoardsEval

function main()
    eval_cfg, _ = figure_config("evaluation"; default = projectdir("configs", "figures", "evaluation.toml"))
    src = eval_sources(eval_cfg)
    force = haskey(ENV, "BB_FORCE")
    println("dataset: $(src.dataset); sources: $(src.keys)")
    write_pose_results(src; force)

    # The occlusion levels are the BlobBoards occlusion sweep's (its run config
    # of record, [experiment] occlusion_levels); every system is evaluated on each.
    levels = String.(load_config(eval_cfg["runs"]["bb"])["experiment"]["occlusion_levels"])
    for level in levels
        println("\nocclusion level $level:")
        write_pose_results(src; occlusion_level = level, force)
    end

    # Collect immediately so a bad cell surfaces here rather than at plot time.
    df = pose_results(src)
    println("\ncollected: $(size(df, 1)) observations x $(size(df, 2)) columns")
    println("systems: ", sort(unique(df.system)))
    println("levels : ", sort(unique(df.level)))
    return nothing
end

main()
