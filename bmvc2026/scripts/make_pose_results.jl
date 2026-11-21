#=
make_pose_results.jl — motion-capture pose-error results for one absolute-pose
dataset and its occlusion levels.

`evaluate_dataset` joins the detections of the three marker systems (their
keyed runs) with the motion-capture records through the hand-eye chain;
nothing is re-detected. Each (system, level) becomes ONE cell, its own product
(src/results.jl `cell_product`: Eval's `Product` keyed on the evaluation's
sources, produced through DrWatson's `produce_or_load`), carrying its
parameters and the upstream keys; its provenance records the code. Analysis
then goes through `pose_results(src)`, which collects the cells into a single
long-form DataFrame — one row per observation, with `system` and `level` as
ordinary columns.

Usage:
  julia --project=bmvc2026 bmvc2026/scripts/make_pose_results.jl [--config=<evaluation toml>] [--force]

--force recomputes cells that are produced (a code change is already a new key).
=#
# reproduce: reads configs/figures/evaluation.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
using .BlobBoardsEval

function main()
    eval_cfg, _ = figure_config("evaluation"; default = projectdir("configs", "figures", "evaluation.toml"))
    src = eval_sources(eval_cfg)
    force = "--force" in ARGS
    println("dataset: $(src.dataset); sources: ", Dict(k => p.key for (k, p) in src.products))
    write_pose_results(src; force)

    # The occlusion levels are the BlobBoards occlusion sweep's (its run config
    # of record, [experiment] occlusion_levels); every system is evaluated on each.
    for level in src.levels
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
