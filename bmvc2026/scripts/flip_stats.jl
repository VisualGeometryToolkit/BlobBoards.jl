#=
flip_stats.jl — planar-pose flip statistics per marker system.

Reads the pose_results cells written by make_pose_results.jl and
prints, per system × tripod spot: detections, rotation median/p90, the flip
count (rotation error > 10° — the secondary mode; the >10° threshold sits in
the gap of the bimodal distribution, so the count is threshold-robust), and
the small-board share of the flips. This is the acceptance metric for the
planar-pose-ambiguity work (PoseGeometry resolve_pose_ambiguity): the 3 m blob-board
flip count / p90 is the number to move without disturbing the 1–2 m rows.

Usage:
  julia --project=bmvc2026 bmvc2026/scripts/flip_stats.jl

Reads the collected results frame (`pose_results`).
The cells are the evaluation's (configs/figures/evaluation.toml, src/runs.jl).
=#

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
using .BlobBoardsEval: pose_results, eval_sources, load_config
using DataFrames
using Statistics

const SYSTEMS = [("blob", "BlobBoards"), ("april", "AprilTag"), ("aruco", "ArUco")]

# The collected results frame (DrWatson collect_results), not a CSV root:
# `level` is a column, so the unoccluded set is a filter.
frame = pose_results(eval_sources(load_config("figures/evaluation.toml")))

for (key, label) in SYSTEMS
    df = filter(
        r -> !ismissing(r.system) && r.system == key &&
            !ismissing(r.level) && r.level == "none", frame
    )
    println(label)
    for spot in (1000, 2000, 3000)
        d = df[df.rig_distances_bin .== spot, :]
        a = abs.(collect(skipmissing(d.abs_error_angles_deg)))
        isempty(a) && continue
        flips = d[
            .!ismissing.(d.abs_error_angles_deg) .&
                (abs.(coalesce.(d.abs_error_angles_deg, 0.0)) .> 10), :,
        ]
        small = count(==("small"), flips.marker_scales)
        println(
            "  $(spot ÷ 1000) m: n=$(length(a))  rot med=$(round(median(a); digits = 2))°" *
                "  p90=$(round(quantile(a, 0.9); digits = 2))°" *
                "  flips(>10°)=$(nrow(flips)) ($(small) small)"
        )
    end
end
