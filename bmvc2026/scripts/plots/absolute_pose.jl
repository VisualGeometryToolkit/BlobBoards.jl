#=
absolute_pose.jl — figures for the motion-capture pose-error evaluation.

Renders, from the collected results frame:
  pose_errors.(png|pdf)     error vs rig angle bin / distance bin, 3 systems
  occlusion_sweep.(png|pdf) median error + detection rate vs occlusion level

Usage:
  julia --project=bmvc2026 bmvc2026/scripts/plots/absolute_pose.jl

Reads the collected results frame (`pose_results` of the evaluation's
sources). Env knob: BB_FIG (output dir, default: plotsdir("results");
the paper includes neither figure).
BB_CSV is gone — a condition is a `level` filter, not a directory.
=#

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

# Guarded: this file defines reusable plot functions and is meant to be
# include()d into a session that may already hold the module.
if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "..", "src", "BlobBoardsEval.jl"))
end
using .BlobBoardsEval: pose_results, eval_sources, load_config
using CairoMakie
using DrWatson: plotsdir
using DataFrames
using Statistics

# Fixed series order and colors (validated categorical palette; the same
# system keeps the same color in every figure).
const SYSTEMS = [
    (key = "blob", label = "BlobBoards", color = "#0072B2"),
    (key = "april", label = "AprilTag", color = "#E69F00"),
    (key = "aruco", label = "ArUco", color = "#009E73"),
]

# One cell of the collected frame: the observations for one marker system at
# one occlusion level. Replaces `CSV.read(joinpath(csv_root, key * ".csv"))` —
# `level` is now a COLUMN, so selecting a condition is a filter rather than a
# directory path. That removes the whole class of layout bug that once
# silently skipped every occlusion table.
_cell(df, key, level = "none") =
    filter(
    r -> !ismissing(r.system) && r.system == key &&
        !ismissing(r.level) && r.level == level, df
)

# Median + IQR of `col` grouped by `by`, detections only. Bins with fewer
# than `min_n` samples are dropped — a lone spurious pose at an extreme angle
# otherwise dominates its bin's "median".
#
# A system with NO detections at all is a real case, not a defect: at >=20%
# occlusion AprilTag and ArUco both fall to a 0.00 detection rate, so
# `dropmissing` leaves zero rows and the summary statistics have nothing to
# summarize. `median` of an empty group THROWS (Vector{Union{}}), and the
# `min_n` filter ran too late to prevent it — it dropped thin bins only after
# the medians had already been computed. This never surfaced while
# `plot_pose_errors` was called on the parent set alone, where all three
# systems detect; it fires immediately once the same figure is drawn per
# occlusion level. Return the empty frame and let the caller draw nothing.
function _binned(df, by, col; min_n = 5)
    d = dropmissing(df, col)
    isempty(d) && return DataFrame(
        by => eltype(df[!, by])[], :n => Int[],
        :med => Float64[], :lo => Float64[], :hi => Float64[]
    )
    gd = combine(
        groupby(d, by),
        nrow => :n,
        col => median => :med,
        col => (x -> quantile(x, 0.25)) => :lo,
        col => (x -> quantile(x, 0.75)) => :hi
    )
    return sort!(filter(:n => >=(min_n), gd), by)
end

function _series!(ax, df, by, col, sys)
    b = _binned(df, by, col)
    # Nothing survived the detection/min_n filters — draw no series. The system
    # still gets its legend entry from the axes that do have data.
    isempty(b) && return nothing
    band!(ax, b[!, by], b.lo, b.hi; color = (sys.color, 0.15))
    scatterlines!(
        ax, b[!, by], b.med; color = sys.color, label = sys.label,
        linewidth = 2, markersize = 8
    )
    return nothing
end

function plot_pose_errors(df, level = "none"; out_dir = plotsdir())
    dfs = Dict(s.key => _cell(df, s.key, level) for s in SYSTEMS)
    fig = Figure(size = (1200, 700))
    # 2x2, in the paper's terms: rotation and translation error against
    # obliquity, translation against camera distance, and probability of
    # detection. Only absolute error is reported anywhere in the paper, so the
    # signed (relative) rotation panel is gone.
    specs = [
        (1, 1, :rig_angles_bin, :abs_error_angles_deg, "obliquity (°)", "rotation error (°)"),
        (1, 2, :rig_angles_bin, :error_distances, "obliquity (°)", "translation error (mm)"),
        (2, 1, :rig_distances_bin, :error_distances, "camera distance", "translation error (mm)"),
    ]
    local ax
    for (r, c, by, col, xl, yl) in specs
        ax = Axis(fig[r, c]; xlabel = xl, ylabel = yl)
        # The stations are at whole metres (`rig_distances_bin` rounds to
        # 1000 mm), so tick only there rather than letting Makie interpolate.
        by === :rig_distances_bin &&
            (ax.xticks = ([1000, 2000, 3000], ["1\u2009m", "2\u2009m", "3\u2009m"]))
        for sys in SYSTEMS
            _series!(ax, dfs[sys.key], by, col, sys)
        end
    end
    ax = Axis(
        fig[2, 2]; xlabel = "obliquity (°)", ylabel = "probability of detection",
        limits = (nothing, (0, 1.05))
    )
    for sys in SYSTEMS
        d = dfs[sys.key]
        gd = combine(
            groupby(d, :rig_angles_bin),
            :dot_product => (x -> count(!ismissing, x) / length(x)) => :rate
        )
        sort!(gd, :rig_angles_bin)
        scatterlines!(
            ax, gd.rig_angles_bin, gd.rate; color = sys.color,
            label = sys.label, linewidth = 2, markersize = 8
        )
    end
    Legend(fig[0, 1:2], ax; orientation = :horizontal, framevisible = false)
    mkpath(out_dir)
    for ext in ("png", "pdf")
        save(joinpath(out_dir, "pose_errors.$ext"), fig)
    end
    println("pose_errors -> $out_dir")
    return fig
end

function plot_occlusion_sweep(df; out_dir = plotsdir())
    # Levels come from the frame, not from a directory listing. "none" is the
    # unoccluded cell and is excluded — level 00 is its occlusion-experiment
    # counterpart and carries the same numbers.
    levels = sort([l for l in unique(skipmissing(df.level)) if l != "none"])
    isempty(levels) && (println("no occlusion levels in the frame — skipped"); return nothing)
    x = parse.(Int, levels)

    fig = Figure(size = (1200, 400))
    ax1 = Axis(fig[1, 1]; xlabel = "occlusion (%)", ylabel = "relative rotation error (°)")
    ax2 = Axis(fig[1, 2]; xlabel = "occlusion (%)", ylabel = "absolute translation error (mm)")
    ax3 = Axis(
        fig[1, 3]; xlabel = "occlusion (%)", ylabel = "detection rate",
        limits = (nothing, (0, 1.05))
    )
    for sys in SYSTEMS
        med_a = Float64[]; med_d = Float64[]; rate = Float64[]
        for l in levels
            d = _cell(df, sys.key, l)
            ea = collect(skipmissing(d.error_angles_deg))
            ed = collect(skipmissing(d.error_distances))
            # same rationale as _binned's min_n: a median over a few straggler
            # detections after the system has effectively died is misleading
            push!(med_a, length(ea) < 20 ? NaN : median(ea))
            push!(med_d, length(ed) < 20 ? NaN : median(ed))
            push!(rate, count(!ismissing, d.dot_product) / nrow(d))
        end
        scatterlines!(ax1, x, med_a; color = sys.color, label = sys.label, linewidth = 2, markersize = 8)
        scatterlines!(ax2, x, med_d; color = sys.color, linewidth = 2, markersize = 8)
        scatterlines!(ax3, x, rate; color = sys.color, linewidth = 2, markersize = 8)
    end
    Legend(fig[0, 1:3], ax1; orientation = :horizontal, framevisible = false)
    mkpath(out_dir)
    for ext in ("png", "pdf")
        save(joinpath(out_dir, "occlusion_sweep.$ext"), fig)
    end
    println("occlusion_sweep -> $out_dir")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    # ONE collected frame for every condition. `pose_results` is DrWatson's
    # `collect_results` over the per-cell artifacts written by
    # `make_pose_results.jl`, so `system` and `level` are columns and no plot
    # function takes a directory path any more.
    df = pose_results(eval_sources(load_config("figures/evaluation.toml")))
    # The paper does not include these figures (its occlusion and error
    # figures are the pgfplots twins error_distributions.jl writes), so they
    # are working figures and land in plotsdir("results"), never in papers/ —
    # an unincluded file there only reads as a paper asset nobody checks.
    # Never inside the dataset either: data/exp_raw is the immutable input
    # tree. BB_FIG still overrides.
    fig_dir = get(ENV, "BB_FIG", plotsdir("results"))
    mkpath(fig_dir)
    # Unoccluded reference figure.
    plot_pose_errors(df, "none"; out_dir = fig_dir)
    # The occlusion SUMMARY: every level, all three systems, one figure. This
    # is the occlusion result of record — the per-level `pose_errors` panels
    # that briefly lived here were dropped because past 20% occlusion both tag
    # baselines detect NOTHING, so two of three series simply are not drawn and
    # the reader has to be told what the absence means. The sweep shows the
    # collapse itself (AprilTag 0.73 -> 0.11 -> 0.00), and the floor-plan
    # figures render absence as gray wedges instead of missing lines.
    plot_occlusion_sweep(df; out_dir = fig_dir)
end
