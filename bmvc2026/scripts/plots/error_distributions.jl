#=
error_distributions.jl — distribution + floor-plan figures for the
motion-capture pose-error evaluation (the 2026-08 analysis set).

Reads the collected results frame (`pose_results`, i.e. DrWatson's
`collect_results` over the per-cell artifacts written by
make_pose_results.jl) and renders:
  trans_distributions.(png|pdf)  pooled translation-error histograms (log x),
                                 one panel per system, median lines
  rot_distributions.(png|pdf)    rotation error, hand-eye ABSOLUTE vs
                                 calibration-free RELATIVE (3×2, log x)
  trans_by_distance.(png|pdf)    translation error per tripod spot (3 panels,
                                 systems overlaid) — separates the Z² depth
                                 mixture the pooled histogram conflates
  error_floorplan.(png|pdf)      polar sector heatmap over the capture floor
                                 plan: annular ring per tripod spot × 10°
                                 station wedges; top = median translation
                                 error (log color), bottom = detections/cell;
                                 × = station visited but ZERO detections,
                                 dot = fewer than 5

Usage:
  julia --project=bmvc2026 bmvc2026/scripts/plots/error_distributions.jl

Env knobs: BB_FIG (output dir, default: DrWatson's
plotsdir() = bmvc2026/plots — never inside the dataset, which is the immutable
input tree; see the comment at `fig_dir`), BB_OCC_LEVELS (occlusion levels to
render floor plans for, default "10,50,70"), BB_TEX_PREFIX (path prefix baked into
the emitted TikZ).

BB_CSV is gone: there is no CSV root any more. A condition is selected with
`with_level`, because `level` is a column of the frame rather than a directory.
=#
# reproduce: reads configs/figures/evaluation.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

include(joinpath(@__DIR__, "..", "..", "src", "BlobBoardsEval.jl"))
using .BlobBoardsEval: pose_results, figure_config, write_figure_records, eval_sources, poses_dir
using VisualGeometryToolkitEval.Experiments: Stage, run_stages
using DrWatson: projectdir
using CairoMakie
using DrWatson: plotsdir
using DataFrames
using Statistics

# Same series order/colors as absolute_pose.jl — one system, one color.
const SYSTEMS = [
    (key = "blob", label = "BlobBoards", color = "#0072B2"),
    (key = "april", label = "AprilTag", color = "#E69F00"),
    (key = "aruco", label = "ArUco", color = "#009E73"),
]

# One cell of the collected frame: the observations for one marker system at
# one occlusion level ("none" = the unoccluded set). Was
# `CSV.read(joinpath(csv_root, key * ".csv"))`; `level` is a COLUMN now, so a
# condition is a filter rather than a directory path.
#
# `LEVEL` is a module-level knob rather than an argument because every function
# in this file reaches its data through `_load(src, key)` and only the
# floor-plan is generated per level — threading a parameter through all nine
# would be churn for one caller. Set it via `with_level`.
const LEVEL = Ref("none")

_load(src::AbstractDataFrame, key) =
    filter(
    r -> !ismissing(r.system) && r.system == key &&
        !ismissing(r.level) && r.level == LEVEL[], src
)

"""
    with_level(f, level)

Run `f()` with `_load` scoped to one occlusion level. Restores the previous
level even if `f` throws, so a failed level cannot silently poison the next.
"""
function with_level(f, level::AbstractString)
    old = LEVEL[]
    LEVEL[] = level
    try
        return f()
    finally
        LEVEL[] = old
    end
end
_vals(df, col) = collect(skipmissing(df[!, col]))

_logbins(lo, hi, n) = 10 .^ range(log10(lo), log10(hi), n)

# --- pooled translation histograms ---------------------------------------------

function plot_trans_distributions(src; out_dir)
    fig = Figure(size = (900, 700))
    Label(
        fig[0, 1], "Hand-eye absolute translation error — evaluation sweep";
        fontsize = 18, font = :bold, tellwidth = false
    )
    axs = Axis[]
    for (i, s) in enumerate(SYSTEMS)
        e = clamp.(_vals(_load(src, s.key), :error_distances), 0.31, 1.2e4)
        ax = Axis(
            fig[i, 1]; xscale = log10, ylabel = "count",
            title = "$(s.label)  (n=$(length(e)), median $(round(median(e); digits = 1)) mm, " *
                "p90 $(round(quantile(e, 0.9); digits = 1)) mm)",
            xlabel = i == length(SYSTEMS) ? "translation error (mm, log scale)" : ""
        )
        hist!(ax, e; bins = _logbins(0.3, 1.2e4, 60), color = (s.color, 0.75))
        vlines!(ax, [median(e)]; color = s.color, linestyle = :dash, linewidth = 2)
        push!(axs, ax)
    end
    linkxaxes!(axs...)
    for ext in ("png", "pdf")
        save(joinpath(out_dir, "trans_distributions.$ext"), fig)
    end
    return println("trans_distributions -> $out_dir")
end

# --- rotation histograms: absolute vs relative ----------------------------------

function plot_rot_distributions(src; out_dir)
    fig = Figure(size = (1200, 700))
    Label(
        fig[0, 1:2], "Rotation error — evaluation sweep";
        fontsize = 18, font = :bold, tellwidth = false
    )
    Label(fig[1, 1], "hand-eye ABSOLUTE"; font = :bold, tellwidth = false)
    Label(fig[1, 2], "calibration-free RELATIVE"; font = :bold, tellwidth = false)
    axs = Axis[]
    for (i, s) in enumerate(SYSTEMS)
        df = _load(src, s.key)
        for (j, col) in enumerate((:abs_error_angles_deg, :error_angles_deg))
            e = max.(abs.(_vals(df, col)), 1.0e-3)
            ax = Axis(
                fig[i + 1, j]; xscale = log10, ylabel = j == 1 ? "count" : "",
                title = "$(s.label)  (median $(round(median(e); digits = 2))°, " *
                    "p90 $(round(quantile(e, 0.9); digits = 2))°)",
                xlabel = i == length(SYSTEMS) ? "rotation error (deg, log scale)" : ""
            )
            hist!(ax, e; bins = _logbins(1.0e-3, 200, 60), color = (s.color, 0.75))
            vlines!(ax, [median(e)]; color = s.color, linestyle = :dash, linewidth = 2)
            push!(axs, ax)
        end
    end
    linkxaxes!(axs...)
    for ext in ("png", "pdf")
        save(joinpath(out_dir, "rot_distributions.$ext"), fig)
    end
    return println("rot_distributions -> $out_dir")
end

# --- per-tripod-spot translation overlays ---------------------------------------

function plot_trans_by_distance(src; out_dir)
    fig = Figure(size = (900, 750))
    Label(
        fig[0, 1], "Translation error by tripod distance";
        fontsize = 18, font = :bold, tellwidth = false
    )
    bins = _logbins(0.3, 400, 55)
    axs = Axis[]
    for (i, spot) in enumerate((1000, 2000, 3000))
        ax = Axis(
            fig[i, 1]; xscale = log10, ylabel = "density",
            title = "$(spot ÷ 1000) m spot",
            xlabel = i == 3 ? "translation error (mm, log scale)" : ""
        )
        for s in SYSTEMS
            df = _load(src, s.key)
            e = clamp.(_vals(df[df.rig_distances_bin .== spot, :], :error_distances), 0.31, 390)
            isempty(e) && continue
            hist!(
                ax, e; bins, normalization = :pdf, color = (s.color, 0.45),
                strokewidth = 1, strokecolor = s.color,
                label = "$(s.label) (med $(round(median(e); digits = 1)))"
            )
            vlines!(ax, [median(e)]; color = s.color, linestyle = :dash, linewidth = 2)
        end
        axislegend(ax; position = :rt, framevisible = false)
        push!(axs, ax)
    end
    linkxaxes!(axs...)
    for ext in ("png", "pdf")
        save(joinpath(out_dir, "trans_by_distance.$ext"), fig)
    end
    return println("trans_by_distance -> $out_dir")
end

# --- polar floor-plan heatmap ----------------------------------------------------

# annular sector: ring at distance d (±300 mm), wedge at station angle a (±5°),
# 0° pointing up — a top-down floor plan of the capture geometry
# In this polar chart only the RADIUS is physical (camera-to-rig distance,
# meters); the Cartesian x/y are just the embedding, so axes are hidden and
# rings are labeled by distance instead.
function _sector(a, d)                       # d in mm; drawn in METERS
    θs = range(deg2rad(a - 5), deg2rad(a + 5); length = 12)
    r1, r2 = (d - 300.0) / 1000, (d + 300.0) / 1000
    return Point2f[
        [Point2f(r1 * sin(θ), r1 * cos(θ)) for θ in θs];
        [Point2f(r2 * sin(θ), r2 * cos(θ)) for θ in reverse(θs)]
    ]
end

# Station assignment: RANK-SLOT protocol. The capture protocol is 19
# nominal stations (−90°:10°:+90°) per ring, rotated by hand; the
# measured angles are treated as noisy labels of the operator's INTENT.
# Within each ring, sort the frames by measured angle and deal them into
# consecutive 10° slots — no gaps, all captured data assumed to fill the
# sweep in order. The first sorted frame anchors at its nearest nominal
# station, so a ring one frame short (bb's 3 m sweep) leaves only the
# final slot empty. NB this deliberately overrides the measured angle
# where the operator mis-rotated (the 2 m ring's near-duplicate pairs
# land in adjacent slots rather than merging), trading geometric
# fidelity for protocol fidelity. Frames are identified by their exact
# rig_angles value (constant across a frame's marker rows, distinct
# between frames).
function _cells(df)
    station = Dict{Tuple{Float64, Int}, Int}()
    for spot in unique(df.rig_distances_bin)
        angs = sort(unique(df.rig_angles[df.rig_distances_bin .== spot]))
        anchor = round(Int, rad2deg(first(angs)) / 10) * 10
        for (k, a) in enumerate(angs)
            station[(a, spot)] = anchor + 10 * (k - 1)
        end
    end
    d = transform(
        df,
        [:rig_angles, :rig_distances_bin] => ByRow((a, s) -> station[(a, s)]) => :station
    )
    _med(e) = (v = collect(skipmissing(e)); isempty(v) ? NaN : median(v))
    g = combine(
        groupby(d, [:station, :rig_distances_bin]),
        :error_distances => (e -> count(!ismissing, e)) => :n,
        :error_distances => length => :ntot,
        :error_distances => _med => :med,
        :abs_error_angles_deg => (e -> _med(abs.(e))) => :medr
    )
    cells = [
        (
                angle = Float64(r.station), dist = Float64(r.rig_distances_bin),
                n = r.n, rate = r.n / r.ntot, med = r.med, medr = r.medr,
            )
            for r in eachrow(g)
    ]
    # The ±90° stations are edge-on — a plane presenting zero area, where no
    # system can detect anything — so they carry no information and are
    # DROPPED from the chart (this also absorbs bb's one uncaptured +90°@3 m
    # frame). Any remaining interior slot without a frame is filled as n = 0
    # so a hole in the fan always means zero detections, never missing data.
    filter!(c -> abs(c.angle) < 90, cells)
    have = Set((c.angle, c.dist) for c in cells)
    sa = ED["station_angles"]
    for spot in unique(df.rig_distances_bin), a in Float64(sa["start"]):Float64(sa["spacing"]):Float64(sa["stop"])
        (a, Float64(spot)) in have || push!(
            cells,
            (angle = a, dist = Float64(spot), n = 0, rate = 0.0, med = NaN, medr = NaN)
        )
    end
    return cells
end

function plot_error_floorplan(src; out_dir)
    crange = (log10(2), log10(120))          # translation, mm
    rrange = (log10(0.2), log10(50))         # absolute rotation, deg
    fig = Figure(size = (1500, 1550))
    Label(
        fig[0, 1:3],
        "Capture floor plan — median translation error / median absolute rotation error / detection rate " *
            "(gray = zero detections)";
        fontsize = 16, font = :bold, tellwidth = false
    )
    for (j, s) in enumerate(SYSTEMS)
        cells = _cells(_load(src, s.key))
        det = [c for c in cells if c.n > 0]
        zed = [c for c in cells if c.n == 0]
        marks = (ax) -> begin
            isempty(zed) || poly!(
                ax, [_sector(c.angle, c.dist) for c in zed];
                color = :gray82, strokewidth = 0.8, strokecolor = (:black, 0.5)
            )
            # no low-n dot: per-cell support is the detection-rate row's job
            scatter!(ax, [Point2f(0, 0)]; marker = :rect, markersize = 14, color = :black)
            for dd in (1000.0, 2000.0, 3000.0)
                r = dd / 1000
                text!(
                    ax, Point2f(r, -0.22);
                    text = "$(Int(dd ÷ 1000)) m", align = (:center, :center),
                    fontsize = 20, color = :gray30, font = :bold
                )
            end
            hidedecorations!(ax); hidespines!(ax)
        end
        ax = Axis(fig[1, j]; title = s.label, titlesize = 24, titlefont = :bold, aspect = DataAspect())
        poly!(
            ax, [_sector(c.angle, c.dist) for c in det];
            color = [clamp(log10(c.med), crange...) for c in det],
            colormap = Reverse(:RdYlGn), colorrange = crange,
            strokewidth = 0.5, strokecolor = (:black, 0.3)
        )
        marks(ax)
        axr = Axis(fig[2, j]; aspect = DataAspect())
        poly!(
            axr, [_sector(c.angle, c.dist) for c in det];
            color = [clamp(log10(c.medr), rrange...) for c in det],
            colormap = Reverse(:RdYlGn), colorrange = rrange,
            strokewidth = 0.5, strokecolor = (:black, 0.3)
        )
        marks(axr)
        axn = Axis(fig[3, j]; aspect = DataAspect())
        poly!(
            axn, [_sector(c.angle, c.dist) for c in cells];
            color = [c.rate for c in cells], colormap = :RdYlGn,
            colorrange = (0, 1), strokewidth = 0.5, strokecolor = (:black, 0.3)
        )
        scatter!(axn, [Point2f(0, 0)]; marker = :rect, markersize = 14, color = :black)
        for dd in (1000.0, 2000.0, 3000.0)
            r = dd / 1000
            text!(
                axn, Point2f(r, -0.22);
                text = "$(Int(dd ÷ 1000)) m", align = (:center, :center),
                fontsize = 20, color = :gray30, font = :bold
            )
        end
        hidedecorations!(axn); hidespines!(axn)
    end
    Colorbar(
        fig[1, 4]; colorrange = crange, colormap = Reverse(:RdYlGn),
        ticks = (log10.(ED["trans_ticks"]), String.(ED["trans_tick_labels"])),
        label = "median translation error (mm, log scale)", height = Relative(0.8)
    )
    Colorbar(
        fig[2, 4]; colorrange = rrange, colormap = Reverse(:RdYlGn),
        ticks = (log10.(ED["rot_ticks"]), String.(ED["rot_tick_labels"])),
        label = "median abs rotation error (deg, log scale)", height = Relative(0.8)
    )
    Colorbar(
        fig[3, 4]; colorrange = (0, 1), colormap = :RdYlGn,
        label = "P(detection)", height = Relative(0.8)
    )
    for ext in ("png", "pdf")
        save(joinpath(out_dir, "error_floorplan.$ext"), fig)
    end
    return println("error_floorplan -> $out_dir")
end

# --- the config of record ------------------------------------------------------------

const EVAL, EVAL_PATH = figure_config("evaluation"; default = projectdir("configs", "figures", "evaluation.toml"))
const DRV, ED = EVAL["drivers"], EVAL["error_distributions"]

# --- TikZ export -------------------------------------------------------------------
#
# Per-panel TikZ files + a LaTeX assembly table, so the paper controls
# layout and typography:
#   floorplan_<sys>_<metric>.tex   bare fan (no title), geometry scaled by
#                                  the \fpscale macro (fonts unaffected)
#   floorplan_cbar_<metric>.tex    pgfplots colorbar-only axis for the row
#   error_floorplan_table.tex      tabular: system column heads, rotated row
#                                  labels (Translation/Rotation/Detection),
#                                  colorbars in the last column
#   error_floorplan_standalone.tex compilable preview of the table
# Requires \usepackage{graphicx,pgfplots} + \pgfplotsset{compat=1.18}.

function write_floorplan_tikz(src; out_dir, labels = true)
    cg = cgrad(:RdYlGn)
    _rgb(x) = let c = get(cg, x)
        "{rgb,1:red,$(round(c.r; digits = 4));green,$(round(c.g; digits = 4));blue,$(round(c.b; digits = 4))}"
    end
    crange = (log10(2), log10(120))
    rrange = (log10(0.2), log10(50))
    _norm(v, (lo, hi)) = clamp((clamp(log10(v), lo, hi) - lo) / (hi - lo), 0, 1)
    cmap(rev) = join(
        [
            "rgb=($(round(get(cg, rev ? 1 - x : x).r; digits = 4)),$(round(get(cg, rev ? 1 - x : x).g; digits = 4)),$(round(get(cg, rev ? 1 - x : x).b; digits = 4)))"
                for x in range(0, 1, 15)
        ], " "
    )
    # Detection first: whether a board was found at all precedes how accurately
    # it was located. Under occlusion the detection panel is the one that
    # carries the result -- the error panels are conditioned on detection, so a
    # system that finds nothing has no error to plot.
    metrics = [
        ("det", :rate, (0.0, 1.0), false),
        ("trans", :med, crange, true), ("rot", :medr, rrange, true),
    ]
    for s in SYSTEMS
        cells = _cells(_load(src, s.key))
        for (mkey, field, rng, islog) in metrics
            io = IOBuffer()
            println(io, "% floorplan fan: $(s.label) / $(mkey) — generated by error_distributions.jl; do not edit by hand")
            println(io, "\\providecommand{\\fpscale}{1}")
            # TOP-anchored, matching the colorbar beside it. Both were centred,
            # so the 1.6cm colorbar floated in the middle of a much taller fan
            # row instead of starting level with it.
            println(io, "\\begin{tikzpicture}[x=\\fpscale cm, y=\\fpscale cm, baseline=(current bounding box.north)]")
            for c in cells
                r1, r2 = (c.dist - 300) / 1000, (c.dist + 300) / 1000
                φ1, φ2 = 95 - c.angle, 85 - c.angle      # 0°=up, cw → TikZ math angles
                fill = c.n == 0 ? "gray!30" :
                    _rgb(field === :rate ? c.rate : (islog ? 1 - _norm(getfield(c, field), rng) : getfield(c, field)))
                println(
                    io, "  \\fill[fill=$(fill), draw=black!30, line width=0.2pt] " *
                        "($(round(φ1; digits = 2)):$(round(r1; digits = 3))) arc[start angle=$(round(φ1; digits = 2)), end angle=$(round(φ2; digits = 2)), radius=$(round(r1; digits = 3))] -- " *
                        "($(round(φ2; digits = 2)):$(round(r2; digits = 3))) arc[start angle=$(round(φ2; digits = 2)), end angle=$(round(φ1; digits = 2)), radius=$(round(r2; digits = 3))] -- cycle;"
                )
            end
            println(io, "  \\fill[black] (-0.09, -0.09) rectangle (0.09, 0.09);")
            # `labels = false` exists for any assembly that renders these fans
            # below roughly \\fpscale 0.3: the ring labels are \\scriptsize and NOT
            # affected by \\fpscale, so they are what collides first. Every
            # current figure renders at Fig. 5's scale and keeps them.
            if labels
                for dd in (1, 2, 3)
                    println(io, "  \\node[font=\\scriptsize\\bfseries, text=black!60] at ($(dd), -0.22) {$(dd)\\,m};")
                end
            end
            println(io, "\\end{tikzpicture}")
            open(joinpath(out_dir, "floorplan_$(s.key)_$(mkey).tex"), "w") do f
                write(f, take!(io))
            end
        end
    end
    bars = [
        (
            "trans", "rdylgnrev", crange, raw"trans.~error [mm]",
            log10.(ED["trans_ticks"]), String.(ED["trans_tick_labels"]),
        ),
        (
            "rot", "rdylgnrev", rrange, raw"rot.~error [$^\circ$]",
            log10.(ED["rot_ticks"]), String.(ED["rot_tick_labels"]),
        ),
        (
            "det", "rdylgn", (0.0, 1.0), "P(detection)", Float64.(ED["det_ticks"]),
            String.(ED["det_tick_labels"]),
        ),
    ]
    for (mkey, cm, rng, lab, ticks, tlabs) in bars
        io = IOBuffer()
        println(io, "% floorplan colorbar: $(mkey) — generated by error_distributions.jl; do not edit by hand")
        println(io, "\\begin{tikzpicture}[baseline=(current bounding box.north)]")
        println(io, "\\pgfplotsset{colormap={rdylgnrev}{$(cmap(true))}, colormap={rdylgn}{$(cmap(false))}}")
        println(io, "\\begin{axis}[hide axis, scale only axis, width=0pt, height=1.6cm,")
        println(io, "  colormap name=$(cm), colorbar, point meta min=$(round(rng[1]; digits = 4)), point meta max=$(round(rng[2]; digits = 4)),")
        println(io, "  colorbar style={height=1.6cm, width=0.25cm, ylabel={$(lab)}, ylabel style={font=\\scriptsize},")
        println(io, "    ytick={$(join(round.(ticks; digits = 4), ","))}, yticklabels={$(join(tlabs, ","))}, tick label style={font=\\scriptsize},\n    yticklabel style={text width=1.7em, align=right}}]")
        println(io, "  \\addplot[draw=none] coordinates {(0,0)};")
        println(io, "\\end{axis}")
        println(io, "\\end{tikzpicture}")
        open(joinpath(out_dir, "floorplan_cbar_$(mkey).tex"), "w") do f
            write(f, take!(io))
        end
    end
    rowlab = Dict(
        "trans" => raw"Median Translation Error [mm]",
        "rot" => raw"Median Rotation Error [$^\circ$]",
        "det" => "Probability of Detection"
    )
    prefix = get(ENV, "BB_TEX_PREFIX", DRV["tex_prefix"])
    # Two assemblies of the same panels: the full three-row table (the
    # supplement's per-level floor plans) and rows (a)+(b) only — the main
    # paper's Fig. floorplan, which omits the per-station median rotation
    # because the BlobBoard median rotation error sits within a factor of two
    # of the hand-eye calibration residual, so that row is not resolvable by
    # the ground truth. Both generated, so neither can drift from the panels.
    for (name, keys, why) in (
            ("error_floorplan_table.tex", ED["full_table_rows"], "all rows"),
            ("error_floorplan_ab.tex", ED["ab_table_rows"], "rows (a) and (b): no per-station rotation row (not resolvable by the hand-eye ground truth)"),
        )
        rows = [m for m in metrics if m[1] in keys]
        io = IOBuffer()
        println(io, "% $(name) — generated by error_distributions.jl ($(why)); do not edit by hand")
        println(io, "% class-agnostic: manual (a)/(b)/(c) caption lines, NO subcaption")
        println(io, "% (bmvc2k's caption machinery mis-numbers subcaption subfigures)")
        println(io, "% requires: \\usepackage{graphicx,pgfplots} \\pgfplotsset{compat=1.18}")
        println(io, "\\providecommand{\\fpscale}{1}\\renewcommand{\\fpscale}{0.42}   % fan geometry scale (fonts unaffected)")
        println(io, "\\setlength{\\tabcolsep}{3pt}")
        println(io, "\\centering")
        println(io, "\\begin{tabular}{@{}ccc l@{}}")
        println(io, join(["\\textbf{$(s.label)}" for s in SYSTEMS], " & ") * " & \\\\[2pt]")
        for (k, (mkey, _, _, _)) in enumerate(rows)
            println(
                io, join(["\\input{$(prefix)floorplan_$(s.key)_$(mkey)}" for s in SYSTEMS], " & ") *
                    " & \\input{$(prefix)floorplan_cbar_$(mkey)} \\\\[2pt]"
            )
            println(
                io, "\\multicolumn{3}{c}{($('a' + k - 1)) $(rowlab[mkey])} & \\\\" *
                    (k < length(rows) ? "[8pt]" : "")
            )
        end
        println(io, "\\end{tabular}")
        open(joinpath(out_dir, name), "w") do f
            write(f, take!(io))
        end
    end
    open(joinpath(out_dir, "error_floorplan_standalone.tex"), "w") do f
        write(
            f, "\\documentclass[a4paper]{article}\n\\usepackage[margin=1cm]{geometry}\n" *
                "\\usepackage{graphicx,subcaption}\n\\usepackage{pgfplots}\n" *
                "\\pgfplotsset{compat=1.18}\n\\newcommand{\\fpscale}{1}\n" *
                "\\pagestyle{empty}\n\\begin{document}\n\\begin{figure}\n" *
                "\\input{error_floorplan_table.tex}\n" *
                "\\caption{Pose accuracy over the capture floor plan.}\n" *
                "\\end{figure}\n\\end{document}\n"
        )
    end
    return println("floorplan_*.tex panels + cbars + error_floorplan_{table,ab}.tex -> $out_dir")
end


# --- TikZ histograms ---------------------------------------------------------------
#
# hist_trans.tex / hist_rot.tex: pooled error histograms (log x, three
# systems overlaid, dashed medians) as editable pgfplots panels, plus the
# error_histograms_table.tex assembly (two panels + (a)/(b) captions).

function write_histograms_tikz(src; out_dir)
    prefix = get(ENV, "BB_TEX_PREFIX", DRV["tex_prefix"])
    colors = Dict(
        "blob" => ("bbBlue", "0072B2"), "april" => ("bbOrange", "E69F00"),
        "aruco" => ("bbGreen", "009E73")
    )
    # Empirical CDFs (log x): no bandwidth choice, and the 0.5 / 0.9
    # crossings read off the median and 90th percentile directly. File
    # names keep the hist_ prefix for Makefile/assembly stability.
    panels = [
        (
            "trans", :error_distances, (0.5, 400.0), identity,
            "xtick={1,10,100}, xticklabels={{1 mm},{10 mm},{100 mm}}",
        ),
        (
            "rot", :abs_error_angles_deg, (0.02, 200.0), abs,
            raw"xtick={0.1,1,10,100}, xticklabels={{0.1$^\circ$},{1$^\circ$},{10$^\circ$},{100$^\circ$}}",
        ),
    ]
    for (mkey, col, (lo, hi), f, xticks) in panels
        io = IOBuffer()
        println(io, "% hist_$(mkey).tex — generated (empirical CDFs) by error_distributions.jl; do not edit by hand")
        for (_, (cname, hex)) in colors
            println(io, "\\providecolor{$(cname)}{HTML}{$(hex)}")
        end
        println(io, "\\begin{tikzpicture}[baseline=(current bounding box.center)]")
        println(io, "\\begin{axis}[width=6.1cm, height=3.9cm, xmode=log, ymin=0, ymax=1,")
        println(io, "  xmin=$(lo), xmax=$(hi), ylabel={\$P(\\mathrm{error} < x)\$},")
        println(io, "  $(xticks),")
        println(io, "  ytick={0,0.25,0.5,0.75,1}, yticklabels={0,,0.5,,1},")
        println(io, "  label style={font=\\scriptsize}, tick label style={font=\\scriptsize},")
        println(io, "  legend style={font=\\scriptsize, draw=none, fill=none, at={(0.97,0.06)}, anchor=south east},")
        println(io, "  legend cell align=left]")
        # quantile guides: crossings give the median and p90
        println(io, "\\addplot[forget plot, black!35, dashed, line width=0.5pt] coordinates {($(lo),0.5) ($(hi),0.5)};")
        for s in SYSTEMS
            sel = _load(src, s.key)[!, col]
            ntot = length(sel)                       # ALL attempts, incl. misses
            e = sort(clamp.(f.(collect(skipmissing(sel))), lo * 1.0001, hi * 0.9999))
            n = length(e)
            idx = unique(vcat(1, collect(1:max(1, n ÷ 250):n), n))
            coords = join(
                [
                    "($(round(e[i]; sigdigits = 5)),$(round(i / ntot; sigdigits = 4)))"
                        for i in idx
                ], " "
            )
            cname = colors[s.key][1]
            println(io, "\\addplot[const plot, draw=$(cname), line width=1.0pt] coordinates {($(lo),0) $(coords) ($(hi),$(round(n / ntot; sigdigits = 4)))};")
            println(io, "\\addlegendentry{$(s.label)}")
        end
        println(io, "\\end{axis}")
        println(io, "\\end{tikzpicture}")
        open(joinpath(out_dir, "hist_$(mkey).tex"), "w") do fio
            write(fio, take!(io))
        end
    end
    io = IOBuffer()
    println(io, "% error_histograms_table.tex — generated by error_distributions.jl; do not edit by hand")
    println(io, "\\centering")
    println(io, "\\begin{tabular}{@{}cc@{}}")
    println(io, "\\input{$(prefix)hist_trans} & \\input{$(prefix)hist_rot} \\\\")
    println(io, "(a) Translation error & (b) Rotation error \\\\")
    println(io, "\\end{tabular}")
    open(joinpath(out_dir, "error_histograms_table.tex"), "w") do fio
        write(fio, take!(io))
    end
    return println("hist_trans/rot.tex (ECDF) + error_histograms_table.tex -> $out_dir")
end


# --- per-scale ECDFs (3×2: board scale × {translation, rotation}) -------------------

# Shared legend row for the ECDF and occlusion-sweep grids. Every panel in
# both figures carries the same three systems, and an in-axis legend covers
# the curves at this panel width, so the key is hoisted out. It sits BELOW the
# panel grid and above the caption, one size up from the tick labels so it
# reads as a key rather than as axis furniture.
function _legend_tikz(io, colors)
    println(io, "\\begin{tikzpicture}[baseline]")
    x = 0.0
    for sy in SYSTEMS
        cname = colors[sy.key][1]
        println(io, "\\draw[$(cname), line width=1.2pt] ($(x)cm,0) -- ($(x + 0.6)cm,0);")
        println(io, "\\node[anchor=west, font=\\small] at ($(x + 0.7)cm,0) {$(sy.label)};")
        x += 3.0
    end
    return println(io, "\\end{tikzpicture}")
end

function write_scale_ecdfs_tikz(src; out_dir)
    prefix = get(ENV, "BB_TEX_PREFIX", DRV["tex_prefix"])
    colors = Dict(
        "blob" => ("bbBlue", "0072B2"), "april" => ("bbOrange", "E69F00"),
        "aruco" => ("bbGreen", "009E73")
    )
    panels = [
        (
            "trans", :error_distances, (0.5, 400.0), identity,
            "xtick={1,10,100}, xticklabels={{1 mm},{10 mm},{100 mm}}",
        ),
        (
            "rot", :abs_error_angles_deg, (0.02, 200.0), abs,
            raw"xtick={0.1,1,10,100}, xticklabels={{0.1$^\circ$},{1$^\circ$},{10$^\circ$},{100$^\circ$}}",
        ),
    ]
    # Third element is the short form used as a column header in the
    # transposed assembly (sizes across, error type down).
    # Column order is smallest first: the difficulty, and the gap between the
    # methods, grows to the left-to-right reader's right.
    scales = [
        ("small", "Small boards (4\\,cm)", "Small (4\\,cm)"),
        ("medium", "Medium boards (6\\,cm)", "Medium (6\\,cm)"),
        ("large", "Large boards (12\\,cm)", "Large (12\\,cm)"),
    ]
    for (mkey, col, (lo, hi), f, xticks) in panels, (scale, _, _) in scales
        io = IOBuffer()
        println(io, "% hist_$(mkey)_$(scale).tex — generated (per-scale ECDF) by error_distributions.jl; do not edit by hand")
        for (_, (cname, hex)) in colors
            println(io, "\\providecolor{$(cname)}{HTML}{$(hex)}")
        end
        println(io, "\\begin{tikzpicture}[baseline=(current bounding box.center)]")
        # 4.0cm x 3 + tabcolsep fits the 12.55cm text block; the transposed
        # 2x3 assembly is also two panel-rows tall instead of three.
        println(io, "\\begin{axis}[width=4.05cm, height=3.4cm, xmode=log, ymin=0, ymax=1,")
        println(io, "  xmin=$(lo), xmax=$(hi), ylabel={\$P(\\mathrm{error} < x)\$},")
        println(io, "  $(xticks),")
        println(io, "  ytick={0,0.25,0.5,0.75,1}, yticklabels={0,,0.5,,1},")
        println(io, "  label style={font=\\scriptsize}, tick label style={font=\\scriptsize},")
        println(io, "  ]")
        println(io, "\\addplot[forget plot, black!35, dashed, line width=0.5pt] coordinates {($(lo),0.5) ($(hi),0.5)};")
        for s in SYSTEMS
            df = _load(src, s.key)
            sel = df[df.marker_scales .== scale, col]
            ntot = length(sel)                       # ALL attempts, incl. misses
            e = sort(clamp.(f.(collect(skipmissing(sel))), lo * 1.0001, hi * 0.9999))
            n = length(e)
            n == 0 && continue
            idx = unique(vcat(1, collect(1:max(1, n ÷ 250):n), n))
            coords = join(
                [
                    "($(round(e[i]; sigdigits = 5)),$(round(i / ntot; sigdigits = 4)))"
                        for i in idx
                ], " "
            )
            cname = colors[s.key][1]
            println(io, "\\addplot[const plot, draw=$(cname), line width=1.0pt] coordinates {($(lo),0) $(coords) ($(hi),$(round(n / ntot; sigdigits = 4)))};")
        end
        println(io, "\\end{axis}")
        println(io, "\\end{tikzpicture}")
        open(joinpath(out_dir, "hist_$(mkey)_$(scale).tex"), "w") do fio
            write(fio, take!(io))
        end
    end
    io = IOBuffer()
    println(io, "% error_scale_ecdfs_table.tex — generated by error_distributions.jl; do not edit by hand")
    println(io, "\\centering")
    println(io, "\\setlength{\\tabcolsep}{2pt}")
    for (_, (cname, hex)) in colors
        println(io, "\\providecolor{$(cname)}{HTML}{$(hex)}")
    end
    println(io, "\\begin{tabular}{@{}ccc@{}}")
    println(io, join(["{\\small $(short)}" for (_, _, short) in scales], " & ") * " \\\\[1pt]")
    rowlab = ["Translation error", "Rotation error"]
    for (k, (mkey, _, _, _, _)) in enumerate(panels)
        println(
            io, join(
                ["\\input{$(prefix)hist_$(mkey)_$(scale)}" for (scale, _, _) in scales], " & "
            ) * " \\\\"
        )
        println(
            io, "\\multicolumn{3}{c}{($(('a' + k - 1))) $(rowlab[k])} \\\\" *
                (k < length(panels) ? "[6pt]" : "")
        )
    end
    println(io, "\\end{tabular}\\\\[4pt]")
    _legend_tikz(io, colors)
    open(joinpath(out_dir, "error_scale_ecdfs_table.tex"), "w") do fio
        write(fio, take!(io))
    end
    return println("hist_{trans,rot}_{scale}.tex + error_scale_ecdfs_table.tex -> $out_dir")
end


# --- paper stat macros ---------------------------------------------------------------
#
# Every evaluation number the paper CITES IN PROSE (captions, results text)
# is emitted here as a LaTeX macro; the paper does \input{bb_stats.tex} and
# uses the macros, so the text can never go stale against the figures — one
# driver, one data read, one truth. Attempt-normalized quantities use the
# same denominator as the ECDF figures (30 markers × tracked frames).
function write_paper_stats(src; out_dir, n_attempts = 1680)
    fmt(x; d = 1) = string(round(x; digits = d))
    io = IOBuffer()
    println(io, "% bb_stats.tex — generated by experiments/plots/error_distributions.jl")
    println(io, "% every number the paper cites in prose; do not edit by hand")
    per_scale_bb = NamedTuple[]
    trans_med = Dict{Tuple{String, String}, Float64}()
    for (key, mac) in (("blob", "bb"), ("april", "apr"), ("aruco", "uco"))
        d = _load(src, key)
        t = sort(collect(skipmissing(d.error_distances)))
        r = sort(abs.(collect(skipmissing(d.abs_error_angles_deg))))
        att(v) = v[ceil(Int, 0.5 * n_attempts)]   # error at attempt-fraction 0.5
        # PoD as an integer percentage (80, not 0.80): easier to read and
        # consistent with the per-scale \bbDet<Scale> percentages below.
        println(io, "\\newcommand{\\$(mac)Det}{$(round(Int, 100 * length(t) / n_attempts))}")
        println(io, "\\newcommand{\\$(mac)AttTransMed}{$(fmt(att(t)))}")
        println(io, "\\newcommand{\\$(mac)AttRotMed}{$(fmt(att(r); d = 2))}")
        # detected-only overall medians — the numbers figure captions cite
        println(io, "\\newcommand{\\$(mac)TransMed}{$(fmt(median(t)))}")
        println(io, "\\newcommand{\\$(mac)RotMed}{$(fmt(median(r); d = 2))}")
        for sc in ("large", "medium", "small")
            ds = d[d.marker_scales .== sc, :]
            td = collect(skipmissing(ds.error_distances))
            rd = abs.(collect(skipmissing(ds.abs_error_angles_deg)))
            S = uppercasefirst(sc)
            println(io, "\\newcommand{\\$(mac)Det$(S)}{$(round(Int, 100 * length(td) / nrow(ds)))}")
            println(io, "\\newcommand{\\$(mac)TransMed$(S)}{$(fmt(median(td)))}")
            println(io, "\\newcommand{\\$(mac)RotMed$(S)}{$(fmt(median(rd); d = 2))}")
            println(io, "\\newcommand{\\$(mac)RotPninety$(S)}{$(fmt(quantile(rd, 0.9); d = 2))}")
            trans_med[(key, sc)] = median(td)
            key == "blob" && push!(per_scale_bb, (rot_med = median(rd),))
        end
    end
    # ceiling of the BB per-scale rotation medians — backs the "below X°
    # at every size" claim with a computed bound (round UP to 1 decimal)
    println(io, "\\newcommand{\\bbScaleRotMedCeil}{$(fmt(ceil(maximum(s.rot_med for s in per_scale_bb) * 10) / 10))}")
    # translation-accuracy gain over AprilTag per scale — the paper's
    # "multiplicative, not marginal" claim, computed rather than asserted
    for sc in ("large", "medium", "small")
        S = uppercasefirst(sc)
        g = trans_med[("april", sc)] / trans_med[("blob", sc)]
        println(io, "\\newcommand{\\gainApr$(S)}{$(fmt(g))}")
        # same reduction expressed as a percentage: 1 - blob/april
        println(io, "\\newcommand{\\gainApr$(S)Pct}{$(round(Int, 100 * (1 - 1 / g)))}")
    end
    open(joinpath(out_dir, "bb_stats.tex"), "w") do f
        write(f, take!(io))
    end
    return println("bb_stats.tex -> $out_dir")
end



# --- occlusion stat macros ----------------------------------------------------------
#
# The occlusion counterparts of bb_stats.tex. Kept SEPARATE from
# `write_paper_stats` rather than folded into it: that function computes
# per-board-scale medians and quantiles with bare `median`/`quantile`, which
# throw the moment a system detects nothing --- which is the normal case here
# from 20% occlusion on. This one is written for absent systems.
#
# LaTeX macro names cannot contain digits, so levels are spelled: 10 -> Ten,
# 70 -> Seventy.
const _LEVEL_WORD = Dict(
    "00" => "Zero", "10" => "Ten", "20" => "Twenty",
    "30" => "Thirty", "40" => "Forty", "50" => "Fifty", "60" => "Sixty",
    "70" => "Seventy", "80" => "Eighty", "90" => "Ninety"
)

function write_occlusion_stats(src; out_dir, levels = Tuple(ED["macro_levels"]))
    fmt(x; d = 1) = isnan(x) ? "n/a" : string(round(x; digits = d))
    io = IOBuffer()
    println(io, "% bb_occ_stats.tex — generated by error_distributions.jl")
    println(io, "% occlusion numbers the paper cites in prose; do not edit by hand")
    for lvl in levels
        word = get(_LEVEL_WORD, lvl, lvl)
        for (key, mac) in (("blob", "bb"), ("april", "apr"), ("aruco", "uco"))
            d = with_level(() -> _load(src, key), lvl)
            n = nrow(d)
            det = n == 0 ? NaN : count(!ismissing, d.dot_product) / n
            t = collect(skipmissing(d.error_distances))
            r = abs.(collect(skipmissing(d.error_angles_deg)))
            println(io, "\\newcommand{\\$(mac)DetOcc$(word)}{$(fmt(det; d = 2))}")
            # integer-percent twin of the fraction above, for prose that reads
            # "56% of markers" rather than "0.56 of boards" (the abstract).
            # Same convention as write_paper_stats' \bbDet<Scale>, which the
            # paper writes as `$\bbDetSmall\%$`.
            println(io, "\\newcommand{\\$(mac)DetPctOcc$(word)}{$(isnan(det) ? "n/a" : string(round(Int, 100 * det)))}")
            println(io, "\\newcommand{\\$(mac)TransMedOcc$(word)}{$(fmt(isempty(t) ? NaN : median(t)))}")
            println(io, "\\newcommand{\\$(mac)RotMedOcc$(word)}{$(fmt(isempty(r) ? NaN : median(r); d = 2))}")
        end
    end
    # BlobBoards translation error is flat across the sweep; the paper cites the
    # range, so derive it rather than let anyone retype it.
    meds = Float64[]
    for lvl in sort(collect(skipmissing(unique(src.level))))
        lvl == "none" && continue
        d = with_level(() -> _load(src, "blob"), lvl)
        t = collect(skipmissing(d.error_distances))
        isempty(t) || push!(meds, median(t))
    end
    if !isempty(meds)
        println(io, "\\newcommand{\\bbTransMedOccMin}{$(fmt(minimum(meds)))}")
        println(io, "\\newcommand{\\bbTransMedOccMax}{$(fmt(maximum(meds)))}")
    end
    # The collapse points the prose cites: the occlusion level from which a
    # tag system detects nothing at every higher level, and the highest level
    # at which BlobBoards still detects. Levels with no rows are skipped.
    all_levels = sort([l for l in unique(skipmissing(src.level)) if l != "none"]; by = l -> parse(Int, l))
    detrate(key, lvl) = (d = with_level(() -> _load(src, key), lvl); nrow(d) == 0 ? NaN :
        count(!ismissing, d.dot_product) / nrow(d))
    for (key, mac) in (("april", "apr"), ("aruco", "uco"))
        rates = [(parse(Int, l), detrate(key, l)) for l in all_levels]
        rates = filter(r -> !isnan(r[2]), rates)
        k = findfirst(i -> all(r -> r[2] == 0, rates[i:end]), eachindex(rates))
        k === nothing || println(io, "\\newcommand{\\$(mac)ZeroFromOccPct}{$(rates[k][1])}")
    end
    # The onset of catastrophic failure: the first occlusion level whose
    # probability of detection is below BB_COLLAPSE_POD (default 1%) and stays
    # below it at every higher level — the levels the prose cites as where a
    # tag system "fails catastrophically".
    collapse_pod = something(tryparse(Float64, get(ENV, "BB_COLLAPSE_POD", "")), Float64(DRV["collapse_pod"]))
    for (key, mac) in (("april", "apr"), ("aruco", "uco"))
        rates = filter(r -> !isnan(r[2]), [(parse(Int, l), detrate(key, l)) for l in all_levels])
        k = findfirst(i -> all(r -> r[2] < collapse_pod, rates[i:end]), eachindex(rates))
        k === nothing || println(io, "\\newcommand{\\$(mac)CollapseFromOccPct}{$(rates[k][1])}")
    end
    bbdet = [parse(Int, l) for l in all_levels if (r = detrate("blob", l); !isnan(r) && r > 0)]
    isempty(bbdet) || println(io, "\\newcommand{\\bbMaxDetectedOccPct}{$(maximum(bbdet))}")
    open(joinpath(out_dir, "bb_occ_stats.tex"), "w") do f
        write(f, take!(io))
    end
    return println("bb_occ_stats.tex -> $out_dir")
end


# --- occlusion sweep, TikZ ----------------------------------------------------
#
# The pgfplots twin of plot_occlusion_sweep (absolute_pose.jl), emitted in the
# same house style as the per-scale ECDFs so the two figures read as one
# system: identical panel box, \scriptsize labels, and ONE shared legend above
# the row rather than a legend inside an axis. Same quantities and the same
# min-20-detections median rule as the Makie version, so the numbers are
# unchanged -- only the rendering moves into LaTeX.
function write_occlusion_sweep_tikz(src; out_dir, prefix = get(ENV, "BB_TEX_PREFIX", DRV["tex_prefix"]))
    # same palette as the ECDF panels (write_scale_ecdfs_tikz), which keeps it
    # local rather than global
    colors = Dict(
        "blob" => ("bbBlue", "0072B2"), "april" => ("bbOrange", "E69F00"),
        "aruco" => ("bbGreen", "009E73")
    )
    levels = sort([l for l in unique(skipmissing(src.level)) if l != "none"])
    isempty(levels) && (println("no occlusion levels in the frame — skipped"); return nothing)
    xs = parse.(Int, levels)

    # (file key, y label, extractor, min-n rule)
    panels = [
        ("occ_rot", raw"median error ($^\circ$)",
            d -> collect(skipmissing(d.error_angles_deg)), true),
        ("occ_trans", "median error (mm)",
            d -> collect(skipmissing(d.error_distances)), true),
        ("occ_pod", raw"$P(\mathrm{detection})$",
            d -> d, false),
    ]
    for (fkey, ylab, extract, use_median) in panels
        io = IOBuffer()
        println(io, "% $(fkey).tex — generated (occlusion sweep) by error_distributions.jl; do not edit by hand")
        for (_, (cname, hex)) in colors
            println(io, "\\providecolor{$(cname)}{HTML}{$(hex)}")
        end
        println(io, "\\begin{tikzpicture}[baseline=(current bounding box.center)]")
        println(io, "\\begin{axis}[width=4.05cm, height=3.4cm, xmin=-5, xmax=95,")
        println(io, "  ylabel={$(ylab)}, xlabel={occlusion level},")
        println(io, "  xtick={0,30,60,90}, xticklabels={{0\\%},{30\\%},{60\\%},{90\\%}},")
        # log y on the error panels: BlobBoards sit near 3.5 mm / 0.5 deg while
        # the tag curves reach ~90 mm / ~24 deg, so a linear axis flattens ours
        # onto the baseline. The PoD panel stays linear on [0, 1].
        # `log ticks with fixed point`: print 1/10/100 rather than 10^0/10^1/10^2.
        if use_median
            yt = fkey == "occ_rot" ? "{0.5,1,2,5,10,25}" : "{5,10,25,50,100}"
            println(io, "  ymode=log, log ticks with fixed point, ytick=$(yt),")
        end
        use_median || println(io, "  ymin=0, ymax=1.05, ytick={0,0.25,0.5,0.75,1}, yticklabels={0,,0.5,,1},")
        println(io, "  label style={font=\\scriptsize}, tick label style={font=\\scriptsize},")
        println(io, "  ]")
        for sy in SYSTEMS
            pts = String[]
            for (x, lvl) in zip(xs, levels)
                d = with_level(() -> _load(src, sy.key), lvl)
                if use_median
                    v = extract(d)
                    length(v) < 20 && continue          # same rule as the Makie figure
                    push!(pts, "($(x),$(round(median(v); sigdigits = 4)))")
                else
                    n = nrow(d)
                    n == 0 && continue
                    push!(pts, "($(x),$(round(count(!ismissing, d.dot_product) / n; sigdigits = 4)))")
                end
            end
            isempty(pts) && continue
            cname = colors[sy.key][1]
            println(io, "\\addplot[draw=$(cname), line width=1.0pt, mark=*, mark size=1.1pt, mark options={fill=$(cname), draw=$(cname)}] coordinates {$(join(pts, " "))};")
        end
        println(io, "\\end{axis}")
        println(io, "\\end{tikzpicture}")
        open(joinpath(out_dir, "$(fkey).tex"), "w") do fio
            write(fio, take!(io))
        end
    end

    io = IOBuffer()
    println(io, "% occlusion_sweep_table.tex — generated by error_distributions.jl; do not edit by hand")
    println(io, "\\centering")
    println(io, "\\setlength{\\tabcolsep}{2pt}")
    for (_, (cname, hex)) in colors
        println(io, "\\providecolor{$(cname)}{HTML}{$(hex)}")
    end
    println(io, "\\begin{tabular}{@{}ccc@{}}")
    println(io, join(["\\input{$(prefix)$(f)}" for (f, _, _, _) in panels], " & ") * " \\\\[1pt]")
    println(io, "(a) Rotation error & (b) Translation error & (c) Probability of detection \\\\")
    println(io, "\\end{tabular}\\\\[4pt]")
    _legend_tikz(io, colors)
    open(joinpath(out_dir, "occlusion_sweep_table.tex"), "w") do fio
        write(fio, take!(io))
    end
    return println("occ_{rot,trans,pod}.tex + occlusion_sweep_table.tex -> $out_dir")
end


# --- unoccluded marginals, TikZ ----------------------------------------------
#
# The pgfplots twin of plot_pose_errors (absolute_pose.jl), in the same house
# style as the ECDF and occlusion panels: identical palette, \scriptsize
# labels, one shared legend above the block. This was the last Makie-rendered
# figure either document still included, and it was the only one whose fonts
# and proportions did not match the rest.
#
# Median with an interquartile band, per system, per bin. The band is emitted
# as a closed polygon (lows forward, highs backward) rather than through the
# fillbetween library, so the figure needs no extra pgfplots library.
function write_unoccluded_tikz(src; out_dir, prefix = get(ENV, "BB_TEX_PREFIX", DRV["tex_prefix"]))
    colors = Dict(
        "blob" => ("bbBlue", "0072B2"), "april" => ("bbOrange", "E69F00"),
        "aruco" => ("bbGreen", "009E73")
    )
    # (file key, bin column, value column, x label, y label, extra axis opts)
    # Ticks are explicit 1-2-5 steps over each panel's data range rather than
    # decades alone: rotation spans 0.1-14 deg, translation 2-100 mm, so a
    # decade-only axis gives two or three labels across the whole panel.
    panels = [
        ("unocc_rot", :rig_angles_bin, :abs_error_angles_deg,
            raw"obliquity ($^\circ$)", raw"rotation error ($^\circ$)",
            "ymode=log, log ticks with fixed point, xtick={-90,-60,-30,0,30,60,90},\n" *
            "             ytick={0.1,0.5,1,5,10},"),
        ("unocc_trans_ang", :rig_angles_bin, :error_distances,
            raw"obliquity ($^\circ$)", "translation error (mm)",
            "ymode=log, log ticks with fixed point, xtick={-90,-60,-30,0,30,60,90},\n" *
            "             ytick={2,5,10,25,50,100},"),
        ("unocc_trans_dist", :rig_distances_bin, :error_distances,
            "camera distance", "translation error (mm)",
            "ymode=log, log ticks with fixed point, ytick={2,5,10,25,50,100},\n" *
            "             xtick={1000,2000,3000}, xticklabels={{1\\,m},{2\\,m},{3\\,m}},"),
        ("unocc_pod", :rig_angles_bin, nothing,
            raw"obliquity ($^\circ$)", raw"$P(\mathrm{detection})$",
            "ymin=0, ymax=1.05, ytick={0,0.25,0.5,0.75,1}, yticklabels={0,,0.5,,1},\n" *
            "             xtick={-90,-60,-30,0,30,60,90},"),
    ]
    for (fkey, by, col, xl, yl, extra) in panels
        io = IOBuffer()
        println(io, "% $(fkey).tex — generated (unoccluded marginals) by error_distributions.jl; do not edit by hand")
        for (_, (cname, hex)) in colors
            println(io, "\\providecolor{$(cname)}{HTML}{$(hex)}")
        end
        println(io, "\\begin{tikzpicture}[baseline=(current bounding box.center)]")
        println(io, "\\begin{axis}[width=5.6cm, height=4.0cm,")
        println(io, "  xlabel={$(xl)}, ylabel={$(yl)},")
        println(io, "  $(extra)")
        println(io, "  label style={font=\\scriptsize}, tick label style={font=\\scriptsize},")
        println(io, "  ]")
        for sy in SYSTEMS
            d = _load(src, sy.key)
            cname = colors[sy.key][1]
            if col === nothing                       # detection rate: no band
                g = combine(
                    groupby(d, by),
                    :dot_product => (x -> count(!ismissing, x) / length(x)) => :rate
                )
                sort!(g, by)
                pts = ["($(r[by]),$(round(r.rate; sigdigits = 4)))" for r in eachrow(g)]
                isempty(pts) && continue
            else
                dd = dropmissing(d, col)
                isempty(dd) && continue
                g = combine(
                    groupby(dd, by), nrow => :n,
                    col => median => :med,
                    col => (x -> quantile(x, 0.25)) => :lo,
                    col => (x -> quantile(x, 0.75)) => :hi
                )
                sort!(filter!(:n => >=(5), g), by)   # same min-n rule as the Makie figure
                isempty(g) && continue
                lo = ["($(r[by]),$(round(r.lo; sigdigits = 4)))" for r in eachrow(g)]
                hi = ["($(r[by]),$(round(r.hi; sigdigits = 4)))" for r in eachrow(g)]
                println(
                    io, "\\addplot[draw=none, fill=$(cname), fill opacity=0.15, forget plot] " *
                        "coordinates {$(join(vcat(lo, reverse(hi)), " "))} \\closedcycle;"
                )
                pts = ["($(r[by]),$(round(r.med; sigdigits = 4)))" for r in eachrow(g)]
            end
            println(
                io, "\\addplot[draw=$(cname), line width=1.0pt, mark=*, mark size=1.1pt, " *
                    "mark options={fill=$(cname), draw=$(cname)}] coordinates {$(join(pts, " "))};"
            )
        end
        println(io, "\\end{axis}")
        println(io, "\\end{tikzpicture}")
        open(joinpath(out_dir, "$(fkey).tex"), "w") do f
            write(f, take!(io))
        end
    end

    io = IOBuffer()
    println(io, "% unoccluded_table.tex — generated by error_distributions.jl; do not edit by hand")
    println(io, "\\centering")
    println(io, "\\setlength{\\tabcolsep}{2pt}")
    for (_, (cname, hex)) in colors
        println(io, "\\providecolor{$(cname)}{HTML}{$(hex)}")
    end
    println(io, "\\begin{tabular}{@{}cc@{}}")
    println(io, "\\input{$(prefix)unocc_rot} & \\input{$(prefix)unocc_trans_ang} \\\\")
    println(io, "(a) Rotation error & (b) Translation error \\\\[6pt]")
    println(io, "\\input{$(prefix)unocc_trans_dist} & \\input{$(prefix)unocc_pod} \\\\")
    println(io, "(c) Translation error by distance & (d) Probability of detection \\\\")
    println(io, "\\end{tabular}\\\\[4pt]")
    _legend_tikz(io, colors)
    open(joinpath(out_dir, "unoccluded_table.tex"), "w") do f
        write(f, take!(io))
    end
    return println("unocc_*.tex + unoccluded_table.tex -> $out_dir")
end



# --- occlusion floor plans ----------------------------------------------------------
#
# Figure 5's floor plan, reproduced per occlusion level. This is the occlusion
# result of record together with `occlusion_sweep` (absolute_pose.jl), and it
# replaced the per-level distribution panels for one concrete reason: past 20%
# occlusion BOTH tag baselines detect nothing, and every other function in this
# file assumes each system has samples (bare `median`/`quantile` throughout).
# The floor plan is the one that was written for absence — `_med` returns NaN on
# an empty cell and zero-detection stations render as GRAY WEDGES, so "detects
# nothing" is visible rather than a missing line the reader must be told about.
#
# Each level writes its own TikZ set under a per-level prefix, so the paper can
# \input them independently of the unoccluded figure.
function write_occlusion_floorplans(src; fig_dir)
    levels = get(ENV, "BB_OCC_LEVELS", DRV["occlusion_levels"])
    for lvl in filter(!isempty, strip.(split(levels, ',')))
        lvl_dir = joinpath(fig_dir, "occ_" * lvl)
        mkpath(lvl_dir)
        with_level(lvl) do
            if all(isempty(_load(src, s.key)) for s in SYSTEMS)
                println("occlusion level $lvl: no observations in the frame — skipped")
                return
            end
            plot_error_floorplan(src; out_dir = lvl_dir)
            prefix = get(ENV, "BB_TEX_PREFIX", DRV["tex_prefix"])
            withenv(
                "BB_TEX_PREFIX" => prefix == "" ? "" : replace(prefix, "floorplan/" => "floorplan_occ_$lvl/")
            ) do
                write_floorplan_tikz(src; out_dir = lvl_dir)
            end
            println("occlusion floor plan @ $lvl% -> $lvl_dir")
        end
    end
end

# --- per-image timing macros --------------------------------------------------------
#
# Mean ± std (and median) of the END-TO-END per-image time, per marker system,
# for the paper's method comparison. Source of record: the detection run's
# telemetry (`telemetry.jsonl` beside its `target_poses.json`; `src/sweep.jl`
# times the full find_boards(…, gray) call, with a warm-up frame so JIT is
# excluded; the tag baselines' telemetry carries the same field, converted from
# their per-image exports). Read from the detection runs rather than the
# pose_results cells because timing is a property of the detection run, not of
# the error join. The median is
# emitted alongside mean/std because the first frame of a run can absorb
# residual compile the warm-up misses (observed: one 18 s outlier against a
# 1.9 s median on 260527/bb, inflating std past the mean).
import JSON3

function write_timing_stats(sources; out_dir)
    io = IOBuffer()
    println(io, "% timing_stats.tex — generated by error_distributions.jl")
    println(io, "% end-to-end per-image detection time from the detection runs' telemetry")
    for (sys, mac) in (("bb", "bb"), ("apriltags", "apr"), ("aruco", "uco"))
        d = poses_dir(sources, sys)
        isdir(d) || (println("timing: no $d — skipped"); continue)
        ts = Float64[]
        tel = joinpath(d, "telemetry.jsonl")
        if isfile(tel)
            # one line per detected image, in run order (sorted here by frame
            # index, the order the per-image exports were read in)
            rows = [JSON3.read(l) for l in eachline(tel) if !isempty(strip(l))]
            for j in sort(rows; by = r -> Int(r.t))
                haskey(j, :elapsed_time_ms) && push!(ts, Float64(j.elapsed_time_ms) / 1000)
            end
        end
        isempty(ts) && (println("timing: no elapsed_time_ms in $d — skipped"); continue)
        fmt(x; d = 2) = string(round(x; digits = d))
        println(io, "\\newcommand{\\$(mac)TimeMeanS}{$(fmt(mean(ts)))}")
        println(io, "\\newcommand{\\$(mac)TimeStdS}{$(fmt(std(ts)))}")
        println(io, "\\newcommand{\\$(mac)TimeMedS}{$(fmt(median(ts)))}")
        println(io, "\\newcommand{\\$(mac)TimeN}{$(length(ts))}")
    end
    open(joinpath(out_dir, "timing_stats.tex"), "w") do f
        write(f, take!(io))
    end
    return println("timing_stats.tex -> $out_dir")
end

# --- main: one stage ------------------------------------------------------------------

function main()
    sources = eval_sources(EVAL)
    # ONE collected frame for every condition (DrWatson `collect_results` over the
    # per-cell artifacts); a condition is selected with `with_level`, not a path.
    src = pose_results(sources)
    # plotsdir(), not a "figures/" dir inside the dataset: figures are DERIVED
    # products of this project, and data/exp_raw is the immutable input tree.
    # BB_FIG still overrides.
    fig_dir = get(ENV, "BB_FIG", plotsdir())
    mkpath(fig_dir)
    plot_trans_distributions(src; out_dir = fig_dir)
    plot_rot_distributions(src; out_dir = fig_dir)
    plot_trans_by_distance(src; out_dir = fig_dir)
    plot_error_floorplan(src; out_dir = fig_dir)
    write_floorplan_tikz(src; out_dir = fig_dir)
    write_histograms_tikz(src; out_dir = fig_dir)
    write_scale_ecdfs_tikz(src; out_dir = fig_dir)
    write_paper_stats(src; out_dir = fig_dir)
    write_occlusion_stats(src; out_dir = fig_dir)
    write_occlusion_sweep_tikz(src; out_dir = fig_dir)
    write_unoccluded_tikz(src; out_dir = fig_dir)
    write_occlusion_floorplans(src; fig_dir)
    write_timing_stats(sources; out_dir = fig_dir)
    write_figure_records(EVAL, EVAL_PATH; driver = "bmvc2026/scripts/plots/error_distributions.jl",
        inputs = [poses_dir(sources, s) for s in ("apriltags", "aruco", "bb")], outputs = [fig_dir])
    return nothing
end

run_stages("error_distributions", [
    Stage(:emit, "floor plans, ECDFs, histograms, occlusion sweep and the macro files from the pose_results cells",
        () -> false, main),
])
