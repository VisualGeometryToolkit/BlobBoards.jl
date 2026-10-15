# =============================================================================
# view_angle_figure — detections per board vs. viewing obliquity
# =============================================================================
#
# The chart `build_tracks_for_sequences` writes at the end of a run (and
# `build_absolute_tracks` per media item under `debug_dir`). Core carries the
# stubs (`function view_angle_figure end`); these are the methods, so the
# detection pipeline still loads with zero graphics and the terminal fallback
# (`print_view_angle_profile`) covers runs with no backend.
#
# Three stacked panels over ONE shared obliquity axis — deliberately not two
# y-scales on one axis, since matched-blob counts, a recall fraction and a
# board-frame count share no scale:
#
#   1. mean matched blobs per board-frame, ±1 sd   — the headline magnitude
#   2. recall = Σ matched / Σ visible GT blobs     — the same thing normalized,
#                                                    so a board leaving the
#                                                    frame doesn't read as a
#                                                    detector failure
#   3. board-frames per bin                        — the support, i.e. which
#                                                    bars are worth believing
#
# One series per panel, so no legend is needed (each panel's y label names it)
# and identity is never carried by color alone. Bars are a single sequential
# hue; grid, axes and value labels stay recessive text ink.

const _VA_SURFACE = Makie.RGBf(0.988, 0.988, 0.984)   # #fcfcfb
const _VA_INK = Makie.RGBf(0.043, 0.043, 0.043)   # #0b0b0b
const _VA_INK_2 = Makie.RGBf(0.322, 0.318, 0.306)   # #52514e
const _VA_MATCHED = Makie.RGBf(0.165, 0.471, 0.839)   # #2a78d6  categorical 1
const _VA_RECALL = Makie.RGBf(0.922, 0.408, 0.204)   # #eb6834  categorical 2
const _VA_SUPPORT = Makie.RGBf(0.604, 0.6, 0.58)   # muted — support, not data
const _VA_GRID = Makie.RGBAf(0, 0, 0, 0.07)

# Axis styling shared by all three panels: y grid only, no top/right spines,
# ticks and labels in secondary ink so the bars carry the eye.
function _va_axis!(fig, row; ylabel, xlabel = "")
    ax = Makie.Axis(
        fig[row, 1];
        ylabel, xlabel,
        ylabelsize = 12, xlabelsize = 12,
        ylabelcolor = _VA_INK_2, xlabelcolor = _VA_INK_2,
        xticklabelsize = 11, yticklabelsize = 11,
        xticklabelcolor = _VA_INK_2, yticklabelcolor = _VA_INK_2,
        xgridvisible = false, ygridvisible = true,
        ygridcolor = _VA_GRID, ygridwidth = 1,
        topspinevisible = false, rightspinevisible = false,
        leftspinecolor = _VA_GRID, bottomspinecolor = _VA_GRID,
        xtickcolor = _VA_GRID, ytickcolor = _VA_GRID,
        backgroundcolor = _VA_SURFACE
    )
    return ax
end

# Bars centred on each bin, `frac` of the bin wide so neighbours keep a visible
# surface gap. Empty bins are dropped rather than drawn as zero — a zero bar
# reads as "measured zero", which is the opposite of "no samples here".
function _va_bars!(ax, bins, value, color; frac = 0.86)
    keep = [b for b in bins if b.n > 0]
    isempty(keep) && return keep
    x = [(b.lo + b.hi) / 2 for b in keep]
    w = [(b.hi - b.lo) * frac for b in keep]
    Makie.barplot!(ax, x, [value(b) for b in keep]; width = w, color, gap = 0)
    return keep
end

# Direct value labels above the bars — legible without reading off the axis, and
# in text ink rather than the series color (the bar already carries identity).
function _va_labels!(ax, keep, value, label; offset = 6)
    isempty(keep) && return
    return Makie.text!(
        ax, [(b.lo + b.hi) / 2 for b in keep], [value(b) for b in keep];
        text = [label(b) for b in keep],
        align = (:center, :bottom), offset = (0, offset),
        fontsize = 10, color = _VA_INK_2
    )
end

function view_angle_figure(
        stats;
        edges = BlobBoards.VIEW_ANGLE_EDGES,
        title = "avg detections per board vs. viewing angle",
        size = (900, 780)
    )
    bins = BlobBoards._bin_view_angles(stats, edges)
    fig = Makie.Figure(; size, backgroundcolor = _VA_SURFACE, figure_padding = 18)

    # Pre-`n_gt_in_image` files carry no recall denominator — drop that panel rather
    # than render an empty one, and say why in the subtitle.
    has_recall = any(b -> b.n > 0 && !isnan(b.recall), bins)

    n_boards = length(Set(s.uid for s in stats))
    Makie.Label(
        fig[0, 1], title;
        fontsize = 16, color = _VA_INK, font = :bold, halign = :left,
        tellwidth = false, justification = :left
    )
    bin_width = (last(edges) - first(edges)) / (length(edges) - 1)
    Makie.Label(
        fig[1, 1],
        "$(length(stats)) board-frames over $n_boards board(s) · " *
            "obliquity 0° = fronto-parallel, 90° = edge-on · " *
            "bins of $(round(bin_width; digits = 1))°" *
            (has_recall ? "" : "\nno `n_gt_in_image` recorded in this file — rebuild tracks for the recall panel");
        fontsize = 11, color = _VA_INK_2, halign = :left,
        tellwidth = false, justification = :left
    )

    # --- 1. mean matched blobs per board-frame, ±1 sd -------------------------
    ax1 = _va_axis!(fig, 2; ylabel = "matched blobs / board-frame")
    keep = _va_bars!(ax1, bins, b -> b.mean_matched, _VA_MATCHED)
    if !isempty(keep)
        xs = [(b.lo + b.hi) / 2 for b in keep]
        ys = [b.mean_matched for b in keep]
        sd = [b.sd_matched for b in keep]
        # Clamp the lower whisker at 0 — a count can't go negative, and an
        # sd-wide bar dipping below the baseline invents impossible values.
        Makie.errorbars!(
            ax1, xs, ys, min.(sd, ys), sd;
            color = _VA_INK_2, whiskerwidth = 8, linewidth = 1.2
        )
        _va_labels!(
            ax1, keep, b -> b.mean_matched + b.sd_matched,
            b -> string(round(b.mean_matched; digits = 1))
        )
        # Headroom so the topmost value label isn't clipped by the axis.
        Makie.ylims!(ax1, 0, maximum(ys .+ sd) * 1.18)
    end

    # --- 2. recall against the board's visible GT blobs -----------------------
    # The y axis is pinned to 0–100% rather than fitted to the tallest bar, so
    # the fall-off is readable at the same slope across runs.
    ax2 = nothing
    if has_recall
        ax2 = _va_axis!(fig, 3; ylabel = "recall  (matched / visible GT)")
        keep2 = _va_bars!(ax2, bins, b -> isnan(b.recall) ? 0.0 : b.recall, _VA_RECALL)
        _va_labels!(
            ax2, keep2, b -> isnan(b.recall) ? 0.0 : b.recall,
            b -> isnan(b.recall) ? "—" : string(round(100b.recall; digits = 1)) * "%"
        )
        Makie.ylims!(ax2, 0, 1.1)
        ax2.yticks = (0:0.25:1, ["0", "25%", "50%", "75%", "100%"])
    end

    # --- 3. support: board-frames per bin ------------------------------------
    ax3 = _va_axis!(
        fig, has_recall ? 4 : 3;
        ylabel = "board-frames", xlabel = "viewing obliquity  [°]"
    )
    keep3 = _va_bars!(ax3, bins, b -> Float64(b.n), _VA_SUPPORT)
    _va_labels!(ax3, keep3, b -> Float64(b.n), b -> string(b.n))
    if !isempty(keep3)
        nmax = maximum(b -> b.n, keep3)
        Makie.ylims!(ax3, 0, nmax * 1.22)
        # Board-frames are a count — don't let Makie label a 2-frame bin "1.5".
        nmax <= 6 && (ax3.yticks = 0:nmax)
    end

    axes = ax2 === nothing ? (ax1, ax3) : (ax1, ax2, ax3)
    for ax in axes
        Makie.xlims!(ax, first(edges), last(edges))
        ax.xticks = collect(edges)
    end
    # One shared obliquity axis: only the bottom panel carries tick labels.
    for ax in axes[1:(end - 1)]
        Makie.hidexdecorations!(ax; grid = false)
    end
    # Headline panel taller than the normalized one, support smaller than both.
    Makie.rowsize!(fig.layout, 2, Makie.Auto(1.5))
    has_recall && Makie.rowsize!(fig.layout, 3, Makie.Auto(1.1))
    Makie.rowsize!(fig.layout, has_recall ? 4 : 3, Makie.Auto(0.7))
    Makie.rowgap!(fig.layout, 8)
    return fig
end

function save_view_angle_figure(path::AbstractString, stats; kwargs...)
    fig = view_angle_figure(stats; kwargs...)
    dir = dirname(path)
    isempty(dir) || isdir(dir) || mkpath(dir)
    Makie.save(path, fig)
    return path
end
