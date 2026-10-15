# =============================================================================
# plot_annotations — moved here from src/annotate.jl (2026-07-26).
#
# It called Makie.Figure/hidedecorations!/colgap! directly from the CORE
# module, where Makie is only a weakdep — a violation of "core loads with zero
# graphics" that JET caught as `BlobBoards.Makie is not defined`. Core now
# carries the stub (`function plot_annotations end`); this is the method.
# =============================================================================

function plot_annotations(sets::Dict; max_blobs::Int = 20, max_obs::Int = 10)
    isempty(sets) && error("No annotations to plot")

    # Sort by number of observations (most first)
    sorted = sort(collect(sets); by = kv -> length(kv[2]), rev = true)
    if length(sorted) > max_blobs
        sorted = sorted[1:max_blobs]
    end

    n_rows = length(sorted)
    # Columns: 1 ref + gap + up to max_obs observations
    max_n_obs = minimum([max_obs, maximum(length(cs) - 1 for (_, cs) in sorted)])
    n_cols = 1 + 1 + max_n_obs  # ref + gap + obs

    fig = Makie.Figure(; figure_padding = 5)

    for (row, (key, cs)) in enumerate(sorted)
        uid, bid = key
        n_obs = min(length(cs) - 1, max_obs)

        # Column 1: reference patch
        ax_ref, _ = imshow(fig[row, 1], _clamp_patch(cs[1]))
        Makie.hidedecorations!(ax_ref); Makie.hidespines!(ax_ref)

        # Column 2: observations in a row
        for j in 1:n_obs
            ax, _ = imshow(fig[row, j + 1], _clamp_patch(cs[j + 1]))
            Makie.hidedecorations!(ax); Makie.hidespines!(ax)
        end
    end

    Makie.colgap!(fig.layout, 2)
    Makie.rowgap!(fig.layout, 2)

    return fig
end
