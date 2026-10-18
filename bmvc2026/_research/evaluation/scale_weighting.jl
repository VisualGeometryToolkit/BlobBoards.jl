#=
Compare spatial_weighting=true vs false for adaptation methods.

Uses detect_blobs with CPU() for detection, then
adapt(dets, img; method=GNMethod(), spatial_weighting=...) for fitting.

Displays a 2x2 grid:
  Row 1: GN without / with spatial weighting
  Row 2: Posterior without / with spatial weighting

Usage:
  include("examples/evaluation/scale_weighting.jl")
  compare_scale_weighting("/path/to/image.png")
=#

using BlobBoards
using BlobBoards: adapt, detect_blobs, filter_blobs   # advanced API (unexported)
using VisualGeometryToolkitCore:
    Circle, Ellipse, Disc, EllipticFrame,
    boundary, imshow!
using GeometryBasics: Point2
using GLMakie
import FileIO
using VisualGeometryToolkitCore: ustrip
using Colors: Gray
using Statistics: median

# =============================================================================
# Comparison runner
# =============================================================================

"""
    compare_scale_weighting(image_path; kwargs...)

Compare spatial_weighting=true vs false for GN and Posterior adaptation.

Detects blobs using CPU(), then fits with adapt() using both
spatial_weighting settings. Displays 2x2 grid of results.
"""
function compare_scale_weighting(
    image_path;
    visible = true,
    cutoff_val = DEFAULT_CUTOFF,
    peak_threshold = 0.04,
    edge_threshold = 10.0,
    overlap_threshold = 0.2,
    polarity = :dark,
    xlims = nothing,
    ylims = nothing,
)
    GLMakie.activate!(; visible)

    println("Loading image: $image_path")
    img = FileIO.load(image_path)
    println("Image size: $(size(img, 2)) x $(size(img, 1))")

    img_gray = Gray.(img)

    # Detect blobs
    println("\nRunning blob detection...")
    detections = detect_blobs(img_gray;
        device=KA.CPU(),
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        overlap_threshold=overlap_threshold,
        polarity=polarity)
    println("Detections after NMS: $(length(detections))")

    # --- Run all 4 variants ---
    variants = [
        ("GN (no spatial)",        GNMethod(),        false),
        ("GN (spatial)",           GNMethod(),        true),
        ("Posterior (no spatial)",  PosteriorMethod(), false),
        ("Posterior (spatial)",     PosteriorMethod(), true),
    ]

    all_ellipses = Dict{String, Vector{Ellipse{Float64}}}()

    println("\n" * "="^70)
    println("Method                       Success  Filtered")
    println("-"^70)

    for (label, method, spatial) in variants
        adapted = adapt(detections, img_gray;
            method=method,
            cutoff=cutoff_val,
            spatial_weighting=spatial)
        filtered = filter_blobs(adapted, detections;
            cutoff=cutoff_val,
            image=img_gray)
        ellipses = [boundary(det.frame, cutoff_val) for det in filtered]
        all_ellipses[label] = ellipses

        println("  $(rpad(label, 26)) $(lpad(length(adapted),4))/$(length(detections))   $(length(ellipses))")
    end
    println("="^70)

    # --- Visualization: 2x2 grid ---
    fig = Figure(size=(1600, 1200))
    axes = Makie.Axis[]

    titles_and_keys = [
        (1, 1, "GN (no spatial)"),
        (1, 2, "GN (spatial)"),
        (2, 1, "Posterior (no spatial)"),
        (2, 2, "Posterior (spatial)"),
    ]
    colors_map = Dict(
        "GN (no spatial)"        => :magenta,
        "GN (spatial)"           => :green,
        "Posterior (no spatial)"  => :magenta,
        "Posterior (spatial)"     => :green,
    )

    for (r, c, key) in titles_and_keys
        n = length(all_ellipses[key])
        ax = Makie.Axis(fig[r, c]; aspect=DataAspect(), yreversed=true,
            title="$key ($n)")
        imshow!(ax, img_gray)
        Makie.poly!(ax, all_ellipses[key];
            strokecolor=colors_map[key], strokewidth=1.0, color=:transparent)
        push!(axes, ax)
    end

    linkaxes!(axes...)

    if !isnothing(xlims)
        Makie.xlims!(axes[1], xlims...)
    end
    if !isnothing(ylims)
        Makie.ylims!(axes[1], ylims...)
    end

    display(fig)
    return fig
end
