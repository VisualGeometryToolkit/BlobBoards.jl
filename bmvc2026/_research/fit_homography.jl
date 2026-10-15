#=
Fit Homography from Saved Correspondences

Loads a JLD2 file containing matched 2D point correspondences
(source -> target) and fits a homography via RANSAC + LO.

Demonstrates the minimal API path for going from raw point pairs to a
robust homography estimate, without the full board-detection pipeline.

Usage:
  include("examples/pipeline/fit_homography.jl")
=#

using BlobBoards  # not strictly needed; pulls in VGC + RVG transitively
using JLD2
using StaticArrays: SVector
using StructArrays: StructArray
using GLMakie; GLMakie.activate!()
using GeometryBasics: Point2f
using CoordinateTransformations: PerspectiveMap
using VisualGeometryCore: HomographyMap, matches!
using RobustVisualGeometry: fit_homography
using Statistics: mean

using Infiltrator

# =============================================================================
# Load correspondences
# =============================================================================

cspond_path = joinpath(@__DIR__, "..", "test", "data", "correspondences.jld2")

src_pts, tgt_pts = jldopen(cspond_path, "r") do f
    # Stored as StructVector{SVector{2,Float64}} in both fields
    collect(f["source"]), collect(f["target"])
end

@assert length(src_pts) == length(tgt_pts)
n = length(src_pts)
println("Loaded $n correspondences from $(basename(cspond_path))")

# =============================================================================
# Fit homography
# =============================================================================
#
# fit_homography accepts a bare StructArray of Pair{src,tgt}; with no
# `scores` kwarg the RANSAC sampler defaults to uniform. Supply `scores=v` to
# enable PROSAC ordered by descending v (e.g. descriptor similarities).
#
# `outlier_halfwidth` is the half-width (in target-pixel units) of the
# RANSAC outlier-rejection band. Tune to the expected pixel noise.

S = SVector{2, Float64}
pairs = StructArray{Pair{S, S}}((src_pts, tgt_pts))

result = fit_homography(pairs; outlier_halfwidth = 5.0)

if isnothing(result) || !result.converged
    error("RANSAC failed to converge (n=$n correspondences)")
end

H = result.value
inlier_mask = result.inlier_mask
n_inliers = count(inlier_mask)

println("Converged in $(result.iterations) trials")
println("  inliers     = $n_inliers / $n")
println("  scale       = $(round(result.scale; digits=3)) px")
println("  quality     = $(round(result.quality; digits=2))")
inlier_rmse = sqrt(mean(abs2, result.residuals[inlier_mask]))
println("  inlier RMSE = $(round(inlier_rmse; digits=3)) px")

# =============================================================================
# Project source points through H and visualize
# =============================================================================

@infiltrate
proj = PerspectiveMap() ∘ HomographyMap(H)
proj_pts = Point2f.(proj.(src_pts))
src_pts_2f = Point2f.(src_pts)
tgt_pts_2f = Point2f.(tgt_pts)

# Source = blue, target = green. Outliers fade out via per-point alpha
# from the RANSAC IRLS weights (≈1 for inliers, ≈0 for outliers).
src_marker_colors = [(:blue, w) for w in result.weights]

fig = Figure(size = (1200, 600))

ax_src = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true,
              title = "Source points")
scatter!(ax_src, src_pts_2f; color = src_marker_colors, markersize = 10)

ax_tgt = Axis(fig[1, 2]; aspect = DataAspect(), yreversed = true,
              title = "Target + reprojected source (H·src)")
matches!(ax_tgt, proj_pts, tgt_pts_2f;
         src_color = :blue, tgt_color = :green,
         weights = result.weights)

display(fig)
fig
