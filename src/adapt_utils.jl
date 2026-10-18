# =============================================================================
# Shared Adaptation Utilities
# =============================================================================
#
# Constants, helpers, and types shared by adapt.jl and the GPU adaptation
# extension. Included before them in BlobBoards.jl.
# =============================================================================

import VisualGeometryToolkitEstimation

# =============================================================================
# Solve-type tags — select the per-blob solver/objective. Shared by the GPU
# adaptation extension and `AdaptConfig`.
# (The CPU paraboloid implementation that consumed these was removed.)
# =============================================================================
# NB no `SolveType` hierarchy. There were three — WhitenedLinearSolveType,
# LinearSolveType, HqSolveType — selectable via `AdaptConfig.solve`. Two of
# them were fiction: adaptation was GPU-only at the time, and the CUDA
# `_adapt` dispatched on `AdaptConfig{T}` alone, so whitened-vs-unwhitened
# never reached a kernel and picking `linear` changed nothing. The third (half-quadratic Welsch) was
# a real alternative arm that measured worse than the Student-t mixture and
# was not used. Removed 2026-08: the solver is the Student-t/uniform mixture
# with the sort-and-sweep posterior partition, and it is not a choice.

using VisualGeometryToolkitCore:
    Disc, EllipticFrame, Paraboloid3D,
    moment_matrix, region, iou, coverage,
    is_valid_shape, is_peak, characteristic_scale, coeffs

using VisualGeometryToolkitFeatures: Detection

using LinearAlgebra: Symmetric, dot, norm
using StaticArrays: SVector, SMatrix

# =============================================================================
# Constants
# =============================================================================

const _MIN_POINTS_FIXED_CENTER = 10
const _MIN_POINTS_ISOTROPIC = 10
const _MIN_POINTS_ANISOTROPIC = 20
const _MIN_REGION_HALFWIDTH = 2

# `saturation_bounds` — the pixel value range a measurement may trust — is
# VisualGeometryToolkitCore's since 0.21.1.
using VisualGeometryToolkitCore: saturation_bounds

# =============================================================================

# Site accessors: VGC's `centroid`/`characteristic_scale` ARE the accessors
# (the local _site_center/_site_max_sigma pair duplicated them — and the two
# spellings were used inconsistently).

# 6-param ISOTROPIC seed Paraboloid3D: C = -1/(2σ²)·I, b = 0.  EllipticFrame
# input is collapsed to a Disc of geomean-σ first — matches the OLD behavior
# that worked, before we tried to seed C from S·S' directly.
@inline _site_seed6(d::Disc, z_max::Real) =
    Paraboloid3D{Float64, 6}(Paraboloid3D(d, z_max))
@inline function _site_seed6(f::EllipticFrame, z_max::Real)
    # Back-projected ellipse area tends to over-estimate the true blob extent,
    # so halve the geomean σ when building the isotropic seed C = -1/(2σ²)·I.
    σ = 0.5 * characteristic_scale(f)
    return _site_seed6(Disc(f.center, σ), z_max)
end
