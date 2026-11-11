# =============================================================================
# Patch Extraction — Canonicalize frames to normalized patches
# =============================================================================
#
# `canonicalize` extracts per-frame patches in one of three parameterizations
# (`mode`), all sampling the same neighborhood out to `outer_extent`·σ:
#
#   :logpolar  — log-polar warp (rotation/scale invariant under translation in
#                θ/log_r). Samples radially from `inner_extent`·σ to
#                `outer_extent`·σ. The default; feeds the descriptor network.
#   :cartesian — affine-rectified patch. The frame's `outer_extent`·σ ellipse
#                is warped into the square patch, removing shape/rotation/scale.
#   :original  — unrectified axis-aligned crop. The frame's `outer_extent`·σ
#                circumscribed circle is mapped to the square patch, so the blob
#                keeps its original image appearance (no shape/rotation removal).
#
# `inner_extent` only affects `:logpolar`; the other modes use `outer_extent`
# as their single field-of-view radius.
#
# ONE backend-generic batched path: every KA backend (CPU, CUDA, Metal, ROCm)
# samples through VGC's `image_warp_batch_to!`, whose kernel dispatches on the
# backend that owns the output cube.  Descriptor gallery and query patches
# therefore share the interpolant AND the boundary policy exactly — the earlier
# CPU path (Interpolations BSpline + hard-edge extrapolation) differed from the
# GPU kernel in the half-open boundary ring.
#
# The patch cube is Float32 on every backend (`_patch_cube`): the same kernel,
# the same store; `canonicalize` hands it back as Gray{Float32} patches, and the
# descriptor stage (`_canon_describe`, pipeline.jl) hands it to the network.
#
# Dispatches on a KA device token — `KA.CPU()` here, `CUDABackend()` when
# `BlobBoardsCUDAExt` is loaded. NOT VGF's `CpuBackend`/`CudaBackend`: those are
# `ComputeBackend`s, not `KA.Backend`s, and match no `_canonicalize` method.

using VisualGeometryToolkitCore: logpolar_map, canonical_map, image_warp,
    image_warp_batch_to!, region, Circle, ka_upload, on_device, device_zeros
# the supersampling downfilter: Feat's block mean, in Float32 for a Gray{Float32} patch
using VisualGeometryToolkitFeatures: box_downsample

# `_interpolant` is the host BSpline sampler shared with src/training.jl's
# `image_warp`-style augmentation; the batched KA path below does not use it.
import Interpolations
using Interpolations: interpolate, extrapolate, BSpline, Cubic, OnGrid

# =============================================================================
# canonicalize — dispatches on device
# =============================================================================

"""
    canonicalize(frames, img; mode=:logpolar, device=default_device(), kwargs...)
        -> Vector{Matrix{Gray{Float32}}}

Extract normalized patches of `img` at `frames` (Detection, EllipticFrame,
Disc, etc.) — the frames are the shapes the patches are sampled on, so they
lead. Both are typed, so a call in the retired `(img, frames)` order is a
MethodError.

# Keyword Arguments
- `mode=:logpolar`: patch parameterization — `:logpolar`, `:cartesian`, or
  `:original` (see this file's header for the exact geometry of each)
- `patch_size=64`: Output patch size (square)
- `inner_extent=2.0`: Inner radius in units of σ (only used by `:logpolar`)
- `outer_extent=DEFAULT_OUTER_EXTENT` (96σ): outer radius in units of σ (field-of-view for all modes); gallery and query patches must share it
- `device = default_device()`: Computation device
- `supersample=1`: Anti-aliasing supersample factor. When `>1`, the patch is
  sampled on a `supersample`× finer grid and box-averaged down to `patch_size`.
  At large `outer_extent` the log-polar (and cartesian/original) maps minify the
  source — one output texel spans many input pixels — so a single interpolated
  tap aliases; supersampling averages that footprint. Cost scales as
  `supersample²`. (See `out_debug/logpolar_aliasing_oe96.png`.)
"""
function canonicalize(
        frames::AbstractVector, img::AbstractMatrix;
        mode::Symbol = :logpolar,
        patch_size = 64,
        inner_extent = 2.0,
        outer_extent = DEFAULT_OUTER_EXTENT,
        device = default_device(),
        supersample::Int = 1
    )
    return _canonicalize(
        device, img, frames;
        patch_size, inner_extent, outer_extent, mode, supersample
    )
end

# Math-space region map for a frame, selected by `mode`. Shared by every
# backend — all of them wrap the result in `image_warp`.
#   :logpolar  — log-polar warp of the `outer_extent`·σ region, radial range
#                `inner_extent`·σ … `outer_extent`·σ
#   :cartesian — affine rectification of the `outer_extent`·σ region ellipse
#   :original  — axis-aligned crop of the `outer_extent`·σ circumscribed circle
function _region_map(mode::Symbol, frame; patch_size, inner_extent, outer_extent)
    mode === :logpolar &&
        return logpolar_map(region(frame, outer_extent), patch_size, inner_extent / outer_extent)
    mode === :cartesian &&
        return canonical_map(region(frame, outer_extent), patch_size)
    mode === :original &&
        return canonical_map(Circle(frame, outer_extent), patch_size)
    throw(
        ArgumentError(
            "canonicalize: unknown mode $(repr(mode)); expected :logpolar, :cartesian, or :original"
        )
    )
end

# Per-frame image-warp BoundedMap: mode's math region map + image_warp wrap.
function _region_bm(mode::Symbol, frame; patch_size, inner_extent, outer_extent)
    return image_warp(_region_map(mode, frame; patch_size, inner_extent, outer_extent), patch_size)
end

# =============================================================================
# Backend-generic batched warp — VGC's one KA bilinear kernel, any device
# =============================================================================

# The image as a Float32 matrix on `device`, uploaded once (an image already
# there in Float32 is used as it is) — Core's one upload.
_device_image(device, img) = on_device(Float32, device, gray_f32(img))

# The (P·s, P·s, N) Float32 patch cube of `frames` on `device`, sampled by VGC's
# one batched bilinear KA kernel (`image_warp_batch_to!`) at `supersample` = s
# times the patch resolution. `img` is uploaded unless it is already there.
function _patch_cube(
        device::KA.Backend, img, frames;
        patch_size = 64, inner_extent = 2.0, outer_extent = DEFAULT_OUTER_EXTENT,
        mode::Symbol = :logpolar, supersample::Int = 1
    )
    P_hi = patch_size * supersample
    warps = [_region_bm(mode, ff; patch_size = P_hi, inner_extent, outer_extent) for ff in frames]
    cube = device_zeros(Float32, device, P_hi, P_hi, length(warps))
    image_warp_batch_to!(cube, ka_upload(eltype(warps), device, warps), _device_image(device, img); fillvalue = 0.5f0)
    return cube
end

# The cube's patches on the host, as the Gray{Float32} matrices `canonicalize`
# returns, box-averaged down by `supersample` (1: a copy).
function _host_patches(cube, supersample::Int = 1)
    c = Array(cube)
    return [box_downsample(Gray{Float32}.(view(c, :, :, k)), supersample) for k in axes(c, 3)]
end

function _canonicalize(
        device::KA.Backend, img, frames;
        patch_size = 64,
        inner_extent = 2.0,
        outer_extent = DEFAULT_OUTER_EXTENT,
        mode::Symbol = :logpolar,
        supersample::Int = 1
    )
    supersample >= 1 || throw(ArgumentError("`supersample` must be ≥ 1"))
    isempty(frames) && return Matrix{Gray{Float32}}[]
    # Sample the same patch domain at `supersample`× resolution, then box-average
    # down. supersample == 1 leaves the batched path byte-identical.
    cube = _patch_cube(device, img, frames; patch_size, inner_extent, outer_extent, mode, supersample)
    return _host_patches(cube, supersample)
end

function _interpolant(img)
    img_f = gray_f32(img)
    return extrapolate(
        interpolate(img_f, BSpline(Cubic(Interpolations.Line(OnGrid())))),
        zero(eltype(img_f)) + Gray{Float32}(0.5),
    )
end
