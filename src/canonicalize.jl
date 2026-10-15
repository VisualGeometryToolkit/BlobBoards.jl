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
# On CPU the cube eltype is Gray{Float32}, so the colorant store path is the
# same one the old per-frame `image_warp!` loop used — byte-identical output.
#
# Dispatches on a KA device token — `KA.CPU()` here, `CUDABackend()` when
# `BlobBoardsCUDAExt` is loaded. NOT VGF's `CpuBackend`/`CudaBackend`: those are
# `ComputeBackend`s, not `KA.Backend`s, and match no `_canonicalize` method.

using VisualGeometryCore: logpolar_map, canonical_map, image_warp,
    image_warp_batch_to!, region, Circle

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
    P = patch_size
    P_hi = P * supersample
    warps = [_region_bm(mode, ff; patch_size = P_hi, inner_extent, outer_extent) for ff in frames]
    N = length(warps)
    if device isa KA.CPU
        # Gray{Float32} cube → the identical colorant store path the old
        # per-frame image_warp! loop ran: byte-identical patches.
        cube = Array{Gray{Float32}, 3}(undef, P_hi, P_hi, N)
        image_warp_batch_to!(cube, warps, gray_f32(img); fillvalue = 0.5f0)
        return [_downsample_patch(cube[:, :, k], supersample) for k in 1:N]
    end
    # GPU backends: upload once (skipped when the caller pre-uploaded Float32
    # on this device), run the same batched kernel, download + reshape into
    # the legacy Vector{Matrix{Gray}} API.
    d_img = (KA.get_backend(img) == device && eltype(img) == Float32) ? img :
        begin
            h = Float32.(Gray{Float32}.(gray_f32(img)))
            d = KA.allocate(device, Float32, size(h)...)
            copyto!(d, h)
            d
        end
    d_warps = KA.allocate(device, eltype(warps), N)
    copyto!(d_warps, warps)
    d_cube = KA.zeros(device, Float32, P_hi, P_hi, N)
    image_warp_batch_to!(d_cube, d_warps, d_img; fillvalue = 0.5f0)
    cube = Array(d_cube)
    return [_downsample_patch(Gray{Float32}.(view(cube, :, :, k)), supersample) for k in 1:N]
end

_downsample_patch(patch::AbstractMatrix, S::Int) =
    S == 1 ? Matrix{Gray{Float32}}(patch) : _box_downsample(patch, S)

# Box-average `S×S` blocks of `hi` (a `P·S × P·S` patch) into a `P × P` patch.
# The area-average of the finer samples over each output texel — the downfilter
# half of supersampling.
function _box_downsample(hi::AbstractMatrix, S::Int)
    P = size(hi, 1) ÷ S
    (P * S == size(hi, 1) && P * S == size(hi, 2)) ||
        throw(DimensionMismatch("supersampled patch $(size(hi)) not divisible by S=$S"))
    out = Matrix{Gray{Float32}}(undef, P, P)
    invn = 1.0f0 / (S * S)
    @inbounds for j in 1:P, i in 1:P
        acc = 0.0f0
        for jj in 1:S, ii in 1:S
            acc += Float32(hi[(i - 1) * S + ii, (j - 1) * S + jj])
        end
        out[i, j] = Gray{Float32}(acc * invn)
    end
    return out
end

function _interpolant(img)
    img_f = gray_f32(img)
    return extrapolate(
        interpolate(img_f, BSpline(Cubic(Interpolations.Line(OnGrid())))),
        zero(eltype(img_f)) + Gray{Float32}(0.5),
    )
end
