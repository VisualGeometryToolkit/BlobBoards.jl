module BlobBoardsCUDAExt

using CUDA: CUDA, CuArray
using Colors: Gray
using FixedPointNumbers: N0f8

using BlobBoards
using VisualGeometryToolkitFeatures: gray_f32, batched_rows
using VisualGeometryToolkitCore: logpolar_map, image_warp, image_warp_batch_to!, region, undistort!,
    CameraModel, AbstractIntrinsics, AbstractLensModel, Pinhole
using CUDA: CUDABackend   # the KA device token — BB's sole GPU dispatch tag
import KernelAbstractions as KA
using ONNXRunTime: ONNXRunTime
const _OCAPI = ONNXRunTime.CAPI

import BlobBoards: _undistort_adapt

# The default-device resolver is registered by VisualGeometryToolkitCoreCUDAExt
# (2026-09-07); this extension no longer needs an `__init__`.

# =============================================================================
# GPU canonicalize→describe fast path (ORT device tensors)
# =============================================================================
#
# The kernel and bilinear sampler live in `VisualGeometryToolkitCoreCUDAExt`
# (`image_warp_batch_to!` for `CuArray`). This file keeps only the DEVICE core
# below — the (P,P,N) patch cube that `_canon_describe` hands to ORT without a
# download. The host-facing `_canonicalize` is the backend-generic one in
# src/canonicalize.jl (a CUDA override here was unreachable: `canonicalize`
# dispatches to BlobBoards' own generic method).

# Device core: returns the (P,P,N) patch cube ON DEVICE (no download).
function _canonicalize_cuda_device(img, warps, P::Int)
    N = length(warps)
    # Skip the CPU conversion + PCIe upload if caller pre-uploaded the image
    # (≈80 ms saved on a 4240×2832 Float32 image).
    d_img = if img isa CUDA.CuArray{Float32, 2}
        img
    else
        CuArray(Float32.(Gray{Float32}.(gray_f32(img))))
    end
    d_warps = CuArray(warps)
    d_patches = CUDA.zeros(Float32, P, P, N)

    # Delegate to VGC's batched bilinear warp (per-thread warp math + sample).
    image_warp_batch_to!(d_patches, d_warps, d_img; fillvalue = 0.5f0)
    return d_patches
end

# =============================================================================
# GPU-resident canonicalize → describe (ORT device tensors)
# =============================================================================
#
# The host path downloads the (P,P,N) patch cube (~440 MB at 26k patches),
# splits it into 26k Gray matrices, then re-gathers 128-patch host batches
# that the ORT CUDA EP uploads again. This path keeps the cube on-device:
# permute to ORT's row-major NCHW byte layout on the GPU and hand ONNX
# Runtime an OrtValue backed by CUDA memory.
#
# ONNXRunTime 1.3.x wraps only CreateCpuMemoryInfo; the CreateMemoryInfo and
# CreateTensorWithDataAsOrtValue entries in the api table accept any
# allocator name, so we ccall them directly ("Cuda" is the CUDA EP's
# allocator name). Graph outputs are not bound, so ORT copies the (small)
# descriptor output to host itself.

function _ort_cuda_meminfo(api::_OCAPI.OrtApi)
    p_ptr = Ref(C_NULL)
    status = @ccall $(api.CreateMemoryInfo)(
        "Cuda"::Cstring,
        _OCAPI.OrtDeviceAllocator::Cint,
        Cint(0)::Cint,                       # device id
        _OCAPI.OrtMemTypeDefault::Cint,
        p_ptr::Ptr{Ptr{Cvoid}},
    )::_OCAPI.OrtStatusPtr
    return _OCAPI.into_julia(_OCAPI.OrtMemoryInfo, api, p_ptr, status, Any[])
end

# OrtValue over device memory. `shape` is the ORT (row-major) shape; `d` must
# already hold the matching byte layout. GC handles keep the CuArray + meminfo
# alive for the OrtValue's lifetime.
function _ort_device_tensor(api::_OCAPI.OrtApi, meminfo, d::CuArray{Float32}, shape)
    shapevec = collect(Int64, shape)
    p_ptr = Ref(C_NULL)
    status = @ccall $(api.CreateTensorWithDataAsOrtValue)(
        meminfo.ptr::Ptr{Cvoid},
        Ptr{Cvoid}(UInt(pointer(d)))::Ptr{Cvoid},
        (length(d) * sizeof(Float32))::Csize_t,
        pointer(shapevec)::Ptr{Int64},
        length(shapevec)::Csize_t,
        _OCAPI.ONNXTensorElementDataType(Float32)::_OCAPI.ONNXTensorElementDataType,
        p_ptr::Ptr{Ptr{Cvoid}},
    )::_OCAPI.OrtStatusPtr
    return _OCAPI.into_julia(_OCAPI.OrtValue, api, p_ptr, status, [d, meminfo, shapevec])
end

# The batch comes from `model.max_batch` — it sizes the arena this network
# reserved at construction (BlobBoards.reserve_workspace!), so it cannot be a
# per-call keyword without invalidating that reservation.
#
# max_batch=64: the first 64-channel conv layer's activation is
# batch×64×64×64×4B per extend request. The ORT arena must coexist with the
# ~13.5 GiB KA detector plan on a 24 GB card, and inside that budget SMALL
# batches win — measured on 3247 real patches (RTX 4090, 3 GiB arena cap,
# detector plan resident), µs/patch by max_batch:
#   32 → 64.1   64 → 65.3   96 → 77.3   128 → 79.1   192 → 83.3   256 → 80.2
# and 384/512 OOM the arena outright. That last fact now reads differently:
# the arena needs 14 + 8*batch MiB, so 384 wants 3.1 GiB and 512 wants 4.1 —
# both above the FIXED 3 GiB cap those runs used. The OOM was the cap, not the
# card. With the cap derived from the batch the curve is worth re-measuring;
# 64 remains the default until it is. Batching is numerically exact here — the
# graph normalizes each patch independently and carries no BatchNorm.
function _describe_device(model, d_patches::CuArray{Float32, 3})
    P, P2, N = size(d_patches)
    N == 0 && return Matrix{Float32}(undef, 0, 0)
    sess = model.session
    api = sess.api
    meminfo = _ort_cuda_meminfo(api)
    return batched_rows(N, model.max_batch) do offset, n
        # ORT logical shape (n,1,P,P) row-major == col-major (P,P,1,n) with
        # w fastest. d_patches is [h,w,k]; permute → [w,h,k], add the
        # singleton channel dim. permutedims materializes contiguously.
        d_chunk = reshape(
            permutedims(view(d_patches, :, :, (offset + 1):(offset + n)), (2, 1, 3)),
            P, P, 1, n
        )
        CUDA.synchronize()   # our stream must finish before ORT's touches it
        t = _ort_device_tensor(api, meminfo, d_chunk, (n, 1, P, P))
        outs = _OCAPI.Run(
            api, sess.session, nothing,
            sess.input_names, [t], sess.output_names
        )
        _OCAPI.GetTensorMutableData(api, outs[1])
    end
end

# CUDABackend + ONNX(:cuda) fast path for the fused canonicalize→describe
# stage. Any other execution provider falls back to the host pipeline.
function BlobBoards._canon_describe(
        device::CUDABackend, model::BlobBoards.DescriptorNetwork,
        frames::AbstractVector, img::AbstractMatrix; inner_extent, outer_extent, patch_size = 64
    )
    if model.session.execution_provider !== :cuda || patch_size != 64
        return BlobBoards.describe(
            model,
            BlobBoards.canonicalize(frames, img; patch_size, inner_extent, outer_extent, device)
        )
    end
    P = 64
    inner_r = inner_extent / outer_extent
    N = length(frames)
    N == 0 && return Matrix{Float32}(undef, 0, 0)
    # The image goes up once; every chunk warps from the resident copy.
    d_img = _upload_f32(img)
    # THE PATCH CUBE IS CHUNKED. One cube for a whole frame is P·P·N·4 B —
    # 3 GiB at 195 000 detections — held while the network runs, beside the
    # detector's plan and the arena, and it was the allocation that failed
    # first on a 24 GB card (2026-09-15). A chunk of `_DESCRIBE_CHUNK` is
    # 256 MB; the network batches inside it as before, so the descriptors are
    # the same bytes.
    out = Matrix{Float32}(undef, 0, 0)
    for start in 1:_DESCRIBE_CHUNK:N
        stop = min(start + _DESCRIBE_CHUNK - 1, N)
        # Logpolar-only by construction (this fast path never receives mode or
        # supersample; the generic BlobBoards.canonicalize handles those). The
        # inline warp build mirrors _region_bm(:logpolar, …) — kept inline on
        # purpose: routing through the generic builder changes warp construction
        # on the production descriptor path (2026-09 audit).
        warps = [
            image_warp(logpolar_map(region(ff, outer_extent), P, inner_r), P)
                for ff in view(frames, start:stop)
        ]
        d_patches = _canonicalize_cuda_device(d_img, warps, P)
        rows = _describe_device(model, d_patches)
        finalize(d_patches)   # back to the pool now, not at the next collection
        isempty(out) && (out = Matrix{Float32}(undef, N, size(rows, 2)))
        out[start:stop, :] .= rows
    end
    return out
end
const _DESCRIBE_CHUNK = 16_384

# The image on the device as Float32. An 8-bit grey raster (a board: 50 Mpx at
# 1200 dpi) goes up as its bytes and is converted there — a quarter of the
# transfer and no host-side Float32 copy (0.34 → 0.20 s per board); the values
# are the same as the host conversion's (N0f8 → Float32 is exact either way).
_upload_f32(img::CuArray{Float32, 2}) = img
_upload_f32(img::AbstractMatrix{Gray{N0f8}}) =
    Float32.(reinterpret(N0f8, CuArray(collect(reinterpret(UInt8, img)))))
_upload_f32(img::AbstractMatrix) = CuArray(Float32.(Gray{Float32}.(gray_f32(img))))

# =============================================================================
# CUDA hooks for the backend-generic KA adaptation (src/adapt_ka.jl)
# =============================================================================
#
# The adaptation kernels are backend-generic; CUDA keeps the two
# bit-exactness hooks of the original implementation.

# __nv_lgamma keeps the sweep S-score and the weight kernels' log_C
# bit-identical to the legacy CUDA kernels.
_lgamma_nv(x::Float64) = ccall("extern __nv_lgamma", llvmcall, Cdouble, (Cdouble,), x)
BlobBoards._device_lgamma(::CUDABackend) = _lgamma_nv

# cuBLAS batched LU — bit-identical to legacy, including which
# near-degenerate blobs getrf's `info` invalidates.
function BlobBoards._solve_6x6_batched!(
        ::CUDABackend, AtWA, AtWz, pivot, info, valid, converged
    )
    CUDA.CUBLAS.getrf_strided_batched!(AtWA, pivot, info)
    CUDA.CUBLAS.getrs_strided_batched!('N', AtWA, AtWz, pivot)
    return nothing
end


# =============================================================================
# GPU undistort — lens distortion + bilinear sampling on device
# =============================================================================
#
# Returns CuArray{Float32,2} that flows directly into scale space (zero-copy).
# For each output pixel: K⁻¹ → normalized → lens → K → bilinear sample.

# Upload image to GPU as Float32 — reinterpret + convert on device, no CPU Float32 copy
function _gpu_upload_f32(img)
    raw = reinterpret(eltype(eltype(img)), img)  # Gray{N0f8} → N0f8, Gray{Float32} → Float32, etc.
    d_raw = CuArray(raw)
    return Float32.(d_raw)
end
_gpu_upload_f32(img::AbstractMatrix{Float32}) = CuArray(img)
_gpu_upload_f32(img::CuArray{Float32, 2}) = img

_undistort_adapt(::CUDABackend, img, ::Nothing) = _gpu_upload_f32(img)
_undistort_adapt(::CUDABackend, img, cam::CameraModel{<:AbstractIntrinsics, Pinhole}) = _gpu_upload_f32(img)

# Lens undistort (any lens with a distortion map):
# VGC's fused KA warp (`undistort!`) on the CUDA backend; this method just
# keeps the CUDABackend contract (device-resident CuArray{Float32} flowing
# into scale space).
function _undistort_adapt(::CUDABackend, img, cam::CameraModel{<:AbstractIntrinsics, <:AbstractLensModel})
    d_img = _gpu_upload_f32(img)
    d_out = CUDA.zeros(Float32, size(d_img))
    undistort!(d_out, Float32, cam, d_img)
    return d_out
end

# =============================================================================
# Training-pipeline backend hooks (src/training.jl)
# =============================================================================
#
# training.jl lives in the core module and must not `using CUDA`; its hooks
# get their CUDA methods here. The device resolver is set in this file's single
# `__init__` (above).

BlobBoards.reclaim_pool!(::CUDABackend) = CUDA.reclaim()

end # module BlobBoardsCUDAExt
