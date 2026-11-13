module BlobBoardsCUDAExt

using CUDA: CUDA, CuArray

using BlobBoards
using VisualGeometryToolkitFeatures: batched_rows
using CUDA: CUDABackend   # the KA device token — BB's sole GPU dispatch tag
using ONNXRunTime: ONNXRunTime
const _OCAPI = ONNXRunTime.CAPI

# The default-device resolver is registered by VisualGeometryToolkitCoreCUDAExt
# (2026-09-07); this extension no longer needs an `__init__`.

# =============================================================================
# GPU-resident canonicalize → describe (ORT device tensors)
# =============================================================================
#
# The patch cube is BlobBoards' one KA warp (`_patch_cube`, src/canonicalize.jl)
# and the chunked driver its `_canon_describe` (src/pipeline.jl); this file adds
# only the HANDOFF of a CUDA cube to an ONNX session on the CUDA provider. The
# host handoff downloads the cube (~440 MB at 26k patches), splits it into 26k
# Gray matrices, then re-gathers 128-patch host batches that the ORT CUDA EP
# uploads again. This one keeps the cube on-device: permute to ORT's row-major
# NCHW byte layout on the GPU and hand ONNX Runtime an OrtValue backed by CUDA
# memory.
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

# The batch comes from `model.max_batch` — it sizes the cap of this network's
# arena (set at construction), so it is not a per-call keyword. Every run goes
# through Features' `onnx_run`, which charges the arena's growth to the ASIFT
# detector's device budget.
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
        outs = BlobBoards.onnx_run(sess) do
            _OCAPI.Run(api, sess.session, nothing, sess.input_names, [t], sess.output_names)
        end
        _OCAPI.GetTensorMutableData(api, outs[1])
    end
end

# A CUDA patch cube to an ONNX session on the CUDA provider: the device tensor
# above. Any other provider takes the host handoff (the same network, read
# through the host).
BlobBoards._describe_patches(model::BlobBoards.DescriptorNetwork, cube::CuArray{Float32, 3}) =
    model.session.execution_provider === :cuda ? _describe_device(model, cube) :
    BlobBoards.describe(model, BlobBoards._host_patches(cube))

end # module BlobBoardsCUDAExt
