# =============================================================================
# Descriptor models — abstract interface + ONNX and Lux backends
# =============================================================================

import ONNXRunTime
# the one ONNX Runtime session of the stack (provider resolution, the sm_100-capable CUDA build)
using VisualGeometryToolkitFeatures: onnx_session, onnx_run

# =============================================================================
# Abstract interface
# =============================================================================

"""
    AbstractDescriptorNetwork

Abstract type for descriptor models. Subtypes must implement:
- `describe(model, patches) -> Matrix{Float32}` (N×D, L2-normalized)
- `network_hash(model) -> String` (unique identifier for cache invalidation)
- `patch_size(model) -> Int` (the input resolution the network requires)

Batching is a property OF THE MODEL, not of the call (`max_batch` sizes its CUDA arena).
"""
abstract type AbstractDescriptorNetwork end

(m::AbstractDescriptorNetwork)(patches) = describe(m, patches)

"""
    network_hash(model::AbstractDescriptorNetwork) -> String

The descriptor network's identity, a hash of its weights file: part of every
gallery cache key, so features computed by one network are never served for
another.
"""
network_hash(m::AbstractDescriptorNetwork) = m.network_hash

"""
    patch_size(model::AbstractDescriptorNetwork) -> Int

The canonical patch resolution the network takes, as a property OF THE
NETWORK rather than a configured number.

It used to be a `[descriptor] patch_size` key in the run config, which
restated something the checkpoint already fixes. That was redundant rather
than dangerous — a mismatch is refused by the backend — but it was refused
*late*: `patch_size` is part of the gallery cache key, so a wrong value first
missed the cache and re-extracted the whole board set, and only then failed at
the first `describe`. Deriving it makes the disagreement impossible instead of
merely detectable.
"""
patch_size(m::AbstractDescriptorNetwork) = m.patch_size

"""
    load_descriptor_network(network_or_path) -> AbstractDescriptorNetwork

The descriptor network a pipeline runs. It is a network as given, or one read from a
path — a `.jld2` checkpoint as a [`LuxDescriptorNetwork`](@ref) (BlobBoardsLuxExt
must be loaded), anything else as an ONNX [`DescriptorNetwork`](@ref). The single
switch point: every caller that has a path and wants a model funnels through here.
"""
load_descriptor_network(m::AbstractDescriptorNetwork) = m

function load_descriptor_network(path::AbstractString)
    return if endswith(lowercase(path), ".jld2")
        LuxDescriptorNetwork(path)
    else
        DescriptorNetwork(path)
    end
end

# =============================================================================
# ONNX backend
# =============================================================================

"""
    DescriptorNetwork <: AbstractDescriptorNetwork

ONNX-based descriptor model. Loads a pre-exported ONNX file for inference
via ONNXRunTime.

# Constructors
- `DescriptorNetwork(path::AbstractString)` — load from ONNX file

`max_batch` is a FIELD rather than a `describe` keyword because it fixes the
size of the CUDA arena this network's session is capped at (`_arena_cap`); a cap
made for one batch size that a later call could exceed would be no cap at all.
"""
struct DescriptorNetwork <: AbstractDescriptorNetwork
    session::ONNXRunTime.InferenceSession
    input_name::String
    output_name::String
    network_hash::String
    patch_size::Int
    max_batch::Int
end

# Ceiling for the ORT CUDA arena, derived from the batch it has to hold.
#
# Measured 2026-08-13 (RTX 4090, log-polar s96 network, patch 64) by stepping
# max_batch and watching `CUDA.free_memory()`; the arena settles at
#
#     arena_MiB = 14 + 8 * max_batch
#
# exact to the megabyte at batch 32/64/128/256. This is a GUARD, not a
# forecast: the slope is a property of one graph's activations, and the actual
# claim is made by the network's runs, each through Features' `onnx_run`, which
charges the arena's growth to the detector's device budget. 4x headroom covers
# a graph up to four times heavier per patch without the cap ever binding.
#
# The cap binding is not hypothetical — at the previous fixed 3 GiB, batch 384
# and 512 OOMed the arena outright (see `_describe_device`) and batch 512
# measured 442 us/patch against 63 at batch 64. Both were read as evidence that
# large batches are intrinsically slow; the formula says 512 wants 4.1 GiB, so
# they are at least partly the cap thrashing against its own ceiling. Worth
# re-measuring now that the ceiling moves with the batch.
#
# NO UPPER CLAMP. An earlier version clamped to 4 GiB, which at max_batch=512
# put the ceiling (4096 MiB) BELOW the predicted need (4110 MiB) and would have
# reintroduced the very OOM this replaced. A cap costs nothing when unused —
# `gpu_mem_limit` is lazy, so an oversized ceiling is never allocated — while a
# cap below the working set is fatal. Err high; the runs make the real claim,
# counted by Features' `onnx_run` in the detector's budget.
function _arena_cap(max_batch::Integer)
    max_batch <= 0 && return 3 << 30      # uncapped batch: keep the old ceiling
    need_mib = 14 + 8 * Int(max_batch)
    return max(512, 4 * need_mib) * (1 << 20)
end

# -----------------------------------------------------------------------------
# Input geometry, read from the graph
# -----------------------------------------------------------------------------
#
# ONNXRunTime.jl surfaces `input_names`/`output_names` and nothing else, so the
# input SHAPE has to come from the C API directly. Four raw ccalls; we
# deliberately do not build ONNXRunTime.jl's own wrapper objects around these
# pointers, because `CastTypeInfoToTensorInfo` returns a BORROWED pointer owned
# by the OrtTypeInfo — wrapping it in a type whose finalizer releases would be
# a double free. Only the OrtTypeInfo is owned, and it is released in `finally`.

function _onnx_input_dims(s::ONNXRunTime.InferenceSession, index::Integer = 0)
    api = s.api
    p_ti = Ref(Ptr{Cvoid}(C_NULL))
    st = @ccall $(api.SessionGetInputTypeInfo)(
        s.session::Ptr{Cvoid}, Csize_t(index)::Csize_t, p_ti::Ptr{Ptr{Cvoid}}
    )::Ptr{Cvoid}
    st == C_NULL || error("ONNX SessionGetInputTypeInfo failed for input $index")
    ti = p_ti[]
    try
        p_tsi = Ref(Ptr{Cvoid}(C_NULL))
        st2 = @ccall $(api.CastTypeInfoToTensorInfo)(
            ti::Ptr{Cvoid}, p_tsi::Ptr{Ptr{Cvoid}}
        )::Ptr{Cvoid}
        st2 == C_NULL || error("ONNX CastTypeInfoToTensorInfo failed for input $index")
        tsi = p_tsi[]          # borrowed — owned by `ti`, must NOT be released
        p_n = Ref{Csize_t}()
        @ccall $(api.GetDimensionsCount)(tsi::Ptr{Cvoid}, p_n::Ptr{Csize_t})::Ptr{Cvoid}
        n = Int(p_n[])
        dims = Vector{Int64}(undef, n)
        @ccall $(api.GetDimensions)(
            tsi::Ptr{Cvoid}, dims::Ptr{Int64}, Csize_t(n)::Csize_t
        )::Ptr{Cvoid}
        return dims
    finally
        ccall(api.ReleaseTypeInfo, Cvoid, (Ptr{Cvoid},), ti)
    end
end

# (batch, channels, height, width); ORT reports a dynamic axis as -1. Batch is
# expected to be dynamic — that is what makes `max_batch` a throughput knob —
# but the spatial axes must be fixed and square, since a log-polar patch has no
# meaning at a non-square resolution.
function _patch_size_from_graph(s::ONNXRunTime.InferenceSession, path)
    dims = _onnx_input_dims(s)
    length(dims) == 4 || throw(
        ArgumentError(
            "$path: expected a 4-D descriptor input (batch, channels, height, width), got $dims"
        )
    )
    h, w = dims[3], dims[4]
    (h > 0 && w > 0) || throw(
        ArgumentError(
            "$path: the network's spatial input is dynamic ($dims); BlobBoards derives " *
                "patch_size from the graph and cannot do so for a model that does not fix it"
        )
    )
    h == w || throw(
        ArgumentError(
            "$path: descriptor input must be square, got $(h)x$(w)"
        )
    )
    return Int(h)
end

"""
    DescriptorNetwork(path; execution_provider = :cuda)

Load an ONNX descriptor model.  Defaults to ONNXRuntime's CUDA EP since
the rest of BlobBoards' hot path is GPU-resident (KA detect, CUDA
adapt + canonicalize).  Saves ~340 ms on the SupCon checkpoint vs CPU
EP (~2× on this batch-1 model; dynamic-batch models would gain more).

The session is VisualGeometryToolkitFeatures' `onnx_session`: on `:cuda` it
runs ONNX Runtime's CUDA build that Features installs (device code from sm_70
to sm_120, the B200's sm_100 among them), which needs a visible CUDA device and
CUDA 12's runtime and cuDNN (CUDA_Runtime_jll, CUDNN_jll) in the environment;
its arena is capped at `gpu_mem_limit` (by default derived from `max_batch`).

THROWS if the :cuda EP cannot initialize. It used to warn and continue on the
CPU EP, which is ~34x slower on this network and reported success while doing
it. Opt out explicitly: `execution_provider = :cpu` on a CUDA-less host, or
`allow_cpu_fallback = true` to restore the warn-and-continue behaviour.
"""
function DescriptorNetwork(
        path::AbstractString;
        execution_provider::Symbol = :cuda,
        max_batch::Integer = 64,
        gpu_mem_limit::Integer = _arena_cap(max_batch),
        allow_cpu_fallback::Bool = false
    )
    h = open(sha256_hex, path)   # the model file's content hash keys the gallery cache (the Lux checkpoint's likewise)
    # The session is VisualGeometryToolkitFeatures' one ONNX Runtime session
    # (`onnx_session`): its provider resolution, and on CUDA its ONNX Runtime
    # build with device code for every device from sm_70 to sm_120 (the B200's
    # sm_100 among them, which Microsoft's GPU builds lack), the arena capped at
    # `gpu_mem_limit` and counted in the reserve the ASIFT detector's plans leave.
    session = try
        onnx_session(path; execution_provider, gpu_mem_limit = Int(gpu_mem_limit))
    catch err
        execution_provider === :cuda || rethrow()
        # THIS THROWS BY DEFAULT, and used to `@warn` and silently continue on
        # the CPU EP. The old comment justified that with "~2x slower". It is
        # not 2x: measured 2026-08-16 on track_logpolar_circ_s96_iter4 over
        # 13,004 patches, the CPU EP runs 1.76 ms/patch against 51.9 us on the
        # CUDA EP — 34x, turning a 1.74 s frame into 23.97 s, 91% of it in
        # `describe`. A degradation that large which REPORTS SUCCESS is worse
        # than a failure: it gets measured, believed, and reported as the
        # pipeline's throughput (it was, in this session).
        #
        # Features' error names why the CUDA provider is unavailable (no device
        # visible, a CUDA 12 runtime or cuDNN missing from the environment, …); a
        # full card at session creation fails here too.
        #
        # Opt out deliberately: `execution_provider = :cpu` for a CUDA-less
        # host, or `allow_cpu_fallback = true` to keep the old behaviour.
        allow_cpu_fallback || throw(
            ErrorException(
                "ONNX CUDA execution provider unavailable and " *
                    "allow_cpu_fallback = false.\n" *
                    "The CPU EP is ~34x slower on this network (1.76 ms vs " *
                    "52 us per patch), so falling back silently would " *
                    "misreport pipeline throughput.\n" *
                    "Fix: the reason below; install BlobBoards' CUDA extension once with " *
                    "`$(EXTENSION_INSTALL.BlobBoardsCUDAExt.install)` and pin the CUDA 12 runtime with `$(CUDA_RUNTIME_PIN)` " *
                    "(then restart Julia); or free the GPU if the session's creation failed for memory.\n" *
                    "Opt out: DescriptorNetwork(path; execution_provider = :cpu) " *
                    "or DescriptorNetwork(path; allow_cpu_fallback = true).\n" *
                    "Underlying error: $(sprint(showerror, err))"
            )
        )
        @warn "ONNX CUDA execution provider unavailable; falling back to CPU EP (~34x slower)" exception = err
        onnx_session(path; execution_provider = :cpu)
    end
    net = DescriptorNetwork(
        session, String(first(session.input_names)),
        String(first(session.output_names)), h,
        _patch_size_from_graph(session, path), Int(max_batch)
    )
    return net
end


# The batching skeleton behind every describe backend is VisualGeometryToolkitFeatures'
# `batched_rows` since 0.7.1 (the four drifting copies collapsed there).
using VisualGeometryToolkitFeatures: batched_rows

"""
    describe(model::DescriptorNetwork, patches) -> Matrix{Float32}

Run ONNX descriptor model on patches in batched mode.
Returns an `N × D` descriptor matrix (L2-normalized by the model).

The per-call batch is `model.max_batch`, set at construction — it is not a
call-site keyword, because it also fixes the cap of this network's CUDA arena
(`_arena_cap`). Measured on 3247 real 64×64 patches (RTX 4090,
log-polar s96 network, 3 GiB arena cap), µs/patch by `max_batch`:

    32 → 61.7   64 → 63.1   128 → 76.8   256 → 91.7   512 → 442.4   0 → 472.0

Hence the default of `64`. Read the right-hand columns with care: those runs
predate the batch-derived cap, and batch 512 wants 4.1 GiB against the 3 GiB
ceiling they were measured under, so part of that cliff is the arena thrashing
rather than the convolutions. Batching is numerically exact (each patch is
normalized independently; the graph has no BatchNorm), so this only ever trades
throughput.
"""
function describe(model::DescriptorNetwork, patches)
    isempty(patches) && return Matrix{Float32}(undef, 0, 0)

    N = length(patches)
    H, W = size(patches[1])
    # One reusable batch buffer — reallocated only for the ragged last batch
    # (or after the B→1 error retry); identical bytes reach the session. Held
    # in a typed Ref: a captured variable REASSIGNED inside the closure boxes
    # (JET flags a possibly-undefined path); the Ref keeps reuse box-free.
    input = Ref(Array{Float32}(undef, 0, 1, H, W))
    return batched_rows(N, model.max_batch; retry_single = true) do offset, n
        size(input[], 1) == n || (input[] = Array{Float32}(undef, n, 1, H, W))
        buf = input[]
        for i in 1:n
            buf[i, 1, :, :] .= Float32.(patches[offset + i])
        end
        onnx_run(model.session, Dict(model.input_name => buf))[model.output_name]   # its arena growth counted (Features)
    end
end

# =============================================================================
# Lux backend (provided by BlobBoardsLuxExt extension)
# =============================================================================
#
# `LuxDescriptorNetwork` is a function stub here; the actual struct + methods
# live in ext/BlobBoardsLuxExt.jl and become available when Lux + NNlib +
# cuDNN are all loaded (deliberately lean — no Reactant/Enzyme, so cuDNN can
# load without the libReactantExtra.so symbol conflict). To use:
#
#   using BlobBoards, Lux, NNlib, cuDNN, CUDA
#   model = LuxDescriptorNetwork("checkpoints/best_model.jld2")  # auto-GPU
#   descriptors = describe(model, patches)
#
# The training loop's heavier stack (Reactant/Enzyme/Optimisers/...) lives in
# BlobBoardsTrainingExt. The two extensions are INDEPENDENT: TrainingExt does
# not trigger on cuDNN, so it compiles the shared model file itself rather
# than reaching into LuxExt (2026-09-07 — the previous "TrainingExt builds on
# top of LuxExt" arrangement could not precompile).

"""
    LuxDescriptorNetwork(...)

Lux-based descriptor model. Implementation provided by the
BlobBoardsLuxExt extension; available when Lux + NNlib + cuDNN are loaded,
and without them the call raises an error naming the install line.
Auto-moves parameters to GPU when functional.
"""
function LuxDescriptorNetwork end
