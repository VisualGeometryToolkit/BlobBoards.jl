# =============================================================================
# Descriptor models — abstract interface + ONNX and Lux backends
# =============================================================================

import ONNXRunTime
using SHA: sha256

# =============================================================================
# Abstract interface
# =============================================================================

"""
    AbstractDescriptorNetwork

Abstract type for descriptor models. Subtypes must implement:
- `describe(model, patches) -> Matrix{Float32}` (N×D, L2-normalized)
- `network_hash(model) -> String` (unique identifier for cache invalidation)
- `patch_size(model) -> Int` (the input resolution the network requires)

Batching is a property OF THE MODEL, not of the call — see `reserve_workspace!`.
"""
abstract type AbstractDescriptorNetwork end

"""
    reserve_workspace!(model) -> nothing

Realize whatever device memory `model` will need at steady state, NOW.

BlobBoards' three GPU consumers are not peers. ASIFT's pyramids
(`_chunk_warps`) and the stacked matcher (`_gallery_chunks`) both size
themselves against `device_free_bytes` at the moment they run, so they adapt to
whatever is left. A descriptor session's arena does not: it is set once, grows
in power-of-two extents, and never shrinks. It is the only inelastic consumer,
and therefore the only one that can FAIL rather than adapt — so it must be
claimed first and the elastic budgets sized against the remainder.

`VisualGeometryFeatures`' `_PYRAMID_BUDGET` already assumes exactly this
("the descriptor session's arena is already allocated before the plan is
built, so it is excluded from free"). That assumption was FALSE until this
function existed: ONNXRuntime's `gpu_mem_limit` is a CAP, not a reservation, so
constructing a session claimed ~20 MiB and the arena did not grow until the
first `describe` — which happens long after the plan is built. Measured
2026-08-13 on a 24 GB card: session construction moved `CUDA.free_memory` by
0.02 GiB against a 3 GiB limit.

The default is a NO-OP, which is correct for every backend that allocates from
CUDA.jl's pool (the Lux backend) or from the host (the CPU execution provider).
Those share the pool the elastic consumers already measure, so there is nothing
separate to reserve.
"""
reserve_workspace!(::AbstractDescriptorNetwork) = nothing

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

_ensure_network(m::AbstractDescriptorNetwork) = m

# Path-based dispatch: .jld2 → Lux (requires BlobBoardsLuxExt loaded),
# anything else → ONNX. This is the single switch point — every caller
# that has a path and wants a model funnels through here.
function _ensure_network(path::AbstractString)
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
size of the CUDA arena this network reserves (`reserve_workspace!`). A
reservation made for one batch size that a later call could exceed would not be
a reservation at all.
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
# claim is made by `reserve_workspace!` running the network. 4x headroom covers
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
# cap below the working set is fatal. Err high; `reserve_workspace!` makes the
# real claim, and an unreasonable `max_batch` now fails at construction on an
# empty card rather than mid-run against a loaded one.
# Content hash of a model file — keys the gallery cache for BOTH backends (ONNX
# here, the Lux checkpoint in BlobBoardsLuxExt).
_file_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

function _arena_cap(max_batch::Integer)
    max_batch <= 0 && return 3 << 30      # uncapped batch: keep the old ceiling
    need_mib = 14 + 8 * Int(max_batch)
    return max(512, 4 * need_mib) * (1 << 20)
end

"""
    reserve_workspace!(model::DescriptorNetwork) -> nothing

Grow the ORT CUDA arena to its steady-state size by running one forward evaluation at
`model.max_batch`. Since the arena never shrinks, that one evaluation converts a lazy
cap into a genuine reservation that `device_free_bytes` can see.

No-op on the CPU execution provider (no device arena) and when `max_batch <= 0`
(the batch is unbounded, so there is no steady state to reserve for — that mode
opts out of the guarantee).
"""
function reserve_workspace!(model::DescriptorNetwork)
    model.session.execution_provider === :cuda || return nothing
    model.max_batch > 0 || return nothing
    P = model.patch_size
    # THREE evaluations, not one. Measured 2026-09-07 on a B200 (CUDA 12.9 runtime):
    # construction + one evaluation claimed 530 MiB, and the NEXT call at the very same
    # batch shape grew the arena by exactly 256 MiB once more, after which every
    # call (64, 256, 512 patches) was flat. One evaluation is therefore not the steady
    # state on that driver stack, and the arena test asserted that it was. The
    # extra evaluations cost two 64-patch forwards; the reservation they buy is what
    # makes `device_free_bytes` truthful for VGF's `_chunk_warps`.
    warm = [zeros(Float32, P, P) for _ in 1:model.max_batch]
    for _ in 1:3
        describe(model, warm)
    end
    return nothing
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

Requires `import CUDA, cuDNN` before the call — BOTH; `import CUDA` alone
leaves ONNXRunTime's CUDA extension inactive and the :cuda EP unavailable.

THROWS if the :cuda EP cannot initialize. It used to warn and continue on the
CPU EP, which is ~34x slower on this network and reported success while doing
it. Opt out explicitly: `execution_provider = :cpu` on a CUDA-less host, or
`allow_cpu_fallback = true` to restore the warn-and-continue behaviour.

The CUDA runtime version must be in the range ONNXRunTime.jl supports
(currently 12.x).  Pin via `CUDA.set_runtime_version!(v"12.x")` if your
default is newer; the preference lives in `LocalPreferences.toml`.
"""
function DescriptorNetwork(
        path::AbstractString;
        execution_provider::Symbol = :cuda,
        max_batch::Integer = 64,
        gpu_mem_limit::Integer = _arena_cap(max_batch),
        allow_cpu_fallback::Bool = false
    )
    h = _file_sha256(path)
    provider_options = execution_provider === :cuda ?
        (; gpu_mem_limit = Int(gpu_mem_limit)) : (;)
    session = try
        ONNXRunTime.load_inference(path; execution_provider, provider_options)
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
        # The trigger is mundane and easy to hit: the :cuda EP needs
        # ONNXRunTime's CUDA extension active in THIS process, which requires
        # BOTH `import CUDA` and `import cuDNN`. `using CUDA` alone is not
        # enough, and nothing else in the pipeline needs cuDNN, so the omission
        # is invisible. A full card at session creation produces the same
        # exception (ORT reserves a fixed arena up front), so contention and a
        # missing package are indistinguishable without reading this text.
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
                    "Fix: `import CUDA, cuDNN` before constructing " *
                    "DescriptorNetwork (cuDNN is required and easy to omit), " *
                    "or free the GPU if the arena reservation failed.\n" *
                    "Opt out: DescriptorNetwork(path; execution_provider = :cpu) " *
                    "or DescriptorNetwork(path; allow_cpu_fallback = true).\n" *
                    "Underlying error: $(sprint(showerror, err))"
            )
        )
        @warn "ONNX CUDA execution provider unavailable; falling back to CPU EP (~34x slower)" exception = err
        ONNXRunTime.load_inference(path; execution_provider = :cpu)
    end
    net = DescriptorNetwork(
        session, String(first(session.input_names)),
        String(first(session.output_names)), h,
        _patch_size_from_graph(session, path), Int(max_batch)
    )
    # Claim the arena HERE, while the card is still empty — every other GPU
    # consumer in the pipeline sizes itself against what is left.
    reserve_workspace!(net)
    return net
end


# The batching skeleton behind every describe backend is VisualGeometryFeatures'
# `batched_rows` since 0.7.1 (the four drifting copies collapsed there).
using VisualGeometryFeatures: batched_rows

"""
    describe(model::DescriptorNetwork, patches) -> Matrix{Float32}

Run ONNX descriptor model on patches in batched mode.
Returns an `N × D` descriptor matrix (L2-normalized by the model).

The per-call batch is `model.max_batch`, set at construction — it is not a
call-site keyword, because it also fixes the arena this network reserved (see
`reserve_workspace!`). Measured on 3247 real 64×64 patches (RTX 4090,
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
        model.session(Dict(model.input_name => buf))[model.output_name]
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
BlobBoardsLuxExt extension; available when Lux + NNlib + cuDNN are loaded.
Auto-moves parameters to GPU when functional.
"""
function LuxDescriptorNetwork end
