module BlobBoardsLuxExt

# Inference-only Lux backend for the descriptor model.
#
# Trigger set: Lux + NNlib + cuDNN. Deliberately excludes Reactant/Enzyme/
# Optimisers/TensorBoardLogger/Zygote (those live in BlobBoardsTrainingExt and
# are training-only). Keeping the inference trigger set lean lets a caller
# load the Lux backend WITHOUT Reactant — and avoid the cuDNN/Reactant
# libReactantExtra.so symbol conflict that prevents cuDNN.jl from loading
# alongside Reactant (Reactant bundles its own cuDNN 9.14 via XLA).

using Lux
using NNlib
using cuDNN  # trigger only — loaded so NNlib's GPU conv path uses real cuDNN

using Random: Random
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra
using SHA: sha256
using JLD2: JLD2

using BlobBoards
import KernelAbstractions as KA
using BlobBoards: AbstractDescriptorNetwork, _board_raster, _board_mask_image, DEFAULT_OUTER_EXTENT,
    _stack_2ch, canonicalize, cutoff, described_features
import BlobBoards: describe, network_hash, LuxDescriptorNetwork,
    extract_board_features

include("training/model.jl")  # BlobDescriptorNetwork

# =============================================================================
# LuxDescriptorNetworkImpl — concrete subtype of AbstractDescriptorNetwork
# =============================================================================

struct LuxDescriptorNetworkImpl{M, P, S} <: AbstractDescriptorNetwork
    model::M
    ps::P
    st::S
    network_hash::String
    # The checkpoint's `cfg.patch_size` — the Lux analogue of the ONNX
    # backend's graph-declared input size. See BlobBoards.patch_size.
    patch_size::Int
end

network_hash(m::LuxDescriptorNetworkImpl) = m.network_hash
BlobBoards.patch_size(m::LuxDescriptorNetworkImpl) = m.patch_size

function LuxDescriptorNetwork(
        model, ps, st; network_hash::String = "lux-model",
        patch_size::Int = 64
    )
    return LuxDescriptorNetworkImpl(model, ps, Lux.testmode(st), network_hash, patch_size)
end

"""
    LuxDescriptorNetwork(path::AbstractString)

Load a Lux descriptor checkpoint (.jld2 with keys `ps`, `st`, `cfg`).
Auto-moves parameters/state to GPU when `Lux.MLDataDevices.gpu_device()` is
functional — otherwise stays on CPU. Caller never has to deal with device
placement manually.
"""
function LuxDescriptorNetwork(checkpoint_path::AbstractString)
    h = BlobBoards._file_sha256(checkpoint_path)
    ckpt = JLD2.load(checkpoint_path)
    cfg = ckpt["cfg"]
    model = BlobDescriptorNetwork(;
        patch_size = cfg.patch_size,
        descriptor_dim = cfg.descriptor_dim,
        dropout = cfg.dropout
    )
    ps = ckpt["ps"]
    st = Lux.testmode(ckpt["st"])
    # Auto-GPU when functional. Use try/catch because gpu_device(force=true)
    # throws if no GPU is present; we want graceful CPU fallback.
    dev = try
        Lux.MLDataDevices.gpu_device(force = true)
    catch
        Lux.MLDataDevices.cpu_device()
    end
    ps = ps |> dev
    st = st |> dev
    return LuxDescriptorNetworkImpl(model, ps, st, h, Int(cfg.patch_size))
end

# =============================================================================
# describe — detection path (1-channel patches) and reference path (2-channel)
# =============================================================================

# Detection path: 1-channel grayscale patches. Stack uniform-1 mask into ch 2.
# Batched to avoid GPU OOM on large detection sets. The ONNX path dropped its
# cap to 64 on measurement (see `BlobBoards.describe`); this backend has not
# been re-measured, so it keeps 128. Supply `max_batch=0` to disable.
function describe(
        model::LuxDescriptorNetworkImpl,
        patches::AbstractVector{<:AbstractMatrix};
        max_batch::Int = 128
    )
    isempty(patches) && return Matrix{Float32}(undef, 0, 0)
    N = length(patches)
    H, W = size(patches[1])
    return batched_rows(N, max_batch) do offset, n
        input = Array{Float32}(undef, W, H, 2, n)
        @inbounds for i in 1:n
            input[:, :, 1, i] .= Float32.(patches[offset + i])
        end
        input[:, :, 2, :] .= 1.0f0
        _lux_forward(model, input)  # n × D CPU matrix
    end
end

# Reference path: pre-stacked (P, P, 2, N) tensor — intensity + real mask.
function describe(
        model::LuxDescriptorNetworkImpl,
        patches::AbstractArray{<:Real, 4};
        max_batch::Int = 128
    )
    isempty(patches) && return Matrix{Float32}(undef, 0, 0)
    size(patches, 3) == 2 ||
        error("LuxDescriptorNetwork expects 2-channel input (intensity + mask), got $(size(patches, 3))")
    N = size(patches, 4)
    return batched_rows(N, max_batch) do offset, n
        _lux_forward(model, Float32.(@view patches[:, :, :, (offset + 1):(offset + n)]))
    end
end

# Move CPU input to the device where ps lives, run forward, copy back to CPU.
function _lux_forward(
        model::LuxDescriptorNetworkImpl,
        input::AbstractArray{Float32, 4}
    )
    target_dev = Lux.MLDataDevices.get_device(model.ps)
    input_dev = target_dev(input)
    emb, _ = model.model(input_dev, model.ps, model.st)  # (D, N)
    return Matrix{Float32}(Array(emb)')                   # CPU N × D
end

# =============================================================================
# Reference-side patch extraction for the masked 2-channel descriptor.
# Mirrors src/annotate.jl's _extract_reference_patches.
# =============================================================================

# `device` is accepted (and ignored — the Lux path places via gpu_device())
# because BoardGallery's insert! forwards it unconditionally. Without it this
# method is unreachable through the gallery: MethodError on an unsupported
# keyword. Pre-existing; the Lux gallery path had never been exercised.
#
# The `canonicalize` calls below supply `device = KA.CPU()` EXPLICITLY. They used
# to rely on its default, which was `KA.CPU()` until `default_device()` made
# the GPU the default (0.15.0). That would not have crashed — `_canonicalize`
# uploads host input on a GPU backend — but it would have changed these
# descriptors' VALUES: the CPU branch runs the colorant store path documented
# as byte-identical to the legacy per-frame loop, while the GPU branch is a
# separate numeric path. These are GALLERY descriptors, which key the feature
# cache and drive matching, so silently recomputing them on a different path
# is the risk. And no gate could have caught it: this extension's Lux/NNlib
# triggers are absent from `[targets] test`, so `Pkg.test` never loads it.
# Placement belongs to Lux's `gpu_device()`, which `describe` applies to the
# stacked patches; canonicalization stays on the host deliberately.
function extract_board_features(
        model::LuxDescriptorNetworkImpl, board;
        patch_size = 64,
        inner_extent = cutoff(board.bundle),
        outer_extent = DEFAULT_OUTER_EXTENT,
        device = nothing,
        image::AbstractMatrix = _board_raster(board.image_path, board.bundle, board.config_hash; density = board.density)
    )
    # `device` is accepted and deliberately IGNORED: this backend canonicalizes
    # on the CPU by design (see above), and `Base.insert!(::BoardGallery, …)`
    # forwards `default_device()` unconditionally — on any GPU host that is
    # `CUDABackend()`, so refusing it (tried in the 2026-09-07 audit) broke
    # every Lux gallery build. The keyword exists for the generic seam, not
    # for this backend.
    mask_image = _board_mask_image(board.bundle, image, board.density)
    intensity_patches = canonicalize(
        board.canvas_frames, image;
        patch_size, inner_extent, outer_extent, device = KA.CPU()
    )
    mask_patches = canonicalize(
        board.canvas_frames, mask_image;
        patch_size, inner_extent, outer_extent, device = KA.CPU()
    )
    patches_2ch = _stack_2ch(intensity_patches, mask_patches, patch_size)
    desc = describe(model, patches_2ch)
    return described_features(board.world_frames, desc)
end

end  # module BlobBoardsLuxExt
