# =============================================================================
# BlobDescriptorNetwork — Lux.jl CNN for patch descriptors
# =============================================================================

using Lux: Lux, Chain, Conv, BatchNorm, MaxPool, Dropout, FlattenLayer,
    WrappedFunction, orthogonal
using NNlib: NNlib
using Statistics: mean, std

function _ortho_init(rng, dims...)
    return reshape(orthogonal(rng, Float32, dims...; gain = 0.6f0), dims...)
end

"""
    _l2_normalize(x)

L2-normalize along the first (descriptor) dimension.
Input: (D, B), Output: (D, B) with unit-norm columns.
"""
_l2_normalize(x) = x ./ (sqrt.(sum(abs2, x; dims = 1)) .+ Float32(1.0e-10))

"""
    _input_normalize_masked(x)

Per-sample mean/std normalization of channel 1 (intensity), with statistics
computed only over the pixels marked valid by channel 2 (validity mask).
Channel 2 is forwarded unchanged.

Input: (W, H, 2, B). Output: (W, H, 2, B).

Computing μ/σ over only the valid region avoids biasing the normalization
toward the 0.5 background fill in masked-out regions (which would dilute
the descriptor's contrast on reference patches near the board edge).
"""
function _input_normalize_masked(x::AbstractArray{T, 4}) where {T}
    intensity = @view x[:, :, 1:1, :]
    mask = @view x[:, :, 2:2, :]
    n = sum(mask; dims = (1, 2, 3)) .+ T(1.0e-7)              # effective N per sample
    μ = sum(intensity .* mask; dims = (1, 2, 3)) ./ n       # mean over valid pixels
    cen = (intensity .- μ) .* mask                          # masked centered intensity
    σ = sqrt.(sum(abs2, cen; dims = (1, 2, 3)) ./ n .+ T(1.0e-7))
    normalized = (intensity .- μ) ./ σ                      # apply globally
    return cat(normalized, mask; dims = 3)
end

"""
    _conv_block(in => out, k; stride=1) -> Tuple of (Conv, BatchNorm)

Returns a tuple of layers (NOT a nested Chain — splat into the outer Chain).
ReLU is baked into the Conv via the `act` arg, mirroring Lux's official
ResNet20 tutorial. Avoids both the WrappedFunction wrapper and an extra
nesting level — both reduce Lux's compile-time type signature.
"""
function _conv_block(channels::Pair{Int, Int}, k::Int; stride::Int = 1)
    pad = div(k, 2)
    return (
        Conv(
            (k, k), channels, NNlib.relu;
            stride, pad, use_bias = false, init_weight = _ortho_init
        ),
        BatchNorm(channels.second),
    )
end

"""
    BlobDescriptorNetwork(; patch_size=64, descriptor_dim=128, dropout=0.1f0)

Create descriptor CNN as a Lux Chain.

Architecture (for patch_size=64, default depth):
```
Input (64, 64, 2, B)  — channel 1 intensity, channel 2 validity mask
  → input_normalize_masked (zero-mean / unit-var over valid pixels only,
                            mask channel forwarded)
  → Conv 5×5  2→32  + BN + ReLU
  → Conv 5×5 32→32  + BN + ReLU
  → Conv 5×5 32→64  stride=2 + BN + ReLU     → (32, 32, 64, B)
  → Conv 5×5 64→64  + BN + ReLU
  → Conv 5×5 64→128 stride=2 + BN + ReLU     → (16, 16, 128, B)
  → Conv 5×5 128→128 + BN + ReLU
  → Dropout(0.1)
  → MaxPool (16, 1)                           → (1, 16, 128, B)
  → Conv (1, 16) 128→128 + BN                → (1, 1, 128, B)
  → Flatten                                   → (128, B)
  → L2 normalize                              → (128, B)
```
"""
function BlobDescriptorNetwork(;
        patch_size::Int = 64,
        descriptor_dim::Int = 128,
        dropout::Float32 = 0.1f0
    )
    k, pool = if patch_size == 32
        3, 8
    elseif patch_size == 64
        5, 16
    elseif patch_size == 128
        9, 32
    else
        error("Unsupported patch_size=$patch_size (must be 32, 64, or 128)")
    end

    depths = (32, 64, 128)

    return Chain(
        WrappedFunction(_input_normalize_masked),
        # Block 1: 2 → 32 (reads intensity + mask jointly)
        _conv_block(2 => depths[1], k)...,
        _conv_block(depths[1] => depths[1], k)...,
        # Block 2: 32 → 64, stride-2 downsample
        _conv_block(depths[1] => depths[2], k; stride = 2)...,
        _conv_block(depths[2] => depths[2], k)...,
        # Block 3: 64 → 128, stride-2 downsample
        _conv_block(depths[2] => depths[3], k; stride = 2)...,
        _conv_block(depths[3] => depths[3], k)...,
        # Head
        Dropout(dropout),
        MaxPool((pool, 1)),
        Conv(
            (1, pool), depths[3] => descriptor_dim; use_bias = false,
            init_weight = _ortho_init
        ),
        BatchNorm(descriptor_dim),
        FlattenLayer(),
        WrappedFunction(_l2_normalize),
    )
end
