# =============================================================================
# Training data — config, data loading, balanced batch sampler
# =============================================================================

using Random: randperm

# =============================================================================
# Training Config
# =============================================================================

"""
    TrainingConfig(; kwargs...)

Hyperparameters for descriptor training.

# Batch composition
- `blobs_per_batch::Int = 64` — M: distinct blob identities per batch
- `views_per_blob::Int = 8` — K: patches sampled per blob (with replacement)
- `confusers_per_batch::Int = 512` — C: unmatched detection patches per batch
- Batch size = M×K + C (default: 64×8 + 512 = 1024)

# Data
- `min_track_length::Int = 2` — minimum observations to include a blob

# Loss
- `temperature::Float32 = 0.07f0` — SupCon softmax temperature τ

# Optimization
- `optimizer::Symbol = :adamw` — `:adamw` or `:sgd`
- `lr::Float64 = 1e-3` — initial learning rate (decays linearly to 0)
- `weight_decay::Float64 = 1e-4` — L2 regularization
- `epochs::Int = 50` — number of training epochs
- `batches_per_epoch::Int = 0` — 0 = auto (ceil(n_blobs / M))
- `eval_interval::Int = 50` — validate & log every N steps (0 = end of epoch only)

# Model
- `patch_size::Int = 64` — input patch resolution
- `descriptor_dim::Int = 128` — output descriptor dimension
- `dropout::Float32 = 0.1f0` — dropout rate before pooling
"""
Base.@kwdef struct TrainingConfig
    # Batch composition
    blobs_per_batch::Int = 64
    views_per_blob::Int = 8
    confusers_per_batch::Int = 512
    # Data
    min_track_length::Int = 2
    # Loss
    temperature::Float32 = 0.07f0
    # Optimization
    optimizer::Symbol = :adamw
    lr::Float64 = 1.0e-3
    weight_decay::Float64 = 1.0e-4
    epochs::Int = 50
    batches_per_epoch::Int = 0      # 0 = auto
    eval_interval::Int = 50         # validate every N steps
    # Model
    patch_size::Int = 64
    descriptor_dim::Int = 128
    dropout::Float32 = 0.1f0
end

BlobDescriptorNetwork(cfg::TrainingConfig) =
    BlobDescriptorNetwork(;
    patch_size = cfg.patch_size,
    descriptor_dim = cfg.descriptor_dim,
    dropout = cfg.dropout
)

function batch_size(cfg::TrainingConfig)
    return cfg.blobs_per_batch * cfg.views_per_blob + cfg.confusers_per_batch
end

# =============================================================================
# Training Data
# =============================================================================

"""
    TrainingData

In-memory training dataset: tracked patches with blob identity labels,
plus an optional pool of confuser patches (unmatched detections).

Loaded from annotation JLD2 files produced by `annotate_blobs`.

# Fields
- `patches::Array{Float32,4}` — (P, P, 2, N) tracked patches. Channel 1 = intensity,
  channel 2 = validity mask (real on the reference side, all-1s on the detection side).
- `labels::Vector{Int32}` — (N,) blob identity per tracked patch
- `label_to_indices::Dict{Int32, Vector{Int}}` — label → patch indices
- `unique_labels::Vector{Int32}` — labels with ≥ min_track_length patches
- `confusers::Array{Float32,4}` — (P, P, 2, C) confuser patches (may be empty)
"""
struct TrainingData
    patches::Array{Float32, 4}
    labels::Vector{Int32}
    label_to_indices::Dict{Int32, Vector{Int}}
    unique_labels::Vector{Int32}
    confusers::Array{Float32, 4}
end

"""
    TrainingData(patches, labels; confusers=nothing, min_track_length=2)

Build training data from raw patches and labels.
Filters to labels with at least `min_track_length` observations.

`patches` is `(P, P, 2, N)` — intensity + validity mask.
`confusers` is an optional `(P, P, 2, C)` array of unmatched detection patches.
These get unique singleton labels in each batch — they act purely as negatives.
"""
function TrainingData(
        patches::Array{Float32, 4}, labels::Vector{Int32};
        confusers::Union{Nothing, Array{Float32, 4}} = nothing,
        min_track_length::Int = 2
    )
    label_to_indices = Dict{Int32, Vector{Int}}()
    for (i, l) in enumerate(labels)
        push!(get!(Vector{Int}, label_to_indices, l), i)
    end

    # Filter to labels with enough observations for positive pairs
    filter!(p -> length(p.second) >= min_track_length, label_to_indices)
    unique_labels = collect(keys(label_to_indices))
    sort!(unique_labels)

    n_patches = sum(length, values(label_to_indices))
    n_labels = length(unique_labels)
    P = size(patches, 1)
    conf = something(confusers, Array{Float32}(undef, P, P, 2, 0))
    n_conf = size(conf, 4)

    @info "TrainingData: $n_patches patches, $n_labels blobs, $n_conf confusers (min_track_length=$min_track_length)"

    return TrainingData(patches, labels, label_to_indices, unique_labels, conf)
end

"""
    load_training_data(path; min_track_length=2)

Load `TrainingData` from annotations produced by `annotate_blobs` /
`split_training_data`. `path` may be:
- a directory of per-image JLD2s + `refs/<uid>.jld2` (from `build_annotations`)
- a single split JLD2 file (`train.jld2`, `val.jld2`, `test.jld2` from
  `split_training_data`)

Identity is `(board_uid, blob_id)` mapped to a contiguous Int32 label.
Confuser patches are loaded from the `confusers` field (if present).
"""
function load_training_data(path::AbstractString; min_track_length::Int = 2)
    data = load_annotations(path)
    N = size(data.patches, 4)   # (P, P, 2, N)

    # Map (board_uid, blob_id) → contiguous Int32 label
    key_to_label = Dict{Tuple{String, Int}, Int32}()
    labels = Vector{Int32}(undef, N)
    next_label = Int32(1)
    for i in 1:N
        key = (data.board_uids[i], data.blob_ids[i])
        label = get!(key_to_label, key, next_label)
        if label == next_label
            next_label += Int32(1)
        end
        labels[i] = label
    end

    confusers = size(data.confusers, 4) > 0 ? data.confusers : nothing
    return TrainingData(data.patches, labels; confusers, min_track_length)
end

# =============================================================================
# Balanced Batch Sampler (with confusers)
# =============================================================================

"""
    BalancedBatchSampler

Iterator that yields balanced batches: M blobs × K views + C confusers.

Each batch contains `M * K + C` patches:
1. Sample M blob identities (without replacement)
2. For each blob, sample K patches (with replacement if fewer than K available)
3. Sample C confuser patches, each assigned a unique singleton label

Tracked blobs have K positives each for SupCon.
Confusers have 0 positives — they only appear as negatives in other anchors'
denominators. The SupCon loss skips them as anchors (no positive pairs).
"""
struct BalancedBatchSampler
    data::TrainingData
    M::Int          # blobs per batch
    K::Int          # views per blob
    C::Int          # confusers per batch
    n_batches::Int  # batches per epoch
    rng::Random.AbstractRNG
end

"""
    BalancedBatchSampler([rng = default_rng(),] data, cfg::TrainingConfig)

Create a balanced batch sampler from training data and config. The RNG leads,
positionally (Julia's `rand(rng, …)` order), never a keyword.
"""
BalancedBatchSampler(data::TrainingData, cfg::TrainingConfig) =
    BalancedBatchSampler(Random.default_rng(), data, cfg)

function BalancedBatchSampler(
        rng::Random.AbstractRNG, data::TrainingData, cfg::TrainingConfig
    )
    n_labels = length(data.unique_labels)
    M = min(cfg.blobs_per_batch, n_labels)
    C = min(cfg.confusers_per_batch, size(data.confusers, 4))

    n_batches = if cfg.batches_per_epoch > 0
        cfg.batches_per_epoch
    else
        max(1, cld(n_labels, M))
    end

    return BalancedBatchSampler(data, M, cfg.views_per_blob, C, n_batches, rng)
end

Base.length(s::BalancedBatchSampler) = s.n_batches

"""
    sample_batch(sampler) -> (batch_patches, batch_labels)

Sample one balanced batch.

Returns:
- `batch_patches::Array{Float32, 4}` — (P, P, 2, M*K+C) for Lux Conv input.
  Channel 1 = intensity, channel 2 = validity mask (real on the reference
  side, all-1s on the detection side).
- `batch_labels::Vector{Int32}` — (M*K+C,) blob identity labels.
  Confuser labels are unique negative integers (no positives exist).
"""
function sample_batch(s::BalancedBatchSampler)
    P = size(s.data.patches, 1)
    N = s.M * s.K + s.C
    batch_patches = Array{Float32}(undef, P, P, 2, N)
    batch_labels = Vector{Int32}(undef, N)

    # --- Tracked blobs: M blobs × K views ---
    n_labels = length(s.data.unique_labels)
    label_perm = randperm(s.rng, n_labels)

    offset = 0
    for m in 1:s.M
        label = s.data.unique_labels[label_perm[m]]
        indices = s.data.label_to_indices[label]
        n_avail = length(indices)

        for k in 1:s.K
            idx = indices[rand(s.rng, 1:n_avail)]
            offset += 1
            batch_patches[:, :, :, offset] .= @view s.data.patches[:, :, :, idx]
            batch_labels[offset] = label
        end
    end

    # --- Confusers: C patches with unique singleton labels ---
    n_conf = size(s.data.confusers, 4)
    if s.C > 0 && n_conf > 0
        conf_perm = randperm(s.rng, n_conf)
        for c in 1:s.C
            cidx = conf_perm[((c - 1) % n_conf) + 1]
            offset += 1
            batch_patches[:, :, :, offset] .= @view s.data.confusers[:, :, :, cidx]
            batch_labels[offset] = -Int32(c)  # unique negative label per confuser
        end
    end

    return batch_patches, batch_labels
end

# Iterator interface
function Base.iterate(s::BalancedBatchSampler, state = 1)
    state > s.n_batches && return nothing
    return sample_batch(s), state + 1
end
