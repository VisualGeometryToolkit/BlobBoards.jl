# =============================================================================
# SupCon Loss
# =============================================================================

"""
    supcon_loss_mat(embeddings, labels; τ=0.07f0) -> scalar loss

Vectorized SupCon loss using matrix operations (Khosla et al. 2020).

Purely functional — no mutation, no data-dependent control flow, no CPU
allocations — so it compiles through Reactant/XLA and differentiates
cleanly with Enzyme.

SupCon expressed as matrix ops (the former loop reference `supcon_loss` was removed: no callers).
"""
function supcon_loss_mat(
        embeddings::AbstractMatrix{T}, labels::AbstractVector;
        τ = T(0.07)
    ) where {T}
    D, N = size(embeddings)

    # (N, N) similarity matrix (embeddings assumed L2-normalized)
    S = embeddings' * embeddings

    # Masks — built from traced broadcasts, no CPU constructors
    indices = T.(1:N)
    self_mask = indices .== indices'                        # (N, N) diagonal
    pos_mask = (labels .== labels') .& .!self_mask         # same label, not self

    # Count positives per anchor
    n_pos = vec(sum(pos_mask; dims = 2))                     # (N,)
    valid = T.(n_pos .> 0)                                 # 1.0 or 0.0

    # Logits: S / τ, with self-similarity masked to -Inf (functional, no mutation).
    # ifelse keeps off-diagonal entries finite — `T(-Inf) .* false` would be NaN
    # (IEEE: -Inf * 0 = NaN), which would poison every row downstream.
    logits = S ./ τ .+ ifelse.(self_mask, T(-Inf), zero(T))

    # Log-sum-exp per row (numerically stable)
    max_logits = maximum(logits; dims = 2)                   # (N, 1)
    log_denom = log.(sum(exp.(logits .- max_logits); dims = 2)) .+ max_logits

    # Log-prob for each pair
    log_prob = logits .- log_denom                         # (N, N)

    # Average log-prob over positives for each anchor
    per_anchor = vec(sum(log_prob .* pos_mask; dims = 2))    # (N,)

    # Per-anchor loss: -mean over positives, masked to valid anchors only
    safe_n_pos = max.(T.(n_pos), one(T))                   # avoid /0
    n_valid = sum(valid)
    return -sum(per_anchor ./ safe_n_pos .* valid) / max(n_valid, one(T))
end
