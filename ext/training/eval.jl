# =============================================================================
# Evaluation — FPR@95 recall
# =============================================================================

"""
    fpr_at_95_recall(embeddings, labels) -> Float64

Compute False Positive Rate at 95% True Positive Recall.

All pairs `(i, j)` with `i < j` are scored by cosine similarity.
Same-label pairs are positives, different-label pairs are negatives.
Returns the FPR when the similarity threshold is set to capture 95% of positives.

Lower is better. 0.0 = perfect, 1.0 = random.
"""
function fpr_at_95_recall(
        embeddings::AbstractMatrix{T},
        labels::AbstractVector
    ) where {T}
    D, N = size(embeddings)
    S = embeddings' * embeddings  # (N, N) cosine similarity

    pos_sims = T[]
    neg_sims = T[]
    for i in 1:N, j in (i + 1):N
        s = S[i, j]
        if labels[i] == labels[j]
            push!(pos_sims, s)
        else
            push!(neg_sims, s)
        end
    end

    isempty(pos_sims) && return 1.0
    isempty(neg_sims) && return 0.0

    # Threshold at 95% recall: 5th percentile of positive similarities
    sort!(pos_sims)
    idx_95 = max(1, floor(Int, 0.05 * length(pos_sims)))
    threshold = pos_sims[idx_95]

    # FPR: fraction of negatives above threshold
    n_fp = count(s -> s >= threshold, neg_sims)
    return Float64(n_fp) / length(neg_sims)
end
