# Synthetic-blob accuracy test: place Gaussian blobs at known (x, y, σ),
# run CPU detect_blobs and GPU VKS detect on the same image, measure the
# per-detector ground-truth error in position and σ.
#
# Eliminates the spatial-matching ambiguity from CPU-vs-GPU comparisons —
# each detector is measured against the SAME ground truth.

using BlobBoards
using Colors: Gray
using FixedPointNumbers: N0f8
using CUDA
using VisualGeometryToolkitCore: ustrip, centroid, scale
using StaticArrays: SVector
import Statistics

include(joinpath(@__DIR__, "driver.jl"))
using .AsiftGpu

# -----------------------------------------------------------------------------
# Synthetic blob generation
# -----------------------------------------------------------------------------

"""
    render_synthetic_blobs(W, H, blobs; bg=0.5f0, contrast=0.4f0)

Render Gaussian dark blobs at COLMAP-convention centers `(cx, cy, σ)` —
i.e., pixel `i` (0-indexed) has center at `(i + 0.5, i + 0.5)` in COLMAP.

For each Julia 1-indexed array pixel `(y, x)`, the COLMAP center of that
pixel is `(x - 0.5, y - 0.5)`. The Gaussian is evaluated at that center.

The blob intensity profile is `bg - contrast * exp(-r²/(2σ²))`.
"""
function render_synthetic_blobs(W::Int, H::Int,
        blobs::Vector{NTuple{3,Float64}};
        bg::Float32 = 0.5f0, contrast::Float32 = 0.4f0)
    img = fill(bg, H, W)
    for (cx, cy, σ) in blobs
        rad = ceil(Int, 4σ)
        x0 = max(1, floor(Int, cx - rad + 0.5)); x1 = min(W, ceil(Int, cx + rad + 0.5))
        y0 = max(1, floor(Int, cy - rad + 0.5)); y1 = min(H, ceil(Int, cy + rad + 0.5))
        two_σ² = 2 * σ * σ
        @inbounds for y in y0:y1, x in x0:x1
            dx = (x - 0.5) - cx; dy = (y - 0.5) - cy
            img[y, x] -= contrast * Float32(exp(-(dx*dx + dy*dy) / two_σ²))
        end
    end
    img
end

# -----------------------------------------------------------------------------
# Ground-truth pairing
# -----------------------------------------------------------------------------

"""
    pair_to_gt(detected_xy::Vector{Tuple{Float64,Float64,Float64}},
               gt::Vector{NTuple{3,Float64}};
               max_dist=5.0)

For each ground-truth blob, find the nearest detection within `max_dist`.
Returns matched pairs `(gt_idx, det_idx, dist, gt_σ, det_σ)` and the set of
unmatched GT indices.
"""
function pair_to_gt(detected, gt; max_dist=5.0)
    matched = NTuple{5,Float64}[]
    unmatched_gt = Int[]
    for (gi, (gx, gy, gσ)) in enumerate(gt)
        best_d = Inf; best_di = 0; best_σ = 0.0
        for (di, (dx, dy, dσ)) in enumerate(detected)
            d = sqrt((dx-gx)^2 + (dy-gy)^2)
            if d < best_d
                best_d = d; best_di = di; best_σ = dσ
            end
        end
        if best_d <= max_dist
            push!(matched, (Float64(gi), Float64(best_di), best_d, gσ, best_σ))
        else
            push!(unmatched_gt, gi)
        end
    end
    matched, unmatched_gt
end

# -----------------------------------------------------------------------------
# Run experiment
# -----------------------------------------------------------------------------

# Generate a regular grid of blobs at multiple σ values
W, H = 1024, 1024
blobs = NTuple{3,Float64}[]
for σ in (1.5, 2.5, 4.0, 6.4, 10.0)
    spacing = max(60.0, 8σ)   # avoid blob overlap
    for cy in spacing:spacing:H-spacing
        for cx in spacing:spacing:W-spacing
            push!(blobs, (cx, cy, σ))
        end
    end
end
println("Generated $(length(blobs)) synthetic blobs at σ ∈ {1.5, 2.5, 4.0, 6.4, 10.0}")

img_f32 = render_synthetic_blobs(W, H, blobs)
img_u8  = round.(UInt8, clamp.(img_f32 .* 255f0, 0f0, 255f0))
img_gray = Gray{Float32}.(img_f32)

# --- CPU: BlobBoards.detect_blobs (CPU()) ---
println("\n=== CPU detect_blobs ===")
@time cpu_dets = BlobBoards.detect_blobs(img_gray;
    device = CUDABackend(),
    peak_threshold = 0.01,
    refine_detections = true,
    max_detections = 0)
cpu_xy = [(Float64(ustrip(centroid(d.frame)[1])),
           Float64(ustrip(centroid(d.frame)[2])),
           Float64(ustrip(scale(d.frame))))
          for d in cpu_dets]
println("  Detections: $(length(cpu_xy))")

# --- GPU: VKS identity warp + detect (no IMAS chain — flat detection) ---
println("\n=== GPU VKS detect ===")
h = AsiftGpu.vks_init(W, H;
    intensity_thr = 0.01f0, edge_thr = 10.0f0,
    seed_sigma = 1.6f0, blur_level = 0.5f0,
    n_scales = 3, n_octaves = 0, upsample = true, use_2d_nms = false)
AsiftGpu.vks_set_warp!(h, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5f0)
AsiftGpu.vks_set_preblur!(h, 0.0, 0.0, 1.0)
buf = vec(permutedims(img_u8, (2, 1)))  # row-major byte stream
@time vks_feats = AsiftGpu.vks_detect!(h, buf, W, H)
gpu_xy = [(Float64(f.x) + 0.5, Float64(f.y) + 0.5, Float64(f.sigma))
          for f in vks_feats]
AsiftGpu.vks_destroy!(h)
println("  Detections: $(length(gpu_xy))")

# --- Compare against ground truth ---
println("\n=== Ground-truth comparison ===")
cpu_matched, cpu_missed = pair_to_gt(cpu_xy, blobs; max_dist=5.0)
gpu_matched, gpu_missed = pair_to_gt(gpu_xy, blobs; max_dist=5.0)

function summarize(name, matched, n_gt)
    pos_err = [m[3] for m in matched]
    σ_ratio = [m[5] / m[4] for m in matched]
    println("$name: matched=$(length(matched))/$n_gt (recall=$(round(length(matched)/n_gt; digits=3)))")
    println("  pos err  median=$(round(Statistics.median(pos_err); digits=3))  p95=$(round(Statistics.quantile(pos_err, 0.95); digits=3))  max=$(round(maximum(pos_err); digits=3))")
    println("  σ ratio  median=$(round(Statistics.median(σ_ratio); digits=3))  p05..p95=($(round(Statistics.quantile(σ_ratio, 0.05); digits=3)) .. $(round(Statistics.quantile(σ_ratio, 0.95); digits=3)))")
end

summarize("CPU BB", cpu_matched, length(blobs))
summarize("GPU VKS", gpu_matched, length(blobs))

# Per-σ breakdown
println("\n--- Per-σ recall and median position error ---")
unique_σ = unique(b[3] for b in blobs)
for σ in unique_σ
    σ_blobs = [(i, b) for (i, b) in enumerate(blobs) if b[3] ≈ σ]
    σ_indices = Set(i for (i, _) in σ_blobs)
    cpu_σ = filter(m -> Int(m[1]) in σ_indices, cpu_matched)
    gpu_σ = filter(m -> Int(m[1]) in σ_indices, gpu_matched)
    n_σ = length(σ_blobs)
    cpu_recall = length(cpu_σ) / n_σ
    gpu_recall = length(gpu_σ) / n_σ
    cpu_pe = isempty(cpu_σ) ? NaN : Statistics.median(m[3] for m in cpu_σ)
    gpu_pe = isempty(gpu_σ) ? NaN : Statistics.median(m[3] for m in gpu_σ)
    println("  σ=$(round(σ; digits=2)) (n=$n_σ):  CPU recall=$(round(cpu_recall; digits=2)) pe=$(round(cpu_pe; digits=3))   GPU recall=$(round(gpu_recall; digits=2)) pe=$(round(gpu_pe; digits=3))")
end

# Return values for further inspection
(cpu_matched, gpu_matched, blobs, cpu_xy, gpu_xy)
