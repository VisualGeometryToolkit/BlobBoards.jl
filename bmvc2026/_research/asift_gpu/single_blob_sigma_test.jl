# Single-blob σ sweep: place ONE Gaussian dark blob at the center of each image,
# vary its σ, and measure each detector's reported σ for the strongest detection.
# Eliminates the multi-detection-per-blob ambiguity that makes population-level
# σ comparisons unreliable.

using BlobBoards
using Colors: Gray
using FixedPointNumbers: N0f8
using CUDA
using VisualGeometryToolkitCore: ustrip, centroid, scale
import Statistics

include(joinpath(@__DIR__, "driver.jl"))
using .AsiftGpu

# Render one blob at COLMAP center (cx, cy) into a (H, W) image.
function render_one_blob(W::Int, H::Int, cx::Float64, cy::Float64, σ::Float64;
        bg::Float32 = 0.5f0, contrast::Float32 = 0.4f0)
    img = fill(bg, H, W)
    rad = ceil(Int, 5σ)
    x0 = max(1, floor(Int, cx - rad + 0.5)); x1 = min(W, ceil(Int, cx + rad + 0.5))
    y0 = max(1, floor(Int, cy - rad + 0.5)); y1 = min(H, ceil(Int, cy + rad + 0.5))
    two_σ² = 2 * σ * σ
    @inbounds for y in y0:y1, x in x0:x1
        dx = (x - 0.5) - cx; dy = (y - 0.5) - cy
        img[y, x] -= contrast * Float32(exp(-(dx*dx + dy*dy) / two_σ²))
    end
    img
end

# Pick the detection with maximum |response| within R px of (cx, cy).
# Returns (x, y, σ, response) or nothing.
function strongest_detection(dets, cx, cy, R)
    best = nothing; best_r = -Inf
    for (x, y, σ, r) in dets
        if sqrt((x-cx)^2 + (y-cy)^2) <= R && abs(r) > best_r
            best = (x, y, σ, r); best_r = abs(r)
        end
    end
    best
end

# BlobBoards' detector (its one KernelAbstractions path, here on CUDA) → list of (x, y, σ, response) in COLMAP
function detect_blobboards(img_f32)
    img_gray = Gray{Float32}.(img_f32)
    ds = BlobBoards.detect_blobs(img_gray;
        device = CUDABackend(),
        peak_threshold = 0.01,
        refine_detections = true, max_detections = 0)
    [(Float64(ustrip(centroid(d.frame)[1])),
      Float64(ustrip(centroid(d.frame)[2])),
      Float64(ustrip(scale(d.frame))),
      Float64(d.metadata.response)) for d in ds]
end

# The VKS reference detector (the AsiftGpu C library): identity warp, no σ_aa, raw VKS detect → COLMAP
function detect_vks(h, img_f32, W, H)
    img_u8 = round.(UInt8, clamp.(img_f32 .* 255f0, 0f0, 255f0))
    buf = vec(permutedims(img_u8, (2, 1)))
    feats = AsiftGpu.vks_detect!(h, buf, W, H)
    [(Float64(f.x) + 0.5, Float64(f.y) + 0.5,
      Float64(f.sigma), Float64(f.intensity)) for f in feats]
end

# Sweep σ values
σ_values = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.4, 8.0, 10.0, 12.0, 16.0]
W, H = 256, 256
cx, cy = 127.5, 127.5   # COLMAP center of image

h = AsiftGpu.vks_init(W, H;
    intensity_thr = 0.005f0, edge_thr = 10.0f0,
    seed_sigma = 1.6f0, blur_level = 0.5f0,
    n_scales = 3, n_octaves = 0, upsample = true, use_2d_nms = false)
AsiftGpu.vks_set_warp!(h, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5f0)
AsiftGpu.vks_set_preblur!(h, 0.0, 0.0, 1.0)

println("σ_true   N_bb   N_vks     BB σ         VKS σ        BB/true   VKS/true   pos_bb   pos_vks")
println("-------- -----  -----   ---------    ---------   ---------  ---------   -------  -------")
results = []
for σ in σ_values
    img = render_one_blob(W, H, cx, cy, σ)
    bb_dets = detect_blobboards(img)
    vks_dets = detect_vks(h, img, W, H)
    R = max(3.0, 2σ)
    bb_best = strongest_detection(bb_dets, cx, cy, R)
    vks_best = strongest_detection(vks_dets, cx, cy, R)
    σ_bb = isnothing(bb_best) ? NaN : bb_best[3]
    σ_vks = isnothing(vks_best) ? NaN : vks_best[3]
    pos_bb = isnothing(bb_best) ? NaN : sqrt((bb_best[1]-cx)^2 + (bb_best[2]-cy)^2)
    pos_vks = isnothing(vks_best) ? NaN : sqrt((vks_best[1]-cx)^2 + (vks_best[2]-cy)^2)
    push!(results, (σ, length(bb_dets), length(vks_dets), σ_bb, σ_vks, pos_bb, pos_vks))
    println(rpad(string(σ), 8), " ",
        lpad(string(length(bb_dets)), 5), "  ",
        lpad(string(length(vks_dets)), 5), "   ",
        lpad(isnan(σ_bb) ? "miss" : string(round(σ_bb; digits=3)), 9), "    ",
        lpad(isnan(σ_vks) ? "miss" : string(round(σ_vks; digits=3)), 9), "    ",
        lpad(isnan(σ_bb) ? "—" : string(round(σ_bb/σ; digits=3)), 8), "   ",
        lpad(isnan(σ_vks) ? "—" : string(round(σ_vks/σ; digits=3)), 8), "    ",
        lpad(isnan(pos_bb) ? "—" : string(round(pos_bb; digits=3)), 5), "    ",
        lpad(isnan(pos_vks) ? "—" : string(round(pos_vks; digits=3)), 5))
end

AsiftGpu.vks_destroy!(h)
println("\nExpected DoG-peak σ for a Gaussian-std-σ_true blob: σ_true · 2^(1/(2·N)) ≈ σ_true · 1.122 (N=3)")
results
