# CPU ↔ GPU adaptation parity — the gate on the one KA implementation of the
# Student-t/uniform mixture run on its two kinds of backend: a 256-item
# workgroup per blob on the GPU, one work item per blob on the CPU
# (adapt_ka.jl "The per-blob kernels"). This testset pins the two together at
# the level the architecture note licenses: frame-level with tolerances, never
# bitwise — CUDA's atomic pixel-slot assignment jitters even GPU-vs-GPU runs.
#
# `adapt` returns VALID frames only, id-indexed (`_finalize_frames` drops
# failed fits) — so parity is judged by MATCHING IDS: a frame present on one
# side only is a validity disagreement (none is allowed),
# and centroids are compared on the common id set.
# Skips without a functional CUDA device (the divergence risk lives on GPU
# machines, which is where this runs).
using Test
using BlobBoards
using BlobBoards.KernelAbstractions: CPU
using CUDA: CUDA, CUDABackend
using VisualGeometryToolkitCore: centroid, feature_frame
using Random
using Colors: Gray

if CUDA.functional()
    @testset "adapt CPU ≡ GPU (frame level)" begin
        # Synthetic blob field: dark Gaussian blobs on a light ground, the
        # detector's native polarity — enough blobs for a statistical claim,
        # small enough to keep the CPU run quick.
        rng = MersenneTwister(42)
        H, W = 480, 640
        img = fill(0.85f0, H, W)
        for _ in 1:60
            σ = 1.5 + 4.0 * rand(rng)
            cx = 20 + (W - 40) * rand(rng)
            cy = 20 + (H - 40) * rand(rng)
            a = 0.3 + 0.4 * rand(rng)
            x0, x1 = max(1, floor(Int, cx - 4σ)), min(W, ceil(Int, cx + 4σ))
            y0, y1 = max(1, floor(Int, cy - 4σ)), min(H, ceil(Int, cy + 4σ))
            for y in y0:y1, x in x0:x1
                img[y, x] -= Float32(a * exp(-((x - cx)^2 + (y - cy)^2) / (2σ^2)))
            end
        end
        clamp!(img, 0.0f0, 1.0f0)
        img_g = Gray.(img)
        dets = BlobBoards.detect_blobs(img_g; device = CPU())
        @test length(dets) >= 30
        cfg = BlobBoards.AdaptConfig()
        cpu = BlobBoards.adapt(dets, img_g; config = cfg, device = CPU())
        gpu = BlobBoards.adapt(dets, img_g; config = cfg, device = CUDABackend())
        by_id(v) = Dict(f.id => f for f in v)
        c, g = by_id(cpu), by_id(gpu)
        common = intersect(keys(c), keys(g))
        oneside = length(symdiff(keys(c), keys(g)))
        dc = [
            hypot((centroid(feature_frame(c[id])) .- centroid(feature_frame(g[id])))...)
                for id in common
        ]
        n = length(dc)
        @test n >= 25
        # One code on both backends (one ν step with Core's ascent safeguard,
        # one lgamma, one 6×6 solve: the owner's ruling of 2026-11-11), so the
        # frames differ by accumulation ORDER only (reduction trees, atomics)
        # and the validity sets agree. The baseline: 8.3e-9 px on this field, on the 2026-11-11
        # B200 (0.188 px when the gate was introduced, 2026-08-31, before the
        # safeguard and the one lgamma/solve); the bound is rounding level, and
        # tightening it is a ratchet.
        @test oneside == 0
        @test maximum(dc) < 1.0e-6
    end
else
    @info "adapt parity: skipped (no functional CUDA device)"
end
