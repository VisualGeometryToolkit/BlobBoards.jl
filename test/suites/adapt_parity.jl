# CPU ↔ GPU adaptation parity — the gate on the two realizations of the
# Student-t/uniform mixture (the cpu = false workgroup kernels vs the threaded
# host path). They share the scalar core (adapt_ka.jl "Shared scalar core");
# this testset pins the two ORCHESTRATIONS together at the level the
# architecture note licenses: frame-level with tolerances, never bitwise —
# CUDA's atomic pixel-slot assignment jitters even GPU-vs-GPU runs.
#
# `adapt` returns VALID frames only, id-indexed (`_finalize_frames` drops
# failed fits) — so parity is judged by MATCHING IDS: a frame present on one
# side only is a validity disagreement (allowed for a few knife-edge blobs),
# and centroids are compared on the common id set.
# Skips without a functional CUDA device (the divergence risk lives on GPU
# machines, which is where this runs).
using Test
using BlobBoards
using BlobBoards.KernelAbstractions: CPU
using CUDA: CUDA, CUDABackend
using VisualGeometryCore: centroid, feature_frame
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
        # Validity may disagree for a few knife-edge blobs (the paths clamp and
        # iterate in different orders at the margins). Agreeing frames differ
        # by accumulation ORDER only (reduction trees, atomics): measured max
        # 0.188 px on this field at the gate's introduction (2026-08-31) —
        # the bound is 2.5× that, and tightening it is a ratchet.
        @test oneside <= max(2, round(Int, 0.05n))
        @test maximum(dc) < 0.5
    end
else
    @info "adapt parity: skipped (no functional CUDA device)"
end
