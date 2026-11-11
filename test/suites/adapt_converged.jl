# The adaptation IRLS records, per blob, whether it CONVERGED (the owner's §14
# ruling for BlobBoards, 2026-11-11). A blob that reaches `max_irls_iter`
# without meeting `rtol` is still returned (validity is not changed), but it is
# flagged `converged = false` and counted, never silently valid. A blob whose
# next step the shape gate refused (reverted to the previous iterate) stopped,
# but it did not converge either, so it is flagged the same way.
using Test
using BlobBoards
using BlobBoards.KernelAbstractions: CPU
using Random
using Colors: Gray

# Dark Gaussian blobs on a light ground (the parity suite's field, smaller).
function _converged_blob_field(rng; H = 240, W = 320, n = 30)
    img = fill(0.85f0, H, W)
    for _ in 1:n
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
    return Gray.(clamp!(img, 0.0f0, 1.0f0))
end

@testset "adapt: per-blob convergence flag" begin
    img = _converged_blob_field(MersenneTwister(7))
    dets = BlobBoards.detect_blobs(img; device = CPU())
    @test length(dets) >= 15

    full = BlobBoards.adapt(dets, img; config = BlobBoards.AdaptConfig(), device = CPU())
    @test !isempty(full)
    @test all(f -> f.converged isa Bool, full)
    # At the default budget (20) some blobs converge; most of this field's do
    # not meet rtol = 1e-6 yet (21 of 30 capped on 2026-11-11, one stopped by
    # the shape gate). With a budget of 100 none is capped.
    @test BlobBoards.count_unconverged(full) < length(full)
    long = BlobBoards.adapt(dets, img; config = BlobBoards.AdaptConfig(max_irls_iter = 100), device = CPU())
    @test [f.id for f in long] == [f.id for f in full]
    @test BlobBoards.count_unconverged(long) < BlobBoards.count_unconverged(full)
    @test count(f -> f.converged, long) >= length(long) - 2

    # One iteration cannot meet rtol = 1e-6 from the seed: every blob still
    # live at the cap is flagged and counted, not reported as converged.
    capped = BlobBoards.adapt(dets, img; config = BlobBoards.AdaptConfig(max_irls_iter = 1), device = CPU())
    @test !isempty(capped)
    @test BlobBoards.count_unconverged(capped) > 0
    @test BlobBoards.count_unconverged(capped) > BlobBoards.count_unconverged(full)
    @test BlobBoards.count_unconverged(capped) == count(f -> !f.converged, capped)

    # The flag changes no validity and no geometry: the frames of a run are
    # the frames of the same run without the flag (ids and centroids).
    again = BlobBoards.adapt(dets, img; config = BlobBoards.AdaptConfig(), device = CPU())
    @test [f.id for f in again] == [f.id for f in full]
    @test [f.converged for f in again] == [f.converged for f in full]

    @test BlobBoards.count_unconverged(BlobBoards.BlobDetection{Float64}[]) == 0
end
