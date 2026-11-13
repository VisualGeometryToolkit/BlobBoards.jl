# The adaptation IRLS records, per blob, how it ended (the owner's §14 rulings
# for BlobBoards, 2026-11-11), in Core's termination record:
#   converged at rtol             ReturnCode.Success        StopRule IncrementSize
#   capped at max_irls_iter       ReturnCode.MaxIters       StopRule IterationCap
#   step refused by the shape gate ReturnCode.Failure        StopRule ShapeGate
#     (refused at the first step from the seed: ReturnCode.InitialFailure)
# with the refusal's reason in `rejection` (:not_peak or :aspect_ratio; :none
# otherwise). Capped blobs are UNCONVERGED (`count_unconverged`); refused ones
# are REJECTED (`count_rejected`), a separate terminal outcome. Neither changes
# validity: both are returned, flagged and counted, never silently valid.
using Test
using BlobBoards
using BlobBoards.KernelAbstractions: CPU
using Random
using Colors: Gray
import VisualGeometryToolkitCore

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

@testset "adapt: per-blob outcome (converged, capped, rejected)" begin
    img = _converged_blob_field(MersenneTwister(7))
    dets = BlobBoards.detect_blobs(img; device = CPU())
    @test length(dets) >= 15
    RC = VisualGeometryToolkitCore.ReturnCode
    retcodes = (RC.Success, RC.MaxIters, RC.Failure, RC.InitialFailure)
    rules = Dict(RC.Success => VisualGeometryToolkitCore.IncrementSize, RC.MaxIters => VisualGeometryToolkitCore.IterationCap,
                 RC.Failure => VisualGeometryToolkitCore.ShapeGate, RC.InitialFailure => VisualGeometryToolkitCore.ShapeGate)

    function consistent(frames)
        all(frames) do f
            f.retcode in retcodes && f.stop_rule == rules[f.retcode] &&
                (f.rejection === :none) == !(f.retcode in (RC.Failure, RC.InitialFailure)) &&
                f.rejection in (:none, :not_peak, :aspect_ratio)
        end
    end
    total(frames) = count(f -> VisualGeometryToolkitCore.successful_retcode(f.retcode), frames) + BlobBoards.count_unconverged(frames) + BlobBoards.count_rejected(frames)

    # The default budget is 100 (the owner's ruling, 2026-11-11): at the old
    # 20, 21 of this field's 30 blobs were still moving at the cap (rtol 1e-6)
    # and one step was refused by the shape gate; at 100 none is capped.
    @test BlobBoards.AdaptConfig().max_irls_iter == 100
    full = BlobBoards.adapt(dets, img; config = BlobBoards.AdaptConfig(), device = CPU())
    @test !isempty(full) && consistent(full) && total(full) == length(full)
    @test BlobBoards.count_unconverged(full) == 0
    @test BlobBoards.count_rejected(full) >= 1                       # the field's shape-gate refusal
    @test BlobBoards.count_rejected(full) ==
        BlobBoards.count_rejected(full, :not_peak) + BlobBoards.count_rejected(full, :aspect_ratio)

    old = BlobBoards.adapt(dets, img; config = BlobBoards.AdaptConfig(max_irls_iter = 20), device = CPU())
    @test [f.id for f in old] == [f.id for f in full]
    @test consistent(old) && total(old) == length(old)
    @test BlobBoards.count_unconverged(old) > 0                      # capped at 20, not rejected
    @test BlobBoards.count_rejected(old) == BlobBoards.count_rejected(full)

    # One iteration cannot meet rtol = 1e-6 from the seed: every live blob is
    # capped (unconverged); a step refused there is an InitialFailure.
    one_ = BlobBoards.adapt(dets, img; config = BlobBoards.AdaptConfig(max_irls_iter = 1), device = CPU())
    @test consistent(one_)
    @test BlobBoards.count_unconverged(one_) > BlobBoards.count_unconverged(old)
    @test all(f -> f.retcode != RC.Failure, one_)                    # no refusal after the first step exists

    # The record changes no validity and no geometry: the same run twice.
    again = BlobBoards.adapt(dets, img; config = BlobBoards.AdaptConfig(), device = CPU())
    @test [(f.id, f.frame, f.retcode, f.rejection) for f in again] == [(f.id, f.frame, f.retcode, f.rejection) for f in full]

    @test BlobBoards.count_unconverged(BlobBoards.BlobDetection{Float64}[]) == 0
    @test BlobBoards.count_rejected(BlobBoards.BlobDetection{Float64}[]) == 0
    @test_throws ArgumentError BlobBoards.count_rejected(full, :aspect)
end

@testset "adapt: the shape gate's reason" begin
    # C = [θ₄ θ₅; θ₅ θ₆], the paraboloid's quadratic part; max_aspect_ratio 10.
    @test BlobBoards._shape_refusal(-1.0, 0.0, -1.0, 10.0) == :none          # a round peak
    @test BlobBoards._shape_refusal(-1.0, 0.0, 1.0, 10.0) == :not_peak       # a saddle
    @test BlobBoards._shape_refusal(1.0, 0.0, 1.0, 10.0) == :not_peak        # a valley
    @test BlobBoards._shape_refusal(-1.0, 0.0, -0.001, 10.0) == :aspect_ratio  # axis ratio √1000 ≈ 31.6 > 10
    @test BlobBoards._shape_refusal(-1.0, 0.0, -0.02, 10.0) == :none           # axis ratio √50 ≈ 7.1
end

@testset "adapt: a blob of 10–19 samples is judged on its isotropic fit" begin
    # Fewer than `_MIN_POINTS_ANISOTROPIC` (20) samples fit the isotropic paraboloid
    # z = a₀ + bᵀx + c·|x|² (4 parameters, Core's Paraboloid3D{T,4}: C = c·I); 20 or more
    # the anisotropic one (6). Two dark Gaussian blobs (σ = 1.2 px) on a flat ground,
    # each darkening only the pixels within `ρ` of its centre: the 13 pixels of
    # |x|² ≤ 4.2 (the isotropic model) and the 21 of |x|² ≤ 5.3 (the anisotropic
    # one); every other pixel of the scan window is the background (no sample). The
    # data are exactly log-quadratic, so both models recover the blob: the shape
    # σ·I, the centre, and a converged, unrefused record. The sites are 1.5 px
    # discs (the seed, which the fit leaves). The image is Float64, so the data
    # are log-quadratic to rounding, and every comparison is held at the IRLS's own
    # stopping tolerance, `rtol` (1e-6, the relative parameter step).
    σ, a, ground = 1.2, 0.4, 0.85
    blobs = [(60.5, 40.5, 4.2, 13), (120.5, 40.5, 5.3, 21)]          # centre (pixel centres), ρ², samples
    img = fill(ground, 80, 180)
    for (cx, cy, ρ2, _) in blobs, y in axes(img, 1), x in axes(img, 2)
        r2 = (x - 0.5 - cx)^2 + (y - 0.5 - cy)^2                       # pixel x's centre is x − ½
        r2 <= ρ2 && (img[y, x] = ground - a * exp(-r2 / (2σ^2)))
    end
    for (cx, cy, ρ2, n) in blobs                                       # the sample counts the models are chosen by
        @test count(I -> (I[2] - 0.5 - cx)^2 + (I[1] - 0.5 - cy)^2 <= ρ2, CartesianIndices(img)) == n
    end
    sites = [BlobBoards.Disc(BlobBoards.Point2(cx, cy), 1.5) for (cx, cy, _, _) in blobs]
    cfg = BlobBoards.AdaptConfig()
    frames = BlobBoards.adapt(sites, Gray.(img); config = cfg, device = CPU())
    @test [f.id for f in frames] == [1, 2]
    for f in frames
        @test f.rejection === :none
        @test VisualGeometryToolkitCore.successful_retcode(f.retcode)
        @test f.frame.S ≈ σ * [1 0; 0 1] rtol = cfg.rtol
        @test f.frame.center ≈ BlobBoards.Point2(blobs[f.id][1], blobs[f.id][2]) rtol = cfg.rtol
    end
    @test BlobBoards.count_rejected(frames) == 0
end
