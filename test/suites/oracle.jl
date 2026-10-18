# =============================================================================
# The serial-reference oracle vs the matching engine.
#
# `_run_matching_phase!` / `_claim_boards` promise results identical to "a
# fully serial eager loop"; `BlobBoards._find_boards_serial` IS that loop,
# and this test asserts the equality the engine's comments have always
# claimed but nothing checked. The scene enters at the PRE-EXTRACTED stage
# (no detect/adapt/describe): two generated board bundles placed into a fake
# image by known similarities, with synthetic unit-norm descriptors —
# matched pairs identical (sim 1.0), distractors random (|sim| ≈ 0.09 ≪
# 0.6 gate). CPU-only, single fixed RNG, so any failure is an engine/oracle
# divergence, not data noise.
# =============================================================================

using Random: Random, MersenneTwister
using LinearAlgebra: normalize
using VisualGeometryToolkitCore: is_map_inlier

@testset "serial oracle ≡ engine" begin
    VGF = BlobBoards.VisualGeometryToolkitFeatures
    VGC = BlobBoards.VisualGeometryToolkitCore

    rng = MersenneTwister(0xB10B)
    randdesc() = Float32.(normalize(randn(rng, Float32, 128)))

    # Real bundles — the geometry authority (cutoff, world Rect, world blobs,
    # polarity) — with synthetic appearance.
    bundles = Dict(
        "aaa" => blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x0001),
        "bbb" => blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x0002),
    )
    # Disjoint similarity placements inside a notional ~2000×1400 px image.
    place = Dict(
        "aaa" => (s = 4.0, t = (60.0, 60.0)),
        "bbb" => (s = 4.0, t = (1000.0, 60.0))
    )

    board_features = Dict{String, Any}()
    img_discs = BlobBoards.Disc{Float64}[]
    img_desc = Vector{Float32}[]
    for uid in ("aaa", "bbb")
        wd = blobs(bundles[uid], :world)          # bare mm by the :world contract
        D = reduce(vcat, (randdesc()' for _ in wd))
        board_features[uid] = VGF.described_features(
            [VGF.Detection(d, (; id = j)) for (j, d) in enumerate(wd)], D
        )
        (; s, t) = place[uid]
        for (j, d) in enumerate(wd)
            c = VGC.centroid(d)
            push!(img_discs, BlobBoards.Disc(s * c[1] + t[1], s * c[2] + t[2], s * d.σ))
            push!(img_desc, D[j, :])
        end
    end
    # Clutter: descriptor-random distractors sprinkled over the image.
    for _ in 1:60
        push!(
            img_discs, BlobBoards.Disc(
                2000.0 * rand(rng), 1400.0 * rand(rng),
                2.0 + 6.0 * rand(rng)
            )
        )
        push!(img_desc, randdesc())
    end
    feats = VGF.described_features(
        [VGF.Detection(d, (; id = j)) for (j, d) in enumerate(img_discs)],
        reduce(vcat, (v' for v in img_desc))
    )

    config = FindBoardsConfig()                   # homography path, all defaults
    engine = find_boards(board_features, feats; board_bundles = bundles, config)
    oracle = BlobBoards._find_boards_serial(
        board_features, feats;
        board_bundles = bundles, config
    )

    # Both find both boards…
    @test sort!([r[1] for r in engine]) == ["aaa", "bbb"]
    @test sort!([r[1] for r in oracle]) == ["aaa", "bbb"]
    # …and the engine matches its own contract. DECISIONS are exact: same
    # boards, same inlier support, same correspondence sets — any drift there
    # means an engine mechanism (veto, cache, round structure) leaked into
    # results. The CONTINUOUS values (quality, model) are tight-≈, not ==,
    # and cannot be ==: the engine fits through phase-sized per-thread
    # workspaces whose `set_n!` views give BLAS different strides than the
    # oracle's exact-size allocations, so the reduction order — and the last
    # ulp — differs (measured |Δq|/q ≈ 1e-13 on this scene). rtol 1e-9 is
    # ~4 orders looser than that floor and ~10 orders tighter than any real
    # divergence the corpus A/B surfaced.
    eng = Dict(r[1] => r for r in engine)
    ser = Dict(r[1] => r for r in oracle)
    for uid in ("aaa", "bbb")
        (_, m1, q1, s1) = eng[uid]
        (_, m2, q2, s2) = ser[uid]
        @test count(is_map_inlier, s1.weights) == count(is_map_inlier, s2.weights)
        @test isapprox(q1, q2; rtol = 1.0e-9)
        @test isapprox(m1, m2; rtol = 1.0e-9)
        @test sort!(collect(zip(s1.src_id, s1.det_id))) ==
            sort!(collect(zip(s2.src_id, s2.det_id)))
    end

    # The retired features-first orders bind no method: the gallery leads the
    # pre-extracted features, the camera leads a single board's match, and the
    # model leads the bundle it outlines.
    @test !applicable(find_boards, feats, board_features)
    @test !applicable(BlobBoards._find_boards_serial, feats, board_features)
    @test !applicable(BlobBoards.find_board, feats, board_features["aaa"], nothing)
    @test !applicable(BlobBoards.board_quad, nothing, bundles["aaa"], eng["aaa"][2])
    @test applicable(BlobBoards.find_board, nothing, board_features["aaa"], feats)
    @test applicable(BlobBoards.board_quad, nothing, eng["aaa"][2], bundles["aaa"])

    # Seeded fits: identical calls give identical results, independent of
    # ambient RNG state — RVG (≥ 0.15.7) routes every sampler draw through
    # the per-board stream `fit_rng(config.ransac_seed, uid)`, so nothing
    # in the fit consults the task-local RNG. EXACT equality on quality is
    # deliberate: same inputs + same streams + same workspace sizes must be
    # the same bits.
    Random.seed!(0x0BAD5EED)
    engine_repeat = find_boards(board_features, feats; board_bundles = bundles, config)
    fingerprint(rs) = sort!(
        [(r[1], count(is_map_inlier, r[4].weights), r[3]) for r in rs]; by = first
    )
    # This one IS `==` on a Float64 quality, unlike the engine-vs-oracle
    # comparison above, because it is the engine against ITSELF: one
    # implementation twice, not two routes. It holds on an invariant worth
    # keeping true — every per-thread workspace is the same size
    # (`Threads.maxthreadid()`, phase-sized) and `set_n!` only TRIMS a view of
    # it, so which thread `:dynamic` scheduling hands a board to cannot change
    # a stride, and therefore cannot change a reduction order. Size a workspace
    # per board, or grow one on demand, and this test becomes a flake
    # generator — the rtol above is what that looks like when the strides
    # genuinely do differ.
    @test fingerprint(engine_repeat) == fingerprint(engine)
end
