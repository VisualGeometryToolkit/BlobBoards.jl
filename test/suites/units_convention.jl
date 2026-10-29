# =============================================================================
# Units convention enforcement: no unaudited bare ustrip
# =============================================================================
#
# Bare `ustrip(x)` hands back base-unit magnitudes (FlexUnits: METRES for a
# length, px/m for a density) with no dimension check — the failure class behind
# the 2026-07-29 incident where a bare strip of a DENSITY yielded px/mm (under
# the old (mm, px) basis) and boards silently rasterized at ~47 dpi instead of
# 1200 (see `_render_dpi` in io.jl).
#
# Convention (matches VisualGeometryToolkitIO): every dimension-boundary crossing
# uses the target-unit form `ustrip(u, x)` — frames, Size2 and Rects included.
# Bare strips are allowed only for audited cases — defensive identity-or-strip
# over bare-by-contract or px data (:world rects and blobs, Extrinsics3.t,
# adapted frames, shape matrices), and "magnitudes in caller-chosen units"
# export loops — each carrying an `(audited)` comment at the call site.
#
# The per-file counts below are golden: exceeding one fails this test and
# forces a conscious re-audit; dropping below just means tightening the
# number here.

@testset "units convention: bare ustrip is audited" begin
    pkg_root = dirname(dirname(@__DIR__))        # test/suites/ → the package root

    allowed = Dict(
        # the _resolve_pose_ambiguity line: board-side blobs are unitless mm by
        # the :world contract, so the bare strip is an identity there
        # (re-audited 2026-08-08; the FlexUnits migration retired the other
        # pipeline strip and every whole-Disc strip in pattern/adapt/coordinates).
        "src/pipeline.jl" => 1,
        # the :world rect identity (the bare-or-strip helpers are
        # VisualGeometryToolkitCore's `strip_or_bare` since 2026-10-15)
        "ext/BlobBoardsMakieExt.jl" => 1,
        "ext/BlobBoardsPythonExt.jl" => 9, # export loops over values already stripped in the requested unit
        # Training pipeline. Audited 2026-08-19: `shape_matrix` strips and
        # defensive centroid strips over adapted `EllipticFrame{Float64}`s,
        # which are bare px by contract off the ASIFT path — `ustrip(px, ...)`
        # throws on those. Each call site carries its own `(audited)` note.
        "src/training.jl" => 7,
    )

    # A call is target-unit if the first argument is a unit expression; the
    # scan is line-based and conservative (multi-line ustrip calls would be
    # flagged — keep them single-line).
    target_re = r"ustrip\.?\(\s*(px|mm|inch|dpi|pt|u\b|unit)"
    call_re = r"ustrip\.?\("

    counts = Dict{String, Int}()
    for (root, _, files) in walkdir(joinpath(pkg_root, "src")), f in files
        endswith(f, ".jl") || continue
        rel = relpath(joinpath(root, f), pkg_root)
        for line in eachline(joinpath(root, f))
            occursin(call_re, line) && !occursin(target_re, line) &&
                (counts[rel] = get(counts, rel, 0) + 1)
        end
    end
    for (root, _, files) in walkdir(joinpath(pkg_root, "ext")), f in files
        endswith(f, ".jl") || continue
        rel = relpath(joinpath(root, f), pkg_root)
        for line in eachline(joinpath(root, f))
            occursin(call_re, line) && !occursin(target_re, line) &&
                (counts[rel] = get(counts, rel, 0) + 1)
        end
    end

    for (file, n) in counts
        @test get(allowed, file, 0) >= n
        get(allowed, file, 0) >= n ||
            @error "unaudited bare ustrip introduced" file found = n allowed = get(allowed, file, 0)
    end
end

# The arithmetic kept by hand, not FlexUnits' native form, because the native
# form moves a stored value (FlexUnits holds a length in metres, so a product or
# difference through it lands an ulp away). Each case shows the native form
# disagreeing with the stored arithmetic on a value the boards in circulation
# carry; the units audit (2026-10-15) keeps these and nothing else by hand.
@testset "units: hand arithmetic that stays bit-identical to stored boards" begin
    BB = BlobBoards
    # A4 with a 10 mm margin: the derived board is 190 × 277 mm exactly;
    # FlexUnits' difference is 276.99999999999994 mm
    cs, mg = BB.Size2(width = 210.0mm, height = 297.0mm), BB.Size2(width = 10.0mm, height = 10.0mm)
    @test BB.ustrip(mm, BB._board_from_margin(cs, mg).height) === 277.0
    @test BB.ustrip(mm, cs.height - 2 * mg.height) !== 277.0
    @test BB._margin_of(cs, BB._board_from_margin(cs, mg)) == mg
    # a physical length in px: mm × px/mm, the arithmetic the sheets were laid out
    # with (tick spacings, padding, grid counts). Through FlexUnits 0.6.15 the
    # native product differed on both values below (at 135.89 mm and 300 dpi
    # enough to move a floor at a whole pixel, 1605); 0.6.16 applies a rational
    # scale with one rounding and agrees on them, and the hand form stays the
    # arithmetic the stored sheets were made with
    q, d = 0.5mm, 300dpi
    @test BB._length_to_px(q, d) === BB.ustrip(mm, q) * BB.ustrip(px / mm, d)
    @test floor(BB._length_to_px(135.89mm, 300dpi)) == 1604
    # points per pixel of a density: a ratio of dpi magnitudes (vector blobs, the
    # PDF's point scale); the native quotient differs at 96 dpi
    @test BB._pt_per(96dpi) === BB.ustrip(dpi, BB.POINT_DENSITY) / 96.0
    @test BB._pt_per(96dpi) !== BB.ustrip(BB.POINT_DENSITY / 96dpi)
    # world blobs: stored px ÷ the stored density's px per mm
    bundle, _, _ = BB.load_bundle(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"))
    bc = BB.board_config(bundle)
    world = BB.blobs(bundle, :world)
    ppm = BB.px_per_mm(BB.stored_density(bc))
    canvas = bundle.blobs .+ Ref(BB._frame_offset(bundle, :canvas))
    @test all(w.σ === BB.ustrip(px, b.σ) / ppm for (w, b) in zip(world, canvas))
    @test any(w.σ !== BB.ustrip(mm, b.σ / BB.stored_density(bc)) for (w, b) in zip(world, canvas))
end
