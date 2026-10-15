# =============================================================================
# Appearance-verification primitives (src/verify.jl) + derived matching gates.
# The end-to-end mandatory gate in `_claim_boards` is exercised by the
# dataset-gated annotate smoke; these pin the arithmetic it reads.
# =============================================================================

@testset "appearance verification" begin
    @testset "appearance_support" begin
        # Empty ⇒ 0.0: a board with no judgeable inliers FAILS the gate.
        @test BlobBoards.appearance_support(Float32[]) == 0.0
        @test BlobBoards.appearance_support(Float32[0.9, 0.9, 0.1, 0.1]) == 0.5
        @test BlobBoards.appearance_support(Float32[0.79]; verify_similarity = 0.8) == 0.0
        @test BlobBoards.appearance_support(Float32[0.8]; verify_similarity = 0.8) == 1.0
    end

    @testset "inlier sims read only inliers, via the id columns" begin
        SA = BlobBoards.StructArray
        Dg = Float32[1 0; 0 1]           # gallery rows: e1, e2
        Di = Float32[1 0; 0 1; 1 0]      # image rows: e1, e2, e1
        scored = SA(
            (
                weights = [1.0, 1.0, 0.0],   # third cspond is an outlier — never judged
                src_id = [1, 2, 2],
                det_id = [1, 3, 2],
            )
        )
        @test BlobBoards._inlier_sims(Dg, Di, scored) == Float32[1.0, 0.0]
        # Out-of-range ids are skipped, not thrown.
        oob = SA((weights = [1.0], src_id = [7], det_id = [1]))
        @test isempty(BlobBoards._inlier_sims(Dg, Di, oob))
        # No id columns ⇒ no judgeable inliers (fails closed downstream:
        # appearance_support of an empty set is 0.0).
        bare = SA((weights = [1.0],))
        @test isempty(BlobBoards._inlier_sims(Dg, Di, bare))
    end

    @testset "derived matching gates" begin
        # Phase 2's gate DERIVES from min_similarity so a sweep cannot invert
        # the phase semantics (a hardcoded 0.5 made "permissive cleanup"
        # STRICTER than phase 1 for any min_similarity < 0.5).
        @test BlobBoards._phase2_similarity(0.6) == 0.5
        @test BlobBoards._phase2_similarity(0.05) == 0.0   # floored at zero
        # The agenda counts EVERY resolved match — there is no second
        # similarity bar. Deleting `_strong_similarity` removed a cliff on the
        # path to the engine's only irreversible decision; the count is
        # monotone, so the floor can only admit more, never fewer.
        @test BlobBoards._n_resolved_matches(1:7) == 7
        @test BlobBoards._n_resolved_matches(()) == 0
    end
end
