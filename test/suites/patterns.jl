@testset "Pattern Generation" begin
    @testset "Basic pattern generation" begin
        pattern, blobs, config = blob_pattern(400, 300; blob_sizes = (12px, nothing), seed = 0x1234, cutoff = 2.0)
        @test size(pattern) == (300, 400)
        @test length(blobs) > 0
        @test config.seed == 0x1234
    end

    @testset "Pattern reproducibility" begin
        p1, b1, _ = blob_pattern(400, 300; blob_sizes = (12px, nothing), seed = 0x5678, cutoff = 2.0)
        p2, b2, _ = blob_pattern(400, 300; blob_sizes = (12px, nothing), seed = 0x5678, cutoff = 2.0)
        @test p1 == p2
        @test b1 == b2
    end

    @testset "Pattern polarity" begin
        p_dark, _, _ = blob_pattern(200, 200; blob_sizes = (12px, nothing), seed = 0x1234, polarity = :dark, cutoff = 2.0)
        p_light, _, _ = blob_pattern(200, 200; blob_sizes = (12px, nothing), seed = 0x1234, polarity = :light, cutoff = 2.0)
        # Light pattern should be inverted
        @test p_dark .+ p_light ≈ ones(eltype(p_dark), size(p_dark)...) atol = 1.0e-6
    end

    @testset "Pattern round-trip" begin
        dir = mktempdir()
        _, blobs1, config1 = blob_pattern(400, 300; blob_sizes = (12px, nothing), seed = 0xABCD, cutoff = 2.0, dir = dir)

        # Load and regenerate
        json_file = only(filter(f -> endswith(f, ".json"), readdir(dir; join = true)))
        bundle = JSON3.read(read(json_file, String), BlobBoards.PatternBundle)

        @test bundle.pattern_config.seed == config1.seed
        @test length(bundle.blobs) == length(blobs1)
    end
end
