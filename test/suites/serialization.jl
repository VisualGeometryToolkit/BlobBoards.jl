# =============================================================================
# Bundle serialization: v2 schema, v1 read, canonical identity, v0 refusal
# =============================================================================
#
# The golden constants below are the point of this file: if a change to the
# serializer (or a JSON3/StructTypes upgrade) would silently re-mint board
# identities, these tests fail loudly instead.

using BlobBoards: PatternConfig, PatternBundle, BoardBundle,
    hash_config, canonical_config_json, uid_from_hash,
    save_json
using VisualGeometryCore: Disc, Point2, px, mm, ustrip

# Fully explicit config — the golden must not depend on any default drifting.
_golden_config() = PatternConfig(
    pattern_resolution = BlobBoards.Size2(width = 64.0px, height = 64.0px),
    min_scale = 3.0px, max_scale = 6.0px, seed = UInt64(42),
    scales_per_octave = 3, max_trials = 100, α = 1.6, cutoff = 2.0,
    raster_aligned = true, polarity = :dark
)

const GOLDEN_CANONICAL = "{\"alpha\":1.6,\"cutoff\":2,\"max_scale\":{\"unit\":\"px\",\"value\":6},\"max_trials\":100,\"min_scale\":{\"unit\":\"px\",\"value\":3},\"pattern_resolution\":{\"height\":{\"unit\":\"px\",\"value\":64},\"width\":{\"unit\":\"px\",\"value\":64}},\"pixel_type\":\"gray8\",\"polarity\":\"dark\",\"raster_aligned\":true,\"scales_per_octave\":3,\"seed\":\"42\"}"
const GOLDEN_HASH = "ba1b3dc4bd7e068ed6f059a15d305e168dea5c0e6e80c96f578931c8f339708f"

# A v1 pattern bundle as the v1 writer emitted it: `max_diameter_fraction`, no
# `max_scale`. It must still load (v1 is READABLE), its fraction converting to the
# top rung it produced — 256 px, cutoff 2, fraction 0.2: σ cap 12.8 px, so 6 rungs
# from 3 px, top 3·2^(5/3) — and its stored hash staying the identity.
const V1_FIXTURE = """
{"schema": "blobboards/pattern/v1",
 "pattern_config": {
   "pattern_resolution": {"width": {"value": 256, "unit": "px"},
                          "height": {"value": 256, "unit": "px"}},
   "min_scale": {"value": 3, "unit": "px"},
   "seed": "42", "pixel_type": "gray8",
   "max_diameter_fraction": 0.2, "scales_per_octave": 3, "max_trials": 100,
   "alpha": 1.6, "cutoff": 2, "raster_aligned": true, "polarity": "dark"},
 "blobs": {"unit": "px", "centers": [[10.5, 20.5]], "sigmas": [3.0]},
 "hashes": {"config": "abc12345000000000000000000000000000000000000000000000000000000ff"}}
"""

# A pre-v1 pattern-bundle JSON exactly as the old writer emitted it: no
# schema key, `α` spelling, Julia type name, row-form blobs with per-scalar
# units. The v0 READER was removed in 0.19 (every known corpus migrated in
# place, 281 bundles on 2026-08-20); this fixture now guards the REFUSAL —
# a schema-less file must error actionably (naming BlobBoards@0.18's
# migrator), never parse to garbage or a bare KeyError.
const V0_FIXTURE = """
{"pattern_config": {
   "pattern_resolution": {"width": {"value": 64.0, "unit": "px"},
                          "height": {"value": 64.0, "unit": "px"}},
   "min_scale": {"value": 3.0, "unit": "px"},
   "seed": "42", "pixel_type": "Gray{N0f8}",
   "max_diameter_fraction": 0.2, "scales_per_octave": 3, "max_trials": 100,
   "α": 1.6, "cutoff": 2.0, "oversampling": 2,
   "raster_aligned": true, "polarity": "dark"},
 "blobs": [
   {"center": [{"value": 10.5, "unit": "px"}, {"value": 20.5, "unit": "px"}],
    "σ": {"value": 3.0, "unit": "px"}},
   {"center": [{"value": 30.25, "unit": "px"}, {"value": 40.0, "unit": "px"}],
    "σ": {"value": 4.5, "unit": "px"}}],
 "hashes": {"config": "feedface00000000000000000000000000000000000000000000000000000000"}}
"""

@testset "bundle serialization" begin

    @testset "canonical identity minting" begin
        pc = _golden_config()
        @test canonical_config_json(pc) == GOLDEN_CANONICAL
        @test hash_config(pc) == GOLDEN_HASH
        @test uid_from_hash(hash_config(pc)) == GOLDEN_HASH[1:3]
    end

    @testset "pre-v1 files are refused with the migration pointer" begin
        # load_bundle: the schema gate fires BEFORE any parse, and the error
        # names the escape hatch (the permanently registered 0.18 migrator).
        mktempdir() do dir
            path = joinpath(dir, "blob_pattern_fee_dface.json")
            write(path, V0_FIXTURE)
            err = try
                BlobBoards.load_bundle(path)
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("migrate_bundles!", err)
            @test occursin("0.18", err)
        end
        # direct typed parse of v0 spellings: also refused, not misparsed
        @test_throws Exception JSON3.read(V0_FIXTURE, PatternBundle)
    end

    @testset "v1 files load: fraction → max_scale, stored hash kept" begin
        mktempdir() do dir
            path = joinpath(dir, "blob_pattern_abc_12345.json")
            write(path, V1_FIXTURE)
            write(joinpath(dir, "blob_pattern_abc_12345.png"), "")   # sidecar present: no re-render
            bundle, _, config_hash = BlobBoards.load_bundle(path)
            pc = bundle.pattern_config
            @test pc.max_scale ≈ 3 * 2^(5 / 3) * px
            @test BlobBoards.calc_blob_scales(3.0, ustrip(pc.max_scale)) == [3.0 * 2^(i / 3) for i in 0:5]
            @test config_hash == "abc12345000000000000000000000000000000000000000000000000000000ff"
        end
    end

    @testset "blob_sizes: one stored config whatever the spelling" begin
        # Sizes are disc diameters, 2·cutoff·σ (σ = d/4 at cutoff 2); a short_side
        # max is a diameter too. Every spelling below is the SAME blobs, so the
        # same identity.
        mk(; kw...) = BlobBoards.blob_pattern(BlobBoards.Size2(width = 300px, height = 200px);
            seed = 0x42, cutoff = 2.0, kw...)[3]
        hashes = hash_config.([
            mk(; blob_sizes = (10px, 40px)),
            mk(; blob_sizes = (10, 40)),                                    # a bare number is px
            mk(; blob_sizes = (10px, 0.2short_side)),                       # 0.2 · 200 px
        ])
        @test allequal(hashes)
        pc = mk(; blob_sizes = (10px, 40px))
        @test pc.min_scale == 2.5px
        @test pc.max_scale == 10px
        # both ends exact; on the 2^(1/3) ladder (10 = 2.5·2^(6/3)) the original steps
        @test BlobBoards.calc_blob_scales(2.5, 10.0) == [2.5 * 2^(i / 3) for i in 0:6]
        # off the ladder: the max is kept, k = round(3·log2(11/2.5)) = 6 steps evenly in log
        off = mk(; blob_sizes = (10px, 44px))
        @test off.max_scale == 11px
        ladder = BlobBoards.calc_blob_scales(2.5, 11.0)
        @test length(ladder) == 7 && first(ladder) == 2.5 && last(ladder) == 11.0
        @test all(≈(log2(11 / 2.5) / 6), diff(log2.(ladder)))
        # the retired keywords and malformed sizes fail loudly
        @test_throws ArgumentError mk(; min_scale = 3px)
        @test_throws ArgumentError mk(; max_diameter_fraction = 0.2)
        @test_throws ArgumentError mk(; blob_sizes = (0.1short_side, 10px))
        @test_throws ArgumentError mk(; blob_sizes = (12px, 40px), blob_measure = :diameter)   # removed keyword
        @test_throws ArgumentError mk(; blob_measure = :diameter)            # removed, with or without sizes
        @test_throws ArgumentError mk(; blob_sizes = (7px, 40px))           # σ = 1.75 px < the 2 px floor
        @test_throws ArgumentError mk(; blob_sizes = (12px, 400px))         # 400 px disc > 200 px side
        # the default range mints one identity whichever constructor builds it
        @test hash_config(mk()) == hash_config(BlobBoards.PatternConfig(;
            pattern_resolution = BlobBoards.Size2(width = 300px, height = 200px), min_scale = 2px,
            seed = 0x42, cutoff = 2.0))
    end

    @testset "v2 write → read round trip" begin
        pb = PatternBundle(
            _golden_config(),
            [
                Disc(Point2(10.5px, 20.5px), 3.0px),
                Disc(Point2(30.25px, 40.0px), 4.5px),
            ]
        )
        str = mktempdir() do dir
            path = joinpath(dir, "blob_pattern_x.json")
            save_json(path, pb)
            read(path, String)
        end
        obj = JSON3.read(str)
        @test obj.schema == "blobboards/pattern/v2"
        @test haskey(obj.pattern_config, :max_scale) && !haskey(obj.pattern_config, :max_diameter_fraction)
        @test String(obj.hashes.config) == GOLDEN_HASH                    # canonical mint
        @test haskey(obj.blobs, :centers) && haskey(obj.blobs, :sigmas)   # columnar
        @test obj.blobs.unit == "px"
        @test haskey(obj.pattern_config, :alpha)                          # v1 spelling
        @test obj.pattern_config.pixel_type == "gray8"

        pb2 = JSON3.read(str, PatternBundle)
        @test pb2.blobs == pb.blobs
        @test hash_config(pb2.pattern_config) == hash_config(pb.pattern_config)
    end


end
