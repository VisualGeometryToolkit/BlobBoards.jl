# =============================================================================
# Identity and byte pins: every hashed or content-addressed output
# =============================================================================
#
# Board identities (`hash_config`, the uid printed in a board's corners and
# carried by its QR code), the annotation config hash, the render-cache key and
# the bytes of the JSON a board or pattern is saved as were pinned on
# 2026-10-24 against the JSON3/StructTypes serializer, before the move to
# JSON.jl. A serializer change that alters any of them fails here: a printed
# board's uid must never move, and a file written today must be byte-identical
# to the one the previous writer wrote for the same board.

using SHA: sha256
using VisualGeometryToolkitCore: Disc, Point2, px, mm, dpi

_sha(bytes) = bytes2hex(sha256(bytes))
_onlyext(dir, ext) = only(filter(endswith(ext), readdir(dir; join = true)))

const _FIXTURES = joinpath(@__DIR__, "..", "fixtures", "v3_render")

# The pattern bundle of serialization.jl's golden config, as `save_json` writes
# it. The integral floats are printed the way JSON3.pretty printed them: an
# object value 3.0 as `3`, an array's first element 3.0 as `3`, but 40.0 after
# the float 30.25 in the same array as `40.0`.
const PINNED_PATTERN_JSON = """
{
    "schema": "blobboards/pattern/v2",
    "pattern_config": {
        "pattern_resolution": {
            "width": {
                "value": 64,
                "unit": "px"
            },
            "height": {
                "value": 64,
                "unit": "px"
            }
        },
        "min_scale": {
            "value": 3,
            "unit": "px"
        },
        "seed": "42",
        "pixel_type": "gray8",
        "max_scale": {
            "value": 6,
            "unit": "px"
        },
        "scales_per_octave": 3,
        "max_trials": 100,
        "alpha": 1.6,
        "cutoff": 2,
        "raster_aligned": true,
        "polarity": "dark"
    },
    "blobs": {
        "unit": "px",
        "centers": [
            [
                10.5,
                20.5
            ],
            [
                30.25,
                40.0
            ]
        ],
        "sigmas": [
            3,
            4.5
        ]
    },
    "hashes": {
        "config": "ba1b3dc4bd7e068ed6f059a15d305e168dea5c0e6e80c96f578931c8f339708f"
    }
}"""

@testset "identity and byte pins" begin

    @testset "pattern bundle JSON bytes" begin
        pc = BlobBoards.PatternConfig(
            pattern_resolution = BlobBoards.Size2(width = 64.0px, height = 64.0px),
            min_scale = 3.0px, max_scale = 6.0px, seed = UInt64(42),
            scales_per_octave = 3, max_trials = 100, α = 1.6, cutoff = 2.0,
            raster_aligned = true, polarity = :dark
        )
        pb = BlobBoards.PatternBundle(pc, [Disc(Point2(10.5px, 20.5px), 3.0px), Disc(Point2(30.25px, 40.0px), 4.5px)])
        mktempdir() do dir
            BlobBoards.save_json(joinpath(dir, "pretty.json"), pb)
            @test read(joinpath(dir, "pretty.json"), String) == PINNED_PATTERN_JSON
            BlobBoards.save_json(joinpath(dir, "compact.json"), pb; pretty_json = false)
            @test _sha(read(joinpath(dir, "compact.json"))) == "63c50b2bed0029640af8cebec94b680cc9fe493865670522c837948b62211da4"
        end
    end

    @testset "point-grid (v4) board: identity, code, JSON bytes" begin
        mktempdir() do dir
            b = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, dir)
            @test hash_config(b.preamble) == "da6009bb31d28be9a193fdfdf1e3f287d1468335e2751b20e6bc2f32830943d4"
            @test _sha(BlobBoards.canonical_config_json(b.preamble)) == "80388c6c81978b0923d85f0a9e95aa2e7c94eff600d09bea40ceaf8e50fc1e4b"
            @test BlobBoards.board_code(b.preamble) ==
                "BB4/148MM/210MM/10MM/10MM/10MM/10MM/0MM/10MM/5MM/10MM/5MM/1234/2.83464566929/18.1415/3/5000/1.2/2/D/DA6009BB"
            @test _sha(read(_onlyext(dir, ".json"))) == "15c9d0e5fddec0b7f5e0278a8693929608a6d9d99d30224c886b45ab095b38cb"
            @test basename(BlobBoards._render_cache_dir(hash_config(b.preamble), true)) == "da6009bb31d28be9_629d0450"
        end
        # a raster output records its density (a quantity) in the render block
        mktempdir() do dir
            blob_board("A5"; blob_sizes = (5.0mm, 0.2short_side), seed = 0x77, dir, output = BlobBoards.Png(150dpi), id_label = true)
            json = read(_onlyext(dir, ".json"))
            @test _sha(json) == "e12212daddfbf928872aa349617163abe8bf3b91c544e5279a2a06557f5c735c"
            @test occursin("cd5a5372d3c7c38f4b2a3724759421f44d354027b127b9a5623543dc37ec4017", String(json))
        end
    end

    @testset "stored pixel-grid (v3) boards: identity, re-save and rebuild bytes" begin
        for (name, h, resave) in (
                ("odd300", "0ed26d0d3b982d10064a872e75e1dfb5215ef91cac71633083a2da5c6ddadc39",
                    "f8dc7c5040c95110f7e8fa1bc9513727cf7a1e477aad8fd40d67d174dbc02b79"),
                ("framed600", "b01c46b2f66b6507175cd68e077f8445cfc538ea550e05101f1e4d062f353768",
                    "d5bb2672d53247cae58647f69f50087c3b93fe25a66964b896a7d6bf32823855"),
            )
            v3, _, stored = BlobBoards.load_bundle(joinpath(_FIXTURES, name * ".json"))
            @test stored == h
            @test hash_config(v3.preamble) == h                 # v3: the configuration alone
            @test _sha(BlobBoards.canonical_config_json(v3.preamble)) == h
            mktempdir() do dir
                BlobBoards.save_json(joinpath(dir, "x.json"), v3; config_hash = stored)
                @test _sha(read(joinpath(dir, "x.json"))) == resave
            end
        end
        mktempdir() do dir
            BlobBoards.rebuild_board(joinpath(_FIXTURES, "odd300.json"); dir)
            @test _sha(read(_onlyext(dir, ".json"))) == "f7f34fa90c7ad22ba5cf48fd946cf4960afa72c110f1b85debb0c556caa031d7"
        end
    end

    @testset "annotation and run-config hashes" begin
        @test BlobBoards.config_hash(;
            config = BlobBoards.FindBoardsConfig(), cutoff = 2.0, patch_size = 64, outer_extent = 1.5, min_iou = 0.5
        ) == "148d5baf77e7d5e18fbbe26735207cc6367a7da2c137eb28b4558c221ed197ce"
        @test _sha(BlobBoards.canonical_config_json(BlobBoards.config_dict(BlobBoards.FindBoardsConfig()))) ==
            "5efb5950c750dce114bc501ea6a761fd2e4eeb0e0ac6a6485d4a364a039be0dd"
    end
end
