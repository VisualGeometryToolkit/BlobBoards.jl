# =============================================================================
# Identity and byte pins: every hashed or content-addressed output
# =============================================================================
#
# Board identities (`hash_config`, the uid printed in a board's corners and
# carried by its QR code), the annotation config hash and the bytes of the JSON a
# board or pattern is saved as are pinned here. The IDENTITIES were pinned on
# 2026-10-24 against the JSON3/StructTypes serializer and have not moved since:
# not for the move to JSON.jl, and not for the file format's bump to board v5 /
# pattern v3 (2026-10-27, the quantity format), which is why `hash_config` of
# every board below is the same string it was, and the stored `hashes.config` of
# every migrated file is untouched. A printed board's uid must never move. The
# FILE BYTES are pinned too; the v5 pins are those of the v5 writer (a file is
# byte-identical to what the writer wrote for the same board, and the migration
# of each tracked file is byte-identical to the file that replaced it).

using SHA: sha256
using VisualGeometryToolkitCore: Disc, Point2, px, mm, dpi, pt, ustrip

_sha(bytes) = bytes2hex(sha256(bytes))
_onlyext(dir, ext) = only(filter(endswith(ext), readdir(dir; join = true)))

const _FIXTURES = joinpath(@__DIR__, "..", "fixtures", "v3_render")

# The pattern bundle of serialization.jl's golden config, as `save_json` writes
# it. The integral floats are printed the way JSON3.pretty printed them: an
# object value 3.0 as `3`, an array's first element 3.0 as `3`, but 40.0 after
# the float 30.25 in the same array as `40.0`.
const PINNED_PATTERN_JSON = """
{
    "schema": "blobboards/pattern/v3",
    "units": {
        "world": "mm",
        "density": "dpi"
    },
    "pattern_config": {
        "pattern_resolution": {
            "width": 64,
            "height": 64
        },
        "min_scale": 3,
        "seed": "42",
        "pixel_type": "gray8",
        "max_scale": 6,
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

    @testset "the writer's layout is JSON3.pretty's, quirks included" begin
        # expected strings as JSON3.pretty printed these values (2026-10-24)
        pretty(x) = sprint(io -> BlobBoards._write_json(io, x))
        @test pretty(Any[3.0, 4.5, 5.0]) == "[\n    3,\n    4.5,\n    5\n]"
        @test pretty(Any[1.5, nothing, 2.0, 2]) == "[\n    1.5,\n    null,\n    2.0,\n    2.0\n]"
        @test pretty(Any[nothing, 2.0]) == "[\n    null,\n    2\n]"
        @test pretty(Any[2.5, "a", 3.0]) == "[\n    2.5,\n    \"a\",\n    3\n]"
        @test pretty((a = 3.0, b = Any[], c = Dict{String, Any}(), d = -0.0, e = 1.0e16, f = Any[Any[1.0], 2.5, 3])) ==
            "{\n    \"a\": 3,\n    \"b\": [\n    ],\n    \"c\": {\n    },\n    \"d\": 0,\n    \"e\": 10000000000000000,\n" *
            "    \"f\": [\n        [\n            1\n        ],\n        2.5,\n        3\n    ]\n}"
        @test sprint(io -> BlobBoards._write_json(io, (a = 3.0, b = Any[1.5, 2]); pretty = false)) == "{\"a\":3.0,\"b\":[1.5,2]}"
    end

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
            @test _sha(read(joinpath(dir, "compact.json"))) == "fe15e2d852977e2cedfef6fe0b84ac9497de03d2551678be56b73f8dba97d16a"
        end
    end

    @testset "point-grid board: identity, code, JSON bytes (board v5)" begin
        mktempdir() do dir
            b = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, dir)
            @test hash_config(b.preamble) == "da6009bb31d28be9a193fdfdf1e3f287d1468335e2751b20e6bc2f32830943d4"
            @test _sha(BlobBoards.canonical_config_json(b.preamble)) == "80388c6c81978b0923d85f0a9e95aa2e7c94eff600d09bea40ceaf8e50fc1e4b"
            @test BlobBoards.board_code(b.preamble) ==
                "BB4/148MM/210MM/10MM/10MM/10MM/10MM/0MM/10MM/5MM/10MM/5MM/1234/2.83464566929/18.1415/3/5000/1.2/2/D/DA6009BB"
            @test _sha(read(_onlyext(dir, ".json"))) == "638105cda9e03fb1507768e825f8ef10fc129ae94be82fbc2a7305053feab87e"
            @test basename(BlobBoards._render_cache_dir(hash_config(b.preamble), true)) == "da6009bb31d28be9_05984910"
            # write → read → write is byte-identical: the lift paths (margin, ruler_width, corner_size, layout_grid,
            # the blob columns) lose nothing
            written = _onlyext(dir, ".json")
            bundle, _, stored = BlobBoards.load_bundle(written)
            BlobBoards.save_json(joinpath(dir, "again.json"), bundle; config_hash = stored, render = BlobBoards._render_record(BlobBoards.Pdf(), true))
            @test read(joinpath(dir, "again.json")) == read(written)
        end
        # a tracked board file (docs/hero_board.json, v5) re-saves to the same bytes
        mktempdir() do dir
            hero = joinpath(pkgdir(BlobBoards), "docs", "hero_board.json")
            bundle, _, stored = BlobBoards.load_bundle(hero)
            BlobBoards.save_json(joinpath(dir, "hero.json"), bundle; config_hash = stored, render = BlobBoards._render_record(BlobBoards.Pdf(), false))
            @test read(joinpath(dir, "hero.json")) == read(hero)
        end
        # a raster output records its density (a quantity) in the render block
        mktempdir() do dir
            blob_board("A5"; blob_sizes = (5.0mm, 0.2short_side), seed = 0x77, dir, output = BlobBoards.Png(150dpi), id_label = true)
            json = read(_onlyext(dir, ".json"))
            @test _sha(json) == "df504c5f3496567f95d13563912dd8e9dcab3fb57fdf91b10a437cf2984775d1"
            @test occursin("cd5a5372d3c7c38f4b2a3724759421f44d354027b127b9a5623543dc37ec4017", String(json))
        end
    end

    @testset "stored pixel-grid boards: identity, re-save and rebuild bytes (board v5)" begin
        for (name, h, resave) in (
                ("odd300", "0ed26d0d3b982d10064a872e75e1dfb5215ef91cac71633083a2da5c6ddadc39",
                    "e0cca7b28d53a1281545a3333a896fe2faa30d99ee9aca9a1bf500b64ce410d7"),
                ("framed600", "b01c46b2f66b6507175cd68e077f8445cfc538ea550e05101f1e4d062f353768",
                    "eb06733bd28ab318b79db8ec489152438da00a5bd3588085953419ccd39b5f52"),
            )
            v3, _, stored = BlobBoards.load_bundle(joinpath(_FIXTURES, name * ".json"))
            @test stored == h
            @test hash_config(v3.preamble) == h                 # a pixel-grid board: the configuration alone
            @test _sha(BlobBoards.canonical_config_json(v3.preamble)) == h
            mktempdir() do dir
                BlobBoards.save_json(joinpath(dir, "x.json"), v3; config_hash = stored)
                @test _sha(read(joinpath(dir, "x.json"))) == resave
            end
        end
        mktempdir() do dir
            BlobBoards.rebuild_board(joinpath(_FIXTURES, "odd300.json"); dir)
            @test _sha(read(_onlyext(dir, ".json"))) == "8a71575e49f3ddfe21456738c906cb739df051a20f64858fe7d78239fb402322"
        end
    end


    @testset "migration: the tracked boards, UIDs invariant, bytes pinned" begin
        legacy = joinpath(@__DIR__, "..", "fixtures", "legacy")
        pkg = pkgdir(BlobBoards)
        # (legacy file, the tracked file that replaced it, its schema before, the sha256 of the new file)
        cases = (
            ("hero_board_v4.json", joinpath(pkg, "docs", "hero_board.json"), "blobboards/board/v4", "289"),
            ("odd300_v3.json", joinpath(_FIXTURES, "odd300.json"), "blobboards/board/v3", "0ed"),
            ("framed600_v3.json", joinpath(_FIXTURES, "framed600.json"), "blobboards/board/v3", "b01"),
            ("v3_raster_render_no_density.json", joinpath(pkg, "test", "fixtures", "v3_raster_render_no_density.json"), "blobboards/board/v3", "6f5"),
        )
        # the two the IO monorepo kept as migration inputs (VisualGeometryToolkitIO at 0d86d1e,
        # test/fixtures/legacy_quantities/): the very same bytes
        @test _sha(read(joinpath(legacy, "odd300_v3.json"))) == "2bedb7da2c2f4348238b59b842c55eae37fcad4dd721312f6a082222ca41d1f9"
        @test _sha(read(joinpath(legacy, "v3_raster_render_no_density.json"))) == "7fbd716d89acd1e9c8b81e660323e7d5d4a709db49694db171beb678b995e077"
        for (old, tracked, from, uid) in cases
            old_path = joinpath(legacy, old)
            old_obj = JSON.parse(read(old_path, String))
            @test old_obj.schema == from
            # loading the retired form (here through JSON.parse, as the Python wrapper does) names the migration
            err = try
                JSON.parse(read(old_path, String), BlobBoards.BoardBundle)
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing && occursin("vgio_migrate", err) && occursin(from, err)
            mktempdir() do dir
                out = joinpath(dir, basename(tracked))
                r = VgtkIO.vgio_migrate(old_path; out)
                @test r.from == from && r.to == "blobboards/board/v5" && r.uid == uid
                # the migration IS the tracked file, byte for byte
                @test read(out) == read(tracked)
                new_obj = JSON.parse(read(tracked, String))
                @test new_obj.schema == "blobboards/board/v5"
                # the uid is invariant: the stored hash is carried over …
                @test new_obj.hashes.config == old_obj.hashes.config && r.config_hash == old_obj.hashes.config
                # … and the identity recomputed from the migrated file is the one the old file's configuration mints
                bundle, stored = BlobBoards._parse_bundle(tracked)
                @test stored == old_obj.hashes.config == BlobBoards._bundle_identity(bundle)
                @test BlobBoards.hash_config(bundle.preamble) == stored
                # every number reads back as the old file read it
                legacy_bundle = BlobBoards._legacy_board_bundle(old_obj)
                @test BlobBoards._same_value(legacy_bundle, bundle)
            end
        end
        # the new files' bytes
        @test _sha(read(joinpath(pkg, "docs", "hero_board.json"))) == "d77843a45ec333562de08e135aa063ddc0b09fd41664420b6e28e16bda76e143"
        @test _sha(read(joinpath(_FIXTURES, "odd300.json"))) == "f3dc087dbc61d5232e82ff8dd6e9ce76dfbd0ef9c63e94ec154e0e0664a58bd4"
        @test _sha(read(joinpath(_FIXTURES, "framed600.json"))) == "fd2e046c3dfb8ed492230527e290252a6b76298e979a4f1a45d2fb46da45050e"
        @test _sha(read(joinpath(pkg, "test", "fixtures", "v3_raster_render_no_density.json"))) == "6762a06795eb5534f5b90264711a2e5a774ea3f9d053890d6b1ade3cc4ab59fc"
    end

    @testset "the file format is not part of an identity" begin
        @test BlobBoards.BOARD_SCHEMA == "blobboards/board/v5" && BlobBoards.PATTERN_SCHEMA == "blobboards/pattern/v3"
        @test BlobBoards.IDENTITY_SCHEMA == "blobboards/board/v4"            # frozen: a bump of the file schema moves no uid
        b = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, dir = mktempdir())
        # the identity tree names the identity schema and holds the quantities in its own form
        tree = BlobBoards._plain(b.preamble)
        @test tree["board_config"]["canvas_size"]["width"] == Dict{String, Any}("value" => 148, "unit" => "mm") ||
            tree["board_config"]["canvas_size"]["width"] == Dict{String, Any}("value" => 148.0, "unit" => "mm")
        # … while the file holds numbers, and the two forms never leak into each other
        @test BlobBoards._lower(148.0mm) == 148.0 && BlobBoards._lower(300dpi) == 300 && BlobBoards._lower(64px) == 64
        @test BlobBoards._plain(148.0mm) == Dict{String, Any}("value" => 148.0, "unit" => "mm")
        @test BlobBoards._lower(148.0mm) == 148.0                              # after `_plain`, still a file's number
    end

    @testset "IO reads what a board writes: units block, roles, numbers" begin
        hero = JSON.parse(read(joinpath(pkgdir(BlobBoards), "docs", "hero_board.json"), String))
        c = VgtkIO.from_json(VgtkIO.UnitContract, hero.units)
        @test c == VgtkIO.UnitContract(world = mm, density = dpi, lengths = (pattern = pt,))
        @test VgtkIO.role_unit(c, :pattern) == pt
        bundle, _, _ = BlobBoards.load_bundle(joinpath(pkgdir(BlobBoards), "docs", "hero_board.json"))
        bc, pc = bundle.preamble.board_config, bundle.preamble.pattern_config
        # IO's decimal-exact reader agrees with the board's own to rounding (an ulp: a board is built from `210.0mm`)
        @test VgtkIO.read_quantity(hero.preamble.board_config.canvas_size.width, :length, c) ≈ bc.canvas_size.width rtol = 4eps()
        @test VgtkIO.read_quantity(hero.preamble.board_config.corner_size, :length, c) ≈ bc.corner_size rtol = 4eps()
        @test VgtkIO.read_quantity(hero.preamble.pattern_config.min_scale, :length, c; role = :pattern) ≈ ustrip(px, pc.min_scale) * pt rtol = 4eps()
        @test hero.preamble.board_config.layout_grid == 0.01                        # a number in the `pattern` role (pt)
    end

    @testset "annotation and run-config hashes" begin
        @test BlobBoards.config_hash(;
            config = BlobBoards.FindBoardsConfig(), cutoff = 2.0, patch_size = 64, outer_extent = 1.5, min_iou = 0.5
        ) == "148d5baf77e7d5e18fbbe26735207cc6367a7da2c137eb28b4558c221ed197ce"
        @test _sha(BlobBoards.canonical_config_json(BlobBoards.config_dict(BlobBoards.FindBoardsConfig()))) ==
            "5efb5950c750dce114bc501ea6a761fd2e4eeb0e0ac6a6485d4a364a039be0dd"
    end
end
