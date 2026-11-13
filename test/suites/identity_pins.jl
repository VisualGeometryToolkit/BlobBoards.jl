# =============================================================================
# Identity and byte pins: every hashed or content-addressed output
# =============================================================================
#
# Board identities (`hash_config`, the uid printed in a board's corners and
# carried by its QR code), the annotation config hash and the bytes of the JSON a
# board or pattern is saved as are pinned here. The IDENTITIES were pinned on
# 2026-10-24 against the JSON3/StructTypes serializer and have not moved since:
# not for the move to JSON.jl, not for the file format's bump to board v5 /
# pattern v3 (2026-10-27, the quantity format), and not for board v6 / pattern
# v4 (2026-11-06, VisualGeometryToolkitIO's writer), which is why `hash_config`
# of every board below is the same string it was, and the stored
# `hashes.config` of every migrated file is untouched. A printed board's uid
# must never move, nor for board v7.0 / pattern v5.0 (2026-11-13, IO 0.10's
# envelope). The FILE BYTES are pinned too, less the `generator` (the writing
# packages' versions, `_pinned`); the pins are those of the current (v7.0 / v5.0)
# writer, IO's `write_record`. The v5 / v3 writer (a
# JSON3.pretty clone) is gone; its files are the tracked fixtures, pinned by
# their sha256 and read through their version step to the value they always
# held. The migration of a retired file and of its tracked v5 successor give the
# same bytes, and those are the bytes a fresh v6 write of the board gives (with
# the render block the file records; for docs/hero_board.json, the canonical
# `_render_record(Pdf(), false)`). Every length round-trips bit for bit, built
# in millimetres or in inches.

using SHA: sha256
using VisualGeometryToolkitCore: Disc, Point2, px, mm, dpi, pt, inch, ustrip

_sha(bytes) = bytes2hex(sha256(bytes))
# A file's bytes less its `generator` (the IO and BlobBoards versions that wrote it, which move every release):
# what a byte pin holds.
_pinned(bytes) = replace(String(copy(bytes)), r"\n    \"generator\": \{.*?\n    \},"s => "")
_onlyext(dir, ext) = only(filter(endswith(ext), readdir(dir; join = true)))

# Two parsed records are the same record: the same keys (the schema aside), every
# number equal as a number (`3` and `3.0`: a v5 file printed an integral float as
# an integer), every other value equal.
# The envelope (`schema`, and IO 0.10's `generator`, `units`, `conventions`) aside: the units are compared by
# the caller (an earlier record declared them in the retired layout).
const _ENVELOPE = ["schema", "generator", "units", "conventions"]
_same_record(a::AbstractDict, b::AbstractDict) =
    setdiff(Set(String.(keys(a))), _ENVELOPE) == setdiff(Set(String.(keys(b))), _ENVELOPE) &&
    all(k -> String(k) in _ENVELOPE || _same_record(a[k], b[String(k)]), keys(a))
_same_record(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(_same_record(x, y) for (x, y) in zip(a, b))
_same_record(a::Real, b::Real) = (a isa Bool || b isa Bool) ? a === b : a == b
_same_record(a, b) = isequal(a, b)

const _FIXTURES = joinpath(@__DIR__, "..", "fixtures", "v3_render")

# The pattern bundle of serialization.jl's golden config, as `save_json` writes
# it less its `generator` (pattern v5.0: IO 0.10's envelope; JSON.jl's printing — a
# Float64 keeps its `.0`, an integral pixel count is an integer).
const PINNED_PATTERN_JSON = """
{
    "schema": "blobboards/pattern/v5.0",
    "units": {
        "length": "mm",
        "time": "s",
        "angle": "rad",
        "pixel": "colmap",
        "density": "dpi",
        "radiance": "relative",
        "length_roles": {}
    },
    "conventions": {
        "rotation_encoding": "quaternion_wxyz_hamilton",
        "transform_convention": "x_to = R * x_from + t",
        "matrix_layout": "rows",
        "grid_layout": "row_major",
        "index_base": 1
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
        "cutoff": 2.0,
        "raster_aligned": true,
        "polarity": "dark"
    },
    "blobs": {
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
            3.0,
            4.5
        ]
    },
    "hashes": {
        "config": "ba1b3dc4bd7e068ed6f059a15d305e168dea5c0e6e80c96f578931c8f339708f"
    }
}"""

# The same bundle as the retired v3 writer wrote it (the pin it was held to until
# 2026-11-06, JSON3.pretty's integral floats printed as integers): a v3 file still
# reads, through its version step, as that bundle.
const PINNED_PATTERN_V3_JSON = """
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

    @testset "pattern bundle JSON bytes; the v3 bytes read as the same bundle" begin
        pc = BlobBoards.PatternConfig(
            pattern_resolution = BlobBoards.Size2(width = 64.0px, height = 64.0px),
            min_scale = 3.0px, max_scale = 6.0px, seed = UInt64(42),
            scales_per_octave = 3, max_trials = 100, α = 1.6, cutoff = 2.0,
            raster_aligned = true, polarity = :dark
        )
        pb = BlobBoards.PatternBundle(pc, [Disc(Point2(10.5px, 20.5px), 3.0px), Disc(Point2(30.25px, 40.0px), 4.5px)])
        mktempdir() do dir
            BlobBoards.save_json(joinpath(dir, "pretty.json"), pb)
            @test _pinned(read(joinpath(dir, "pretty.json"))) == PINNED_PATTERN_JSON
            write(joinpath(dir, "v3.json"), PINNED_PATTERN_V3_JSON)
            old, stored = BlobBoards._parse_bundle(joinpath(dir, "v3.json"))
            @test VgtkIO.same_value(old, pb; length_ulps = 0) && stored == "ba1b3dc4bd7e068ed6f059a15d305e168dea5c0e6e80c96f578931c8f339708f"
            # and the v3 file migrates to the current bytes
            r = VgtkIO.vgio_migrate(joinpath(dir, "v3.json"); out = joinpath(dir, "v4.json"))
            @test r.to == "blobboards/pattern/v5.0" && _pinned(read(joinpath(dir, "v4.json"))) == PINNED_PATTERN_JSON
        end
    end

    @testset "point-grid board: identity, code, JSON bytes (board v7.0)" begin
        mktempdir() do dir
            b = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, dir)
            @test hash_config(b.preamble) == "da6009bb31d28be9a193fdfdf1e3f287d1468335e2751b20e6bc2f32830943d4"
            @test _sha(BlobBoards.canonical_config_json(b.preamble)) == "80388c6c81978b0923d85f0a9e95aa2e7c94eff600d09bea40ceaf8e50fc1e4b"
            @test BlobBoards.board_code(b.preamble) ==
                "BB4/148MM/210MM/10MM/10MM/10MM/10MM/0MM/10MM/5MM/10MM/5MM/1234/2.83464566929/18.1415/3/5000/1.2/2/D/DA6009BB"
            @test _sha(_pinned(read(_onlyext(dir, ".json")))) == "e6760fb39d52f82d2b7678eaf7dc65df295730a010ecbc2d655568384c9b7537"
            @test basename(BlobBoards._render_cache_dir(hash_config(b.preamble), true)) == "da6009bb31d28be9_05984910"
            # write → read → write is byte-identical: the lift paths (margin, ruler_width, corner_size, layout_grid,
            # the blob columns) lose nothing
            written = _onlyext(dir, ".json")
            bundle, _, stored = BlobBoards.load_bundle(written)
            BlobBoards.save_json(joinpath(dir, "again.json"), bundle; config_hash = stored, render = BlobBoards._render_record(BlobBoards.Pdf(), true))
            @test read(joinpath(dir, "again.json")) == read(written)
        end
        # a tracked board file (docs/hero_board.json, v5) reads, through its version step, as the board it holds:
        # its re-save is the same record under the current schema, and re-saves to itself
        mktempdir() do dir
            hero = joinpath(pkgdir(BlobBoards), "docs", "hero_board.json")
            bundle, _, stored = BlobBoards.load_bundle(hero)
            BlobBoards.save_json(joinpath(dir, "hero.json"), bundle; config_hash = stored, render = BlobBoards._render_record(BlobBoards.Pdf(), false))
            new, old = JSON.parse(read(joinpath(dir, "hero.json"), String)), JSON.parse(read(hero, String))
            @test new.schema == "blobboards/board/v7.0" && old.schema == "blobboards/board/v5"
            @test _same_record(new, old) && new.units == JSON.parse(JSON.json(VgtkIO.migrate_units(old.units)))
            # its migration is byte for byte the fresh write of the board it holds, under the render record it
            # records (`_render_record(Pdf(), false)`: a PDF, no id label)
            @test VgtkIO.vgio_migrate(hero; out = joinpath(dir, "migrated.json")).to == "blobboards/board/v7.0"
            @test read(joinpath(dir, "migrated.json")) == read(joinpath(dir, "hero.json"))
            again, _, s2 = BlobBoards.load_bundle(joinpath(dir, "hero.json"))
            @test VgtkIO.same_value(again, bundle; length_ulps = 0) && s2 == stored
        end
        # a raster output records its density (a quantity) in the render block
        mktempdir() do dir
            blob_board("A5"; blob_sizes = (5.0mm, 0.2short_side), seed = 0x77, dir, output = BlobBoards.Png(150dpi), id_label = true)
            json = read(_onlyext(dir, ".json"))
            @test _sha(_pinned(json)) == "7e36d78899345c78525dfe99b9230112939f55edcdf6cf8dbe09c0f1d4d2c639"
            @test occursin("cd5a5372d3c7c38f4b2a3724759421f44d354027b127b9a5623543dc37ec4017", String(json))
        end
    end

    @testset "stored pixel-grid boards: identity, re-save and rebuild bytes (board v7.0)" begin
        for (name, h, resave) in (
                ("odd300", "0ed26d0d3b982d10064a872e75e1dfb5215ef91cac71633083a2da5c6ddadc39",
                    "e488f70acce15c63978b8d4835f9c90f82cd51ff26fb15095c2a464483a7cf80"),
                ("framed600", "b01c46b2f66b6507175cd68e077f8445cfc538ea550e05101f1e4d062f353768",
                    "452b56dc8d43707e39ad90875841893baae5daeea63da7d36ba79b6e421d568d"),
            )
            v3, _, stored = BlobBoards.load_bundle(joinpath(_FIXTURES, name * ".json"))
            @test stored == h
            @test hash_config(v3.preamble) == h                 # a pixel-grid board: the configuration alone
            @test _sha(BlobBoards.canonical_config_json(v3.preamble)) == h
            mktempdir() do dir
                BlobBoards.save_json(joinpath(dir, "x.json"), v3; config_hash = stored)
                @test _sha(_pinned(read(joinpath(dir, "x.json")))) == resave
            end
        end
        mktempdir() do dir
            BlobBoards.rebuild_board(joinpath(_FIXTURES, "odd300.json"); dir)
            @test _sha(_pinned(read(_onlyext(dir, ".json")))) == "3e9d12ec9cd09abba42ba74be3680005effc5fae9e30e9160df3ee51060a120e"
        end
    end


    @testset "migration: the tracked boards, UIDs invariant, bytes pinned" begin
        legacy = joinpath(@__DIR__, "..", "fixtures", "legacy")
        pkg = pkgdir(BlobBoards)
        # (legacy file, the tracked v5 file that replaced it, its schema before, its uid)
        cases = (
            ("hero_board_v4.json", joinpath(pkg, "docs", "hero_board.json"), "blobboards/board/v4", "289"),
            ("odd300_v3.json", joinpath(_FIXTURES, "odd300.json"), "blobboards/board/v3", "0ed"),
            ("framed600_v3.json", joinpath(_FIXTURES, "framed600.json"), "blobboards/board/v3", "b01"),
            ("v3_raster_render_no_density.json", joinpath(pkg, "test", "fixtures", "v3_raster_render_no_density.json"), "blobboards/board/v3", "6f5"),
        )
        # all four legacy inputs are byte-pinned; the first two the IO monorepo kept as migration inputs (VisualGeometryToolkitIO at 0d86d1e,
        # test/fixtures/legacy_quantities/): the very same bytes
        @test _sha(read(joinpath(legacy, "odd300_v3.json"))) == "2bedb7da2c2f4348238b59b842c55eae37fcad4dd721312f6a082222ca41d1f9"
        @test _sha(read(joinpath(legacy, "v3_raster_render_no_density.json"))) == "7fbd716d89acd1e9c8b81e660323e7d5d4a709db49694db171beb678b995e077"
        @test _sha(read(joinpath(legacy, "hero_board_v4.json"))) == "2316eaa2cf49965f5a59c3d91495f6960241a39acf7a2d25605cd9a4c44fc6cf"
        @test _sha(read(joinpath(legacy, "framed600_v3.json"))) == "d90d048e520892bae055ed5a0c2aa377516edf4a1cc192202ae09aef2622fc7d"
        for (old, tracked, from, uid) in cases
            old_path = joinpath(legacy, old)
            old_obj = JSON.parse(read(old_path, String))
            @test old_obj.schema == from
            # lifting the retired form without its version step names the migration
            err = try
                JSON.lift(BlobBoards._FILE_STYLE, BlobBoards.BoardBundle, old_obj)
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing && occursin("vgio_migrate", err) && occursin(from, err)
            mktempdir() do dir
                out = joinpath(dir, basename(tracked))
                r = VgtkIO.vgio_migrate(old_path; out)
                @test r.from == from && r.to == "blobboards/board/v7.0" && r.uid == uid
                # the migration of the retired file and of its tracked v5 successor are the same file, byte for byte
                # (each render quantity written as the current writer writes it: v5's `"bleed": 0` is `0.0`)
                out5 = joinpath(dir, "from_v5.json")
                r5 = VgtkIO.vgio_migrate(tracked; out = out5)
                @test r5.from == "blobboards/board/v5" && r5.to == "blobboards/board/v7.0" && r5.uid == uid
                @test read(out) == read(out5)
                # … and the file a fresh write of the loaded board with its render block gives
                fresh = joinpath(dir, "fresh.json")
                loaded, h = BlobBoards._parse_bundle(tracked)
                BlobBoards.save_json(fresh, loaded; config_hash = h, render = JSON.parse(read(out5, String)).render)
                @test read(fresh) == read(out5)
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
                @test VgtkIO.same_value(legacy_bundle, bundle; length_ulps = 0)
                migrated, migrated_stored = BlobBoards._parse_bundle(out)
                @test VgtkIO.same_value(migrated, bundle; length_ulps = 0) && migrated_stored == stored
                # the old file itself loads (its version step, in memory) as the migrated file, uid kept
                old_bundle, old_stored = BlobBoards._parse_bundle(old_path)
                @test VgtkIO.same_value(old_bundle, bundle; length_ulps = 0) && old_stored == stored
                @test BlobBoards.uid_from_hash(old_stored) == uid
            end
        end
        # the tracked v5 files' bytes (unchanged: an old version's fixtures)
        @test _sha(read(joinpath(pkg, "docs", "hero_board.json"))) == "eb09ac5696c83c75f314ed564b6bcfb75d912c4e9f56a76e9408b606a7fc73ce"
        @test _sha(read(joinpath(_FIXTURES, "odd300.json"))) == "6b2e3dfcf442bcdea35d7d9472fd05827cfdeef792de22d4832df4292f86473c"
        @test _sha(read(joinpath(_FIXTURES, "framed600.json"))) == "c2829bab15087ca8656add8cf73a32f4fead9351414f9711b8f576a923579249"
        @test _sha(read(joinpath(pkg, "test", "fixtures", "v3_raster_render_no_density.json"))) == "10eece32c2d719fdfa96867c02c0bcc3f768ba3a2822a15308cc71a3917f467a"
    end


    @testset "a v5 file migrates in the units it declares, its render numbers as it states them" begin
        mktempdir() do dir
            sq(v) = BlobBoards.Size2(width = v, height = v)
            b = blob_board("Letter"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x5150, margin = sq(0.33inch), dir = joinpath(dir, "b"))
            o = JSON.parse(read(_onlyext(joinpath(dir, "b"), ".json"), String); dicttype = Dict{String, Any})
            @test o["units"]["length"] == "inch"
            # a v5 file declaring inches, its PDF's bleed in them: the retired units layout, no envelope
            o["schema"] = "blobboards/board/v5"
            o["units"] = Dict{String, Any}("world" => "inch", "density" => "dpi", "lengths" => Dict{String, Any}("pattern" => "pt"))
            delete!(o, "generator"); delete!(o, "conventions")
            o["render"]["bleed"] = 0.125
            v5 = joinpath(dir, "v5.json")
            write(v5, JSON.json(o))
            @test VgtkIO.vgio_migrate(v5; out = joinpath(dir, "v6.json")).to == "blobboards/board/v7.0"
            n = JSON.parse(read(joinpath(dir, "v6.json"), String))
            @test n.units.length == "inch" && n.render.bleed === 0.125
            old, _ = BlobBoards._parse_bundle(v5)
            new, _ = BlobBoards._parse_bundle(joinpath(dir, "v6.json"))
            @test VgtkIO.same_value(old, new; length_ulps = 0) && VgtkIO.same_value(new, b; length_ulps = 0)
            @test BlobBoards._render_options(BlobBoards._read_record(joinpath(dir, "v6.json"))).bleed == 0.125inch
            # a raster's density, migrated as the file states it (52.4 dpi does not survive a unit round trip)
            o["render"] = Dict{String, Any}(
                "format" => "png", "icc_profile" => nothing, "id_label" => false, "shading_function" => nothing,
                "density" => 52.4, "bleed" => nothing,
            )
            write(v5, JSON.json(o))
            VgtkIO.vgio_migrate(v5; out = joinpath(dir, "v6b.json"))
            @test JSON.parse(read(joinpath(dir, "v6b.json"), String)).render.density === 52.4
        end
    end

    @testset "the migration enforces the uid: stored hash carried, identity(migrated) == identity(old); a doctored output is refused" begin
        legacy = joinpath(@__DIR__, "..", "fixtures", "legacy", "hero_board_v4.json")
        mig(src, dst; kw...) = BlobBoards._migrate_bundle_file(src, dst, "blobboards/board/v4"; kw...)
        mktempdir() do dir
            # a doctored MIGRATED file (a blob moved after the write) is refused and leaves no output
            dst = joinpath(dir, "out.json")
            moved(path) = (o = JSON.parse(read(path, String); dicttype = Dict{String, Any}); o["blobs"]["centers"][1][1] += 0.01; write(path, JSON.json(o)))
            @test_throws ArgumentError mig(legacy, dst; after_write = moved)
            @test !ispath(dst)
            # … so is one whose stored hash was changed
            rehash(path) = (o = JSON.parse(read(path, String); dicttype = Dict{String, Any}); o["hashes"]["config"] = "0" ^ 64; write(path, JSON.json(o)))
            @test_throws ArgumentError mig(legacy, dst; after_write = rehash)
            @test !ispath(dst)
            # a stored hash that does not recompute is authoritative, not a refusal: it is carried over (and logged)
            obj = JSON.parse(read(legacy, String); dicttype = Dict{String, Any})
            obj["hashes"]["config"] = "0" ^ 64
            old = joinpath(dir, "old.json")
            write(old, JSON.json(obj))
            r = @test_logs (:info, r"not the stored hashes.config") match_mode = :any VgtkIO.vgio_migrate(old; out = joinpath(dir, "new.json"))
            @test r.config_hash == "0" ^ 64 && JSON.parse(read(joinpath(dir, "new.json"), String)).hashes.config == "0" ^ 64
        end
    end

    @testset "the identity tree is the frozen {value, unit} form, whole (no quantity leaks a bare number)" begin
        quantity_object(d, k) = d[k] isa AbstractDict && keys(d[k]) == Set(["value", "unit"])
        pt_board = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, dir = mktempdir())
        bc = BlobBoards._identity_lowering(pt_board.preamble)["board_config"]
        pc = BlobBoards._identity_lowering(pt_board.preamble)["pattern_config"]
        @test all(k -> quantity_object(pc, k), ("min_scale", "max_scale"))
        @test pc["min_scale"]["unit"] == "pt" && bc["layout_grid"] == Dict{String, Any}("value" => 0.01, "unit" => "pt")
        @test all(k -> quantity_object(bc["canvas_size"], k), ("width", "height")) && all(k -> quantity_object(bc["margin"], k), ("width", "height"))
        @test all(k -> quantity_object(bc, k), ("ruler_width", "corner_size", "frame_padding"))
        @test all(t -> all(k -> quantity_object(bc["ruler_spec"][t], k), ("spacing", "width")), ("major_tick", "minor_tick"))
        px_board, _, _ = BlobBoards.load_bundle(joinpath(_FIXTURES, "odd300.json"))
        tree = BlobBoards._identity_lowering(px_board.preamble)
        @test quantity_object(tree["board_config"], "print_density") && quantity_object(tree["board_config"], "frame_width")
        @test all(k -> quantity_object(tree["pattern_config"]["pattern_resolution"], k), ("width", "height"))
        @test all(k -> quantity_object(tree["pattern_config"], k), ("min_scale", "max_scale"))
        # and the blob columns of the identity declare their unit, as they always have; the FILE's do not
        @test BlobBoards._identity_lowering(BlobBoards._identity_blobs(pt_board.blobs, "pt"))["unit"] == "pt"
        @test !haskey(BlobBoards._file_blobs(pt_board.blobs), :unit)
    end

    @testset "the file format is not part of an identity" begin
        @test BlobBoards.BOARD_SCHEMA == "blobboards/board/v7.0" && BlobBoards.PATTERN_SCHEMA == "blobboards/pattern/v5.0"
        @test BlobBoards.IDENTITY_SCHEMA == "blobboards/board/v4"            # frozen: a bump of the file schema moves no uid
        b = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, dir = mktempdir())
        # the identity tree names the identity schema and holds the quantities in its own form
        tree = BlobBoards._identity_lowering(b.preamble)
        @test tree["board_config"]["canvas_size"]["width"] == Dict{String, Any}("value" => 148, "unit" => "mm") ||
            tree["board_config"]["canvas_size"]["width"] == Dict{String, Any}("value" => 148.0, "unit" => "mm")
        # … while the file holds numbers, and the two forms never leak into each other: they are two styles
        file(q) = BlobBoards._lowered(BlobBoards._FILE_STYLE, q)
        @test file(148.0mm) == 148.0 && file(300dpi) == 300 && file(64px) == 64
        @test BlobBoards._identity_lowering(148.0mm) == Dict{String, Any}("value" => 148.0, "unit" => "mm")
        @test file(148.0mm) == 148.0                              # after `_identity_lowering`, still a file's number
        # no BlobBoards lowering lands on JSON.jl's own style
        for T in (typeof(b), typeof(b.preamble), typeof(b.preamble.board_config), typeof(b.preamble.pattern_config), typeof(148.0mm))
            @test parentmodule(which(JSON.lower, Tuple{JSON.JSONStyle, T})) !== BlobBoards
        end
    end

    @testset "memory → file → memory is exact: a board file declares the world unit its lengths read back in" begin
        # the unit a length is written in is the first of mm, inch and m in which it reads back bit for bit
        misses = 0
        for v in 0:0.01:200, u in (mm, inch)      # millimetre- and inch-built lengths
            q = BlobBoards.base_quantity(v * u)
            w = BlobBoards._exact_length_unit((q,))
            misses += BlobBoards.@with(BlobBoards._FILE_UNITS => (w, dpi), BlobBoards._file_quantity(BlobBoards._file_number(q; unit = w), :length)) !== q
        end
        @test misses == 0
        @test all(v -> BlobBoards._exact_length_unit((BlobBoards.base_quantity(v * mm),)) == mm, 0:0.01:200)   # every mm-built length stays mm
        # boards built in inches, or mixing inches and millimetres: saved and loaded to the very same values, the
        # values they were built with (a 0.33 inch margin is 0.33 inch), under the unit the file declares
        sq(v) = BlobBoards.Size2(width = v, height = v)
        for (paper, kw) in (("Letter", (; margin = sq(0.33inch))), ("A4", (; margin = sq(0.33inch), corner_size = 0.37inch, frame_padding = 0.03inch)))
            mktempdir() do dir
                b = blob_board(paper; blob_sizes = (4.0mm, 0.2short_side), seed = 0x5150, dir, kw...)
                @test b.preamble.board_config.margin.width == 0.33inch
                file = _onlyext(dir, ".json")
                # not millimetres: no file number in them reads back to a 0.33 inch margin
                @test BlobBoards.parse_unit(JSON.parse(read(file, String)).units.length) != mm
                back, stored = BlobBoards._parse_bundle(file)
                @test VgtkIO.same_value(back, b; length_ulps = 0) && stored == hash_config(b.preamble) == BlobBoards._bundle_identity(back)
            end
        end
        # the Letter board's code spells its margin as built
        b = blob_board("Letter"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x5150, margin = sq(0.33inch))
        @test occursin("/0.33INCH/0.33INCH/", BlobBoards.board_code(b.preamble))
    end

    @testset "IO reads what a board writes: units object, roles, numbers" begin
        hero = VgtkIO.read_record(joinpath(pkgdir(BlobBoards), "docs", "hero_board.json"))   # a v6 file, through its step
        s = VgtkIO.VgioStyle(hero)
        @test s.contract == VgtkIO.UnitContract(world = mm, density = dpi, lengths = (pattern = pt,))
        @test VgtkIO.role_unit(s.contract, :pattern) == pt
        bundle, _, _ = BlobBoards.load_bundle(joinpath(pkgdir(BlobBoards), "docs", "hero_board.json"))
        bc, pc = bundle.preamble.board_config, bundle.preamble.pattern_config
        # IO's decimal-exact reader agrees with the board's own to rounding (an ulp: a board is built from `210.0mm`)
        Q = BlobBoards.Quantity
        @test JSON.lift(s, Q, hero.preamble.board_config.canvas_size.width, (; quantity = :length)) ≈ bc.canvas_size.width rtol = 4eps()
        @test JSON.lift(s, Q, hero.preamble.board_config.corner_size, (; quantity = :length)) ≈ bc.corner_size rtol = 4eps()
        @test JSON.lift(s, Q, hero.preamble.pattern_config.min_scale, (; quantity = :length, role = :pattern)) ≈ ustrip(px, pc.min_scale) * pt rtol = 4eps()
        @test hero.preamble.board_config.layout_grid == 0.01                        # a number in the `pattern` role (pt)
    end

    # Re-pinned 2026-11-13 for VisualGeometryToolkitIO 0.10's canonical JSON (RFC 8785 numbers: 1.0e-6 is written
    # 0.000001), the stack's one deliberate re-id; the trees are unchanged (checked against 0.6.2's text).
    @testset "annotation and run-config hashes" begin
        @test BlobBoards.config_hash(;
            config = BlobBoards.FindBoardsConfig(), cutoff = 2.0, patch_size = 64, outer_extent = 1.5, min_iou = 0.5
        ) == "beae63bcd83efb6e1983a85db1f4af538115fb17d0fad5cefd7aae2cefd71162"
        @test _sha(BlobBoards.canonical_config_json(BlobBoards.config_dict(BlobBoards.FindBoardsConfig()))) ==
            "2fe33da39ad147966ea25292f5280ad6d1b7613b45f37a0d14abafe3dbd147fa"
    end
end
