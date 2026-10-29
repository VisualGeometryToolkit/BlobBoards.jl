# =============================================================================
# The design resolver: a board uid → its nominal plane points (pipeline.jl).
# A stored target rig (VisualGeometryToolkitIO rig/v5) names a board by reference
# and carries the board it is printed on; BlobBoards resolves the reference to the
# board's blob centres in its world frame, in the bundle's blob order — the order
# a measurement's point index names.
# =============================================================================
using Test
using StaticArrays: SVector
using VisualGeometryToolkitCore: EuclideanMap, PlanarTarget, PosedTarget, Rig, BodyFrame, is_resolved, points, mm, dpi, Vec3, model, pose,
    board, canvas, reference, ustrip, m, px, Board, Size2
using SHA: sha256

import VisualGeometryToolkitIO as VgtkIO
using Rotations: QuatRotation

@testset "board design resolver" begin
    mktempdir() do dir
        bundle = blob_board("A5"; blob_sizes = (4.0mm, 0.2BlobBoards.short_side), seed = 0x0000d351, dir, output = Tiff(100dpi))
        lib = board_library(dir)
        @test length(lib) == 1
        uid = only(keys(lib))
        pts = board_points(bundle)
        plain = [SVector{2, Float64}(d.center) for d in BlobBoards.blobs(bundle, :world)]   # world frame, blob order, plain mm
        @test pts == [p .* mm for p in plain] && all(p -> eltype(p) === typeof(1.0mm), pts)
        @test length(pts) == length(bundle.blobs)
        @test board_points(uid, dir) == pts && board_points(uid, lib) == pts
        @test_throws ArgumentError board_points(:zzz, lib)
        resolve = board_resolver(dir)
        @test resolve(uid) == pts && resolve(:zzz) === nothing
        t = board_target(uid, lib)
        @test reference(t) === uid && points(t) == pts
        @test board_target(uid, dir) == t && board_target(uid, bundle) == t
        # the board: the board proper (inside the margins, frame included), in the world chart (canvas
        # centre, y up), lengths; inside the canvas, holding every blob centre
        bd = board(t)
        cfg = BlobBoards.board_config(bundle)
        @test all(isapprox.(ustrip.(mm, (bd.size.width, bd.size.height)), ustrip.(mm, (cfg.board_size.width, cfg.board_size.height)); atol = 0.01))   # the point grid: 1/7200 inch
        lo = ustrip.(mm, bd.origin .* m)
        hi = lo .+ ustrip.(mm, (bd.size.width, bd.size.height))
        W, H = BlobBoards._canvas_mm(cfg)
        @test all(lo .>= (-W / 2, -H / 2) .- 1.0e-9) && all(hi .<= (W / 2, H / 2) .+ 1.0e-9)
        @test all(p -> all(lo .<= p .<= hi), plain)
        # the canvas: the page, centred on the world origin, without a texture unless one is given
        cv = canvas(t)
        @test ustrip.(mm, (cv.size.width, cv.size.height)) == (W, H) && ustrip.(mm, cv.origin .* m) == SVector(-W / 2, -H / 2)
        @test cv.texture === nothing

        # the printed page as a texture: the file written, its SHA-256, the rectangle it covers — whole pixels
        # at the density from the canvas's top-left corner, the canvas to within one pixel
        png = joinpath(dir, "textures", "page.png")
        mkpath(dirname(png))
        tex = BlobBoards.save_canvas_texture(png, bundle; density = 100dpi, relative_to = dir)
        @test tex.path == joinpath("textures", "page.png") && tex.sha256 == bytes2hex(open(sha256, png))
        img = BlobBoards.PNGFiles.load(png)
        @test (tex.size.width, tex.size.height) == BlobBoards.base_quantity.((size(img, 2) * px / 100dpi, size(img, 1) * px / 100dpi))
        tlo = ustrip.(mm, tex.origin .* m)
        @test tlo[1] == -W / 2 && tlo[2] + ustrip(mm, tex.size.height) ≈ H / 2
        pitch = ustrip(mm, 1px / 100dpi)
        @test 0 <= ustrip(mm, tex.size.width) - W < pitch && 0 <= ustrip(mm, tex.size.height) - H < pitch
        tt = board_target(uid, lib; texture = tex)
        @test canvas(tt).texture == tex && board(tt) == board(t) && points(tt) == points(t)

        # a stored rig names the board by reference; the resolver gives its points back. The mount is built in
        # its stored form (a QuatRotation with metre translations), so the round trip through the file is exact.
        q = SVector(0.9, 0.1, -0.2, 0.3) / sqrt(0.95)
        tm = EuclideanMap(QuatRotation(q..., false), Vec3{Float64}(10.0, -5.0, 2.0) .* mm).t
        X = EuclideanMap{3, Float64, QuatRotation{Float64}, typeof(tm)}(QuatRotation(q..., false), tm)
        rig = Rig{BodyFrame{:board_rig}}(Dict(uid => PosedTarget(tt, X)))
        f = joinpath(dir, "board_rig.json")
        VgtkIO.save(f, rig)
        unresolved = VgtkIO.load(Rig, f)
        @test !is_resolved(model(unresolved[uid])) && reference(model(unresolved[uid])) === uid
        back = VgtkIO.load(Rig, f; resolve)
        @test points(model(back[uid])) == points(tt) && reference(model(back[uid])) === uid
        @test board(model(back[uid])) == board(tt)
        # the canvas and its texture: the rig/v5 planar body's `canvas` block, back bit for bit (FlexUnits
        # 0.6.16 rounds `x * mm` to the decimal-nearest metre value, Deduction42/FlexUnits.jl#131)
        bt = canvas(model(back[uid])).texture
        @test (bt.path, bt.sha256, bt.size) == (tex.path, tex.sha256, tex.size) && bt.origin ≈ tex.origin
        @test canvas(model(back[uid])).origin ≈ canvas(tt).origin
        @test canvas(model(back[uid])) == canvas(tt)
        @test model(back[uid]) == tt

        @test pose(back[uid]) == pose(rig[uid])
        @test points(back, uid) == points(rig, uid)

        # The example boards' 142 mm board survives the file bit for bit: 142mm is exact (71//500 m) and
        # stored as the decimal-nearest 0.142 m, which the millimetre file number reads back to.
        t142 = PlanarTarget(uid; nominal = [SVector(0.0, 0.0) .* mm], board = Board(Size2(width = 142mm, height = 142mm), SVector(-71.0, -71.0) .* mm))
        f142 = joinpath(dir, "board_142.json")
        VgtkIO.save(f142, Rig{BodyFrame{:board_rig}}(Dict(uid => PosedTarget(t142, X))))
        @test board(model(VgtkIO.load(Rig, f142)[uid])) == board(t142)
    end
end
