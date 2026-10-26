# =============================================================================
# The design resolver: a board uid → its nominal plane points (pipeline.jl).
# A stored target rig (VisualGeometryToolkitIO rig/v4) names a board by reference;
# BlobBoards resolves it to the board's blob centres in its world frame, in the
# bundle's blob order — the order a measurement's point index names.
# =============================================================================
using Test
using StaticArrays: SVector
using VisualGeometryToolkitCore: EuclideanMap, PlanarTarget, PosedTarget, Rig, BodyFrame, is_resolved, points, mm, dpi, Vec3, model, pose
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
        @test t == PlanarTarget(uid; nominal = pts) && points(t) == pts

        # a stored rig names the board by reference; the resolver gives its points back. The mount is built in
        # its stored form (a QuatRotation with metre translations), so the round trip through the file is exact.
        q = SVector(0.9, 0.1, -0.2, 0.3) / sqrt(0.95)
        tm = EuclideanMap(QuatRotation(q..., false), Vec3{Float64}(10.0, -5.0, 2.0) .* mm).t
        X = EuclideanMap{3, Float64, QuatRotation{Float64}, typeof(tm)}(QuatRotation(q..., false), tm)
        rig = Rig{BodyFrame{:board_rig}}(Dict(uid => PosedTarget(t, X)))
        f = joinpath(dir, "board_rig.json")
        VgtkIO.save(f, rig)
        unresolved = VgtkIO.load(Rig, f)
        @test !is_resolved(model(unresolved[uid])) && model(unresolved[uid]) == PlanarTarget(uid)
        back = VgtkIO.load(Rig, f; resolve)
        @test model(back[uid]) == t
        @test pose(back[uid]) == pose(rig[uid])
        @test points(back, uid) == points(rig, uid)
    end
end
