# =============================================================================
# The design resolver: a board uid → its nominal plane points (pipeline.jl).
# A stored target rig (VisualGeometryIo rig/v4) names a board by reference;
# BlobBoards resolves it to the board's blob centres in its world frame, in the
# bundle's blob order — the order a measurement's point index names.
# =============================================================================
using Test
using StaticArrays: SVector
using VisualGeometryCore: EuclideanMap, PlanarTarget, Mounted, Rig, BodyFrame, is_resolved, points, mm, dpi, Vec3
import VisualGeometryIo as VGIo
using Rotations: QuatRotation

@testset "board design resolver" begin
    mktempdir() do dir
        bundle = blob_board("A5"; blob_sizes = (4.0mm, 0.2BlobBoards.short_side), seed = 0x0000d351, dir, output = Tiff(100dpi))
        lib = board_library(dir)
        @test length(lib) == 1
        uid = only(keys(lib))
        pts = board_points(bundle)
        @test pts == [SVector{2, Float64}(d.center) for d in BlobBoards.blobs(bundle, :world)]   # world frame, blob order
        @test length(pts) == length(bundle.blobs)
        @test board_points(uid, dir) == pts && board_points(uid, lib) == pts
        @test_throws ArgumentError board_points(:zzz, lib)
        resolve = board_resolver(dir)
        @test resolve(uid) == pts && resolve(:zzz) === nothing
        t = board_target(uid, lib)
        @test t == PlanarTarget(uid; nominal = pts) && points(t) == pts

        # a stored rig names the board by reference; the resolver gives its points back
        # (a mount stored as its quaternion: the value VisualGeometryIo reads back, exactly)
        q = SVector(0.9, 0.1, -0.2, 0.3) / sqrt(0.95)
        X = EuclideanMap{3, Float64, QuatRotation{Float64}, Vec3{Float64}}(QuatRotation(q..., false), Vec3{Float64}(10.0, -5.0, 2.0))
        rig = Rig{BodyFrame{:board_rig}}(Dict(uid => Mounted(t, X)))
        f = joinpath(dir, "board_rig.json")
        VGIo.save(f, rig)
        unresolved = VGIo.load(Rig, f)
        @test !is_resolved(unresolved[uid].target) && unresolved[uid].target == PlanarTarget(uid)
        back = VGIo.load(Rig, f; resolve)
        @test back[uid].target == t && back[uid].mount == rig[uid].mount
        @test points(back, uid) == points(rig, uid)
    end
end
