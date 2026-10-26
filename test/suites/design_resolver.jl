# =============================================================================
# The design resolver: a board uid → its nominal plane points (pipeline.jl).
# A stored target rig (VisualGeometryToolkitIO rig/v4) names a board by reference;
# BlobBoards resolves it to the board's blob centres in its world frame, in the
# bundle's blob order — the order a measurement's point index names.
# =============================================================================
using Test
using StaticArrays: SVector, SMatrix
using VisualGeometryToolkitCore: EuclideanMap, PlanarTarget, PosedTarget, Rig, BodyFrame, is_resolved, points, ustrip, with_unit, mm, dpi, m, Vec3, model, pose
import VisualGeometryToolkitIO as VgtkIO
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
        # (a mount stored as its quaternion, read back as a rotation matrix)
        q = SVector(0.9, 0.1, -0.2, 0.3) / sqrt(0.95)
        X = EuclideanMap(QuatRotation(q..., false), Vec3{Float64}(10.0, -5.0, 2.0) .* m)
        # VisualGeometryToolkitIO writes and reads lengths as quantities (a plain target is refused at save):
        # the board's millimetres are attached here, the stack's unitful form of the same points.
        tq = with_unit(t, mm)
        resolve_mm = id -> (p = resolve(id); p === nothing ? nothing : [x .* mm for x in p])
        rig = Rig{BodyFrame{:board_rig}}(Dict(uid => PosedTarget(tq, X)))
        f = joinpath(dir, "board_rig.json")
        VgtkIO.save(f, rig)
        unresolved = VgtkIO.load(Rig, f)
        @test !is_resolved(model(unresolved[uid])) && model(unresolved[uid]) == PlanarTarget(uid)
        back = VgtkIO.load(Rig, f; resolve = resolve_mm)
        # the mount is metres stored in the file's unit: equal to rounding, not bitwise
        @test isapprox(SMatrix{3, 3, Float64}(pose(back[uid]).R), SMatrix{3, 3, Float64}(pose(rig[uid]).R); rtol = 1.0e-12)
        @test all(isapprox.(ustrip.(m, pose(back[uid]).t), ustrip.(m, pose(rig[uid]).t); rtol = 1.0e-12))
        # the points are millimetres stored in metres: equal to the last bit or two, not bitwise
        @test all(zip(points(back, uid), points(rig, uid))) do (a, b)
            all(isapprox.(ustrip.(mm, a), ustrip.(mm, b); rtol = 1.0e-12))
        end
    end
end
