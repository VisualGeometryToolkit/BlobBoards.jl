# =============================================================================
# The cameras of record (scripts/import_cameras.jl): each is the native
# vgio/cameras/v3 port of its session's upstream calibration export, and the
# port is EXACT — K and the lens equal, bit for bit, the camera every
# detection of record was made with (the previous record, stated here by
# value: 260527's vgio/cameras/v3 of record version 1.1.0's draft metadata
# tier, 260521's vgio/cameras/v2 store). The port moves only the principal
# point, from OpenCV's pixel frame into the canonical one (+0.5 px, exact in
# Float64 for these values).
# =============================================================================

using Test
using DrWatson: projectdir
import TOML
import VisualGeometryToolkitIo as VGIo
using VisualGeometryToolkitCore: BoundedCamera, CameraModel, Size2
using VisualGeometryToolkitIo: opencv_lens
using StaticArrays: SMatrix

# The previous cameras of record (canonical frame), by value.
const PREVIOUS = Dict(
    "260527" => (K = SMatrix{3, 3, Float64}(2461.4310993500767, 0, 0, 0, 2460.563045263503, 0, 2140.1593652825422, 1400.3009044208025, 1),
        k = [-0.15680625941667098, 0.04565665542192153, -0.0003527415347489064, 0.0008423616851487834], size = (4240, 2832)),
    "260521" => (K = SMatrix{3, 3, Float64}(2468.2077040546915, 0, 0, 0, 2468.247581880029, 0, 2147.861813010314, 1402.5157043838487, 1),
        k = [-0.15749700505351585, 0.048447063332657615, -0.0004763899767071182, 0.0008518580750734769], size = (4240, 2832)),
)

if !isdefined(Main, :BlobBoardsEval)
    include(projectdir("src", "BlobBoardsEval.jl"))
end
using .BlobBoardsEval: upstream_camera, upstream_source

@testset "cameras of record: the exact native port of the upstream export" begin
    cfg = TOML.parsefile(projectdir("configs", "import_cameras.toml"))
    @test sort([c["session"] for c in cfg["cameras"]]) == sort(collect(keys(PREVIOUS)))
    for entry in cfg["cameras"]
        prev = PREVIOUS[entry["session"]]
        @test !isabspath(entry["source"])                                                  # data_root()-relative, never a machine path
        if !isfile(upstream_source(entry))
            @info "upstream export of $(entry["session"]) not on this machine — conversion check skipped"
            continue
        end
        name, camera, _ = upstream_camera(entry, cfg["upstream"])
        bc = getfield(camera, :value)
        @test String(name) == entry["name"]
        @test SMatrix{3, 3, Float64}(bc.camera.intrinsics.K) == prev.K                     # bit for bit
        @test bc.camera.lens == CameraModel(prev.K; lens = opencv_lens(prev.k)).lens
        @test bc.extent == Size2(prev.size...)
        cal = getfield(camera, :attributes).calibration
        @test !haskey(cal, :reprojection_rms)                                              # the export's error is a mean
        @test occursin("mean reprojection error", cal.note) && occursin("modification date", cal.note)
        @test cal.date == entry["date"] && startswith(cal.source, basename(entry["source"]))
        # the native file reads back to the same camera
        mktempdir() do dir
            f = joinpath(dir, "c.json")
            VGIo.save(f, Dict(name => camera))
            @test VGIo.load(Dict{Symbol, BoundedCamera}, f) == Dict(name => camera)
        end
    end
end
