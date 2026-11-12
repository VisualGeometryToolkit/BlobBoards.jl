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

@testitem "cameras of record: the exact native port of the upstream export" tags = [:data] setup = [BBEval] begin
    using DrWatson: projectdir
    import TOML
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: BoundedCamera, CameraModel, Size2
    using VisualGeometryToolkitIO: opencv_lens
    using VisualGeometryToolkitEval.Experiments: is_primary_copy
    using StaticArrays: SMatrix
    using .BBEval.BlobBoardsEval: upstream_camera, upstream_source

    # The previous cameras of record (canonical frame), by value.
    const PREVIOUS = Dict(
        "260527" => (K = SMatrix{3, 3, Float64}(2461.4310993500767, 0, 0, 0, 2460.563045263503, 0, 2140.1593652825422, 1400.3009044208025, 1),
            k = [-0.15680625941667098, 0.04565665542192153, -0.0003527415347489064, 0.0008423616851487834], size = (4240, 2832)),
        "260521" => (K = SMatrix{3, 3, Float64}(2468.2077040546915, 0, 0, 0, 2468.247581880029, 0, 2147.861813010314, 1402.5157043838487, 1),
            k = [-0.15749700505351585, 0.048447063332657615, -0.0004763899767071182, 0.0008518580750734769], size = (4240, 2832)),
    )

    cfg = TOML.parsefile(projectdir("configs", "import_cameras.toml"))
    @test sort([c["session"] for c in cfg["cameras"]]) == sort(collect(keys(PREVIOUS)))
    for entry in cfg["cameras"]
        prev = PREVIOUS[entry["session"]]
        src_rel = cfg["inputs"][entry["input"]]["path"]
        @test !isabspath(src_rel)                                                          # data_root()-relative, never a machine path
        # the export is an input under the data root: without a bound one, or
        # without the file, a recorded skip (never an install of the artifact
        # view, which upstream_source would trigger)
        # (Eval's input_path refuses a missing input: absent here, the check is a recorded skip)
        if !is_primary_copy() || (try upstream_source(cfg, entry); false catch e; e isa ArgumentError || rethrow(); true end)
            @info "upstream export of $(entry["session"]) not on this machine — conversion check skipped"
            @test_skip false
            continue
        end
        name, camera, _ = upstream_camera(cfg, entry)
        bc = getfield(camera, :value)
        @test String(name) == entry["name"]
        @test SMatrix{3, 3, Float64}(bc.camera.intrinsics.K) == prev.K                     # bit for bit
        @test bc.camera.lens == CameraModel(prev.K; lens = opencv_lens(prev.k)).lens
        @test bc.bound == Size2(prev.size...)
        cal = getfield(camera, :attributes).calibration
        @test !haskey(cal, :reprojection_rms)                                              # the export's error is a mean
        @test occursin("mean reprojection error", cal.note) && occursin("modification date", cal.note)
        @test cal.date == entry["date"] && startswith(cal.source, basename(src_rel))
        # the native file reads back to the same camera
        mktempdir() do dir
            f = joinpath(dir, "c.json")
            VgtkIO.save(f, Dict(name => camera))
            @test VgtkIO.load(Dict{Symbol, BoundedCamera}, f) == Dict(name => camera)
        end
    end
end
