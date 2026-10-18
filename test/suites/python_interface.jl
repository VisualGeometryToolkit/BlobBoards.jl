# Python interface tests
# These tests verify that the Python interface produces identical results to Julia.
# `using PythonCall: pyconvert` at the top of runtests.jl triggers
# BlobBoardsPythonExt, so the *_python wrappers are defined when this file
# runs. Inside `pixi run -e py julia ... Pkg.test()`, JULIA_PYTHONCALL_EXE
# points PythonCall at pixi's Python (numpy + pint preinstalled). The default
# pixi env has no Python: without `-e py` every test here errors with
# `No module named 'numpy'`.

@testset "Python Interface" begin
    @testset "Pattern Julia-Python equivalence" begin
        # Two cases (small + large) are sufficient to validate the Julia↔Python
        # boundary; additional sizes test the same code paths.
        test_cases = [
            # (name, width, height, smallest σ in px, seed)
            ("400x300 default", 400, 300, 3, 0x1234),
            ("1024x768 small scale", 1024, 768, 2, 0xBEEF),
        ]

        for (name, width, height, σ_min, seed) in test_cases
            @testset "$name" begin
                jdir = mktempdir()
                pdir = mktempdir()

                # Generate with Julia interface
                _, julia_blobs, julia_config = blob_pattern(
                    width, height;
                    blob_sizes = (4σ_min * px, nothing), seed = seed, cutoff = 2.0, dir = jdir
                )

                # Generate with Python interface
                BlobBoards.blob_pattern_python(
                    width, height;
                    blob_sizes = (4σ_min, nothing), seed = seed, cutoff = 2.0, dir = pdir
                )

                # Load JSONs
                jfile = only(filter(f -> endswith(f, ".json"), readdir(jdir; join = true)))
                pfile = only(filter(f -> endswith(f, ".json"), readdir(pdir; join = true)))

                jjson = JSON3.read(read(jfile, String))
                pjson = JSON3.read(read(pfile, String))

                # Compare pattern_config
                @test JSON3.write(jjson.pattern_config) == JSON3.write(pjson.pattern_config)

                # Compare blobs
                @test JSON3.write(jjson.blobs) == JSON3.write(pjson.blobs)

                # Compare UID (from filename)
                j_uid = match(r"blob_pattern_([^_]+)_", basename(jfile)).captures[1]
                p_uid = match(r"blob_pattern_([^_]+)_", basename(pfile)).captures[1]
                @test j_uid == p_uid

                # Verify counts match
                @test length(jjson.blobs.centers) == length(julia_blobs)
            end
        end
    end

    @testset "Board Julia-Python equivalence" begin
        # Trimmed to the two configurations that exercise the most surface:
        # - ruler frame (default), Letter paper, base options
        # - no marks, A5, custom seed (covers the corner_size and ruler_width
        #   kwargs + non-A4 paper size)
        test_cases = [
            (
                "Letter ruler",
                (paper = "Letter", blob_sizes = (4.0mm, 0.2short_side), seed = 0x1111),
                (paper = "Letter", blob_sizes = (4.0, 0.2 * pyimport("blobboards").short_side), seed = 0x1111),
            ),

            (
                "A5 no frame",
                (paper = "A5", blob_sizes = (4.0mm, 0.2short_side), seed = 0x5678, corner_size = 0mm, ruler_width = 0mm),
                (paper = "A5", blob_sizes = (4.0, 0.2 * pyimport("blobboards").short_side), seed = 0x5678, corner_size = 0, ruler_width = 0),
            ),
        ]

        for (name, jkw, pkw) in test_cases
            @testset "$name" begin
                jdir = mktempdir()
                pdir = mktempdir()

                # Julia kwargs: the case's, but the paper
                julia_kwargs = Dict{Symbol, Any}(pairs(Base.structdiff(jkw, (; paper = 0))))
                julia_kwargs[:dir] = jdir

                # Generate with Julia interface
                julia_bundle = blob_board(jkw.paper; julia_kwargs...)

                # Python kwargs: the case's, but the paper
                python_kwargs = Dict{Symbol, Any}(pairs(Base.structdiff(pkw, (; paper = 0))))
                python_kwargs[:dir] = pdir

                # Generate with Python interface
                BlobBoards.blob_board_python(pkw.paper; python_kwargs...)

                # Load JSONs
                jfile = only(filter(f -> endswith(f, ".json"), readdir(jdir; join = true)))
                pfile = only(filter(f -> endswith(f, ".json"), readdir(pdir; join = true)))

                jjson = JSON3.read(read(jfile, String))
                pjson = JSON3.read(read(pfile, String))

                # Compare preamble
                @test JSON3.write(jjson.preamble) == JSON3.write(pjson.preamble)

                # Compare blobs
                @test JSON3.write(jjson.blobs) == JSON3.write(pjson.blobs)

                # Compare config hash
                @test jjson.hashes.config == pjson.hashes.config

                # Compare UID (from filename)
                j_uid = match(r"blob_board_([^_]+)_", basename(jfile)).captures[1]
                p_uid = match(r"blob_board_([^_]+)_", basename(pfile)).captures[1]
                @test j_uid == p_uid

                # Verify counts match
                @test length(jjson.blobs.centers) == length(pjson.blobs.centers)
            end
        end
    end

    @testset "Board sizes across units and measures (1200 dpi, no frame)" begin
        # The SAME blobs spelled two ways: Julia gives disc diameters in mm and a
        # short-side max; Python gives Pint INCHES and the same short-side
        # fraction. One stored config, one identity, one blob list — the stored
        # pair is σ in px, not the spelling.
        ureg = pyimport("pint").UnitRegistry()
        ss = pyimport("blobboards").short_side
        c = 3.0
        jdir, pdir = mktempdir(), mktempdir()
        jb = blob_board(
            50mm, 40mm; corner_size = 0mm, ruler_width = 0mm, cutoff = c, seed = 0x1200,
            blob_sizes = (2c * 0.0635mm, 0.2short_side), dir = jdir
        )
        presult = BlobBoards.blob_board_python(
            (50, 40); corner_size = 0, ruler_width = 0, cutoff = c, seed = 0x1200,     # bare numbers: mm
            blob_sizes = (2c * 0.0025 * ureg.inch, 0.2 * ss), dir = pdir
        )
        jjson = JSON3.read(read(only(filter(endswith(".json"), readdir(jdir; join = true))), String))
        pjson = JSON3.read(read(only(filter(endswith(".json"), readdir(pdir; join = true))), String))
        @test jjson.schema == "blobboards/board/v4"
        @test jjson.preamble.pattern_config.min_scale.value ≈ 0.18                # σ = 0.0025 inch, in points
        @test jjson.preamble.pattern_config.min_scale.unit == "pt"
        @test jjson.render.density === nothing                                   # a PDF has no density
        @test !haskey(jjson.preamble.pattern_config, :max_diameter_fraction)
        @test JSON3.write(jjson.preamble) == JSON3.write(pjson.preamble)
        @test JSON3.write(jjson.blobs) == JSON3.write(pjson.blobs)
        @test jjson.hashes.config == pjson.hashes.config == pyconvert(String, presult.config_hash)
        # id_label crosses the boundary: on by default (dark glyphs in the margin
        # under the board), off on request; the identity is unchanged either way
        odir = mktempdir()
        oresult = BlobBoards.blob_board_python(
            (120, 80); corner_size = 0, ruler_width = 0, cutoff = c, seed = 0x1200, board_size = (40, 30),
            blob_sizes = (2c * 0.0025 * ureg.inch, 0.2 * ss), id_label = false, dir = odir
        )
        ldir = mktempdir()
        lresult = BlobBoards.blob_board_python(
            (120, 80); corner_size = 0, ruler_width = 0, cutoff = c, seed = 0x1200, board_size = (40, 30),
            blob_sizes = (2c * 0.0025 * ureg.inch, 0.2 * ss), dir = ldir
        )
        margin(d) = let img = BlobBoards.rasterize_pdf(only(filter(endswith(".pdf"), readdir(d; join = true))), 300)
            Float64.(img[(end - 290):(end - 150), :]) ./ Float64(maximum(img))   # 25 mm margin: 1.5-13.5 mm under the board
        end
        @test pyconvert(String, oresult.config_hash) == pyconvert(String, lresult.config_hash)
        # the render block records how each sheet was made, from Python too
        rblock(d) = JSON3.read(read(only(filter(endswith(".json"), readdir(d; join = true))), String)).render
        @test rblock(odir).id_label === false && rblock(ldir).id_label === true
        @test rblock(ldir).format == "pdf" && !haskey(rblock(ldir), :pdf_standard)
        @test minimum(margin(odir)) > 0.99
        @test minimum(margin(ldir)) < 0.5
    end

    @testset "blobs_python extraction" begin
        dir = mktempdir()
        bundle = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, dir = dir)
        json_file = only(filter(f -> endswith(f, ".json"), readdir(dir; join = true)))

        # pyconvert imported from runtests.jl

        @testset "Pixel units: at a density the caller names" begin
            @test_throws Exception BlobBoards.blobs_python(json_file; units = "px")
            blobs_py = BlobBoards.blobs_python(json_file; units = "px", reference_frame = "pattern", image_origin = "opencv", density = 100)
            blobs_arr = pyconvert(Matrix{Float64}, blobs_py)
            @test size(blobs_arr, 1) == length(bundle.blobs)
            @test size(blobs_arr, 2) == 3  # x, y, σ
        end

        @testset "Physical units (mm)" begin
            blobs_py = BlobBoards.blobs_python(json_file; units = "mm", reference_frame = "pattern")
            blobs_arr = pyconvert(Matrix{Float64}, blobs_py)
            @test size(blobs_arr, 1) == length(bundle.blobs)
        end

        @testset "Different reference frames" begin
            b_pattern = pyconvert(Matrix{Float64}, BlobBoards.blobs_python(json_file; units = "mm", reference_frame = "pattern"))
            b_board = pyconvert(Matrix{Float64}, BlobBoards.blobs_python(json_file; units = "mm", reference_frame = "board"))
            b_canvas = pyconvert(Matrix{Float64}, BlobBoards.blobs_python(json_file; units = "mm", reference_frame = "canvas"))

            # All should have same number of blobs
            @test size(b_pattern, 1) == size(b_board, 1) == size(b_canvas, 1)
            # a v4 board's pattern is the board; the canvas is offset by the margin
            @test b_pattern[1, :] == b_board[1, :]
            @test b_pattern[1, 1] != b_canvas[1, 1]
        end

        @testset "Image origin conventions" begin
            b_opencv = pyconvert(Matrix{Float64}, BlobBoards.blobs_python(json_file; units = "px", image_origin = "opencv", density = 100))
            b_colmap = pyconvert(Matrix{Float64}, BlobBoards.blobs_python(json_file; units = "px", image_origin = "colmap", density = 100))
            b_julia = pyconvert(Matrix{Float64}, BlobBoards.blobs_python(json_file; units = "px", image_origin = "julia", density = 100))

            # OpenCV: center at (0, 0)
            # COLMAP: center at (0.5, 0.5)
            # Julia: center at (1, 1)
            # So julia = colmap + 0.5 = opencv + 1.0
            @test b_colmap[1, 1] ≈ b_opencv[1, 1] + 0.5 atol = 1.0e-6
            @test b_julia[1, 1] ≈ b_opencv[1, 1] + 1.0 atol = 1.0e-6
        end
    end

    # The `opencv` export (blobboards.correspondences): OpenCV is fed the
    # observations converted from their DECLARED frame — COLMAP's xy − 0.5 —
    # and the fit is written natively, the two files migrate_opencv writes.
    # cv2 itself is not in the py env, so the fit is replaced by a known camera
    # and known poses; everything around it runs.
    @testset "load_board / render_board: a v4 board's pixels at its own raster density or the caller's" begin
        dir = mktempdir()
        blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), seed = 0x40, dir)      # 9 mm discs
        json_file = only(filter(endswith(".json"), readdir(dir; join = true)))
        # a PDF has no density: its pixels are at the board's own (smallest disc 16 px across) …
        r = BlobBoards.load_board_python(json_file)
        @test pyconvert(Float64, r.density_dpi) == ceil(16 / (9 / 25.4))
        # … or at the one the caller names
        r = BlobBoards.load_board_python(json_file; density = 300)
        @test pyconvert(Float64, r.density_dpi) == 300
        out = mktempdir()
        BlobBoards.render_board_python(json_file; dir = out, format = "png", density = 300)
        @test any(endswith(".png"), readdir(out))
        # the output is named by strings and numbers from Python, with the Julia types' rules
        @test_throws Exception BlobBoards.render_board_python(json_file; dir = out, format = "png")                       # a raster needs its density
        @test_throws Exception BlobBoards.render_board_python(json_file; dir = out, format = "png", density = 300, icc_profile = "linear")
        @test_throws Exception BlobBoards.render_board_python(json_file; dir = out, format = "tiff", density = 300, shading_function = "sampled")
        @test_throws Exception BlobBoards.render_board_python(json_file; dir = out, format = "svg")
        # a file is written at the OUTPUT's density, whatever raster sits beside the JSON:
        # a board kept as a 300 dpi PNG re-renders as a 600 dpi TIFF, and its blobs come back at 600
        pngdir = mktempdir()
        blob_board("A5"; blob_sizes = (9mm, 0.2short_side), seed = 0x40, dir = pngdir, output = Png(300dpi))
        png_json = only(filter(endswith(".json"), readdir(pngdir; join = true)))
        tiff = mktempdir()
        rt = BlobBoards.render_board_python(png_json; dir = tiff, format = "tiff", density = 600)
        @test any(endswith(".tiff"), readdir(tiff)) && pyconvert(Float64, rt.density_dpi) == 600
        @test pyconvert(Float64, BlobBoards.load_board_python(png_json).density_dpi) == 300
        pdf = mktempdir()
        BlobBoards.render_board_python(json_file; dir = pdf, shading_function = "sampled")
        @test JSON3.read(read(only(filter(endswith(".json"), readdir(pdf; join = true))), String)).render.shading_function == "sampled"
    end

    @testset "opencv export: OpenCV's pixels from the declared frame, native v2 out" begin
        import PythonCall
        VGIo, VGC, Rot = BlobBoards.VisualGeometryToolkitIo, BlobBoards.VisualGeometryToolkitCore, BlobBoards.Rotations
        corr = PythonCall.pyimport("blobboards.correspondences")
        np = PythonCall.pyimport("numpy")
        W, H, f, cx, cy = 1280, 960, 1000.0, 640.5, 480.5                       # canonical (COLMAP) principal point
        grid = [VGC.Vec3{Float64}(20.0 * i, 20.0 * j, 0.0) for i in 0:4 for j in 0:3]
        poses = [(Rot.RotMatrix{3, Float64}(Rot.RotXYZ(0.1 * k, -0.15, 0.05 * k)), VGC.Vec3{Float64}(-40.0, -30.0, 600.0 + 50k)) for k in 1:3]
        pix(R, t, X) = (P = R * X + t; VGC.Vec2{Float64}(f * P[1] / P[3] + cx, f * P[2] / P[3] + cy))
        function run_dir(two_targets)
            dir = mktempdir()
            ids = two_targets ? (:b1, :b2) : (:b1,)
            idpose = VGC.EuclideanMap(one(Rot.RotMatrix{3, Float64}), VGC.Vec3{Float64}(0, 0, 0))
            plane = [VGC.Vec2{Float64}(X[1], X[2]) for X in grid]
            rig = VGC.Rig{VGC.TargetFrame{:b1}}(Dict(id => VGC.Mounted(VGC.PlanarTarget(plane), idpose) for id in ids))
            tp = joinpath(dir, "targets_1.json")
            VGIo.save(tp, rig)
            obs = VGC.TargetPoints2D{VGC.PixelFrame{:colmap}}(
                Dict(
                    k => [(target = id, index = i, xy = pix(R, t, grid[i]), weight = 1.0) for id in ids for i in eachindex(grid)]
                        for (k, (R, t)) in enumerate(poses)
                );
                images = Dict(k => "f$(k).png" for k in eachindex(poses)), sensors = "cam", targets = [VGIo.content_id(tp)],
            )
            VGIo.save(joinpath(dir, "observations.json"), obs)
            cam = VGC.CameraModel(VGC.LogicalIntrinsics(VGC.CameraCalibrationMat(f * px, [cx * px, cy * px])), VGC.Pinhole())
            VGIo.save(joinpath(dir, "camera_mocap.json"), Dict(:cam => VGC.BoundedCamera(cam, VGC.Size2(W, H))))
            return dir
        end
        run = run_dir(false)
        frames = corr.load_correspondences(run, "opencv")
        @test pyconvert(Int, PythonCall.pybuiltins.len(frames)) == length(poses)
        for (k, (R, t)) in enumerate(poses)
            U = pyconvert(Matrix{Float64}, frames[k - 1]["image_points"])
            @test all(U[i, :] == pix(R, t, grid[i]) .- 0.5 for i in eachindex(grid))   # xy − 0.5, exactly
        end
        # a frame seeing two targets is refused, before OpenCV is even imported
        @test_throws PythonCall.PyException corr.calibrate_opencv(run_dir(true), "default")
        # the fit's output, written natively, reads back as the camera and poses it came from
        K_cv = np.array(PythonCall.pylist([PythonCall.pylist([f, 0.0, cx - 0.5]), PythonCall.pylist([0.0, f, cy - 0.5]), PythonCall.pylist([0.0, 0.0, 1.0])]))
        pp = PythonCall.pylist([
            PythonCall.pydict(Dict("image" => "f$(k).png", "target" => "b1", "rvec" => np.array(PythonCall.pylist(collect(Rot.params(Rot.RotationVec(R))))), "tvec" => np.array(PythonCall.pylist(collect(t)))))
                for (k, (R, t)) in enumerate(poses)
        ])
        out = mktempdir()
        paths = corr.write_native_calibration(out, "calibration_opencv_default", K_cv, np.zeros(5), pp, PythonCall.pytuple((W, H)), PythonCall.pydict(Dict("source" => "opencv", "reprojection_error" => 0.0, "error_statistic" => "rms")))
        cams, posesf = pyconvert(String, paths[0]), pyconvert(String, paths[1])
        @test VGIo.validate(cams) == "cameras" && VGIo.validate(posesf) == "transforms"
        store = VGIo.load(Dict{Symbol, VGC.BoundedCamera}, cams)
        rec = only(values(store))
        @test only(keys(store)) == :calibration_opencv_default && VGC.extent(rec) == VGC.Size2(W, H)
        @test VGC.model(rec).intrinsics.K ≈ VGC.CameraCalibrationMat(f * px, [cx * px, cy * px]) atol = 1.0e-12   # canonical: OpenCV's cx + 0.5
        # the writer's lens record IS VisualGeometryToolkitIo's opencv_lens, for every OpenCV vector length
        for n in (4, 5, 8, 12, 14)
            v = [1.0e-3 * (-1)^i * i for i in 1:n]
            pyrec = corr.opencv_lens_record(np.array(PythonCall.pylist(v)))
            rec_jl = JSON3.read(pyconvert(String, PythonCall.pyimport("json").dumps(pyrec)), Dict{String, Any})
            @test VGC.lens_from_record(rec_jl) === VGIo.opencv_lens(v)
        end
        @test_throws PythonCall.PyException corr.opencv_lens_record(np.zeros(6))
        recs = VGIo.load(Vector{VGC.EuclideanMap}, posesf)
        @test all(((r, k),) -> VGC.geom(r) isa VGC.EuclideanMap{3, Float64, <:Any, <:Any, VGC.TargetFrame{:b1}, VGC.ViewFrame{Symbol("f$(k).png")}}, zip(recs, eachindex(poses)))
        @test all(VGC.geom(r).R ≈ R && VGC.geom(r).t ≈ t for (r, (R, t)) in zip(recs, poses))
    end

    @testset "find_boards_python: the homography's pixel convention is stated" begin
        # BlobBoards' H maps the board's world plane (mm) to COLMAP pixels; the
        # binding re-expresses it in `image_origin`'s convention, image side only.
        ext = Base.get_extension(BlobBoards, :BlobBoardsPythonExt)
        SA = BlobBoards.StaticArrays
        H = SA.SMatrix{3, 3, Float64}(900.0, -8.0, 1.0e-4, 12.0, 905.0, 2.0e-4, 2100.0, 1400.0, 1.0)   # column-major
        X = SA.SVector(35.0, -20.0, 1.0)
        pix(G) = (u = G * X; SA.SVector(u[1] / u[3], u[2] / u[3]))
        @test pix(ext._homography_in(:opencv, H)) ≈ pix(H) .- 0.5
        @test pix(ext._homography_in(:colmap, H)) == pix(H)
        @test pix(ext._homography_in(:julia, H)) ≈ pix(H) .+ 0.5
        @test_throws ArgumentError BlobBoards.find_boards_python(["b.json"], "m.onnx", zeros(UInt8, 4, 4); gallery_path = "g.jld2", image_origin = "matlab")
    end

    @testset "find_boards_python: board paths and model lead, the image follows" begin
        img = zeros(UInt8, 4, 4)
        @test applicable(BlobBoards.find_boards_python, ["board.json"], "model.onnx", img)
        # the retired image-first order binds no method
        @test !applicable(BlobBoards.find_boards_python, img, ["board.json"], "model.onnx")
    end
end
