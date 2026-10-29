# =============================================================================
# FindBoardsConfig from TOML — round trip, refusal, and concreteness
# =============================================================================

import VisualGeometryToolkitCore as VgtkCore
using TOML
using Accessors: @set
using BlobBoards: FindBoardsConfig, write_config, read_run_config, config_dict,
    DetectConfig, EllipseNms, MxmNms, PoseGeometry, HomographyGeometry,
    AdaptConfig, DetectConfig
using VisualGeometryToolkitEstimation: RansacConfig
using LocalOptimizationDoneRight: StudentTUniform, HQ, tuning_constant
import VisualGeometryToolkitCore
using VisualGeometryToolkitCore: CameraModel, CameraCalibrationMat, Lens, Perspective, Additive, Radial, Tangential, WelschShape
using StaticArrays: SMatrix

const _CAM = CameraModel(
    CameraCalibrationMat(SMatrix{3, 3, Float64}(2461.4, 0, 0, 0, 2460.6, 0, 2139.7, 1399.8, 1));
    lens = Lens(Perspective(), Additive(Radial((-0.157, 0.046)), Tangential(-3.5e-4, 8.4e-4))), image_origin = :opencv
)

_write(dir, name, s) = (p = joinpath(dir, name); write(p, s); p)

@testset "FindBoardsConfig from TOML" begin

    @testset "reproduces the hand-built sweep config exactly" begin
        # This is the migration-safety property: the TOML must produce the SAME
        # config the driver builds by hand today, or porting the driver silently
        # changes results. Compared by config_dict AND by type, since the type
        # parameters are what keep the struct concrete.
        mktempdir() do dir
            p = _write(
                dir, "run.toml", """
                [find_boards]
                min_similarity = 0.6

                [find_boards.detect]
                peak_threshold = 0.01
                edge_threshold = 5.0
                nms = { type = "ellipse", cutoff = 2.0 }

                [find_boards.geometry]
                type = "pose"
                min_inliers = 8
                resolve_pose_ambiguity = true

                [find_boards.ransac]
                min_trials = 100
                max_trials = 10000
                """
            )
            from_toml = FindBoardsConfig(p; camera = _CAM)
            handbuilt = FindBoardsConfig(
                geometry = PoseGeometry(_CAM; min_inliers = 8, resolve_pose_ambiguity = true),
                ransac_config = RansacConfig(min_trials = 100, max_trials = 10_000),
                detect = DetectConfig(
                    peak_threshold = 0.01,
                    edge_threshold = 5.0, nms = EllipseNms(cutoff = 2.0f0)
                )
            )
            @test config_dict(from_toml) == config_dict(handbuilt)
            @test typeof(from_toml) == typeof(handbuilt)
        end
    end

    @testset "round trip" begin
        mktempdir() do dir
            for (geom, cam) in (
                        (PoseGeometry(_CAM; min_inliers = 8, resolve_pose_ambiguity = true), _CAM),
                        (HomographyGeometry(min_inliers = 6), nothing),
                    ),
                    nms in (nothing, EllipseNms(cutoff = 2.5f0), MxmNms(window_half = 4))

                cfg = FindBoardsConfig(
                    geometry = geom, detect = DetectConfig(nms = nms),
                    adapt = AdaptConfig(), min_similarity = 0.55
                )
                p = joinpath(dir, "rt.toml")
                write_config(p, cfg)
                back = FindBoardsConfig(p; camera = cam)
                @test config_dict(back) == config_dict(cfg)
                @test typeof(back) == typeof(cfg)     # concreteness survives the trip
            end
        end
    end

    @testset "the config stays concretely typed" begin
        # An abstract field would box and dynamically dispatch inside the
        # per-board fitting loop, which is why the struct is parametric at all.
        mktempdir() do dir
            p = _write(dir, "c.toml", "[find_boards.geometry]\ntype = \"homography\"\n")
            cfg = FindBoardsConfig(p)
            @test isconcretetype(typeof(cfg))
            @test isconcretetype(typeof(cfg.detect))
            @test isconcretetype(typeof(cfg.adapt))
            @test isconcretetype(typeof(cfg.geometry))
        end
    end

    @testset "local_optimization: the file's tag on every arm, overridable, round-trips" begin
        # The four Student-t + uniform IRLS literals that pipeline.jl carried
        # until 2026-09-04 are now `[find_boards.geometry] local_optimization`,
        # read through RVG's tagged loader over LocalOptimizationDoneRight's tags
        # (its `StudentTUniform`). The default names ONE thing.
        toml = TOML.parsefile(joinpath(pkgdir(BlobBoards), "configs", "default.toml"))["find_boards"]
        @test toml["geometry"]["local_optimization"] == Dict("type" => "student_t_uniform")
        @test HomographyGeometry().local_optimization isa StudentTUniform
        @test PoseGeometry(_CAM).local_optimization isa StudentTUniform
        @test BlobBoards.RadialHomographyGeometry((4240, 2832)).local_optimization isa StudentTUniform
        @test isconcretetype(typeof(HomographyGeometry())) && isconcretetype(typeof(PoseGeometry(_CAM)))
        mktempdir() do dir
            # A different strategy by tag; `scale_multiple = "cross_validated"` resolves per ARM.
            p = _write(
                dir, "hq.toml",
                "[find_boards.geometry]\ntype = \"pose\"\nlocal_optimization = { type = \"hq\", loss = \"welsch\", scale_multiple = \"cross_validated\", max_iter = 7 }\n"
            )
            g = FindBoardsConfig(p; camera = _CAM).geometry
            @test g.local_optimization isa HQ{WelschShape}
            @test g.local_optimization.width.value == tuning_constant(:absolute_pose, "welsch").scale_multiple   # a ScaleMultiple width
            @test g.local_optimization.max_iter == 7
            @test g.min_inliers == PoseGeometry(_CAM).min_inliers          # siblings inherit
            # It is part of the run record and reloads to the same strategy.
            d = config_dict(FindBoardsConfig(p; camera = _CAM))["geometry"]["local_optimization"]
            @test d["type"] == "hq" && d["loss"] == "welsch" && d["max_iter"] == 7
            q = write_config(joinpath(dir, "back.toml"), FindBoardsConfig(p; camera = _CAM))
            @test FindBoardsConfig(q; camera = _CAM).geometry.local_optimization == g.local_optimization
            # Unknown tags and knobs are refused, never defaulted.
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "bad.toml", "[find_boards.geometry]\ntype = \"homography\"\nlocal_optimization = { type = \"student_t\" }\n")
            )
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "bad2.toml", "[find_boards.geometry]\ntype = \"homography\"\nlocal_optimization = { type = \"student_t_uniform\", max_itr = 3 }\n")
            )
        end
    end

    @testset "extra tables survive and are readable" begin
        mktempdir() do dir
            cfg = FindBoardsConfig(geometry = HomographyGeometry())
            p = joinpath(dir, "run.toml")
            write_config(
                p, cfg; extra = Dict(
                    "experiment" => Dict("dataset" => "260527", "subsets" => ["bb"]),
                    "descriptor" => Dict("model" => "models/x.onnx", "patch_size" => 64)
                )
            )
            # `patch_size` is RETIRED (the ONNX graph declares it): the sidecar
            # still loads — records are immutable — but with a warning.
            rc = @test_logs (:warn, r"retired") match_mode = :any read_run_config(p)
            @test rc.experiment["dataset"] == "260527"
            @test rc.descriptor["model"] == "models/x.onnx"
            @test haskey(rc.config_toml, "find_boards")
            @test config_dict(FindBoardsConfig(p)) == config_dict(cfg)
        end
    end

    @testset "bad configs are refused, never silently defaulted" begin
        mktempdir() do dir
            # A typo'd knob must NOT run at the default and report success -- that
            # is the BB_LIMIT class of failure this loader exists to prevent.
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "typo.toml", "[find_boards.detect]\npeak_thresh = 0.01\n"); camera = _CAM
            )
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "typo2.toml", "[find_boards]\nmin_similarty = 0.6\n"); camera = _CAM
            )
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "nms.toml", "[find_boards.detect]\nnms = { type = \"blur\" }\n"); camera = _CAM
            )
            # `solve` is gone entirely: the Student-t/uniform mixture is the
            # solver, not a choice. Two of the three former options never reached
            # a kernel (the CUDA _adapt dispatches on AdaptConfig{T} alone) and the
            # third measured worse. A config naming one is now refused.
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "solve.toml", "[find_boards.adapt]\nsolve = { type = \"whitened_linear\" }\n"); camera = _CAM
            )
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "geom.toml", "[find_boards.geometry]\ntype = \"p3p\"\n"); camera = _CAM
            )
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "nofb.toml", "[experiment]\ndataset = \"x\"\n"); camera = _CAM
            )
            @test_throws ArgumentError FindBoardsConfig(joinpath(dir, "absent.toml"))

            # type = "pose" without a camera must RAISE, not fall back to the
            # uncalibrated path -- silently estimating a homography where the
            # caller asked for metric pose is a wrong answer that looks right.
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "nocam.toml", "[find_boards.geometry]\ntype = \"pose\"\n")
            )

            # `polarity` belongs to the BOARD, not the run: it is recorded in each
            # bundle and derived per run, so a config stating it could only agree
            # or disagree-and-silently-detect-nothing.
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "pol.toml", "[find_boards.detect]\npolarity = \"dark\"\n")
            )
            # `adapt.cutoff` is refused for the SAME reason, and the argument
            # that once made it a run knob has been answered. It is not merely
            # the paraboloid scan window: it is simultaneously the clip test's
            # extent and the canonicalization inner extent, and those are the
            # board's own printed boundary. A configured value that disagrees
            # with the gallery's makes the two sides' descriptors incomparable
            # SILENTLY (the example set spans 1.75..1.995, so adapt scanned at
            # 2.0 while filter_blobs and describe used the 1.8725 mean).
            #
            # This key was dropped once before and that removal WAS a bug —
            # nothing fed the field and the CUDA kernel kept reading it. It is
            # gone again only because `adapt_detections` now feeds it from the
            # gallery (`_shared_image_cutoff`). To widen or narrow the window,
            # use `scan_padding`, which keeps the extent anchored to the print.
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "cut.toml", "[find_boards.adapt]\ncutoff = 2.5\n")
            )

            # resolve_pose_ambiguity is calibrated-only: the two-fold ambiguity is
            # a property of decomposing H into (R, t, n) given K, so the
            # uncalibrated arm has no second branch. Refused with a message that
            # says INAPPLICABLE, not "unknown key" -- the latter sends the reader
            # hunting for a typo in a correctly-spelled knob.
            err = try
                FindBoardsConfig(
                    _write(
                        dir, "hamb.toml",
                        "[find_boards.geometry]\ntype = \"homography\"\nresolve_pose_ambiguity = true\n"
                    )
                )
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("applies only to", sprint(showerror, err))
            @test !occursin("unknown key", sprint(showerror, err))
        end
    end

    @testset "max_detections round-trips through its sentinel" begin
        # Uncapped is typemax(Int), written as "unbounded" because the literal
        # 9223372036854775807 is opaque in a config file. Both spellings read.
        mktempdir() do dir
            cfg = FindBoardsConfig(geometry = HomographyGeometry())   # uncapped
            p = joinpath(dir, "m.toml"); write_config(p, cfg)
            @test occursin("unbounded", read(p, String))
            @test FindBoardsConfig(p).max_detections == typemax(Int)
            @test FindBoardsConfig(
                _write(dir, "m2.toml", "[find_boards]\nmax_detections = 500\n")
            ).max_detections == 500
            @test_throws ArgumentError FindBoardsConfig(
                _write(dir, "m3.toml", "[find_boards]\nmax_detections = \"lots\"\n")
            )
        end
    end

    # =============================================================================
    # Overriding a loaded config
    # =============================================================================
    #
    # `find_boards` takes a whole `FindBoardsConfig`, never a per-knob keyword, so
    # a caller who wants to change one setting must rebuild the struct. Doing that
    # by hand means restating every field you are KEEPING -- which is how a sweep
    # ends up quietly diverging from the config of record it claims to extend.
    #
    # `Accessors.@set` is the supported way, and it needs no code in this package:
    # ConstructionBase's generic `setproperties` already handles these structs
    # because each is a plain `@kwdef` with no inner constructor, so its positional
    # constructor matches its field order. That is a real invariant and it is easy
    # to break by accident -- adding a validating inner constructor to any of them
    # breaks every override site with no compile-time signal. These tests are that
    # signal.

    @testset "there is exactly ONE default" begin
        # `FindBoardsConfig()` and configs/default.toml must name the SAME
        # configuration. They used to differ -- peak 0.02 vs 0.01, edge 10 vs 5,
        # nms off vs ellipse, 2_000 vs 10_000 iterations, homography vs pose -- so
        # "the default" meant one thing if you constructed and another if you
        # loaded, while default.toml's own header claimed to be what every entry
        # point uses. Entry points in fact took only the [descriptor] model from it
        # and hand-built the knobs: the four-mechanisms problem in miniature.
        # default.toml is `homography` precisely so a zero-argument constructor can
        # express it; `pose` would require a camera.
        @test config_dict(FindBoardsConfig()) ==
            config_dict(FindBoardsConfig(joinpath(pkgdir(BlobBoards), "configs", "default.toml")))
        # That equality is now true BY CONSTRUCTION, not by maintenance: the
        # `@kwdef` defaults READ the file (src/config_defaults.jl), so there is
        # no second set of literals to drift. Assert the mechanism directly, so
        # a regression to restated literals fails here rather than silently
        # reintroducing the two-sources problem.
        toml = TOML.parsefile(BlobBoards.default_config_path())["find_boards"]
        cfg = FindBoardsConfig()
        @test cfg.min_similarity == toml["min_similarity"]
        @test cfg.verify_similarity == toml["verify_similarity"]
        @test cfg.min_appearance_support == toml["min_appearance_support"]
        @test cfg.detect.peak_threshold == toml["detect"]["peak_threshold"]
        @test cfg.detect.edge_threshold == toml["detect"]["edge_threshold"]
        @test cfg.adapt.max_scale_fraction == toml["adapt"]["max_scale_fraction"]
        @test cfg.adapt.max_scale_ratio == toml["adapt"]["max_scale_ratio"]
        @test cfg.geometry.min_inliers == toml["geometry"]["min_inliers"]
        @test cfg.geometry.outlier_halfwidth == toml["geometry"]["outlier_halfwidth"]
        @test cfg.ransac_config.max_trials == toml["ransac"]["max_trials"]
        # Route 1: partial keyword construction inherits the FILE for every
        # field it does not name.
        one_off = FindBoardsConfig(min_similarity = 0.71)
        @test one_off.min_similarity == 0.71
        @test one_off.verify_similarity == toml["verify_similarity"]
        @test one_off.adapt.max_scale_fraction == toml["adapt"]["max_scale_fraction"]
        @test DetectConfig(polarity = :light).peak_threshold ==
            toml["detect"]["peak_threshold"]
        # The deliberate non-default keeps a Julia literal: `detect.polarity`
        # is a board property, not configuration.
        @test !haskey(toml["detect"], "polarity")
    end

    @testset "config_dict covers every field of every config struct" begin
        # The file-vs-config_dict test below is anchored on `config_dict`, so it is
        # structurally blind to a field `config_dict` itself forgets. That blind
        # spot was real: AdaptConfig's cutoff/min_intensity/scan_padding/store and
        # DetectConfig's fill_value/tilts were consumed by the pipeline (the CUDA
        # kernel reads cfg.cutoff) yet emitted by nothing, so write_config produced
        # sidecars that reloaded to different values. Anchor on the STRUCTS.
        cd_ = config_dict(FindBoardsConfig(geometry = PoseGeometry(_CAM)))
        sub(sec) = Set(Symbol.(keys(cd_[sec])))
        # `polarity` is derived from the board bundles by design; `camera` is data.
        derived = Set([:polarity])
        @test setdiff(Set(fieldnames(DetectConfig)), sub("detect"), derived) == Set{Symbol}()
        # `cutoff` went the other way since that history was written: it is now
        # DATA from the gallery, so the writer skips it and the reader refuses
        # it (both halves pinned above). Exclude it here so the check stays
        # honest about every OTHER AdaptConfig field.
        @test setdiff(Set(fieldnames(AdaptConfig)), sub("adapt"), Set([:cutoff])) == Set{Symbol}()
        @test setdiff(
            Set(fieldnames(PoseGeometry)), sub("geometry"),
            Set([:camera, :type])
        ) == Set{Symbol}()
        @test setdiff(Set(fieldnames(RansacConfig)), sub("ransac")) == Set{Symbol}()
        @test setdiff(
            Set(fieldnames(FindBoardsConfig)), Set(Symbol.(keys(cd_))),
            Set([:detect, :adapt, :geometry, :ransac_config])
        ) == Set{Symbol}()
    end

    @testset "the shipped configs state every parameter" begin
        # Reproducibility requires that nothing run at an unstated default: a
        # default that moves in a later BlobBoards silently reinterprets an old
        # config (max_trials already went 10_000 -> 2_000 once), and an
        # artifact's sidecar would then reload to a configuration that did not
        # produce it. `config_dict` is the authority on what a run consists of,
        # so every key it emits must be named in the file.
        #
        # `polarity`, `adapt.cutoff` and `patch_size` are absent BY DESIGN, not by
        # default -- they come from the board bundles and the ONNX graph, which
        # are themselves recorded artifacts. config_dict does not emit them.
        for (f, kw) in (
                (joinpath(pkgdir(BlobBoards), "configs", "default.toml"), (; camera = _CAM)),
                (
                    joinpath(
                        pkgdir(BlobBoards), "bmvc2026", "configs",
                        "bmvc2026_260527.toml"
                    ), (; camera = _CAM),
                ),
                (
                    joinpath(
                        pkgdir(BlobBoards), "bmvc2026", "configs",
                        "bmvc2026_260527_noflip.toml"
                    ), (; camera = _CAM),
                ),
            )
            authored = get(TOML.parsefile(f), "find_boards", Dict{String, Any}())
            effective = config_dict(FindBoardsConfig(f; kw...))
            unstated = String[]
            for (k, v) in effective
                if v isa AbstractDict
                    sub = get(authored, k, Dict{String, Any}())
                    for k2 in keys(v)
                        haskey(sub, k2) || push!(unstated, "$k.$k2")
                    end
                else
                    haskey(authored, k) || push!(unstated, k)
                end
            end
            @test (basename(f), sort(unstated)) == (basename(f), String[])
        end
    end

    @testset "board-derived polarity and cutoff" begin
        _bundle(pol, cut) = (;
            preamble = (; pattern_config = (; polarity = pol)),
            _cutoff = cut,
        )
        BlobBoards.cutoff(b::NamedTuple) = b._cutoff
        _det(resp) = (; metadata = (; response = resp))

        @testset "loader defaults ARE the struct defaults (one authority)" begin
            # The loader supplies only PRESENT keys (_present_kwargs); an absent key
            # must resolve to the struct's @kwdef default. The restated-literal
            # pattern this pins against drifted once (geometry min_inliers,
            # 2026-08-11): the structs moved and the loader fallbacks did not.
            mktemp() do path, io
                write(
                    io, """
                    [find_boards.geometry]
                    type = "homography"
                    """
                )
                close(io)
                @test BlobBoards.config_dict(FindBoardsConfig(path)) ==
                    BlobBoards.config_dict(FindBoardsConfig())
            end
            # And the forbidden derived key stays refused (already enforced via
            # _skip_fields → _allowed_keys; pinned here because the docstring once
            # taught it).
            mktemp() do path, io
                write(
                    io, """
                    [find_boards.geometry]
                    type = "homography"
                    [find_boards.detect]
                    polarity = "dark"
                    """
                )
                close(io)
                @test_throws ArgumentError FindBoardsConfig(path)
            end
        end

        @testset "grouping is by POLARITY only" begin
            # Polarity partitions the DETECTIONS, so adaptation is shared within
            # a polarity. The descriptor sampling extent is ONE shared value per
            # gallery (see _shared_image_cutoff below) — per-cutoff description
            # was removed 2026-08-11 (each distinct cutoff cost a full inference
            # sweep over every detection).
            gal = Dict("a" => nothing, "b" => nothing, "c" => nothing, "d" => nothing)
            bun = Dict(
                "a" => _bundle(:dark, 1.75), "b" => _bundle(:dark, 1.995),
                "c" => _bundle(:dark, 1.75), "d" => _bundle(:light, 2.0)
            )
            gs = BlobBoards._board_groups(gal, bun)
            @test first.(gs) == [:dark, :light]                  # polarity outer, sorted
            @test gs[1][2] == ["a", "b", "c"]                    # uids sorted, ONE group
            @test gs[2][2] == ["d"]
        end

        @testset "shared image cutoff: uniform gallery => exact, mixed => mean" begin
            bun_u = Dict("a" => _bundle(:dark, 2.0), "b" => _bundle(:dark, 2.0))
            @test BlobBoards._shared_image_cutoff(bun_u, ["a", "b"]) == 2.0

            # 50 distinct cutoffs is real: evaluation/blobboards_original spans
            # 1.75..1.995 in 0.005 steps — the image is described ONCE at the
            # mean, which minimizes the worst-case annulus mismatch.
            bun_m = Dict("b$i" => _bundle(:dark, 1.75 + 0.005 * (i - 1)) for i in 1:50)
            # (warns once per session — maxlog=1 — so only the VALUE is asserted)
            c = BlobBoards._shared_image_cutoff(bun_m, ["b$i" for i in 1:50])
            @test c ≈ (1.75 + 1.995) / 2

            # Empty gallery falls back to the package default.
            @test BlobBoards._shared_image_cutoff(Dict{String, Any}(), String[]) ==
                BlobBoards.DEFAULT_CUTOFF
        end

        @testset "a gallery board with no bundle is refused" begin
            @test_throws ArgumentError BlobBoards._board_groups(
                Dict("a" => nothing), Dict("b" => _bundle(:dark, 2.0))
            )
        end

        @testset "polarity partitions the detections disjointly" begin
            # VGF: intensity > 0 is :dark, < 0 is :light; BB carries it as
            # detection.response. Disjoint => a mixed set costs no extra
            # description work, and a dark board never sees a light blob.
            dets = [_det(1.0), _det(-1.0), _det(2.0), _det(-2.0)]
            dark = BlobBoards._polarity_subset(dets, Val(:dark))
            light = BlobBoards._polarity_subset(dets, Val(:light))
            @test [d.metadata.response for d in dark] == [1.0, 2.0]
            @test [d.metadata.response for d in light] == [-1.0, -2.0]
            @test length(dark) + length(light) == length(dets)      # a partition
            @test BlobBoards._polarity_subset(dets, Val(:none)) === dets
        end

        @testset "detection keeps the NARROWEST polarity the board set needs" begin
            # Detecting at the superset unconditionally would make VGF convert
            # every feature to a Detection and BB throw half away. A set that
            # agrees (every real one) detects at exactly its own polarity.
            bun = Dict(
                "a" => _bundle(:dark, 2.0), "b" => _bundle(:dark, 2.0),
                "c" => _bundle(:light, 2.0)
            )
            @test BlobBoards._detection_polarity(bun, ["a", "b"]) === :dark
            @test BlobBoards._detection_polarity(bun, ["c"]) === :light
            @test BlobBoards._detection_polarity(bun, ["a", "c"]) === :none
            # NB :none, not :both -- VGF's asift_detect accepts :none/:dark/:light
            # and throws on anything else, so :both would die at detection.
            @test BlobBoards._POLARITY_SUPERSET === :none

            base = FindBoardsConfig(
                geometry = HomographyGeometry(),
                detect = DetectConfig(polarity = :dark)
            )
            c = BlobBoards._detect_config_for_boards(base, :none)
            @test c.detect.polarity === :none
            @test c.detect.peak_threshold == base.detect.peak_threshold
            @test isconcretetype(typeof(c))
        end
    end

    @testset "config override via Accessors" begin
        base = FindBoardsConfig(
            geometry = PoseGeometry(_CAM; min_inliers = 8),
            detect = DetectConfig(
                polarity = :dark,
                peak_threshold = 0.01,
                edge_threshold = 5.0,
                nms = EllipseNms(cutoff = 2.0f0)
            )
        )

        @testset "nested set changes one field and nothing else" begin
            c = @set base.detect.peak_threshold = 0.02
            @test c.detect.peak_threshold == 0.02
            # siblings within the nested struct
            @test c.detect.edge_threshold == base.detect.edge_threshold
            @test c.detect.polarity == base.detect.polarity
            @test c.detect.upsample == base.detect.upsample
            @test c.detect.nms === base.detect.nms
            # siblings in the outer struct
            @test c.min_similarity == base.min_similarity
            @test c.geometry === base.geometry
            @test c.ransac_config === base.ransac_config
            @test typeof(c) === typeof(base)
        end

        @testset "top-level and geometry fields set too" begin
            @test (@set base.min_similarity = 0.8).min_similarity == 0.8
            @test (@set base.detect.upsample = false).detect.upsample == false
            @test (@set base.geometry.min_inliers = 12).geometry.min_inliers == 12
            # the camera must survive an override of a sibling geometry knob --
            # losing it downgrades a metric-pose run to the uncalibrated path
            @test (@set base.geometry.min_inliers = 12).geometry.camera === _CAM
            @test (@set base.adapt.cutoff = 2.5).adapt.cutoff == 2.5
        end

        @testset "a type-changing override re-specializes the struct" begin
            # `nms` is a type PARAMETER of DetectConfig, so setting it must rebuild
            # rather than convert. Concreteness is the point of the parametrization
            # (this config is read inside the per-board fitting loop).
            c = @set base.detect.nms = MxmNms(window_half = 4)
            @test c.detect.nms isa MxmNms
            @test isconcretetype(typeof(c))
            @test isconcretetype(typeof(c.detect))
            c2 = @set base.detect.nms = nothing
            @test c2.detect.nms === nothing
            @test isconcretetype(typeof(c2))
        end

        @testset "an overridden config still round-trips through TOML" begin
            # An override is only useful if the result is still recordable -- a
            # driver writes its config beside the artifact it produced.
            mktempdir() do dir
                c = @set base.detect.peak_threshold = 0.005
                c = @set c.min_similarity = 0.7
                p = joinpath(dir, "override.toml"); write_config(p, c)
                back = FindBoardsConfig(p; camera = _CAM)
                @test back.detect.peak_threshold == 0.005
                @test back.min_similarity == 0.7
                @test config_dict(back) == config_dict(c)
            end
        end
    end

end


@testset "descriptor model source forms" begin
    # Pure parts of the artifact resolver — no network, no data_dir.
    # Missing keys are refused loudly.
    @test_throws ArgumentError BlobBoards._resolve_model(Dict("name" => "x"))
    # Unknown keys are refused (the typo'd-knob guarantee, one level down).
    @test_throws ArgumentError BlobBoards._resolve_model(
        Dict("artifact" => "x", "artefact" => "y")
    )
    # A name outside the catalog is refused with the published entries named,
    # BEFORE any download is attempted.
    err = try
        BlobBoards._resolve_model(Dict("artifact" => "no_such_network"))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("track_logpolar_circ_s96_iter4", err.msg)
    # Every catalog entry is complete: pinned tree hash plus a sha256-pinned
    # download stanza from a numbered Zenodo record version — the immutability
    # the config form relies on.
    catalog = TOML.parsefile(BlobBoards._artifacts_toml())
    @test !isempty(catalog)
    @test haskey(catalog, "track_logpolar_circ_s96_iter4")
    # DECLARED pending entries: hash-pinned, no download stanza YET, each with
    # the reason. Recorded as skipped (Broken in the summary), never silently
    # counted as green — and a pending entry that gains a download must leave
    # this list, so the list cannot go stale. None is pending today.
    pending = Dict{String, String}()
    for (name, entry) in catalog
        @test haskey(entry, "git-tree-sha1")
        @test length(entry["git-tree-sha1"]) == 40
        if haskey(pending, name)
            @test !haskey(entry, "download")
            @info "catalog entry $name: download stanza SKIPPED — $(pending[name])"
            @test_skip haskey(entry, "download")
            continue
        end
        @test haskey(entry, "download") && !isempty(entry["download"])
        dl = entry["download"][1]
        @test length(dl["sha256"]) == 64
        # The stack's two hosts, each pinned: a numbered Zenodo record version
        # (zenodo_url's form) or a Hugging Face dataset at a full commit
        # (huggingface_url's form); no placeholder id or branch may ship.
        zenodo = "https://zenodo\\.org/records/\\d+/files/$name\\.tar\\.gz\\?download=1"
        hf = "https://huggingface\\.co/datasets/[^/]+/[^/]+/resolve/[0-9a-f]{40}/$name\\.tar\\.gz"
        @test occursin(Regex("^($zenodo|$hf)\$"), dl["url"])
    end
    @test all(in(keys(catalog)), keys(pending))
end

@testset "retired config keys: warn, never resurrect" begin
    # A RECORDED sidecar may carry a retired knob (`adapt.init` was settable,
    # serialized, and read by nothing). It must reload with a warning — the
    # record is immutable — while a typo'd key still fails loudly.
    mktempdir() do dir
        base = read(BlobBoards.default_config_path(), String)
        p1 = joinpath(dir, "retired.toml")
        write(p1, replace(base, "[find_boards.adapt]" => "[find_boards.adapt]\ninit = \"disc\""))
        cfg = @test_logs (:warn, r"retired") match_mode = :any FindBoardsConfig(p1)
        @test cfg isa FindBoardsConfig
        p2 = joinpath(dir, "typo.toml")
        write(p2, replace(base, "[find_boards.adapt]" => "[find_boards.adapt]\ninitt = \"disc\""))
        @test_throws ArgumentError FindBoardsConfig(p2)
    end
end

@testset "one default authority through every door (2026-09 audit)" begin
    mktempdir() do dir
        base = read(BlobBoards.default_config_path(), String)
        # `nms = { type = "ellipse" }` with no cutoff must inherit the FILE's
        # cutoff (2.0), not EllipseNms()'s struct literal (3.0) — the
        # loader-vs-struct divergence the audit found, one nesting level down.
        p1 = joinpath(dir, "nms_inherit.toml")
        write(
            p1, replace(
                base,
                r"nms = \{[^}]*\}" => "nms = { type = \"ellipse\" }"
            )
        )
        cfg = FindBoardsConfig(p1)
        @test cfg.detect.nms.cutoff == Float32(BlobBoards._dflt("detect", "nms")["cutoff"])
        # Retirement is (section, key)-scoped: patch_size is retired from
        # [descriptor] only — in [find_boards.detect] it is a typo and throws.
        p2 = joinpath(dir, "scoped_retire.toml")
        write(p2, replace(base, "[find_boards.detect]" => "[find_boards.detect]\npatch_size = 64"))
        @test_throws ArgumentError FindBoardsConfig(p2)
        # The top-level table surface is enforced through THIS door too, not
        # just read_run_config: a typo'd [descripter] must fail loudly.
        p3 = joinpath(dir, "toplevel_typo.toml")
        write(p3, base * "\n[descripter]\nmodel = \"x\"\n")
        @test_throws ArgumentError FindBoardsConfig(p3)
        # the board/pattern authoring tables belong to the file of record: in a
        # run config they would be knobs no run reads
        p4 = joinpath(dir, "authoring_table.toml")
        write(p4, base * "\n[blob_board]\nmargin_mm = 5.0\n")
        @test_throws ArgumentError FindBoardsConfig(p4)
    end
end

@testset "TrackConfig: the tracker's knobs are in the config of record (2026-09)" begin
    t = VgtkCore.toml_table(BlobBoards.TrackConfig())
    @test sort(collect(keys(t))) == sort(String.(collect(fieldnames(BlobBoards.TrackConfig))))
    # the file's values ARE the struct's (one authority), and a run config overrides per key
    dflt = TOML.parsefile(joinpath(pkgdir(BlobBoards), "configs", "default.toml"))["find_boards"]["track"]
    @test t == dflt
    path = joinpath(mktempdir(), "t.toml")
    write(path, "[find_boards.track]\npatience = 9\n")
    c = FindBoardsConfig(path)
    @test c.track.patience == 9 && c.track.keyframe_every == BlobBoards.TrackConfig().keyframe_every
    write(path, "[find_boards.track]\npatiense = 9\n")
    @test_throws ArgumentError FindBoardsConfig(path)
end

@testset "annotation config_hash covers every non-geometry field (2026-09)" begin
    using BlobBoards: config_hash
    h(cfg) = config_hash(; config = cfg, cutoff = 2.0, patch_size = 64, outer_extent = 96.0, min_iou = 0.5)
    base = h(FindBoardsConfig())
    @test base == h(FindBoardsConfig())                                   # deterministic
    # Every FindBoardsConfig field except geometry moves the hash — derived from
    # config_dict, so a field nobody listed by hand still counts.
    for f in fieldnames(FindBoardsConfig)
        f === :geometry && continue
        d = config_dict(FindBoardsConfig())
        key = f === :ransac_config ? "ransac" : String(f)
        @test haskey(d, key)
    end
    @test h(FindBoardsConfig(verify_similarity = 0.71)) != base
    @test h(FindBoardsConfig(min_appearance_support = 0.31)) != base
    @test h(FindBoardsConfig(ransac_seed = 12345)) != base
    @test h(FindBoardsConfig(min_similarity = 0.61)) != base
    # geometry is the per-source part and must NOT participate
    @test h(FindBoardsConfig(geometry = HomographyGeometry(min_inliers = 12))) == base
    # the annotation parameters do
    @test config_hash(; config = FindBoardsConfig(), cutoff = 2.5, patch_size = 64, outer_extent = 96.0, min_iou = 0.5) != base
end


# A GeometryConfig defined outside the package is loaded and written through
# its own TOML tag (`geometry_from_toml` / `geometry_tag`), and a driver that
# reads its own run config hands over the `[find_boards]` table as a Dict.
struct _ForeignArm <: BlobBoards.GeometryConfig
    image_size::Tuple{Int, Int}
    outlier_halfwidth::Float64
    min_inliers::Int
end
VgtkCore.toml_skip_fields(::Type{_ForeignArm}) = (:image_size,)
BlobBoards.geometry_tag(::_ForeignArm) = "foreign_arm"
function BlobBoards.geometry_from_toml(::Val{:foreign_arm}, d; camera = nothing, image_size = nothing)
    VgtkCore.check_keys(d, ["type", "outlier_halfwidth", "min_inliers"], "find_boards.geometry")
    return _ForeignArm(image_size, Float64(d["outlier_halfwidth"]), Int(d["min_inliers"]))
end

struct _UntaggedArm <: BlobBoards.GeometryConfig end

@testset "a geometry arm defined outside the package (2026-10)" begin
    path = joinpath(mktempdir(), "fa.toml")
    write(path, "[find_boards.geometry]\ntype = \"foreign_arm\"\noutlier_halfwidth = 3.0\nmin_inliers = 8\n")
    c = FindBoardsConfig(path; image_size = (640, 480))
    @test c.geometry === _ForeignArm((640, 480), 3.0, 8)
    t = config_dict(c)
    @test t["geometry"] == Dict("type" => "foreign_arm", "outlier_halfwidth" => 3.0, "min_inliers" => 8)
    # the parsed table round-trips through the Dict constructor
    c2 = FindBoardsConfig(t; image_size = (640, 480))
    @test c2.geometry === c.geometry && config_dict(c2) == t
    # an unknown key inside the foreign table is the foreign loader's refusal
    write(path, "[find_boards.geometry]\ntype = \"foreign_arm\"\noutlier_halfwidth = 3.0\nmin_inliers = 8\nmin_inlers = 9\n")
    @test_throws ArgumentError FindBoardsConfig(path; image_size = (640, 480))
    # a tag nobody registered is refused with the valid set
    write(path, "[find_boards.geometry]\ntype = \"radial_camera\"\n")
    err = try
        FindBoardsConfig(path; image_size = (640, 480)); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("radial_homography", err.msg)
    # a GeometryConfig with no tag cannot be written
    @test_throws ArgumentError config_dict(FindBoardsConfig(geometry = _UntaggedArm()))
end

@testset "read_run_config: a driver's own top-level tables (2026-10)" begin
    path = joinpath(mktempdir(), "rc.toml")
    write(path, "[find_boards]\nmin_similarity = 0.6\n\n[mydriver]\nk = 2\n")
    @test_throws ArgumentError read_run_config(path)                       # not admitted by default
    rc = read_run_config(path; tables = ("mydriver",))
    @test rc.config_toml["mydriver"]["k"] == 2
    @test_throws ArgumentError FindBoardsConfig(path)                     # the package's own door stays strict
    # the package no longer owns a calibration table
    write(path, "[find_boards.refine]\nmax_iter = 7\n")
    @test_throws ArgumentError FindBoardsConfig(path)
    write(path, "[calibrate]\nncoef = 2\n")
    @test_throws ArgumentError read_run_config(path)
end
