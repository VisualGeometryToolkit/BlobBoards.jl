# The tuned operating points (src/operating_points.jl): the joint detection
# CV's config is well-formed and its artifact clears the gate, and every tuned
# value of the configs of record IS the value of the artifact it was adopted
# from, read through VisualGeometryToolkitEval's one gated reader (`deployed_setting`)
# — never a literal that drifted from its tuning. The artifact reads are the
# `:results` items: without the artifact (results root not bound, or the run
# not made) they record a skip.

@testitem "the joint detection CV: its spec and its run directory" setup = [OperatingPoints] begin
    import VisualGeometryToolkitEval as VgtkEval
    using DrWatson: projectdir
    import TOML
    using VisualGeometryToolkitEval.Tuning: tuning_spec, JointCoarseFine, MeanAverageAccuracy

    cfg = OperatingPoints.load_config("detection_cv.toml")
    s = tuning_spec(cfg, "detection").scheme
    @test s isa JointCoarseFine && s.criterion == MeanAverageAccuracy(10.0)
    @test sort!(collect(keys(s.axes))) == ["min_similarity", "outlier_halfwidth", "peak_threshold"]
    @test [n for (n, a) in s.axes if a.expensive] == ["peak_threshold"]
    p = OperatingPoints.detection_cv_product(cfg)
    @test endswith(VgtkEval.Experiments.product_dir(p), joinpath("tuning", OperatingPoints.EXPERIMENT, "detection_cv", p.key))
    # the run config's key is folded in: the search is around it
    @test haskey(OperatingPoints.detection_cv_upstream(cfg), "bb")
    @test p.key != VgtkEval.Experiments.run_key(cfg)
    # the adopted values come from two records: the half-width CV's and the joint CV's (0.8 key)
    @test length(unique(op.artifact for op in OperatingPoints.ADOPTED_OPERATING_POINTS)) == 2
    @test sort!([String(op.axis) for op in OperatingPoints.ADOPTED_OPERATING_POINTS]) == sort!(collect(keys(s.axes)))
end

@testitem "the joint detection CV: its gate" tags = [:results] setup = [OperatingPoints] begin
    using DrWatson: projectdir
    import TOML
    using VisualGeometryToolkitEval.Evaluation: deployed_setting
    using VisualGeometryToolkitEval.Tuning: tuning_spec

    cfg = OperatingPoints.load_config("detection_cv.toml")
    s = tuning_spec(cfg, "detection").scheme
    # without a bound results root there is no run to read: a recorded skip
    path = try OperatingPoints.detection_cv_artifact(OperatingPoints.detection_cv_dir(cfg)) catch; "" end
    if !isfile(path)
        @info "no joint detection CV artifact at $path (results root not bound or not run): gate check SKIPPED"
        @test_skip false
    else
        sel = deployed_setting(path; context = "bmvc2026 joint detection CV")
        @test sort!(collect(String.(keys(sel)))) == sort!(collect(keys(s.axes)))
    end
end

# Each run config of record that carries the operating point names the run
# its value came from.
@testitem "operating points: the configs of record name their tuning runs" setup = [OperatingPoints] begin
    using DrWatson: projectdir
    for op in OperatingPoints.ADOPTED_OPERATING_POINTS, rc in OperatingPoints.RUN_CONFIGS
        @test occursin(basename(dirname(op.artifact)), read(OperatingPoints.config_path(rc), String))
    end
end

@testitem "operating points: the configs of record carry their adopted tuning" tags = [:results] setup = [OperatingPoints] begin
    using DrWatson: projectdir
    import TOML
    using VisualGeometryToolkitEval.Evaluation: deployed_setting

    root = try OperatingPoints.results_root() catch; nothing end   # unbound: every check a recorded skip
    for op in OperatingPoints.ADOPTED_OPERATING_POINTS
        path = root === nothing ? "" : joinpath(root, op.artifact)
        if !isfile(path)
            @info "no tuning artifact at $path (results root not bound here): config-vs-artifact check SKIPPED"
            @test_skip false
            continue
        end
        tuned = deployed_setting(path; context = "bmvc2026 $(op.axis)")[op.axis]
        for rc in OperatingPoints.RUN_CONFIGS
            @test foldl(getindex, op.key; init = OperatingPoints.load_config(rc)) == tuned
        end
    end
end

# The two CVs of the detector of record: the scale ladder (configs/detector_levels.toml)
# and the half-width on it (configs/halfwidth_cv.toml), each a keyed product and a
# stage of configs/reproduce.toml.
@testitem "the ladder and half-width CVs: their specs, products and stages" setup = [OperatingPoints] begin
    import VisualGeometryToolkitEval as VgtkEval
    import TOML
    using DrWatson: projectdir
    using VisualGeometryToolkitEval.Tuning: tuning_spec, JointCoarseFine, Factorial, MeanAverageAccuracy, grid_spec

    hw = OperatingPoints.load_config("halfwidth_cv.toml")
    s = tuning_spec(hw, "detection").scheme
    @test s isa JointCoarseFine && s.criterion == MeanAverageAccuracy(10.0)
    @test collect(keys(s.axes)) == ["outlier_halfwidth"] && s.incumbent === nothing
    p = OperatingPoints.detection_cv_product(hw)
    @test endswith(VgtkEval.Experiments.product_dir(p), joinpath("tuning", OperatingPoints.EXPERIMENT, "detection_cv", p.key))
    @test p.key == OperatingPoints.detection_cv_product(OperatingPoints.load_config("halfwidth_cv.toml")).key
    @test p.key != OperatingPoints.detection_cv_product(OperatingPoints.load_config("detection_cv.toml")).key

    dl = OperatingPoints.load_config("detector_levels.toml")
    f = tuning_spec(dl, "detector").scheme
    @test f isa Factorial && f.criterion == MeanAverageAccuracy(10.0) && collect(keys(f.axes)) == ["setting"]
    @test grid_spec(f.axes["setting"]).values == 1:length(dl["settings"])
    @test all(st -> issubset(["name", "upsample", "nb_scales", "nb_octaves"], keys(st)), dl["settings"])
    @test allunique(st["name"] for st in dl["settings"])
    # the run configs of record detect with one of the settings
    for rc in OperatingPoints.RUN_CONFIGS
        d = OperatingPoints.load_config(rc)["find_boards"]["detect"]
        @test any(st -> all(st[k] == d[k] for k in ("upsample", "nb_scales", "nb_octaves")), dl["settings"])
    end
    q = OperatingPoints.detector_levels_product(dl)
    @test endswith(VgtkEval.Experiments.product_dir(q), joinpath("tuning", OperatingPoints.EXPERIMENT, "detector_levels", q.key))
    @test q.key == OperatingPoints.detector_levels_product(OperatingPoints.load_config("detector_levels.toml")).key

    stages = Dict(st["name"] => st for st in TOML.parsefile(projectdir("configs", "reproduce.toml"))["stage"])
    for (name, driver, conf) in (("halfwidth_cv", "detection_cv.jl", "halfwidth_cv.toml"),
                                 ("detector_levels", "detector_levels.jl", "detector_levels.toml"))
        st = stages[name]
        @test st["driver"] == "bmvc2026/scripts/$(driver)" && st["config"] == "bmvc2026/configs/$(conf)"
        @test st["gpu"] && !st["default"]
        @test conf == "detector_levels.toml" || "--config=$(conf)" in st["command"]
        @test occursin("reproduce: reads configs/$(conf)", read(projectdir("scripts", driver), String))
    end
end

# The ladder's searched scales and its setter, without a GPU.
@testitem "the scale ladders: searched σ and the setter" setup = [BBEval] begin
    using .BBEval.BlobBoardsEval: with_ladder, ladder_sigma_range
    using BlobBoards: FindBoardsConfig
    # VisualGeometryToolkitFeatures `AsiftPlan`'s worked example, 4240×2832: 1.13 … 230 px at the
    # defaults, 0.57 … 230 at upsample = 2, the top doubled by an octave more.
    base = Dict("upsample" => 1, "nb_scales" => 3, "nb_octaves" => 0)
    A = Dict("upsample" => 2, "nb_scales" => 3, "nb_octaves" => 10)
    E = merge(A, Dict("nb_octaves" => 0))
    @test all(isapprox.(ladder_sigma_range(base, 4240, 2832), (1.13, 230); rtol = 0.01))
    @test all(isapprox.(ladder_sigma_range(E, 4240, 2832), (0.566, 230); rtol = 0.01))
    @test all(isapprox.(ladder_sigma_range(A, 4240, 2832), (0.566, 460); rtol = 0.01))
    c = with_ladder(FindBoardsConfig(), A)
    @test (c.detect.upsample, c.detect.nb_scales, c.detect.nb_octaves) == (2, 3, 10)
end

# The record of the ladder CV: the setting it selected is the one the run configs detect with.
@testitem "the detector of record is the ladder CV's selection" tags = [:results] setup = [OperatingPoints] begin
    using JLD2: load
    root = try OperatingPoints.results_root() catch; nothing end
    path = root === nothing ? "" : joinpath(root, OperatingPoints.ADOPTED_DETECTOR_LEVELS)
    if !isfile(path)
        @info "no detector-ladder artifact at $path (results root not bound here): check SKIPPED"
        @test_skip false
    else
        sel = load(path)["settings"][load(path)["selected"]]
        for rc in OperatingPoints.RUN_CONFIGS
            d = OperatingPoints.load_config(rc)["find_boards"]["detect"]
            @test all(sel[k] == d[k] for k in ("upsample", "nb_scales", "nb_octaves"))
        end
    end
end
