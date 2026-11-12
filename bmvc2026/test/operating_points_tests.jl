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
    key = OperatingPoints.detection_cv_key(cfg)
    dir = OperatingPoints.detection_cv_dir(cfg)
    @test endswith(dir, joinpath(OperatingPoints.EXPERIMENT, "detection_cv-" * key))
    # the run config's key is folded in: the search is around it
    @test haskey(OperatingPoints.detection_cv_upstream(cfg), "bb")
    @test key != VgtkEval.Experiments.run_key(cfg)
    # every adopted value comes from one record: the config's under its 0.8 key
    @test length(unique(op.artifact for op in OperatingPoints.ADOPTED_OPERATING_POINTS)) == 1
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
