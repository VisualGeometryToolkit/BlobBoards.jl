# The tuned operating points (src/operating_points.jl): the joint detection
# CV's config is well-formed and its artifact clears the gate, and every tuned
# value of the configs of record IS the value of the artifact it was adopted
# from, read through VisualGeometryEval's one gated reader (`deployed_setting`)
# — never a literal that drifted from its tuning.
using Test
using DrWatson: projectdir
import TOML
import VisualGeometryEval
using VisualGeometryEval.Evaluation: deployed_setting
using VisualGeometryEval.Tuning: tuning_spec, JointCoarseFine, MeanAverageAccuracy

module OperatingPoints
    include(joinpath(@__DIR__, "..", "src", "operating_points.jl"))
end

const CV_CONFIG = projectdir("configs", "detection_cv.toml")
# The run configs of record that carry the operating point.
const RUN_CONFIGS = ["bmvc2026_260527.toml", "bmvc2026_260527_noflip.toml"]

@testset "the joint detection CV: its spec, its run directory, its gate" begin
    cfg = TOML.parsefile(CV_CONFIG)
    s = tuning_spec(cfg, "detection").scheme
    @test s isa JointCoarseFine && s.criterion == MeanAverageAccuracy(10.0)
    @test sort!(collect(keys(s.axes))) == ["min_similarity", "outlier_halfwidth", "peak_threshold"]
    @test [n for (n, a) in s.axes if a.expensive] == ["peak_threshold"]
    key = VisualGeometryEval.Experiments.run_key(cfg)
    dir = OperatingPoints.detection_cv_dir(cfg)
    @test endswith(dir, "detection_cv-" * key)
    # the record of this config is the one the configs of record deploy
    @test all(op -> op.artifact == relpath(OperatingPoints.detection_cv_artifact(dir), OperatingPoints.results_root()),
              OperatingPoints.ADOPTED_OPERATING_POINTS)
    path = OperatingPoints.detection_cv_artifact(dir)
    if !isfile(path)
        @info "no joint detection CV artifact at $path (results root not bound or not run): gate check SKIPPED"
        @test_skip false
    else
        sel = deployed_setting(path; context = "bmvc2026 joint detection CV")
        @test sort!(collect(String.(keys(sel)))) == sort!(collect(keys(s.axes)))
    end
end

@testset "operating points: the configs of record carry their adopted tuning" begin
    for op in OperatingPoints.ADOPTED_OPERATING_POINTS
        # each run config names the run its value came from
        for rc in RUN_CONFIGS
            @test occursin(basename(dirname(op.artifact)), read(projectdir("configs", rc), String))
        end
        path = joinpath(OperatingPoints.results_root(), op.artifact)
        if !isfile(path)
            @info "no tuning artifact at $path (results root not bound here): config-vs-artifact check SKIPPED"
            @test_skip false
            continue
        end
        tuned = deployed_setting(path; context = "bmvc2026 $(op.axis)")[op.axis]
        for rc in RUN_CONFIGS
            @test foldl(getindex, op.key; init = TOML.parsefile(projectdir("configs", rc))) == tuned
        end
    end
end
