# The paper-project conventions of the stack (VisualGeometryToolkitEval PaperProjects):
# layout, data policy, the asset index, configs of record, provenance and the
# stage runner, and the published-data catalog — one check, asserted empty.
@testitem "bmvc2026 paper-project conventions (PaperProjects)" begin
    import VisualGeometryToolkitEval as VgtkEval
    using DrWatson: projectdir
    import VisualGeometryToolkitEval

    # Deferred / justified exceptions, each with its reason (rule, file, match).
    const PAPER_PROJECT_ALLOW = Tuple{String, String, String}[
        # L4 follows test/runtests.jl's `include`s, and the suite is ReTestItems
        # items: runtests.jl includes nothing, it runs every `*_tests.jl` file of
        # test/ — this one, which calls the check, among them. Deferred until
        # PaperProjects follows a ReTestItems `runtests(…, @__DIR__)`.
        ("L4", "test/runtests.jl", "the suite does not call paper_project_violations"),
    ]

    if isdefined(VisualGeometryToolkitEval, :PaperProjects)
        vs = VgtkEval.PaperProjects.paper_project_violations(projectdir();
            papers = ["papers/blob_boards_bmvc.tex", "papers/bmvc_supp.tex"],
            catalog_kinds = ["bb_tier"],
            allow = [VgtkEval.Experiments.DataPolicyAllow(a...) for a in PAPER_PROJECT_ALLOW])
        foreach(v -> @info(sprint(show, v)), vs)
        @test isempty(vs)
    else
        # The resolved VisualGeometryToolkitEval predates PaperProjects (it lands with
        # the release after 1.0.0); the check runs once it resolves.
        @info "VisualGeometryToolkitEval $(pkgversion(VisualGeometryToolkitEval)) has no PaperProjects — paper-project check skipped"
        @test_skip false
    end
end

# The paper environment is headless-reproducible: no package that needs a
# display (GLMakie fails to precompile without X11). The interactive training
# viewers add it in an environment of their own.
@testitem "bmvc2026 environment is headless" begin
    using DrWatson: projectdir
    import TOML
    proj = TOML.parsefile(projectdir("Project.toml"))
    @test !haskey(get(proj, "deps", Dict()), "GLMakie")
    @test !haskey(get(proj, "compat", Dict()), "GLMakie")
end
