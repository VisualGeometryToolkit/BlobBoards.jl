# =============================================================================
# test_jet.jl — static analysis gate (JET.jl)
#
# WHY THIS EXISTS (2026-07-26): the authoring code (then BlobBoardsAuthoringExt,
# since 2026-10-14 the submodule BlobBoards.Authoring) had NEVER been loaded by
# this test suite. `blob_board` is a stub whose implementation lives in that
# extension, and the extension only activates on `using CairoMakie` — which
# nothing here did; CairoMakie merely arrived transitively via
# VisualGeometryCore, and VGC 0.4 dropped it. Consequence: none of the
# extension's bindings had ever been validated. Fourteen missing imports and
# one commented-out-but-still-called function (`compute_logical_saddle_points`)
# had accumulated silently, and `blob_board` was broken for anyone using it in
# a clean environment.
#
# Running the suite finds those one at a time (Julia reports only the first
# undefined name per execution). JET finds them ALL in one run, including on
# code paths the tests never execute — the 3D-preview (`RotX`, `Point3f`) and
# QR-resize (`BSpline`) bugs were found exactly that way.
# =============================================================================
using JET

@testset "JET static analysis" begin
    ext = BlobBoards.Authoring             # board authoring: part of the package since 2026-10-14
    @test ext isa Module

    # ---- authoring: must be CLEAN -------------------------------------------
    # Nothing is allowed: the one former exception (Makie's untyped SpecApi
    # layout callback around `plot_blank_frame`) went when the sheet stopped
    # being a Makie figure (2026-10).
    ALLOWED_EXT = String[]
    if ext !== nothing
        # The vendored QR encoder (src/authoring/qrencode, VENDORED.toml) is
        # third-party code kept verbatim, so its own two reports stay where
        # they are (an undefined `InfoError` in a version-range error path; a
        # `findlast` that can return `nothing` on the zero polynomial). Only
        # errors INSIDE it are ignored: a bad call into it from the extension
        # is reported at the extension's frame.
        reports = JET.get_reports(JET.report_package(ext; target_modules = (ext,), ignored_modules = (ext.QREncode,)))
        unexpected = filter(reports) do r
            !any(a -> occursin(a, sprint(show, r)), ALLOWED_EXT)
        end
        for r in unexpected
            @info "JET (BlobBoards.Authoring)" report = first(replace(sprint(show, r), "\n" => " "), 300)
        end
        @test isempty(unexpected)
    end

    # ---- core: RATCHET ------------------------------------------------------
    # A ceiling, not an endorsement: it may only ever go DOWN. Lower it
    # whenever you fix some.
    #
    # Real bugs found this way and ALREADY FIXED (2026-07-26): `_load_tracks`
    # (called in io.jl but deleted with training.jl in 197beeb) and
    # `BlobBoards.Makie` (plot_annotations called Makie from CORE, where it is
    # only a weakdep — moved to BlobBoardsMakieExt).
    #
    # Known FALSE positives that remain in the count — do NOT "fix" these:
    #  * `blob`, `s`: `@index(Group, Linear)` results inside KernelAbstractions
    #    `@kernel` bodies; JET cannot see through the macro.
    #  * haskey/getindex/String over Union{Nothing,Bool,Float64,...}: dynamic
    #    JSON3 access, which JET cannot narrow.
    # RATCHET: only ever DECREASES. Lowered 25 -> 15 on 2026-08-16; a baseline
    # sitting 10 above the real count is 10 reports of headroom in which a
    # regression lands silently and still succeeds.
    #
    # MEASURE IT IN THE TEST ENV, not a plain `--project=.` REPL. That REPL
    # reports 13; under `Pkg.test` it is 15, because the test target loads
    # CUDA/CairoMakie/PythonCall and their extensions contribute reports the
    # bare env never sees. Setting the ratchet from the REPL number failed the
    # suite on the very run that introduced it.
    # 2026-08-31 (package-cleanup audit): a standalone measurement with CUDA +
    # cuDNN + CairoMakie + PythonCall loaded gave 14; the suite itself still
    # reports 15 (it loads more) — and the suite is the authority, so 15 stays.
    # 2026-09-07 (package-cleanup audit): re-measured at 15 in the suite env
    # after the audit's edits. Routing refine.jl's pinhole through
    # `_pinhole_camera` briefly added a `find_absolute_pose` union-split report
    # (16); pinning the argument type removed it again. The remaining
    # `AbsolutePoseProblem(::Nothing, ::Any)` split is the pre-existing
    # `_fit_workspace_builder` three-way dispatch, not new.
    # 2026-09-21: the suite had reached 16. The regression was a79bbc4
    # (2026-09-10): training.jl's perspective augmentation called
    # `forward(_model_to_homography(model, nothing))`, and that union includes
    # `RadialHomography`, which has no `forward` — a real MethodError on radial-arm
    # tracks (two reports, one site). `_augmentation_homography` now refuses
    # non-homography models by name; that removes both reports, 16 -> 14.
    CORE_BASELINE = 14
    # (the vendored QR encoder is third-party code kept verbatim: its own reports
    # are the authoring section's to ignore, above)
    core_reports = JET.get_reports(
        JET.report_package(BlobBoards; target_modules = (BlobBoards,), ignored_modules = (BlobBoards.Authoring.QREncode,))
    )
    if length(core_reports) > CORE_BASELINE
        for r in core_reports
            @info "JET (BlobBoards core)" report = first(replace(sprint(show, r), "\n" => " "), 300)
        end
    end
    @test length(core_reports) <= CORE_BASELINE
end
