# =============================================================================
# jet.jl — static analysis gate (JET.jl)
#
# WHY THIS EXISTS (2026-07-26): the authoring code (then BlobBoardsAuthoringExt,
# since 2026-10-14 the submodule BlobBoards.Authoring) had NEVER been loaded by
# this test suite. `blob_board` is a stub whose implementation lives in that
# extension, and the extension only activates on `using CairoMakie` — which
# nothing here did; CairoMakie merely arrived transitively via
# VisualGeometryToolkitCore, and VGC 0.4 dropped it. Consequence: none of the
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
    # ---- authoring (BlobBoards.Authoring): must be CLEAN, nothing allowed ----
    # The vendored QR encoder (src/authoring/qrencode, VENDORED.toml) is
    # third-party code kept verbatim, so its own two reports stay where they
    # are (an undefined `InfoError` in a version-range error path; a `findlast`
    # that can return `nothing` on the zero polynomial). Only errors INSIDE it
    # are ignored: a bad call into it from the authoring code is reported at
    # the caller's frame.
    ext = BlobBoards.Authoring
    reports = JET.get_reports(JET.report_package(ext; target_modules = (ext,), ignored_modules = (ext.QREncode,)))
    for r in reports
        @info "JET (BlobBoards.Authoring)" report = first(replace(sprint(show, r), "\n" => " "), 300)
    end
    @test isempty(reports)

    # ---- core: the whole-package JET ratchet is VisualGeometryToolkitCore's
    # `test_quality_gates` (suites/quality_gates.jl, `jet_bound`).
end
