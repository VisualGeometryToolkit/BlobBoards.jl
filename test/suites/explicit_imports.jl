using ExplicitImports

@testset "ExplicitImports" begin
    # No implicit imports: every name states which package it comes from.
    @test ExplicitImports.check_no_implicit_imports(BlobBoards) === nothing
    # No stale explicit imports — an import nothing uses is where the last
    # refactor's residue hides (11 accumulated before this gate existed).
    # `pdftocairo` is a KNOWN FALSE POSITIVE: it is called at
    # src/io.jl inside Cmd interpolation (`$(pdftocairo())` in backticks), which
    # the analysis does not see through.
    @test ExplicitImports.check_no_stale_explicit_imports(
        BlobBoards; ignore = (:pdftocairo,)
    ) === nothing
    # NB deliberately NOT enforced: check_all_qualified_accesses_are_public —
    # the package knowingly reaches into non-public upstream names
    # (KernelAbstractions.zeros/supports_float64, SpatialIndexing.Rect/id,
    # Threads.maxthreadid, …). That list is an accepted
    # risk inventory, not an error class; see the package-cleanup notes.
end
