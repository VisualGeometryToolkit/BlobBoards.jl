# The stack's iterative-estimator vocabulary (CLAUDE.md, "Function Verb
# Conventions"), checked by VisualGeometryToolkitCore's `check_naming`: every name
# BlobBoards binds, and every line of its stack text (paper trees excluded).
using VisualGeometryToolkitCore: check_naming

# bmvc2026/ is the BMVC 2026 paper project, pinned to its released environment
# (BlobBoards 0.7.4 on the Core 0.13 stack, where it reproduces): it is not
# checked against this release's vocabulary.
_paper(f) = startswith(f, "bmvc2026/")
@testset "naming" begin
    found = filter(v -> !_paper(v.file), check_naming(pkgdir(BlobBoards)))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end
