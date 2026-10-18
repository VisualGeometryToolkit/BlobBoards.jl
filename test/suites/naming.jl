# The stack's iterative-estimator vocabulary (CLAUDE.md, "Function Verb
# Conventions"), checked by VisualGeometryToolkitCore's `check_naming`: every name
# BlobBoards binds, and every line of its stack text (paper trees excluded).
using VisualGeometryToolkitCore: check_naming

@testset "naming" begin
    found = check_naming(pkgdir(BlobBoards))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end
