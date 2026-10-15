using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        BlobBoards;
        ambiguities = false,       # Skip — many come from upstream (Makie, FlexUnits)
        # No stale_deps ignore list: every [deps] entry loads with the package.
        # An EXT-ONLY dep would need one (an extension may use only the
        # parent's hard deps plus its own triggers); the last, QRCoders for
        # the board QR corner, was replaced by the vendored encoder
        # (src/authoring/qrencode, VENDORED.toml). Anything unloaded
        # is a rotting dependency — fail the build.
        # (AcceleratedKernels left [deps] with the dead segmented-primitive
        # files; SpecialFunctions is now consumed by src/ — the KA adapt's
        # CPU loggamma — so it loads with the package and needs no ignore.)
        #
        # NB this list cannot protect you from the opposite error. Aqua asks
        # whether a dep is LOADED, so a dep pulled in transitively by another
        # dep reads as live no matter what: HDF5 sat in [deps] unused across
        # several releases, kept alive by VisualGeometryFeatures, and was
        # found by grepping import lines instead (at a release audit).
        # NB no :CUDA/:cuDNN here — they live in [weakdeps], which Aqua's
        # stale-deps check never inspects; listing them was inert (2026-09 audit).
        stale_deps = true,
        # ON since the 2026-09 audit: the old justification ("we extend
        # GeometryBasics.Rect, the units package's ustrip") described methods
        # that either carry a BB-owned type (not piracy) or never existed;
        # the one real pirated method (Size2(::Makie.LScene), dead) was deleted.
        piracies = true,
        # Skip — the CUDA/cuDNN/ONNXRunTime runtime stack holds background
        # tasks by design; the precompile-hang risk this check probes is
        # already exercised by CI precompilation of the package itself.
        persistent_tasks = false,
    )
end
