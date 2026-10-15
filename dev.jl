#!/usr/bin/env julia
# Development helper script for BlobBoards.jl + VisualGeometryCore
#
# Usage:
#   julia --project=. dev.jl              # Start REPL with Revise
#   julia --project=. dev.jl test         # Run tests
#   julia --project=. dev.jl test-strict  # Run tests with --depwarn=error
#   julia --project=. dev.jl coverage     # Pkg.test(coverage=true) + per-file table vs dev/gates_baseline.toml
#   julia --project=. dev.jl invalidations # @snoop_invalidations `using BlobBoards` vs the recorded ratchet
#   julia --project=. dev.jl clean        # Clean precompilation cache

using Pkg

if length(ARGS) > 0
    cmd = ARGS[1]

    if cmd == "test"
        println("Running tests...")
        Pkg.test()

    elseif cmd == "test-strict"
        # --depwarn=error. In a stack that cascades (VGC -> VGF/RVG -> BB) an
        # upstream deprecation should surface at the release that INTRODUCES
        # it, not at the one that removes it. Worth running periodically: on
        # 2026-08-14 it caught `CUDA.available_memory` (deprecated for
        # `free_memory`) in the arena-reservation testset — three call sites
        # that had been warning into the log for weeks with nothing reading it.
        #
        # NOT the default gate, and deliberately so: CUDA.jl's own
        # `link_libraries!` override trips a GPUCompiler deprecation on EVERY
        # kernel compilation, so every GPU testset errors out for a reason
        # BlobBoards cannot fix. Re-check when CUDA.jl migrates to the 2-arg
        # form; promote this to `test` the day it comes back clean.
        println("Running tests with --depwarn=error (expect upstream CUDA/GPUCompiler noise)...")
        Pkg.test(julia_args = ["--depwarn=error"])

    elseif cmd == "coverage"
        # Coverage floor (package-cleanup Phase 6): the ONE gate that names what
        # the suite silently does NOT touch. Read it per FILE — a healthy
        # headline hides an extension sitting at zero (0.11.0 shipped LuxExt,
        # GLMakieExt and TrainingExt entirely uncovered; nothing else reports
        # that). A separate target, not a suite gate: it re-runs the whole
        # suite. Ratchet: a file may only ever RISE above its recorded floor.
        # Coverage.jl comes from the shared env (never a package dep).
        Pkg.activate("v1.12"; shared = true)
        haskey(Pkg.project().dependencies, "Coverage") || Pkg.add("Coverage")
        Pkg.activate(@__DIR__)
        @eval using Coverage, TOML, Logging
        # A red testset must not hide the table: the .cov files are written
        # regardless, and the suite reports its own failures.
        try
            Pkg.test(; coverage = true)
        catch err
            @warn "coverage: the suite did not succeed; reading the .cov files it wrote anyway" exception = err
        end
        floors = get(TOML.parsefile(joinpath(@__DIR__, "dev", "gates_baseline.toml")), "coverage_floor", Dict{String, Any}())
        rows = Tuple{String, Float64}[]
        # CoverageTools logs one Info line per file; the table below is the output.
        files = Logging.with_logger(Logging.NullLogger()) do
            vcat(process_folder(joinpath(@__DIR__, "src")), process_folder(joinpath(@__DIR__, "ext")))
        end
        for fc in files
            covered, total = get_summary(fc)
            rel = relpath(fc.filename, @__DIR__)
            push!(rows, (rel, total == 0 ? 100.0 : 100 * covered / total))
        end
        sort!(rows; by = last)
        failed = String[]
        for (rel, pct) in rows
            floor = get(floors, rel, 0.0)
            flag = pct < floor ? "  UNDER FLOOR ($floor)" : ""
            isempty(flag) || push!(failed, rel)
            println(rpad(rel, 44), lpad(string(round(pct; digits = 1)), 6), "%", flag)
        end
        clean_folder(@__DIR__)
        isempty(failed) || error("coverage under the recorded floor: $(join(failed, ", "))")
        println("✓ Coverage at or above every recorded floor")

    elseif cmd == "invalidations"
        # SnoopCompile invalidation ratchet (package-cleanup Phase 6): a load
        # that invalidates large upstream method trees is what a multi-minute
        # cold start is made of, and no other gate sees it. Count against the
        # recorded ceiling; lower the ceiling whenever you fix some.
        Pkg.activate("v1.12"; shared = true)
        for dep in ("SnoopCompileCore", "SnoopCompile")
            haskey(Pkg.project().dependencies, dep) || Pkg.add(dep)
        end
        Pkg.activate(@__DIR__)
        @eval using SnoopCompileCore, TOML
        invs = @eval @snoop_invalidations using BlobBoards
        @eval using SnoopCompile
        trees = @eval invalidation_trees($invs)
        n = length(trees)
        ceiling = get(TOML.parsefile(joinpath(@__DIR__, "dev", "gates_baseline.toml")), "invalidation_trees_max", typemax(Int))
        println("invalidation trees on `using BlobBoards`: ", n, "  (ceiling ", ceiling, ")")
        for t in last(trees, min(10, n))   # largest last
            println("  ", t)
        end
        n <= ceiling || error("invalidations regressed: $n trees > ceiling $ceiling")

    elseif cmd == "clean"
        println("Cleaning precompilation cache...")
        rm(joinpath(homedir(), ".julia/compiled/v1.12/VisualGeometryCore"),
           recursive=true, force=true)
        rm(joinpath(homedir(), ".julia/compiled/v1.12/BlobBoards"),
           recursive=true, force=true)
        println("✓ Cache cleaned. Next load will trigger full recompilation.")

    elseif cmd == "recompile"
        println("Forcing precompilation...")
        Pkg.precompile()
        println("✓ Precompilation complete")

    else
        println("Unknown command: $cmd")
        println("Available commands: test, test-strict, coverage, invalidations, clean, recompile")
    end
else
    # Interactive development mode with Revise
    println("Loading Revise for auto-reloading...")
    try
        @eval using Revise
        println("✓ Revise loaded - changes will auto-reload")
    catch
        @warn "Revise not available. Install with: Pkg.add(\"Revise\")"
    end

    println("Loading BlobBoards...")
    @eval using BlobBoards
    @eval using VisualGeometryCore: dpi, mm, cm, Size2

    println("\n" * "="^60)
    println("Development environment ready!")
    println("="^60)
    println("\nQuick commands:")
    println("  blob_pattern(200, 150; blob_sizes=(12px, nothing))")
    println("  blob_board(\"A4\"; blob_sizes = suggest_blob_sizes(600dpi, \"A4\"), dir = \"./out\")")
    println("\nEdit VisualGeometryCore files - changes auto-reload with Revise")
    println("="^60 * "\n")
end
