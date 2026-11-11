# =============================================================================
# The suite as ReTestItems test items (the `_test.jl` name is what ReTestItems
# discovers) — one per file of `suites/`, run across
# worker processes by runtests.jl (2026-09-15).
#
# Each item evaluates its suite file in a fresh module: `using Test` and
# `using BlobBoards` are implicit, and the names a body took from the old
# serial runtests.jl preamble are stated per item. Items tagged `:gpu` run on
# ONE worker after
# the fan-out: two suites on a shared card is a false-failure machine. The
# Python interface suite is NOT an item: PythonCall initialised inside a piped
# worker process hangs (measured 2026-09-15), so runtests.jl runs it serially
# in its own process, where the pixi `py` environment's interpreter is found.
# =============================================================================

@testitem "naming" begin
    include("suites/naming.jl")
end

# One implementation per concept across the stack: VisualGeometryToolkitCore's check_duplication,
# its open findings listed (each waits for a hoist; .release/duplication-checklist.md).
@testitem "duplication" begin
    include("suites/duplication.jl")
end

@testitem "quality gates" begin
    include("suites/quality_gates.jl")
end

@testitem "agent docs" begin
    include("suites/agent_docs.jl")
end

# The install table and the examples of the README and the docs, statically;
# then each install row's examples, in a fresh environment built from that
# row's install line alone (a child process: this target is not that env).
@testitem "readme" begin
    include("suites/readme.jl")
end

# Every public name documented, every docstring attached, every shown signature callable.
@testitem "docs" begin
    include("suites/docs.jl")
end

@testitem "readme row: author boards" begin
    include("suites/readme_rows.jl")
    run_row("author boards (pattern, PDF, PNG)")
end

# The CPU row's environment has no extension's packages: every function whose
# methods live in an extension raises the error naming its install line.
@testitem "readme row: find boards on the CPU, and the missing-extension errors" begin
    include("suites/readme_rows.jl")
    run_row("find boards on the CPU"; after = """
        fallbacks = [collect(BlobBoards.EXTENSION_FUNCTIONS);
                     (BlobBoards.board, :BlobBoardsMakieExt); (BlobBoards.board!, :BlobBoardsMakieExt)]
        for (f, ext) in fallbacks
            err = try f("a", 1); nothing catch e; e end
            @test err isa ErrorException
            @test occursin(getproperty(BlobBoards.EXTENSION_INSTALL, ext).install, sprint(showerror, err))
        end
        # The GPU: the network's CUDA provider names the GPU row's line and the runtime pin.
        err = try BlobBoards.DescriptorNetwork(BlobBoards.descriptor_network_path()); nothing catch e; e end
        @test err isa ErrorException
        @test occursin(BlobBoards.EXTENSION_INSTALL.BlobBoardsCUDAExt.install, sprint(showerror, err))
        @test occursin(BlobBoards.CUDA_RUNTIME_PIN, sprint(showerror, err))
        """)
end

@testitem "readme row: track boards in a video" begin
    include("suites/readme_rows.jl")
    run_row("track boards in a video")
end

# With the extension loaded, a call no method fits is the MethodError it always was.
@testitem "readme row: plot or visualize" begin
    include("suites/readme_rows.jl")
    run_row("plot or visualize"; after = """
        @test_throws MethodError BlobBoards.plot_boards("a", 1)
        """)
end

# Runs where a CUDA device is functional (CUDA_VISIBLE_DEVICES=0 on the release
# machine); records a skip otherwise. The runtime pin is part of the row.
# Tagged :gpu, so it runs with the GPU items, one at a time, never beside another item.
@testitem "readme row: find boards on the GPU" tags=[:gpu] begin
    using CUDA: CUDA
    include("suites/readme_rows.jl")
    if get(ENV, "BB_TEST_SKIP_GPU", "0") != "1" && CUDA.functional()
        run_row("find boards on the GPU"; setup = BlobBoards.CUDA_RUNTIME_PIN)
    else
        @test_skip "find boards on the GPU: no functional CUDA device"
    end
end

# The Python script, run by the pixi `py` environment's interpreter (the one
# JULIA_PYTHONCALL_EXE names); a skip without it, as for the Python suite.
@testitem "readme row: use from Python" begin
    include("suites/readme_rows.jl")
    python = get(ENV, "JULIA_PYTHONCALL_EXE", nothing)
    if python === nothing || get(ENV, "BB_TEST_SKIP_PYTHON", "0") == "1"
        @test_skip "use from Python: no JULIA_PYTHONCALL_EXE (run under pixi run -e py)"
    else
        script = joinpath(mktempdir(), "find_board.py")
        write(script, only(e.code for e in examples() if e.lang == "python"))
        # juliacall is given this Julia's own binary, not juliaup's launcher
        # (whose sys.so it cannot load: python/README.md).
        out = withenv("JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing,
                      "PYTHON_JULIAPKG_EXE" => joinpath(Sys.BINDIR, Base.julia_exename())) do
            read(`$python $script`, String)
        end
        @test last(split(strip(out), '\n')) == "True"
    end
end

# The clone row's last line, run in this checkout's root environment: only in a
# clone whose workspace is instantiated (an installed copy has no root Manifest).
@testitem "readme row: work in a clone" begin
    include("suites/readme_rows.jl")
    clone = only(e.code for e in examples() if e.row == "work in a clone")
    code = match(r"^julia --project=\. -e '(using BlobBoards;.*)'$"m, clone)[1]
    if isfile(joinpath(README_ROOT, "Manifest.toml"))
        @test VgtkCore.run_readme_child(README_ROOT, "-e", code) == 0
    else
        @test_skip "work in a clone: no instantiated workspace at $(README_ROOT)"
    end
end

@testitem "JET static analysis" begin
    using JSON
    include("suites/jet.jl")
end

@testitem "bundle serialization" begin
    using JSON
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: mm, px
    include("suites/serialization.jl")
end

@testitem "identity and byte pins" begin
    using JSON
    import VisualGeometryToolkitIO as VgtkIO
    include("suites/identity_pins.jl")
end

@testitem "units convention" begin
    using VisualGeometryToolkitCore: mm, px, dpi
    include("suites/units_convention.jl")
end

@testitem "config from TOML" begin
    include("suites/config_toml.jl")
end

@testitem "detector pixel convention" begin
    include("suites/pixel_convention.jl")
end

@testitem "appearance verification" begin
    include("suites/verify.jl")
end

@testitem "retired argument orders" begin
    include("suites/argument_order.jl")
end

@testitem "raw pose arm" begin
    using VisualGeometryToolkitCore: px
    include("suites/raw_pose_arm.jl")
end

@testitem "serial oracle ≡ engine" begin
    using VisualGeometryToolkitCore: mm, px
    include("suites/oracle.jl")
end

@testitem "patterns" begin
    using JSON
    include("suites/patterns.jl")
end

@testitem "boards" begin
    using TiffImages: TiffImages          # reads the TIFF back (test-only)
    using JSON
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: mm, dpi, px, inch, ustrip, uconvert, intersects, sha256_hex
    include("suites/boards.jl")
end

@testitem "board design resolver" begin
    using CairoMakie                      # BlobBoardsAuthoringExt: `blob_board`
    using Libtiff_jll                     # BlobBoardsTiffExt: `format = "tiff"`
    include("suites/design_resolver.jl")
end

@testitem "adapt: per-blob convergence flag" begin
    include("suites/adapt_converged.jl")
end

@testitem "adapt CPU ≡ GPU" tags=[:gpu] begin
    using VisualGeometryToolkitCore: px
    include("suites/adapt_parity.jl")
end

@testitem "annotate smoke (detection on the device)" tags=[:gpu] begin
    using JSON
    using VisualGeometryToolkitCore: mm, px, dpi
    include("suites/annotate_smoke.jl")
end
