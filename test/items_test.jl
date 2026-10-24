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

@testitem "aqua" begin
    include("suites/aqua.jl")
end

@testitem "explicit imports" begin
    using JSON
    include("suites/explicit_imports.jl")
end

@testitem "naming" begin
    include("suites/naming.jl")
end

@testitem "agent docs" begin
    include("suites/agent_docs.jl")
end

@testitem "JET static analysis" begin
    using JSON
    include("suites/jet.jl")
end

@testitem "bundle serialization" begin
    using JSON
    using VisualGeometryToolkitCore: mm, px
    include("suites/serialization.jl")
end

@testitem "identity and byte pins" begin
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
    using VisualGeometryToolkitCore: mm, dpi, px, inch, ustrip, uconvert, VgcQuantity
    include("suites/boards.jl")
end

@testitem "board design resolver" begin
    using CairoMakie                      # BlobBoardsAuthoringExt: `blob_board`
    using Libtiff_jll                     # BlobBoardsTiffExt: `format = "tiff"`
    include("suites/design_resolver.jl")
end

@testitem "adapt CPU ≡ GPU" tags=[:gpu] begin
    using VisualGeometryToolkitCore: px
    include("suites/adapt_parity.jl")
end

@testitem "annotate smoke (detection on the device)" tags=[:gpu] begin
    using VisualGeometryToolkitCore: mm, px, dpi
    include("suites/annotate_smoke.jl")
end
