# ReTestItems: the suite's files are test items (test/items_test.jl), run across
# worker processes (2026-09-15), so the release gate is not on the critical path
# of development. Items fan out over workers WITH threads (the detector
# multithreads), and the items that use the GPU run afterwards on ONE worker:
# two suites on a shared card is a false-failure machine (the 2026-08-12
# gates). `BB_TEST_SKIP_GPU=1` skips the GPU run — a development convenience
# when the card is busy, never the gate on a GPU machine.
using Test, ReTestItems, BlobBoards

const ITEMS = joinpath(@__DIR__, "items_test.jl")
const NWORKERS = clamp(Sys.CPU_THREADS ÷ 4, 1, 6)

runtests(
    ti -> !(:gpu in ti.tags), ITEMS;
    nworkers = NWORKERS, nworker_threads = 4, testitem_timeout = 3600, report = false,
)
# The Python interface, serially in this process: PythonCall's initialisation
# hangs inside a piped ReTestItems worker (2026-09-15), and the suite is six
# seconds — not worth a worker anyway. It needs an interpreter with numpy and
# pint: inside `pixi run -e py`, JULIA_PYTHONCALL_EXE names one. Without it the
# suite is SKIPPED (a CPU CI runner without the pixi env), loudly, rather than
# left to PythonCall's own conda bootstrap; `BB_TEST_SKIP_PYTHON=1` skips it too.
const RUN_PYTHON = get(ENV, "BB_TEST_SKIP_PYTHON", "0") != "1" && haskey(ENV, "JULIA_PYTHONCALL_EXE")
if RUN_PYTHON
    @testset "python interface" begin
        using JSON3
        using VisualGeometryCore: mm, px, dpi
        using PythonCall: pyconvert, pyimport
        include("suites/python_interface.jl")
    end
else
    @warn "python interface SKIPPED: no JULIA_PYTHONCALL_EXE (run the suite as `pixi run -e py julia --project=. -e 'using Pkg; Pkg.test()'`)"
end

# The GPU items: `BB_TEST_SKIP_GPU=1` skips them, and so does a machine with no
# functional CUDA device (a CPU CI runner) — with a warning, never silently.
using CUDA: CUDA
if get(ENV, "BB_TEST_SKIP_GPU", "0") == "1"
    @warn "GPU test items SKIPPED (BB_TEST_SKIP_GPU=1)"
elseif !CUDA.functional()
    @warn "GPU test items SKIPPED: no functional CUDA device"
else
    runtests(
        ti -> :gpu in ti.tags, ITEMS;
        nworkers = 1, nworker_threads = 8, testitem_timeout = 3600, report = false,
    )
end
