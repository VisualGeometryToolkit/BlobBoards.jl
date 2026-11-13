# ReTestItems: the suite's files are test items (test/items_test.jl), run across
# worker processes (2026-09-15), so the release gate is not on the critical path
# of development. Items fan out over workers WITH threads (the detector
# multithreads), and the items that use the GPU run afterwards on ONE worker:
# two suites on a shared card is a false-failure machine (the 2026-08-12
# gates). `BB_TEST_SKIP_GPU=1` skips the GPU run — a development convenience
# when the card is busy, never the gate on a GPU machine.
using Test, ReTestItems, BlobBoards

# The items come from the package (test/items_test.jl; ReTestItems does not
# descend into bmvc2026/, which has its own Project.toml). Named by the module,
# not by the file: from a path ReTestItems walks up to the nearest Project.toml,
# which is the unnamed workspace member test/Project.toml, and fails to build a
# test env for it.
# ReTestItems' own variables size the pool when set (a gate task sized to its
# allocation: RETESTITEMS_NWORKERS, RETESTITEMS_NWORKER_THREADS); else from the machine.
const NWORKERS = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(clamp(Sys.CPU_THREADS ÷ 4, 1, 6))))
const NWORKER_THREADS = get(ENV, "RETESTITEMS_NWORKER_THREADS", "4")
# A selection — `Pkg.test(; test_args = [regex, …])` or `julia --project=test
# test/runtests.jl regex …` — runs only the items whose name matches one of the
# patterns (and the Python interface suite only when "python interface" does): a
# gate task runs one group of items. A part with nothing selected is skipped;
# a selection that matches nothing anywhere is an error.
const SELECTED = isempty(ARGS) ? (_ -> true) : let r = Regex(join(ARGS, "|")); name -> occursin(r, name) end
const RAN = Ref(false)
function run_items(filter; kwargs...)
    try
        runtests(filter, BlobBoards; testitem_timeout = 3600, report = false, kwargs...)
        RAN[] = true
    catch e
        (!isempty(ARGS) && nameof(typeof(e)) === :NoTestException) || rethrow()
    end
    return nothing
end

run_items(ti -> SELECTED(ti.name) && !(:gpu in ti.tags); nworkers = NWORKERS, nworker_threads = NWORKER_THREADS)
# The Python interface, serially in this process: PythonCall's initialisation
# hangs inside a piped ReTestItems worker (2026-09-15), and the suite is six
# seconds — not worth a worker anyway. It needs an interpreter with numpy and
# pint: inside `pixi run -e py`, JULIA_PYTHONCALL_EXE names one. Without it the
# suite is SKIPPED (a CPU CI runner without the pixi env), loudly, rather than
# left to PythonCall's own conda bootstrap; `BB_TEST_SKIP_PYTHON=1` skips it too.
const RUN_PYTHON = get(ENV, "BB_TEST_SKIP_PYTHON", "0") != "1" && haskey(ENV, "JULIA_PYTHONCALL_EXE")
if !SELECTED("python interface")
elseif RUN_PYTHON
    RAN[] = true
    @testset "python interface" begin
        using JSON
        using VisualGeometryToolkitCore: mm, px, dpi
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
    run_items(ti -> SELECTED(ti.name) && :gpu in ti.tags; nworkers = 1, nworker_threads = get(ENV, "RETESTITEMS_NWORKER_THREADS", "8"))
end
RAN[] || isempty(ARGS) || error("runtests.jl: no test item matches $(repr(ARGS)) (or its part was skipped)")
