# =============================================================================
# bmvc2026's tests, as ReTestItems test items. Run from the repository root:
#   julia --project=bmvc2026 bmvc2026/test/runtests.jl      (or `make check` in papers/)
#   julia --project=bmvc2026 bmvc2026/test/runtests.jl 'cameras|formats'   # items whose name matches
#   RETESTITEMS_NWORKERS=0 → serial, in-process                            # for debugging
#
# Every `*_tests.jl` file here holds `@testitem`s and `@testsetup`s only. Each
# item is evaluated in its own module on a pool of worker processes: items share
# no bindings and run in no fixed order, so an item imports everything it uses.
# The shared fixtures are the `@testsetup` modules of setups_tests.jl (BBEval:
# the project's src/BlobBoardsEval.jl; OperatingPoints), each loaded once per
# worker.
#
# The project has no package module (bmvc2026/Project.toml is unnamed), so the
# items are found by path; run as this file, ReTestItems takes the active
# environment as the test environment and builds none.
#
# Tags:
#   :data     reads the input data on this machine (the bound data_root(), or
#             the installed inputs tier); records a skip when it is not there
#             (the data root not bound — VGTK_DATA_ROOT, the data/exp_raw link
#             and BlobBoards' data_dir all unset — or the tier not installed)
#   :results  reads a run's artifact under results_root(); records a skip when
#             the artifact is not there (the results root not bound, or the run
#             not made)
#   :gpu      the device items, on ONE worker with CUDA_VISIBLE_DEVICES=0, after
#             the pool has exited: two processes on one card is a false-failure
#             machine. No item carries it today; the GPU run is kept so that
#             one that does runs alone.
# Every run: testitem_timeout = 3600 s (BlobBoards' setting).
# =============================================================================

using ReTestItems

const NAME = isempty(ARGS) ? nothing : Regex(join(ARGS, "|"))
const NWORKERS = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(max(1, Sys.CPU_THREADS ÷ 2))))
const ONE = NWORKERS == 0 ? 0 : 1          # the GPU run: one worker (serial mode: in-process)

# One run of a selection of items. A run may match no item (the GPU run today,
# or a name pattern), which ReTestItems reports by throwing; that run is then
# skipped, and only a selection that matches no item in any run is an error.
const RAN = Ref(false)
function run_items(select; kw...)
    try
        runtests(select, @__DIR__; name = NAME, testitem_timeout = 3600, nworker_threads = 2, kw...)
        RAN[] = true
    catch e
        e isa ReTestItems.NoTestException || rethrow()
    end
end

run_items(ti -> !(:gpu in ti.tags); nworkers = NWORKERS)

withenv("CUDA_VISIBLE_DEVICES" => "0") do
    run_items(ti -> :gpu in ti.tags; nworkers = ONE)
end

RAN[] || error("no test item matches $(repr(NAME))")
