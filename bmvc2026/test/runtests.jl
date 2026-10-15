# =============================================================================
# bmvc2026's tests. Run from the repository root:
#   julia --project=bmvc2026 bmvc2026/test/runtests.jl      (or `make check` in papers/)
# =============================================================================
using Test

include("reproduce_tests.jl")
include("paper_project_tests.jl")
include("operating_points_tests.jl")
include("formats_tests.jl")
include("data_tests.jl")
include("cameras_tests.jl")
