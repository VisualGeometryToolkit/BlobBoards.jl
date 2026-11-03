# =============================================================================
# The shared fixtures of bmvc2026's test items (runtests.jl), each evaluated
# once per worker that runs an item naming it.
# =============================================================================

# The project's evaluation library, src/BlobBoardsEval.jl. An item takes it
# with `setup = [BBEval]` and `using .BBEval.BlobBoardsEval: …`.
@testsetup module BBEval
    using DrWatson: projectdir
    include(projectdir("src", "BlobBoardsEval.jl"))
end

# src/operating_points.jl alone: it is self-contained, so the operating-point
# items read it without loading the detection stack.
@testsetup module OperatingPoints
    include(joinpath(@__DIR__, "..", "src", "operating_points.jl"))
    # The run configs of record that carry the operating point.
    const RUN_CONFIGS = ["bmvc2026_260527.toml", "bmvc2026_260527_noflip.toml"]
end
