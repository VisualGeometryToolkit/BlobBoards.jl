# docs.jl — the docstring gates of the julia-docs-pass skill (suites/docs_gates.jl):
# every public name has a docstring, every docstring is attached to its
# definition, and every signature line a docstring shows is accepted by a method.
# Documenter's checkdocs = :public only checks the docstrings that exist, so a
# public name with none would pass the docs build; this item finds it.
#
# EXCEPTIONS: a finding allowed for good, each with its reason. BACKLOG: work
# not done yet; it only shrinks. Both are printed, so they stay visible.

using Test
include("docs_gates.jl")

const EXCEPTIONS = Dict{String, String}()
const BACKLOG = String[]

for (finding, reason) in EXCEPTIONS
    @info "docs exception: $finding — $reason"
end
isempty(BACKLOG) || @info "docs backlog ($(length(BACKLOG))): $(join(BACKLOG, ", "))"

allowed = union(keys(EXCEPTIONS), BACKLOG)
@test isempty(setdiff(undocumented_public(BlobBoards), allowed))
@test isempty(setdiff(signature_mismatches(BlobBoards), allowed))
@test isempty(setdiff(detached_docstrings(joinpath(pkgdir(BlobBoards), "src")), allowed))
