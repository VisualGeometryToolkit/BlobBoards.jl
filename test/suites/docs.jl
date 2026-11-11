# docs.jl — the stack's docstring gates, VisualGeometryToolkitCore's `test_docs`
# (its Test extension): every public name has a docstring, every docstring is
# attached to its definition, and every signature line a docstring shows is
# accepted by a method. Documenter's checkdocs = :public only checks the
# docstrings that exist, so a public name with none goes unnoticed by the docs
# build; this item finds it. (BlobBoards' own copy of the gates,
# suites/docs_gates.jl, is deleted: the stack has one.)
#
# Every list is empty: a new exception takes Core's docs_tests.jl form
# (`exceptions`: allowed for good, with its reason; `backlog`: only shrinks).

using Test
using VisualGeometryToolkitCore: test_docs

test_docs(BlobBoards; dirs = ("src", "ext"))
