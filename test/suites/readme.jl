# readme.jl — the README's examples and the documentation's Getting started run
# as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code; it
# once began with shell lines inside a julia block and loaded placeholder files
# ("board.json", "cameras.json", "descriptor.onnx") it never wrote (General
# registration review, 2026-11).
#
# The examples import names from VisualGeometryToolkitCore, and a reader's
# environment holds only what the install line adds, not this test target. So
# the item builds a temporary environment — this package by path, every other
# package of the install line from the registries — and runs the blocks in a
# child Julia process there. The install lines are `pkg>` blocks
# (```julia-repl), so they are not run as code: the item checks instead that the
# `pkg> add` line names every non-standard-library package the examples load,
# and nothing else, and that the documentation shows the same line.
#
# A ```julia block whose first line begins `# Illustrative` is not part of the
# runnable example (the calibrated GPU pipeline on the reader's own camera and
# images, the API signatures, the training pipeline on the reader's videos): it
# is excluded here, by that marker and by count, so a new block is classified
# rather than silently skipped.

using Test
root = pkgdir(BlobBoards)
julia_blocks(path) = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, read(path, String))]
illustrative(block) = startswith(block, "# Illustrative")
readme_path = joinpath(root, "README.md")
docs_path = joinpath(root, "docs", "src", "guide", "getting-started.md")
readme = read(readme_path, String)
example = filter(!illustrative, julia_blocks(readme_path))
docs = filter(!illustrative, julia_blocks(docs_path))
# Patterns and boards, Blob Extraction, Detection on the CPU; then the GPU
# pipeline, the two signature blocks and the training pipeline.
@test length(example) == 3
@test count(illustrative, julia_blocks(readme_path)) == 4
@test length(docs) == 7
@test !any(illustrative, julia_blocks(docs_path))

# The install line adds exactly the packages the examples load, and the
# documentation shows the same line.
install = match(r"^pkg> add (.+)$"m, readme)
@test install !== nothing
@test [m[1] for m in eachmatch(r"^pkg> add (.+)$"m, read(docs_path, String))] == [install[1]]
packages = split(install[1])
loaded = Set{String}()
for code in [example; docs], m in eachmatch(r"^(?:using|import)\s+([^:\n]+)"m, code)
    for part in split(m[1], ',')
        push!(loaded, first(split(strip(part), r"[.\s]")))
    end
end
stdlib(name) = isdir(joinpath(Sys.STDLIB, name))
@test Set(filter(!stdlib, loaded)) == Set(packages)

env = mktempdir()
write(joinpath(env, "readme_example.jl"), join(example, "\n"))
write(joinpath(env, "docs_example.jl"), join(docs, "\n"))
registered = filter(!=("BlobBoards"), String.(packages))
script = joinpath(env, "run.jl")
write(script, """
using Pkg
Pkg.develop(path = $(repr(root)); io = devnull)
Pkg.add($(repr(registered)); io = devnull)
readme = Module(:ReadmeSandbox)
Base.include(readme, $(repr(joinpath(env, "readme_example.jl"))))
docs = Module(:DocsSandbox)
Base.include(docs, $(repr(joinpath(env, "docs_example.jl"))))
using Test
@testset "README example" begin
    @test length(readdir(readme.board_dir)) == 2            # the PDF and its sidecar
    @test length(readme.blobs_mm) == length(readme.blobs_px) > 0
    # COLMAP → Julia moves every centre by half a pixel on both axes.
    c = readme.ustrip.(readme.px, readme.centroid(first(readme.blobs_px)))
    @test first(readme.centres_julia) ≈ c .+ 0.5
    # The rendered board is found in its own image, by most of its blobs.
    @test readme.uid == readme.scene.uid
    @test readme.n_inliers >= length(readme.scene.world_frames) ÷ 2
end
@testset "Getting started examples" begin
    @test length(docs.X) == length(docs.world) > 0
    @test all(p -> p[3] == 0, docs.X)                       # on the board's plane
    @test docs.config == docs.pattern_config(docs.bundle)
    @test "A4" in keys(docs.BlobBoards.PRINT_SIZES)
end
""")
julia = joinpath(Sys.BINDIR, Base.julia_exename())
cmd = `$julia --startup-file=no --project=$env $script`
# A clean environment: none of the test run's load path or project.
proc = withenv("JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing) do
    run(ignorestatus(cmd))
end
@test proc.exitcode == 0
