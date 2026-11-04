# readme.jl — the install table and the examples, checked without running them
# (the readme row items run them: suites/readme_rows.jl).
#
# - The README's and Install for your task's tables agree, row for row, and
#   hold the eight tasks.
# - A row whose task is an extension names the extension's own install line
#   (src/extensions.jl), the line its fallbacks print.
# - Every code example of the README and of the pages that show the install
#   lines names its install row in its first line; only the training example is
#   illustrative (it needs the reader's recordings), and the counts are pinned
#   so a new block is classified rather than skipped.
# - Each row's install line names exactly the non-standard-library packages its
#   examples load.
# - The README's example and the landing page's are the same block, and so is
#   the CPU row's.

include("readme_rows.jl")
using TOML

readme_table = install_table(joinpath(README_ROOT, "README.md"))
guide_table = install_table(joinpath(README_ROOT, "docs", "src", "guide", "install.md"))
@test sort(collect(keys(readme_table))) == sort(INSTALL_ROWS)
@test readme_table == guide_table

# The extension rows print the lines the fallbacks print.
ext_install(ext) = getproperty(BlobBoards.EXTENSION_INSTALL, ext).install
@test occursin("`$(ext_install(:BlobBoardsMakieExt))`", readme_table["plot or visualize"])
@test occursin("`$(ext_install(:BlobBoardsTrainingExt))`", readme_table["train the descriptor network"])
gpu = readme_table["find boards on the GPU"]
@test issubset(split(chopprefix(ext_install(:BlobBoardsCUDAExt), "pkg> add ")), install_packages(gpu))
@test occursin("`$(BlobBoards.CUDA_RUNTIME_PIN)`", gpu)

# The table covers every extension of the Project.toml, and each install line
# names its extension's triggers (a Makie backend stands in for Makie).
extensions = TOML.parsefile(joinpath(README_ROOT, "Project.toml"))["extensions"]
@test Set(String.(keys(BlobBoards.EXTENSION_INSTALL))) == Set(keys(extensions))
for (ext, triggers) in extensions
    installed = split(chopprefix(ext_install(Symbol(ext)), "pkg> add "))
    @test issubset(filter(!=("Makie"), vcat(triggers)), installed)
end

all_examples = examples()
@test all(e -> e.row in row_key.(INSTALL_ROWS), all_examples)
@test count(e -> e.illustrative, all_examples) == 1
@test only(filter(e -> e.illustrative, all_examples)).row == "train the descriptor network"
counts = Dict(row => count(e -> e.row == row, all_examples) for row in row_key.(INSTALL_ROWS))
@test counts == Dict(
    "author boards" => 8, "find boards on the CPU" => 3, "find boards on the GPU" => 1,
    "track boards in a video" => 1, "plot or visualize" => 1, "train the descriptor network" => 1,
    "use from Python" => 2, "work in a clone" => 1,
)
# Julia rows hold Julia examples; the Python row a shell line and a Python
# script; the clone row a shell session.
@test all(e -> e.lang == "julia", filter(e -> !(e.row in ("use from Python", "work in a clone")), all_examples))
@test sort([e.lang for e in all_examples if e.row == "use from Python"]) == ["bash", "python"]
@test only(e.lang for e in all_examples if e.row == "work in a clone") == "bash"

# Each row's line names exactly what its examples load.
for task in INSTALL_ROWS
    packages = install_packages(readme_table[task])
    isempty(packages) && continue
    julia = [e.code for e in all_examples if e.lang == "julia" && e.row == row_key(task)]
    @test Set(packages) == union(loaded_packages.(julia)...)
end
# The Python script imports only the pixi environment's packages.
python = only(e.code for e in all_examples if e.lang == "python")
imported = Set(first(split(m[1], '.')) for m in eachmatch(r"^(?:from|import)\s+(\S+)"m, python))
@test imported == Set(["os", "tempfile", "matplotlib", "blobboards"])

# One example, shown in the README, on the landing page and in the CPU row.
page_julia(page) = [e.code for e in all_examples if e.page == page && e.lang == "julia"]
readme_example = only(page_julia("README.md"))
@test page_julia(joinpath("docs", "src", "index.md")) == [readme_example]
@test readme_example in page_julia(joinpath("docs", "src", "guide", "install.md"))

# The README's and the landing page's install line is the example's row; the
# Getting started line is the authoring row's, which its examples use.
install_lines(page) = [m[1] for m in eachmatch(r"^pkg> add (.+)$"m, read(joinpath(README_ROOT, page), String))]
@test install_lines("README.md") == [join(install_packages(readme_table["find boards on the CPU"]), " ")]
@test install_lines(joinpath("docs", "src", "index.md")) == install_lines("README.md")
@test install_lines(joinpath("docs", "src", "guide", "getting-started.md")) ==
    [join(install_packages(readme_table["author boards (pattern, PDF, PNG)"]), " ")]

# The clone row's members are the workspace's.
clone = only(e for e in all_examples if e.row == "work in a clone")
guide = read(joinpath(README_ROOT, "docs", "src", "guide", "install.md"), String)
members = [m[1] for m in eachmatch(r"^\| `([^`]+)` \| "m, guide)]
workspace = TOML.parsefile(joinpath(README_ROOT, "Project.toml"))["workspace"]["projects"]
@test !isempty(members) && issubset(filter(!=("."), members), workspace)
@test occursin("julia --project=. -e 'using BlobBoards;", clone.code)
