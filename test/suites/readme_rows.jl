# readme_rows.jl — what the readme items share: the install table of the README
# and of docs/src/guide/install.md, the examples of the pages that show them
# (each block names its install row in its first line), and the runner that
# executes one row's examples in a fresh environment built from that row's
# install line alone.
#
# Why a fresh environment per row: a reader's environment holds exactly what the
# install line adds, not this test target. A colleague of the owner had to add
# package after package by hand to author and find boards, because the README's
# one install line did not say what the extensions (GPU, plots, training,
# Python) need. Each row's line is now checked against what its examples load,
# and the examples run in an environment made from that line.

using Test

const README_ROOT = pkgdir(BlobBoards)

# The pages whose code examples follow the pattern: a first line naming the
# install row (`# Install row: <task>`), or `# Illustrative` then that line for
# a block that cannot run as written (the reader's own data).
const EXAMPLE_PAGES = [
    "README.md",
    joinpath("docs", "src", "index.md"),
    joinpath("docs", "src", "overview.md"),
    joinpath("docs", "src", "guide", "getting-started.md"),
    joinpath("docs", "src", "guide", "install.md"),
]

const INSTALL_ROWS = [
    "author boards (pattern, PDF, PNG)",
    "find boards on the CPU",
    "find boards on the GPU",
    "track boards in a video",
    "plot or visualize",
    "train the descriptor network",
    "use from Python",
    "work in a clone",
]

# A row's name as an example names it: the task without its parenthesis.
row_key(task) = strip(replace(task, r"\s*\(.*\)$" => ""))

# The install table of a page (task => its install entry), the examples of the
# pages, the packages an entry adds and an example loads, a row's fresh
# environment and its run: VisualGeometryToolkitCore's README toolkit (its Test
# extension), the stack's one copy.
import VisualGeometryToolkitCore as VgtkCore

install_table(path) = VgtkCore.install_table(path, INSTALL_ROWS)
examples() = VgtkCore.readme_examples(README_ROOT, EXAMPLE_PAGES)

# Build the environment of `task`'s install line (BlobBoards by path, the rest
# from the registries), then run each distinct Julia example of the row in its
# own module, so each is shown to be complete on its own. An example whose last
# line ends `# true` must return `true`. `after` is code run in the same child
# after the examples (assertions of the row), BlobBoards imported.
function run_row(task; setup = "", after = "")
    packages = VgtkCore.install_packages(install_table(joinpath(README_ROOT, "README.md"))[task])
    env = VgtkCore.readme_environment(README_ROOT, packages; setup)
    blocks = unique(e.code for e in examples() if e.lang == "julia" && e.row == row_key(task) && !e.illustrative)
    VgtkCore.test_readme_row(env, task, blocks; preamble = "import BlobBoards    # for `after`; each example loads its packages itself", after)
    return nothing
end
