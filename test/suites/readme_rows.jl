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

# The install table of a page: task => its install entry.
function install_table(path)
    rows = Dict{String, String}()
    for line in eachline(path)
        m = match(r"^\| ([^|]+?) \| (.+?) \| [^|]+ \|$", line)
        m === nothing && continue
        m[1] in INSTALL_ROWS || continue
        rows[m[1]] = m[2]
    end
    return rows
end

# The `pkg> add` packages of an install entry (empty for the Python and clone rows).
function install_packages(entry)
    m = match(r"`pkg> add ([^`]+)`", entry)
    return m === nothing ? String[] : String.(split(m[1]))
end

struct Example
    page::String
    lang::String        # "julia", "python" or "bash"
    row::String         # the install row it names
    illustrative::Bool
    code::String
end

function examples()
    out = Example[]
    for page in EXAMPLE_PAGES
        text = read(joinpath(README_ROOT, page), String)
        for m in eachmatch(r"^```(julia|python|bash)\n(.*?)^```"ms, text)
            code = m[2]
            lines = split(code, '\n')
            illustrative = startswith(lines[1], "# Illustrative")
            header = illustrative ? lines[2] : lines[1]
            row = match(r"^# Install row: (.+)$", header)
            push!(out, Example(page, m[1], row === nothing ? "" : String(row[1]), illustrative, code))
        end
    end
    return out
end

# The non-standard-library packages a Julia example loads.
function loaded_packages(code)
    loaded = Set{String}()
    for m in eachmatch(r"^(?:using|import)\s+([^:\n#]+)"m, code), part in split(m[1], ',')
        push!(loaded, first(split(strip(part), r"[.\s]")))
    end
    return Set(filter(name -> !isdir(joinpath(Sys.STDLIB, name)), loaded))
end

# A child Julia in `env`: the test run's flags (Base.julia_cmd), and a load path
# of the project and the standard library only, so no default environment can
# supply a package the install line does not add.
function run_child(env, args...)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$env $args`
    proc = withenv("JULIA_LOAD_PATH" => "@:@stdlib", "JULIA_PROJECT" => nothing) do
        run(ignorestatus(cmd))
    end
    return proc.exitcode
end

# Build the environment of `task`'s install line (BlobBoards by path, the rest
# from the registries), then run each distinct Julia example of the row in its
# own module, so each is shown to be complete on its own. An example whose last
# line ends `# true` must return `true`. `after` is code run in the same child
# after the examples (assertions of the row).
function run_row(task; setup = "", after = "")
    table = install_table(joinpath(README_ROOT, "README.md"))
    packages = install_packages(table[task])
    blocks = unique(e.code for e in examples() if e.lang == "julia" && e.row == row_key(task) && !e.illustrative)
    @test !isempty(blocks)
    env = mktempdir()
    registered = filter(!=("BlobBoards"), packages)
    write(joinpath(env, "build.jl"), """
    using Pkg
    Pkg.develop(path = $(repr(README_ROOT)); io = devnull)
    isempty($(repr(registered))) || Pkg.add($(repr(registered)); io = devnull)
    $setup
    """)
    @test run_child(env, joinpath(env, "build.jl")) == 0
    write(joinpath(env, "run.jl"), """
    using Test
    import BlobBoards    # for `after`; each example loads its packages itself
    @testset $(repr("examples of: " * task)) begin
        for (i, code) in enumerate($(repr(blocks)))
            value = include_string(Module(:Example), code, "example_\$i.jl")
            if endswith(strip(last(split(strip(code), '\\n'))), "# true")
                @test value === true
            end
        end
        $after
    end
    """)
    @test run_child(env, joinpath(env, "run.jl")) == 0
    return nothing
end
