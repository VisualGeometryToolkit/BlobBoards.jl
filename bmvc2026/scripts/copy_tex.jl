#=
copy_tex.jl — the generated tables and macro files into the paper's figures
tree: every `plots/*.tex` into papers/figures/floorplan, every
`plots/occ_<L>/*.tex` into papers/figures/floorplan_occ_<L>, the drivers'
compilable previews (`*_standalone.tex`) excepted. A GLOB, not a hand list: an
explicit list is how a generated table once aged behind its neighbours
(flip_table, 2026-08-17). The stage `copy_tex` of configs/reproduce.toml, after
every generator that writes into plots/.

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/copy_tex.jl [--config=<toml>]

# reproduce: reads configs/figures/copy_tex.toml
# reproduce: writes papers/figures/floorplan/*.tex
# reproduce: writes papers/figures/floorplan_occ_*/*.tex
=#

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: paper_figures_dir, figure_config, write_figure_records
using VisualGeometryToolkitEval.Experiments: Stage, run_stages

const CFG, CFG_PATH = figure_config("copy_tex"; default = projectdir("configs", "figures", "copy_tex.toml"))
const PLOTS = plotsdir()

_copied(f) = endswith(f, ".tex") && !any(x -> endswith(f, x), CFG["copy"]["exclude_suffixes"])

function copy_tex()
    copied = String[]
    mkpath(paper_figures_dir("floorplan"))
    for f in readdir(PLOTS)
        _copied(f) && push!(copied, cp(joinpath(PLOTS, f), paper_figures_dir("floorplan", f); force = true))
    end
    for d in readdir(PLOTS; join = true)
        isdir(d) && startswith(basename(d), "occ_") || continue
        dst = paper_figures_dir("floorplan_$(basename(d))")
        mkpath(dst)
        for f in readdir(d)
            _copied(f) && push!(copied, cp(joinpath(d, f), joinpath(dst, f); force = true))
        end
    end
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/copy_tex.jl", outputs = copied)
    println("copy_tex: $(length(copied)) files into papers/figures")
    return copied
end

run_stages("copy_tex", [Stage(:copy, "plots/*.tex (and plots/occ_<L>/*.tex) into papers/figures", () -> false, () -> (copy_tex(); nothing))])
