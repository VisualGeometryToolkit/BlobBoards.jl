# =============================================================================
# Paper assets — where a figure generator writes.
#
# Every generator of a paper asset writes under `paper_images_dir(...)`, which
# is `papers/images` unless `BB_PAPER_IMAGES` points elsewhere. The override is
# how a regeneration is REVIEWED before it replaces the submitted asset: run the
# generators into a scratch tree, compare, and only then let them write into
# papers/images (REPRODUCE.md, "Reviewing a regeneration").
# =============================================================================

import VisualGeometryToolkitEval as VgtkEval
using DrWatson: projectdir, plotsdir

"""
    paper_images_dir(parts...) -> String

The directory the paper's `\\includegraphics` assets live in
(`papers/images/<parts...>`), or `ENV["BB_PAPER_IMAGES"]/<parts...>` when the
override is set. Creates nothing.
"""
paper_images_dir(parts::AbstractString...) =
    joinpath(get(ENV, "BB_PAPER_IMAGES", projectdir("papers", "images")), parts...)

"""
    paper_figures_dir(parts...) -> String

The directory the paper's `\\input` tables and macro files live in
(`papers/figures/<parts...>`), or `ENV["BB_PAPER_FIGURES"]/<parts...>`.
"""
paper_figures_dir(parts::AbstractString...) =
    joinpath(get(ENV, "BB_PAPER_FIGURES", projectdir("papers", "figures")), parts...)

# =============================================================================
# Figure configs of record + run records.
#
# Every paper-figure generator reads ONE TOML config of record,
# `configs/figures/<generator>.toml` (or `--config=<path>`), holding every
# parameter it uses — frame picks, crops, seeds, panel lists, sizes, inputs,
# output names — and writes a run record for each run: VisualGeometryToolkitEval's
# `write_run_records` (`_config.toml` = the effective config,
# `_provenance.toml` = DrWatson `tag!` gitcommit (+ `_dirty.patch`), config
# source, run key, inputs) into the KEYED directory
# `plotsdir("records", "<generator>-<run_key>")`, plus `_outputs.toml` listing
# what it wrote. The run key is `Experiments.run_key(config)` — SHA-256 of
# the sorted TOML — so the same config always lands in the same record.
# =============================================================================

import TOML
import VisualGeometryToolkitEval

"""
    figure_config(generator; args = ARGS, default) -> (cfg::Dict, path::String)

The config of record of a paper-figure generator: `default`
(`configs/figures/<generator>.toml`; scripts spell it out, so the config each
reads is visible in its source),
or the `--config=<path>` among `args` whose `[experiment] generator` names this
generator (several `--config=` may be given — regenerate_paper.jl forwards its
arguments to every generator, and each takes only its own). The `[experiment]`
table must name the generator, so a config cannot be read by the wrong script.
"""
function figure_config(generator::AbstractString; args = ARGS,
        default::AbstractString = projectdir("configs", "figures", generator * ".toml"))
    _gen(p) = get(get(TOML.parsefile(p), "experiment", Dict()), "generator", "")
    given = [abspath(a[(length("--config=") + 1):end]) for a in args if startswith(a, "--config=")]
    foreach(p -> isfile(p) || error("figure config not found: $p"), given)
    mine = filter(p -> _gen(p) == generator, given)
    length(mine) <= 1 || error("several --config= name the generator $(repr(generator)): $mine")
    path = isempty(mine) ? default : only(mine)
    isfile(path) || error("figure config not found: $path")
    cfg = TOML.parsefile(path)
    exp = get(cfg, "experiment", nothing)
    exp isa AbstractDict || error("$path: no [experiment] table")
    get(exp, "generator", "") == generator ||
        error("$path is the config of $(repr(get(exp, "generator", ""))), not of $(repr(generator))")
    return cfg, path
end

"""
    write_figure_records(cfg, path; driver, inputs, outputs) -> String

The run record of one generator run (see the section header): returns the
keyed record directory.
"""
function write_figure_records(cfg::AbstractDict, path::AbstractString;
        driver::AbstractString, inputs::Vector{String} = String[], outputs::Vector{String} = String[])
    E = VgtkEval.Experiments
    key = E.run_key(cfg)
    dir = plotsdir("records", string(cfg["experiment"]["generator"], "-", key))
    E.write_run_records(dir, cfg; driver, config_source = path, key, inputs)
    open(joinpath(dir, "_outputs.toml"), "w") do io
        TOML.print(io, Dict("outputs" => outputs, "sizes" => [isfile(o) ? filesize(o) : -1 for o in outputs]))
    end
    return dir
end
