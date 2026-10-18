#=
make_param_variations.jl — Fig. 2 (fig:params): the pattern-parameter ablation.

# reproduce: writes papers/images/blob_variations_squared/blob_pattern_*.png
# reproduce: reads configs/figures/param_variations.toml

Nine patterns: a baseline (cutoff c = 1.5, α = 1.2, max diameter fraction 0.4,
minimum blob scale 0.2 mm) and two variations of each of the four parameters,
every other setting held at the baseline. The paper shows the baseline once per
row, so the nine files fill a 4 × 3 grid.

    julia --project=bmvc2026 bmvc2026/scripts/make_param_variations.jl [--config=<path>]

Every parameter — the seed, the pattern resolution, dpi and ladder, the nine
panels and their file names — is in the config of record,
configs/figures/param_variations.toml; each run writes its record under
plotsdir("records").

Env: BB_PAPER_IMAGES (output root, default papers/images — see
src/paper_assets.jl; point it at a scratch tree to review a regeneration).
CPU only; no dataset needed.

## What was recovered, and what was not

The submitted PNGs (2026-08) predate this script: their generator and the
pattern JSONs that carried the seeds were lost. Everything except the seeds is
recovered from the images themselves (2026-10-08, measured scale ladders):

  * pattern: 1890 × 1890 px — 40 mm (the small board) at 1200 dpi.
  * minimum σ: `round(s_mm · 1200 / 25.4)` px — 9, 33 and 80 px for s = 0.2,
    0.7 and 1.7 mm. The measured smallest rungs are 9.0, 32.8 and 79.5 px, not
    the unrounded 9.45/33.07/80.3: the original rounded to whole pixels.
  * maximum σ: the TOP RUNG of the 2^(1/3) ladder whose disc diameter fits in
    `max_diameter_fraction` of the short side — the v1 `max_diameter_fraction`
    semantics, which the package still reads (a v1 `pattern_config` converts
    to the rung it always produced). The measured tops (181, 144, 144, 288,
    229, 166, 160 px) match this rule for every panel.
  * scales per octave 3, dark polarity, 8-bit gray, the PNG written linear
    (`file_gamma = 1.0`, the package's pattern writer).

The seeds are NOT recoverable: the original configs drew them from
`RandomDevice` (64-bit), and the `<uid>_<hash5>` in each file name is a SHA-256
over the whole config INCLUDING the seed, in the encoding of that code era. So
the config fixes ONE documented seed for all nine panels (a shared
seed also makes the panels within a row differ only by the varied parameter),
and the regenerated patterns are statistically equivalent to the submitted ones
— same scale ladder, same per-rung counts up to placement — but not
bit-identical. The output file names are the PAPER's (`[[panels]]` in the config), kept
verbatim so `\includegraphics` does not change; their `<uid>_<hash5>` is the
historical identity of the lost originals, not the hash of the regenerated
config (which is in each pattern's JSON sidecar under plots/param_variations/).
=#

using DrWatson
@quickactivate
using DrWatson: projectdir

using BlobBoards
using BlobBoards: PatternConfig
using VisualGeometryToolkitCore: ustrip, px, mm, dpi
import JSON3
if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: paper_images_dir, figure_config, write_figure_records
using VisualGeometryToolkitEval.Experiments: Stage, run_stages
using DrWatson: plotsdir

const StructTypes = BlobBoards.StructTypes
const CFG, CFG_PATH = figure_config("param_variations"; default = projectdir("configs", "figures", "param_variations.toml"))
const PAT = CFG["pattern"]

# The panel's PatternConfig, through the package's v1 reader: a v1
# `pattern_config` carries `max_diameter_fraction`, which the reader converts
# to the ladder's top rung — exactly the rule the originals were made with.
function panel_config(p)
    sq = Dict("value" => PAT["resolution_px"], "unit" => "px")
    v1 = Dict(
        "pattern_resolution" => Dict("width" => sq, "height" => sq),
        "min_scale" => Dict("value" => round(ustrip(px, p["min_scale_mm"] * mm * (PAT["dpi"] * dpi))), "unit" => "px"),
        "seed" => PAT["seed"], "pixel_type" => PAT["pixel_type"],
        "max_diameter_fraction" => p["max_diameter_fraction"], "scales_per_octave" => PAT["scales_per_octave"],
        "max_trials" => PAT["max_trials"], "alpha" => p["alpha"], "cutoff" => p["cutoff"],
        "raster_aligned" => PAT["raster_aligned"], "polarity" => PAT["polarity"],
    )
    return StructTypes.construct(PatternConfig, JSON3.read(JSON3.write(v1)))
end

function main()
    out_dir = paper_images_dir(CFG["output"]["dir"])
    sidecars = plotsdir(CFG["output"]["sidecars"])    # the regenerated configs + blobs (JSON)
    mkpath(out_dir)
    outputs = String[]
    for p in CFG["panels"]
        config = panel_config(p)
        # The package's pattern writer: linear-gamma 8-bit PNG + JSON sidecar
        # named by the regenerated config's hash. The PNG is then published
        # under the paper's name.
        _, blobs = blob_pattern(config; dir = sidecars)
        png = joinpath(sidecars, BlobBoards.make_filebase(BlobBoards.PatternBundle(config, blobs)) * ".png")
        out = joinpath(out_dir, p["name"])
        cp(png, out; force = true)
        push!(outputs, out)
        @info "panel" name = p["name"] n_blobs = length(blobs)
    end
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_param_variations.jl", outputs)
    return out_dir
end

run_stages("make_param_variations", [Stage(:render, "the nine pattern panels", () -> false, main)])
