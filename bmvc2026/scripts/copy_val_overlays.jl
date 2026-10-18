#=
copy_val_overlays.jl — the detection-overlay panels of fig:evaluation_setup_results
(main paper) and fig:supp:setup (supplement), cropped from the detection runs'
overlays and downscaled to the paper's 1500 px width.

The overlays themselves are the detection runs' (`<run>/val/<series>/`: the
BlobBoards sweep's `src/sweep.jl` `save_val_overlay`, the baselines'
`scripts/baselines/detect_markers.py`), the runs found through the evaluation's
config of record (src/runs.jl). This script only SELECTS frames, CROPS them to
the marker panel and resizes, so the panels always depict the runs the paper's
numbers come from. The selection and the crops are the
figure's and live in the config of record, configs/figures/val_overlays.toml
(`--config=<path>` overrides): three captures per system, the first of each row
being the main paper's panel, each with its crop. Each run writes its record
(effective config, provenance, outputs) under plotsdir("records").

# reproduce: writes papers/images/experiment/bb/*_val.png
# reproduce: writes papers/images/experiment/apr/*_val.png
# reproduce: writes papers/images/experiment/aru/*_val.png
# reproduce: reads configs/figures/val_overlays.toml

Usage:
  julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/copy_val_overlays.jl [--config=<path>]
Env: BB_PAPER_IMAGES (output root, default papers/images). Data: the detection runs.
=#

using DrWatson
@quickactivate
using DrWatson: projectdir

using FileIO: load, save
using Images: imresize, RGB, N0f8

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: paper_images_dir, figure_config, write_figure_records, eval_sources, val_dir, load_config
using VisualGeometryToolkitEval.Experiments: Stage, run_stages

const CFG, CFG_PATH = figure_config("val_overlays"; default = projectdir("configs", "figures", "val_overlays.toml"))
const SRC = eval_sources(load_config(CFG["experiment"]["evaluation"]))
const WIDTH = CFG["output"]["width"]

function main()
    outputs, inputs = String[], String[]
    for sub in sort(collect(keys(CFG["systems"])))   # deterministic order
        sys = CFG["systems"][sub]
        out_dir = paper_images_dir(CFG["output"]["dir"], sub)
        mkpath(out_dir)
        for pick in sys["frames"]
            f = pick["frame"]
            r0, c0, nr, nc = pick["crop"]
            src = joinpath(val_dir(SRC, sys["series"]), f * "_val.png")
            isfile(src) || error("no overlay $src — run the detection stages first")
            img = RGB{N0f8}.(load(src))
            crop = img[(r0 + 1):min(r0 + nr, size(img, 1)), (c0 + 1):min(c0 + nc, size(img, 2))]
            h, w = size(crop)
            out = joinpath(out_dir, f * "_val.png")
            save(out, imresize(crop, (round(Int, h * WIDTH / w), WIDTH)))
            push!(outputs, out); push!(inputs, src)
            println("$(src) -> $out")
        end
    end
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/copy_val_overlays.jl", inputs, outputs)
    return nothing
end

run_stages("copy_val_overlays", [Stage(:render, "the overlay panels, cropped from the detection runs", () -> false, main)])
