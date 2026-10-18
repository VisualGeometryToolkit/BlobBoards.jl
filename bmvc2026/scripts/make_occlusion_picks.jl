#=
make_occlusion_picks.jl — the bottom row of fig:occexamples: one frame per
marker system at the 70% occlusion level, cropped to the CENTRAL HALF of the
frame (the marker panel; 4240×2832 → 2120×1416, centred) and downscaled 2×
to 1060×708, so the three panels stay under a megabyte each in the PDF. (Crop
and scale recovered by matching the submitted panels against the frames.)

The frames are the occlusion sweep's synthetic occlusions
(`<dataset>/occlusion/70/images/<sys>/`, made by make_occlusion.jl); the
selection is the figure's and is recorded here — one capture per system with
the panel facing the camera.

# reproduce: writes papers/images/occ_setup/70/compressed_bb_DSC04549.png
# reproduce: writes papers/images/occ_setup/70/compressed_apr_DSC04451.png
# reproduce: writes papers/images/occ_setup/70/compressed_uco_DSC04362.png
# reproduce: reads configs/figures/occlusion_picks.toml

Usage:
  julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/make_occlusion_picks.jl [--config=<path>]
Env: BB_PAPER_IMAGES (output root, default papers/images). Data: the occlusion
tier of the level (bb_absolute_pose_occlusion_70_260527).
=#

using DrWatson
@quickactivate
using DrWatson: projectdir

using FileIO: load, save
using Images: imresize, RGB, N0f8

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: data_root, paper_images_dir, figure_config, write_figure_records
using VisualGeometryToolkitEval.Experiments: Stage, run_stages

const CFG, CFG_PATH = figure_config("occlusion_picks"; default = projectdir("configs", "figures", "occlusion_picks.toml"))
const EXP = CFG["experiment"]
const ROOT = joinpath(data_root(; tiers = String.(EXP["tiers"])),
    "absolute_pose_experiments", EXP["dataset"], "occlusion", EXP["level"], "images")
const KEEP = CFG["crop"]["keep_fraction_inverse"]
const DOWN = CFG["crop"]["downscale"]

function main()
    out_dir = paper_images_dir(CFG["output"]["dir"])
    mkpath(out_dir)
    outputs, inputs = String[], String[]
    for tag in sort(collect(keys(CFG["picks"])))
        subset, frame = CFG["picks"][tag]["subset"], CFG["picks"][tag]["frame"]
        src = only(filter(f -> startswith(f, frame * "."), readdir(joinpath(ROOT, subset))))
        img = RGB{N0f8}.(load(joinpath(ROOT, subset, src)))
        h, w = size(img)
        ch, cw = h ÷ KEEP, w ÷ KEEP
        y0, x0 = (h - ch) ÷ 2, (w - cw) ÷ 2
        crop = img[(y0 + 1):(y0 + ch), (x0 + 1):(x0 + cw)]
        out = joinpath(out_dir, replace(CFG["output"]["name"], "{tag}" => tag, "{frame}" => frame))
        save(out, imresize(crop, (ch ÷ DOWN, cw ÷ DOWN)))
        push!(outputs, out); push!(inputs, joinpath(ROOT, subset, src))
        println("$subset/$src -> $out")
    end
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_occlusion_picks.jl", inputs, outputs)
    return nothing
end

run_stages("make_occlusion_picks", [Stage(:render, "the occlusion-level panels", () -> false, main)])
