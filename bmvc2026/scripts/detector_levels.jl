#=
detector_levels.jl — cross-validate the detector's scale ladder: the ASIFT
pyramid ([find_boards.detect] upsample, nb_scales, nb_octaves) of the run
config of record, against the inter-board invariance mAA on the TUNING
session (configs/detector_levels.toml: the half-width CV's split, criterion
and panel).

Each setting of the config's [[settings]] (the listed axis `setting` of its
`factorial` spec indexes them) is one detection run over the validation frames
(the features of every frame under that ladder), then `find_boards` at the run
config's every other value; its errors are VisualGeometryToolkitEval
`rigid_invariance_errors` (src/invariance_cv.jl, the machinery
scripts/detection_cv.jl runs on). The selection is the setting of least
objective (the spec's mean_average_accuracy, negated; the first on a tie), and
the frame bootstrap (`bootstrap_argmin`) gives the share of resamples each
setting wins.

Outputs, in the product `detector_levels_product(cfg)` (Eval's `Product`:
`…/tuning/bmvc2026/detector_levels/<key>`): the tuning artifact
`detector_levels.jld2` (VisualGeometryToolkitEval `save_tuning_artifact`: the spec,
the settings, each one's objective, unit statistics, boards found per frame,
seconds per frame and peak GPU memory, the selection and the bootstrap shares)
and (stage `table`) `table_detector_levels.tex`: per setting the searched σ
range, the mAA, the bootstrap share, the boards found, the boards found in
each frame where the settings disagree, the median seconds per frame and the
peak GPU memory.

Usage:
  julia --startup-file=no --project=bmvc2026 -t auto bmvc2026/scripts/detector_levels.jl [--config=<name under configs/>] [--plan] [--force]
=#
# reproduce: reads configs/detector_levels.toml

using DrWatson
@quickactivate
using BlobBoards
using CUDA
import cuDNN
using CUDA: CUDABackend
import VisualGeometryToolkitCore
import VisualGeometryToolkitFeatures   # its code identity is recorded
using FileIO: load as load_image
using JLD2: load as load_jld2
using Printf: @sprintf
using Random: Xoshiro
using Statistics: median
using TOML: TOML
using VisualGeometryToolkitEval.Tuning: tuning_spec, Factorial, MeanAverageAccuracy, ListedGrid, QueryUnit,
    grid_spec, bootstrap_argmin, unit_statistics, stats_objective, save_tuning_artifact
using VisualGeometryToolkitEval.Experiments: Stage, run_stages, produce, product_dir, product_output, Produced
using VisualGeometryToolkitEval: cli_option

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: config_path, invariance_cv, detection_run, invariance_errors, run_inputs,
    with_ladder, ladder_sigma_range, detector_levels_product

const CFG_PATH = config_path(cli_option(ARGS, "config", "detector_levels.toml"))
const CFG = TOML.parsefile(CFG_PATH)
const SPEC = tuning_spec(CFG, "detector")
SPEC.scheme isa Factorial && collect(keys(SPEC.scheme.axes)) == ["setting"] ||
    error("[tuning.detector] must be a factorial spec over the one axis `setting`")
SPEC.scheme.criterion isa MeanAverageAccuracy ||
    error("[tuning.detector] criterion must be mean_average_accuracy (the invariance mAA, mm)")
const SETTINGS = CFG["settings"]
const INDICES = Int.(grid_spec(SPEC.scheme.axes["setting"]).values)
INDICES == collect(eachindex(SETTINGS)) || error("[tuning.detector.axes.setting] must list 1 … $(length(SETTINGS)), the [[settings]]")
const PRODUCT = detector_levels_product(CFG)
const OUTDIR = product_dir(PRODUCT)
const TABLE = joinpath(OUTDIR, "table_detector_levels.tex")

# The device's peak memory in use while `f` runs (s, bytes): sampled every 0.5 s
# on another thread (the descriptor network's arena included).
# Each measurement starts from the previous ladder's memory returned to the
# driver (its cached ASIFT plan evicted, the pool reclaimed); what stays is the
# descriptor network's arena, common to every setting.
function peak_memory(f, dev)
    Threads.nthreads() > 1 || @warn "one thread: the memory sampler runs only when detection yields (run with -t auto)"
    VisualGeometryToolkitFeatures.clear_asift_plans!(); GC.gc(); VisualGeometryToolkitCore.reclaim_device!(dev)
    lo = Ref(VisualGeometryToolkitCore.device_free_bytes(dev)); done = Threads.Atomic{Bool}(false)
    sampler = Threads.@spawn while !done[]
        lo[] = min(lo[], VisualGeometryToolkitCore.device_free_bytes(dev)); sleep(0.5)
    end
    out = try f() finally done[] = true; wait(sampler) end
    return out, CUDA.total_memory() - lo[]
end

function run_cv()
    ctx = invariance_cv(CFG, SPEC.scheme.cross_validation; device = CUDABackend())
    crit = SPEC.scheme.criterion
    stats, boards, seconds, memory = Matrix{Float64}[], Vector{Int}[], Vector{Float64}[], Int[]
    pairs = nothing
    for s in SETTINGS
        cfg = with_ladder(ctx.base, s)
        t = Float64[]
        features, peak = peak_memory(() -> detection_run(ctx, cfg; on_frame = (_, dt) -> push!(t, dt)), ctx.device)
        r = invariance_errors(ctx, features, cfg)
        pairs = r.pairs
        push!(stats, unit_statistics(crit, r.translation)); push!(boards, [length(r.poses[st]) for st in ctx.stems])
        push!(seconds, t); push!(memory, peak)
        @info "setting" s["name"] mAA = -stats_objective(crit, last(stats)) boards = sum(last(boards)) median_s = round(median(t); digits = 2) peak_GiB = round(peak / 2^30; digits = 1)
    end
    objective = [stats_objective(crit, S) for S in stats]
    selected = argmin(objective)
    unc = CFG["uncertainty"]
    wins = bootstrap_argmin(crit, stats, Xoshiro(unc["seed"]); n_resamples = unc["n_resamples"])
    share = [count(==(i), wins) / length(wins) for i in eachindex(SETTINGS)]
    H, W = size(load_image(joinpath(ctx.image_dir, first(ctx.frames))))   # Julia images are rows × columns
    mkpath(OUTDIR)
    save_tuning_artifact(product_output(PRODUCT), SPEC; grids = Dict(:setting => ListedGrid(INDICES)),
        unit = QueryUnit(), code = (BlobBoards, VisualGeometryToolkitCore, VisualGeometryToolkitFeatures),
        settings = SETTINGS, frames = ctx.stems, pairs = [collect(p) for p in pairs], image_size = [W, H],
        objective = objective, unit_stats = stats, boards = reduce(hcat, boards), seconds = seconds,
        gpu_memory_peak = memory, selected = selected, selected_name = SETTINGS[selected]["name"],
        bootstrap_share = share,
        objective_label = "mean_average_accuracy of the inter-board invariance translation errors [mm] over (frame, board pair), negated; undetected = failure")
    @info "selected" SETTINGS[selected]["name"] share
    return Produced(nothing; inputs = run_inputs(CFG, ctx))
end

# One row per setting; the frames where the settings find different numbers of boards, one column each.
function write_table()
    a = load_jld2(product_output(PRODUCT))
    B, S = a["boards"], a["settings"]
    hard = [k for k in axes(B, 1) if !allequal(B[k, :])]
    W, H = a["image_size"]
    rows = map(eachindex(S)) do i
        lo, hi = ladder_sigma_range(S[i], W, H)
        star = i == a["selected"] ? "\$^\\star\$" : ""
        join([S[i]["name"] * star, string(S[i]["upsample"]), string(S[i]["nb_scales"]),
              S[i]["nb_octaves"] == 0 ? "--" : string(S[i]["nb_octaves"]), @sprintf("%.2f--%.0f", lo, hi),
              @sprintf("%.4f", -a["objective"][i]), @sprintf("%.2f", a["bootstrap_share"][i]), string(sum(B[:, i])),
              string.(B[hard, i])..., @sprintf("%.2f", median(a["seconds"][i])), @sprintf("%.0f", a["gpu_memory_peak"][i] / 2^30)],
             " & ") * " \\\\"
    end
    cols = ["setting", "up", "\$S\$", "oct", "\$\\sigma\$ [px]", "mAA", "boot", "boards",
            [replace(a["frames"][k], "_" => "\\_") for k in hard]..., "s/frame", "GiB"]
    head = join(cols, " & ") * " \\\\"
    mkpath(dirname(TABLE))
    write(TABLE, """
        % Auto-generated by bmvc2026/scripts/detector_levels.jl from $(basename(product_output(PRODUCT))) (product $(PRODUCT.key)).
        \\begin{tabular}{@{}l$("r"^(length(cols) - 1))@{}}
        \\toprule
        $head
        \\midrule
        $(join(rows, "\n"))
        \\bottomrule
        \\end{tabular}
        """)
    return nothing
end

run_stages("detector_levels", [
    Stage(:cv, "the detector-ladder CV on the tuning session (GPU) -> $(OUTDIR)";
        items = () -> [PRODUCT], product = identity,
        run = p -> produce(_ -> run_cv(), p; driver = "bmvc2026/scripts/detector_levels.jl", config_source = CFG_PATH)),
    Stage(:table, "the ladders' mAA, boards, cost -> $(TABLE)";
        check = () -> (write_table(); nothing)),   # rewritten from the artifact every time: never stale after a --force
])
