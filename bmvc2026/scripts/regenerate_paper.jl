#=
regenerate_paper.jl — THE entry point that regenerates every result and every
generated paper asset FROM THE INPUTS ALONE (REPRODUCE.md indexes each asset:
label → asset → generator → data), so no number, table, figure or macro in the
paper can silently disagree with the runs it claims to describe. `make
figures` in papers/ runs it; `make figures-gpu` adds the GPU figures.

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/regenerate_paper.jl [--gpu]

Steps, in dependency order. Every RESULT is produced into a keyed run
directory under results_root() and found again through its config of record
(src/runs.jl); a stage whose run is complete is skipped, so a rerun redoes
only what changed. A failing step is FATAL.

  results (the inputs → keyed runs; GPU for the BlobBoards sweeps):
   1. convert_formats.jl            the motion-capture records (convert_formats-<key>)
   2. go_bmvc2026.jl                BlobBoards on bb + calibration_bb (bb-<key>)
   3. go_bmvc2026.jl --run-config=  the no-disambiguation ablation (bb-<key>)
   4. go_bmvc2026_occlusion.jl      BlobBoards on every occlusion level (occlusion-<key>)
   5. detect_baselines.jl           AprilTag + ArUco on every subset (baselines-<key>)
   6. make_hand_eye.jl              hand-eye calibrations (hand_eye-<key>)
   7. make_pose_results.jl          per-(system, level) pose-error cells (pose_results-<key>)
  paper assets:
   8. plots/error_distributions.jl  floor plans, ECDFs, histograms, occlusion sweep and
                                    the macro files bb_stats / bb_occ_stats / timing_stats
   9. flip_table.jl                 flip table + flip_stats
  10. make_detection_stats.jl       det_stats.tex
  11. make_hand_eye_table.jl        handeye_table.tex + handeye_stats.tex
  12. make_supp_tables.jl           board_params.tex + run_settings.tex
  13. copy plots/*.tex (and plots/occ_<L>/*.tex) into papers/figures/
  14. copy_val_overlays.jl          experiment/{bb,apr,aru}/DSC*_val.png (the runs' overlays)
  15. make_occlusion_picks.jl       occ_setup/70/compressed_*.png (occlusion-70 tier)
  16. make_param_variations.jl      blob_variations_squared/*.png (fig:params)
 --gpu (the descriptor network and the board gallery):
  17. make_bmvc2026_intro.jl        intro_overlay.png
  18. make_adapted_panels.jl        adapted/{original.jpg, adapted_*.png}
  19. make_pipeline_panels.jl       bb_pipe_{detect,adapt,logpolar,register}.pdf

Outputs go to papers/figures and papers/images unless BB_PAPER_FIGURES /
BB_PAPER_IMAGES point elsewhere — how a regeneration is reviewed against the
submitted assets before it replaces them (REPRODUCE.md). CAU_RESULTS_ROOT
places the results; CAU_DATA_ROOT, or the artifact view, the inputs.

The run is a stage runner (`--plan`, `--stage=<name>[,…]`, `--force`; each
stage runs one generator in its own module). Every generator reads its config
of record (configs/*.toml, configs/figures/<name>.toml); this script's own is
configs/figures/regenerate_paper.toml. `--config=<path>` arguments are
forwarded, each taken by the generator its `[experiment] generator` names.
Every figure run writes its record under plotsdir("records").

# reproduce: reads configs/figures/regenerate_paper.toml
# reproduce: writes papers/figures/floorplan/*.tex
# reproduce: writes papers/figures/floorplan_occ_*/*.tex
=#

using DrWatson
@quickactivate
import TOML

const SCRIPTS = @__DIR__
const PLOTS = projectdir("plots")
if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: paper_figures_dir, figure_config, write_figure_records
using VisualGeometryEval.Experiments: Stage, run_stages

const CFG, CFG_PATH = figure_config("regenerate_paper"; default = projectdir("configs", "figures", "regenerate_paper.toml"))

# Run a standalone script in its OWN module namespace: the paper drivers each
# define top-level `const DATASET`, `const BBE`, … meant for a fresh process, so
# including several into one shared Main collides. A throwaway module with its
# own `include`/`eval` gives each the clean top-level it expects while the loaded
# packages (BlobBoards, CairoMakie, …) stay warm — no re-precompile, no clash.
function run_isolated(path::AbstractString; env::Dict{String,String} = Dict{String,String}(), args::Vector{String} = String[])
    for (k, v) in env
        ENV[k] = v
    end
    m = Module(gensym(basename(path)))
    Core.eval(m, :(include(p) = Base.include(@__MODULE__, p)))
    Core.eval(m, :(eval(x) = Core.eval(@__MODULE__, x)))
    # Some drivers refer to Main.BlobBoardsEval unqualified; make it resolve in
    # the throwaway module too. And turn a CLI exit() into a catchable throw so
    # a script's exit(0) does not tear the process down mid-pipeline.
    isdefined(Main, :BlobBoardsEval) && Core.eval(m, :(const BlobBoardsEval = Main.BlobBoardsEval))
    Core.eval(m, :(exit(code::Integer = 0) = error("script called exit(", code, ")")))
    # Each script runs its own stages; it sees only the --config= arguments
    # (this run's --stage= / --gpu / --force are regenerate_paper's own) and
    # the stage's own `args`.
    saved_args = copy(ARGS)
    try
        @info "regenerate_paper: running $(basename(path))" env args
        empty!(ARGS); append!(ARGS, filter(a -> startswith(a, "--config="), saved_args)); append!(ARGS, args)
        Base.include(m, path)
    finally
        empty!(ARGS); append!(ARGS, saved_args)
        for k in keys(env)
            delete!(ENV, k)
        end
    end
    return m
end

_s(p...) = joinpath(SCRIPTS, p...)
const GPU = "--gpu" in ARGS

# --- copy generated .tex into the paper's figures tree ------------------------
# A GLOB, not a hand list: an explicit list is how a generated table once aged
# behind its neighbours (flip_table, 2026-08-17). The drivers' compilable
# previews (`*_standalone.tex`) are never read by the paper: they stay in plots/.
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
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/regenerate_paper.jl", outputs = copied)
    return copied
end

_stage(name, desc, script...; env = Dict{String, String}(), args = String[]) =
    Stage(name, desc, () -> false, () -> (run_isolated(_s(script...); env, args); nothing))

# In dependency order. Every stage always runs (`done` is false): a result
# stage skips a run that is already complete (its keyed directory holds its
# records), and the asset stages are cheap — a stale output is the failure
# this exists to stop. `--stage=<name>` runs a subset; a failing stage aborts.
# The ablation's run config is the evaluation's ([runs] bb_noflip): the sweep
# writes the run the flip table then finds through the same config.
const NOFLIP = "--run-config=" * projectdir("configs", TOML.parsefile(projectdir("configs", "figures", "evaluation.toml"))["runs"]["bb_noflip"])
stages = [
    _stage(:records, "the motion-capture records (stage 0)", "convert_formats.jl"),
    _stage(:bb, "BlobBoards detections: bb + calibration_bb (GPU)", "go_bmvc2026.jl"),
    _stage(:bb_noflip, "BlobBoards without disambiguation: the flip table's base (GPU)", "go_bmvc2026.jl"; args = [NOFLIP]),
    _stage(:occlusion, "BlobBoards on every occlusion level (GPU)", "go_bmvc2026_occlusion.jl"),
    _stage(:baselines, "AprilTag + ArUco on every subset", "detect_baselines.jl"),
    _stage(:hand_eye, "hand-eye calibrations from the calibration subsets", "make_hand_eye.jl"),
    _stage(:pose_results, "per-(system, level) pose-error cells", "make_pose_results.jl";
        env = CFG["pose_results"]["force"] ? Dict("BB_FORCE" => "1") : Dict{String, String}()),
    _stage(:error_distributions, "floor plans, ECDFs, histograms, occlusion sweep, macro files", "plots", "error_distributions.jl"),
    _stage(:flip_table, "flip table + flip_stats.tex", "flip_table.jl"),
    _stage(:detection_stats, "det_stats.tex", "make_detection_stats.jl"),
    _stage(:hand_eye_table, "handeye_table.tex + handeye_stats.tex", "make_hand_eye_table.jl"),
    _stage(:supp_tables, "board_params.tex + run_settings.tex", "make_supp_tables.jl"),
    Stage(:copy_tex, "copy plots/*.tex into papers/figures", () -> false, () -> (copy_tex(); nothing)),
    _stage(:val_overlays, "the detection-overlay panels", "copy_val_overlays.jl"),
    _stage(:occlusion_picks, "the occlusion-example panels", "make_occlusion_picks.jl"),
    _stage(:param_variations, "the pattern-parameter panels", "make_param_variations.jl"),
]
gpu_stages = [
    _stage(:intro, "intro_overlay.png (GPU)", "make_bmvc2026_intro.jl"),
    _stage(:adapted_panels, "adapted/{original.jpg, adapted_*.png} (GPU)", "make_adapted_panels.jl"),
    _stage(:pipeline_panels, "bb_pipe_{detect,adapt,logpolar,register}.pdf (GPU)", "make_pipeline_panels.jl"),
]

run_stages("regenerate_paper", GPU ? vcat(stages, gpu_stages) : stages)

if !("--plan" in ARGS)
    println("\nregenerate_paper: DONE. Every result and generated asset regenerated from the inputs",
        GPU ? "." : " (GPU figures skipped; add --gpu).")
    println("Now: cd bmvc2026/papers && make")
end
