#=
import VisualGeometryToolkitEval as VgtkEval
detection_cv_replay.jl — the record of configs/detection_cv.toml produced from
the saved evaluations of earlier runs, without running detection.

The search of detection_cv.jl is deterministic given the objective of each
cell it asks for, so it is replayed exactly from two saved artifacts
(results_root()-relative, fixed here: they are inputs, not protocol):

  SEARCH     tuning/detection_cv-c80d42e9055f8cd5/detection.jld2 — every cell
             of the joint search under this config's spec less its
             incumbent: each cell's per-frame mAA statistics (score sum, pair
             count), from which the criterion and the frame bootstrap are exact
  INCUMBENT  tuning/outlier_halfwidth_cv-3a346460b90dc25d/outlier_halfwidth_pose.jld2
             — the (frame × pair) errors at the incumbent half-width under the
             run config of record (whose peak threshold and similarity are the
             incumbent's), on the same frames and board pairs

A cell's errors are rebuilt from its per-frame statistics (per frame ⌊s⌋ exact
hits, one error carrying the fraction, misses after: the same mAA score sum
and pair count); the incumbent's are its saved errors. A cell the search asks
for that neither artifact holds is an error — never interpolated — and the
replayed coarse and fine objectives must equal the source run's. The record
lands where detection_cv.jl would write it (same config, same run key), its
records (Eval's write_run_records, with its code trees) naming both sources by
path and sha256.

Usage:
  julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/detection_cv_replay.jl
=#
# reproduce: reads configs/detection_cv.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)
using TOML: TOML
using Random: Xoshiro
using Statistics: quantile
import VisualGeometryToolkitEval
using VisualGeometryToolkitEval.Tuning: tuning_spec, tuning_table, load_tuning_artifact, grid_spec, bootstrap_argmin,
    unit_statistics, QueryUnit
using VisualGeometryToolkitEval.Evaluation: coarse_fine_search, save_coarse_fine_artifact, deployed_setting
using VisualGeometryToolkitEval.Experiments: Stage, run_stages, run_complete, results_root, write_run_records
using VisualGeometryToolkitEval.Render: emit_tuning_table
import VisualGeometryToolkitCore

module ReplayOperatingPoints
    include(joinpath(@__DIR__, "..", "src", "operating_points.jl"))
end
const OP = ReplayOperatingPoints

const CFG_PATH = OP.config_path("detection_cv.toml")
const CFG = TOML.parsefile(CFG_PATH)
const SPEC = tuning_spec(CFG, "detection")
const SEARCH = joinpath("tuning", "detection_cv-c80d42e9055f8cd5", "detection.jld2")
const INCUMBENT = joinpath("tuning", "outlier_halfwidth_cv-3a346460b90dc25d", "outlier_halfwidth_pose.jld2")
const OUTDIR = OP.detection_cv_dir(CFG)

# A units × items error matrix with the per-unit mAA@T statistics `S`.
function errors_of(S, T)
    E = fill(Inf, size(S, 1), Int(S[1, 2]))
    for (u, s) in enumerate(view(S, :, 1))
        k = floor(Int, s); E[u, 1:k] .= 0.0
        k < size(E, 2) && (E[u, k + 1] = T * (1 - (s - k)))
    end
    return E
end


function replay()
    src = load_tuning_artifact(joinpath(results_root(), SEARCH))
    inc = load_tuning_artifact(joinpath(results_root(), INCUMBENT))
    t = tuning_table(SPEC); delete!(t, "incumbent")
    t == tuning_table(src.spec) || error("$SEARCH ran another search than [tuning.detection] (less its incumbent)")
    src["frames"] == inc["frames"] && src["pairs"] == inc["pairs"] || error("the two sources differ in frames or board pairs")
    v = SPEC.scheme.incumbent.values
    rc = OP.load_config(CFG["runs"]["bb"])["find_boards"]
    (rc["detect"]["peak_threshold"], rc["min_similarity"]) == (v["peak_threshold"], v["min_similarity"]) ||
        error("the incumbent's peak threshold and similarity are not the run config of record's, under which $INCUMBENT ran")
    j = findfirst(==(v["outlier_halfwidth"]), inc["evaluated_x"])
    j === nothing && error("$INCUMBENT holds no errors at the incumbent half-width $(v["outlier_halfwidth"])")
    T = SPEC.scheme.criterion.max_error; C = src["cells"]; ST = src["unit_statistics"]
    names = Tuple(Symbol.(src["axes"]))
    function lookup(p)
        all(n -> p[n] == v[String(n)], names) && return inc["evaluated_errors"][j]
        i = findfirst(r -> all(isapprox(C[r, a], p[n]; rtol = 1e-9) for (a, n) in enumerate(names)), axes(C, 1))
        i === nothing && error("replay: the source run never evaluated $p — no honest replay")
        return errors_of(ST[i], T)
    end
    cells = Dict(:pose => NamedTuple{names}(Tuple(grid_spec(SPEC.scheme.axes[String(n)].coarse) for n in names)))
    rec = coarse_fine_search(ps -> Dict(k => [lookup(p) for p in q] for (k, q) in ps), cells, SPEC.scheme;
                             label = _ -> "detection (replay)")[:pose]
    isapprox(rec.coarse_objective, src["coarse_objective"]; rtol = 1e-12) &&
        isapprox(rec.fine_objective, src["fine_objective"]; rtol = 1e-12) ||
        error("the replayed objective is not the source run's")

    # the frame bootstrap of detection_cv.jl, over the same evaluated cells
    unc = CFG["uncertainty"]
    stats = [unit_statistics(SPEC.scheme.criterion, E) for E in last.(rec.evaluated)]
    boot = first.(rec.evaluated)[bootstrap_argmin(SPEC.scheme.criterion, stats, Xoshiro(unc["seed"]);
                                                  n_resamples = unc["n_resamples"])]
    a = (1 - unc["interval"]) / 2
    ns = String.(collect(names))
    interval = Dict{String,Any}(n => quantile([b[Symbol(n)] for b in boot], [a, 1 - a]) for n in ns)
    sha(p) = VisualGeometryToolkitEval.file_sha256(joinpath(results_root(), p))
    mkpath(OUTDIR)
    save_coarse_fine_artifact(OP.detection_cv_artifact(OUTDIR), SPEC, rec;
        unit = QueryUnit(), code = (VisualGeometryToolkitCore,), frames = src["frames"], pairs = src["pairs"],
        detection_runs = src["detection_runs"], bootstrap_cells = [Float64(b[Symbol(n)]) for b in boot, n in ns],
        bootstrap_interval = interval, objective = src["objective"], arm = "pose",
        replayed_from = Dict{String,Any}(SEARCH => sha(SEARCH), INCUMBENT => sha(INCUMBENT)))
    sel = deployed_setting(OP.detection_cv_artifact(OUTDIR); context = "joint detection CV (replay)")
    # The run's records through Eval (its code trees name the repositories and their trees); the replay's own
    # payload: its sources by results_root-relative path and sha256, and the method.
    write_run_records(OUTDIR, CFG; driver = "bmvc2026/scripts/detection_cv_replay.jl", config_source = CFG_PATH,
        key = OP.detection_cv_key(CFG),
        fields = Dict{String,Any}(
            "data" => Dict{String,Any}("results_root_relative" => Dict{String,Any}(SEARCH => sha(SEARCH), INCUMBENT => sha(INCUMBENT))),
            "method" => "replay of the joint search from the saved evaluations (no detection run)"))
    @info "replayed" selected = sel kept_incumbent = rec.kept_incumbent gain = rec.gain interval
    return OUTDIR
end

const AXES = sort!(collect(keys(SPEC.scheme.axes)))
const TABLES = vcat([[n] for n in AXES], [[AXES[i], AXES[j]] for i in eachindex(AXES) for j in i+1:length(AXES)])
table_path(ax) = joinpath(OUTDIR, "table_" * join(ax, "__") * ".tex")
write_tables() = foreach(TABLES) do ax
    emit_tuning_table(OP.detection_cv_artifact(OUTDIR), table_path(ax); axes = ax,
                      labels = [CFG["tables"]["labels"][n] for n in ax], fmt = "%.4g")
end

run_stages("detection_cv_replay", [
    Stage(:replay, "the joint detection CV replayed from $SEARCH and $INCUMBENT -> $(OUTDIR)",
        () -> run_complete(OUTDIR; sidecar = basename(OP.detection_cv_artifact(OUTDIR))),
        () -> (replay(); nothing)),
    Stage(:tables, "the curve of each axis and the surface of each pair -> $(OUTDIR)/table_*.tex",
        () -> all(isfile ∘ table_path, TABLES),
        () -> (write_tables(); nothing)),
])
