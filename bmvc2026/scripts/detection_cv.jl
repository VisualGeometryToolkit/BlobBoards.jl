#=
detection_cv.jl — cross-validate the detection operating point JOINTLY:
(outlier_halfwidth, peak_threshold, min_similarity) of the pose arm, against
the inter-board invariance mAA.

The search is VisualGeometryToolkitEval's coarse-fine search over the product of the
three axes (`coarse_fine_search` under the `joint_coarse_fine` spec of
configs/detection_cv.toml): the coarse product grid, an axis extended while
the joint argmin lies within the margin of its face, then the product of each
axis's fine bracket about the joint argmin and its free argmin.
`peak_threshold` is a DETECTION parameter (`expensive` in the spec): the
search asks for one value at a time, and this driver runs one detection run
per value — features of every validation frame, kept for the run and reused
by every (half-width, similarity) cell at that value. The half-width and the
similarity gate are post-detection: everything they touch runs per cell inside
`find_boards`, exactly as deployed (the descriptor gate on the seed matches,
the RANSAC scorer, the local optimization, the pose-ambiguity refine and the
guided association, by ellipse overlap alone since 2026-11-11; the record's
run, before that, also used the half-width as its radius).

The criterion (VisualGeometryToolkitEval `rigid_invariance_errors`): in every frame
the relative pose of two detected boards is a constant of the rigid panel, so
its deviation from the leave-one-out reference over the other frames is
estimation error. Errors are the translation of that relative pose, mm, one
per (frame, board pair) of the C(n_boards, 2) possible; an undetected pair is
a failure (Inf). The spec's `mean_average_accuracy` criterion turns them into
the continuous mAA@10 mm.

Outputs, in the product `detection_cv_product(cfg)` (Eval's `Product`: `…/tuning/bmvc2026/detection_cv/<key>`)
(src/operating_points.jl): the tuning artifact `detection.jld2`
(VisualGeometryToolkitEval `save_coarse_fine_artifact`: the spec, the coarse and fine
grids of every axis, the objective over both product grids, every evaluated
cell with its per-frame sufficient statistics, the selection and its boundary
flags) with the frame bootstrap of the joint selection (`bootstrap_argmin`),
the run records, and (stage `tables`) the curve of each axis and the surface
of each pair of axes (`emit_tuning_table`). Consumers read the selection
through VisualGeometryToolkitEval `deployed_setting`.

The spec may tune a SUBSET of the three axes (`--config=halfwidth_cv.toml`:
the half-width alone, as the 2026-10-09 one-axis CV did); an axis it does not
tune keeps the run config's value, and without `peak_threshold` the search
runs one detection run, at the run config's threshold.

Usage:
  julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/detection_cv.jl [--config=<name under configs/>] [--plan] [--force]
=#
# reproduce: reads configs/detection_cv.toml
# reproduce: reads configs/halfwidth_cv.toml

using DrWatson
@quickactivate
using BlobBoards
using CUDA
import cuDNN
using CUDA: CUDABackend
import VisualGeometryToolkitCore
using Accessors: @set
using Random: Xoshiro
using Statistics: quantile
using TOML: TOML
import VisualGeometryToolkitEval
using VisualGeometryToolkitEval.Tuning: tuning_spec, JointCoarseFine, MeanAverageAccuracy,
    QueryUnit, grid_spec, bootstrap_argmin, unit_statistics
using VisualGeometryToolkitEval.Evaluation: coarse_fine_search, save_coarse_fine_artifact, deployed_setting
using VisualGeometryToolkitEval.Render: emit_tuning_table
using VisualGeometryToolkitEval.Experiments: Stage, run_stages, produce, product_dir, Produced
using VisualGeometryToolkitEval: cli_option

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: input_path, config_path, invariance_cv, detection_run, invariance_errors,
    detection_cv_product, detection_cv_artifact, detection_cv_table

const CFG_PATH = config_path(cli_option(ARGS, "config", "detection_cv.toml"))
const CFG = TOML.parsefile(CFG_PATH)
const SPEC = tuning_spec(CFG, "detection")
SPEC.scheme isa JointCoarseFine || error("[tuning.detection] must be a joint_coarse_fine spec")
SPEC.scheme.criterion isa MeanAverageAccuracy ||
    error("[tuning.detection] criterion must be mean_average_accuracy (the invariance mAA, mm)")
# Each axis's setter over the deserialized run config; `peak_threshold` is
# the one detection axis, and the spec must say so. The spec tunes a subset.
const SETTERS = Dict(
    "outlier_halfwidth" => (c, x) -> @set(c.geometry.outlier_halfwidth = x),
    "peak_threshold" => (c, x) -> @set(c.detect.peak_threshold = x),
    "min_similarity" => (c, x) -> @set(c.min_similarity = x))
issubset(keys(SPEC.scheme.axes), keys(SETTERS)) ||
    error("[tuning.detection] axes $(sort!(collect(keys(SPEC.scheme.axes)))) are not among $(sort!(collect(keys(SETTERS))))")
[n for (n, a) in SPEC.scheme.axes if a.expensive] == (haskey(SPEC.scheme.axes, "peak_threshold") ? ["peak_threshold"] : String[]) ||
    error("[tuning.detection]: peak_threshold, and only it, is the expensive (detection) axis")
const PRODUCT = detection_cv_product(CFG)
const OUTDIR = product_dir(PRODUCT)

function run_cv()
    ctx = invariance_cv(CFG, SPEC.scheme.cross_validation; device = CUDABackend())
    base = ctx.base
    apply(cfg, p) = foldl((c, n) -> SETTERS[String(n)](c, p[n]), keys(p); init = cfg)

    # One detection run per peak threshold, kept for the run: the search asks
    # for the cells of one value at a time, and a value revisited (a fine
    # bracket's endpoints are coarse values) reuses its run.
    detections = Pair{Float64,Dict{String,Any}}[]
    function detected(pk)
        i = findfirst(d -> isapprox(d.first, pk; rtol = 1e-9), detections)
        i === nothing || return detections[i].second
        t = @elapsed prepared = detection_run(ctx, @set base.detect.peak_threshold = pk)
        @info "detection run" peak_threshold = pk seconds = round(t; digits = 1) runs = length(detections) + 1
        push!(detections, pk => prepared)
        return prepared
    end
    pairs = Ref{Any}()   # the error matrices' columns (board-uid pairs)
    # (frame × board pair) translation errors of the inter-board relative pose,
    # mm; Inf = not observed or no reference — a failure under the criterion.
    function errors_at(prepared, p)
        r = invariance_errors(ctx, prepared, apply(base, p))
        pairs[] = r.pairs
        return r.translation
    end
    function errorsfn(points)
        ps = points[:pose]
        pk = only(unique(get(p, :peak_threshold, base.detect.peak_threshold) for p in ps))
        prepared = detected(pk)
        t = @elapsed out = [errors_at(prepared, p) for p in ps]
        @info "evaluated" peak_threshold = pk cells = length(ps) seconds = round(t; digits = 1)
        return Dict(:pose => out)
    end
    names = sort!(collect(keys(SPEC.scheme.axes)))
    cells = Dict(:pose => NamedTuple{Tuple(Symbol.(names))}(Tuple(grid_spec(SPEC.scheme.axes[n].coarse) for n in names)))
    rec = coarse_fine_search(errorsfn, cells, SPEC.scheme; label = _ -> "detection")[:pose]

    # Frame bootstrap of the joint selection over every evaluated cell
    # (VisualGeometryToolkitEval `bootstrap_argmin` over the cells' unit statistics:
    # per frame the mAA score sum and pair count, exact for the mAA).
    unc = CFG["uncertainty"]
    stats = [unit_statistics(SPEC.scheme.criterion, E) for E in last.(rec.evaluated)]
    boot = first.(rec.evaluated)[bootstrap_argmin(SPEC.scheme.criterion, stats, Xoshiro(unc["seed"]);
                                                  n_resamples = unc["n_resamples"])]
    a = (1 - unc["interval"]) / 2
    interval = Dict{String,Any}(n => [quantile([b[Symbol(n)] for b in boot], a), quantile([b[Symbol(n)] for b in boot], 1 - a)]
                                for n in names)
    mkpath(OUTDIR)
    save_coarse_fine_artifact(detection_cv_artifact(OUTDIR), SPEC, rec;
        unit = QueryUnit(), code = (BlobBoards, VisualGeometryToolkitCore), frames = ctx.stems,
        pairs = [collect(p) for p in pairs[]], detection_runs = first.(detections),
        bootstrap_cells = [Float64(b[Symbol(n)]) for b in boot, n in names], bootstrap_interval = interval,
        objective = "mean_average_accuracy of the inter-board invariance translation errors [mm] over (frame, board pair); undetected = failure",
        arm = "pose")
    # the read-side gate, before anything is recorded complete
    @info "selected" deployed_setting(detection_cv_artifact(OUTDIR); context = "joint detection CV") interval
    return Produced(nothing; inputs = [ctx.image_dir, input_path(CFG, "camera"), input_path(CFG, "boards"), ctx.run_config])
end

# The curve of each axis and the surface of each pair of axes through the
# coarse argmin (VisualGeometryToolkitEval `emit_tuning_table`): supplementary tables.
const AXES = sort!(collect(keys(SPEC.scheme.axes)))
const TABLES = vcat([[n] for n in AXES], [[AXES[i], AXES[j]] for i in eachindex(AXES) for j in i+1:length(AXES)])
write_tables() = foreach(TABLES) do ax
    emit_tuning_table(detection_cv_artifact(OUTDIR), detection_cv_table(OUTDIR, ax); axes = ax,
                      labels = [CFG["tables"]["labels"][n] for n in ax], fmt = "%.4g")
end

run_stages("detection_cv", [
    Stage(:cv, "joint (half-width, peak threshold, similarity) CV, pose arm (GPU) -> $(OUTDIR)";
        items = () -> [PRODUCT], product = identity,
        run = p -> produce(_ -> run_cv(), p; driver = "bmvc2026/scripts/detection_cv.jl", config_source = CFG_PATH)),
    Stage(:tables, "the curve of each axis and the surface of each pair -> $(OUTDIR)/table_*.tex";
        check = () -> (all(ax -> isfile(detection_cv_table(OUTDIR, ax)), TABLES) || write_tables(); nothing)),
])
