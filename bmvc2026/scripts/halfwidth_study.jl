#=
halfwidth_study.jl — why is the cross-validated outlier half-width small?
A STUDY (configs/halfwidth_study.toml): it selects nothing, and no run
config, record or release reads it.

The half-width CV of record (scripts/detection_cv.jl --config=halfwidth_cv.toml)
selects a ≈ 1.7 px for the uniform outlier component of the pose arm's
Student-t + uniform scorer, where the uniform was expected to span the image.
Four readings, and what tests each:

  H1 selection: the scored candidates passed detection, descriptor matching
     and the guided association, so a wrong candidate lies near its
     prediction; the true outlier support is the gate, not the image
     (stage `residuals`: the labelled residuals of each candidate population
     against the gate and the lattice pitch; the truncated mixture EM's
     profile over the half-width).
  H2 a is confounded with the inlier prior: the inlier odds are
     π·t(r)·(2a)^d/(1 − π), so with π not pinned a is a decision threshold
     (stage `arm`: the CV curve under a profiled and a Beta prior on π; stage
     `residuals`: the MAP decision boundary at each half-width).
  H3 the Student-t tails absorb moderate errors (the fitted ν, and where the
     labelled outliers fall against the boundary).
  H4 the criterion (pose mAA) is not the likelihood (the CV optimum against
     the EM's).

Stages (VisualGeometryToolkitEval's Experiments: one keyed Product each):

  residuals  (GPU)  one detection run and find_boards at the run config of
             record; per board the candidate populations `final` (what the
             board was fitted on, weighted), `guided` (guided_search of the
             refined pose, no keep set, no claims) and `descriptor` (the
             board's mutual-NN matches over the frame), each with the pose
             problem's own residual (mahalanobis_distances!) at the refined
             pose and its label by the board's geometry: an inlier iff the
             detection's nearest projected blob centre is its own blob's. Per
             board the gate (the centre offset at which two equal blob regions
             reach the IoU gate: Core's iou, bisected) and the pitch (the
             median nearest-neighbour distance of the projected blob
             centres). The truncated mixture EM (Core's PosteriorWeightState
             blocks, π estimated; a residual beyond the support disc of the
             uniform's area is an inlier) profiled over a, pooled and per
             frame; the MAP decision boundary (Core's decision_boundary) at
             the half-widths of [thresholds].
  arm        (GPU, --arm=<name>)  the CV curve over the extended grid
             ([grid]) under one scorer-prior arm ([[arms]]), its own
             detection run: one cluster job per arm.
  report     the figures and summary of bmvc2026/notes/halfwidth-study.md.

The scorer is not modified. An arm's label prior is one of Core's
(`FlatLabelPrior`, the configuration of record; `ProfiledLabelPrior`;
`BetaLabelPrior`), given to the scorer the pipeline builds through this
driver's method of Estimation's `default_student_t_uniform_scoring` (the
configuration of record when no arm is active: the same table read the same
way). An arm's LO inlier prior is Estimation's posterior-weight iteration
(`posterior_irls!`, LODR's Stu-tU-IRLS) under another update policy of π:
Core's `Estimated()`, or the Beta MAP block below.

Usage:
  julia --startup-file=no --project=bmvc2026 -t auto bmvc2026/scripts/halfwidth_study.jl --stage=residuals
  julia … halfwidth_study.jl --stage=arm --arm=deployed      (one job per arm)
  julia … halfwidth_study.jl --stage=report
=#
# reproduce: reads configs/halfwidth_study.toml
# reproduce: reads configs/halfwidth_cv.toml

using DrWatson
@quickactivate
using BlobBoards
using CUDA
import cuDNN
using CUDA: CUDABackend
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using Accessors: @set
using CairoMakie
using JLD2: load as load_jld2
using FileIO: load as load_image
using Printf: @sprintf
using Random: Xoshiro
using Statistics: median, quantile, mean
using TOML: TOML
using VisualGeometryToolkitEval.Tuning: tuning_spec, MeanAverageAccuracy, unit_statistics, stats_objective, bootstrap_argmin
import VisualGeometryToolkitEval.Tuning as Tuning
using VisualGeometryToolkitEval.Experiments: Stage, run_stages, produce, product_dir, product_output, produced, Product,
    run_key, Produced
using VisualGeometryToolkitEval: cli_option

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: config_path, load_config, invariance_cv, detection_run, invariance_errors, run_inputs, EXPERIMENT

const CFG_PATH = config_path(cli_option(ARGS, "config", "halfwidth_study.toml"))
const CFG = TOML.parsefile(CFG_PATH)
const CV = load_config(CFG["study"]["cv"])
const SPEC = tuning_spec(CV, "detection")
const CRIT = SPEC.scheme.criterion
CRIT isa MeanAverageAccuracy || error("the CV of record's criterion must be mean_average_accuracy")
const ARMS = Dict(a["name"] => a for a in CFG["arms"])
const ARM_NAMES = [a["name"] for a in CFG["arms"]]
const EXP = join([String.(CFG["experiment"]["layout"]); EXPERIMENT], "/")
const UPSTREAM_CV = Dict{String, Any}("cv" => run_key(CV), "bb" => run_key(load_config(CV["runs"]["bb"])))
const DRIVER = "bmvc2026/scripts/halfwidth_study.jl"
const POPULATIONS = ("final", "guided", "descriptor")

# --max_frames=N (a smoke run): the first N validation frames only, a part of every product's key.
const MAX_FRAMES = parse(Int, cli_option(ARGS, "max_frames", "0"))
_parts(d = Dict{String, Any}()) = MAX_FRAMES > 0 ? merge(d, Dict{String, Any}("max_frames" => MAX_FRAMES)) : (isempty(d) ? nothing : d)
residuals_product() = Product(EXP, "halfwidth_residuals", CFG; output = "residuals.jld2", upstream = UPSTREAM_CV, parts = _parts())
arm_product(name) = Product(EXP, "halfwidth_arm", CFG; output = "arm.jld2", parts = _parts(Dict{String, Any}("arm" => name)),
                            upstream = merge(UPSTREAM_CV, Dict{String, Any}("residuals" => residuals_product())))
report_product() = Product(EXP, "halfwidth_report", CFG; output = "summary.toml", parts = _parts(),
                           upstream = merge(Dict{String, Any}("residuals" => residuals_product()),
                                            Dict{String, Any}("arm_" * n => arm_product(n) for n in ARM_NAMES)))

# =============================================================================
# The scorer-prior arms (no scorer is modified: its label prior is one of Core's)
# =============================================================================

# The label prior of every Student-t + uniform scorer the pipeline builds;
# `nothing` is the configuration of record, read exactly as Estimation reads it.
const LABEL_PRIOR = Ref{Union{Nothing, VgtkCore.AbstractLabelPrior}}(nothing)
function VgtkEstimation.default_student_t_uniform_scoring(::Type{P}, outlier_halfwidth::Real) where {P}
    table = Dict{String, Any}("type" => "student_t_uniform", "outlier_halfwidth" => Float64(outlier_halfwidth))
    LABEL_PRIOR[] === nothing && return VgtkEstimation.scoring_from_toml(table; problem = P)
    return VgtkCore.StudentTUniformScoring(P, Float64(outlier_halfwidth);
        scale_prior = VgtkEstimation._scale_prior_from_toml(table, "halfwidth_study"), label_prior = LABEL_PRIOR[])
end

"""
    BetaMapPrior(alpha, beta)

The inlier-prior block of the posterior-weight EM under a Beta(alpha, beta)
prior on π: the MAP update `π = (Σ pᵢ + α − 1)/(n + α + β − 2)` of the
E-step's inlier posteriors pᵢ (through Core's one clamp).
"""
struct BetaMapPrior
    alpha::Float64
    beta::Float64
end
VgtkCore.update_inlier_prior!(pws::VgtkCore.PosteriorWeightState, n::Int, p::BetaMapPrior) =
    pws.inlier_prior = VgtkCore._clamp_inlier_prior((VgtkCore._sum_inlier_posterior(pws, n) + p.alpha - 1) /
                                                    (n + p.alpha + p.beta - 2))

"""
    PriorStudentTUniform(inner, prior_update)

LODR's Stu-tU-IRLS (`inner`) with the LO's inlier prior π under the update
policy `prior_update` instead of held at the seed's 1/2: the same
posterior-weight iteration (Estimation's `posterior_irls!` under its
`PosteriorLikelihood`), the same workspace, the same identity hooks.
"""
struct PriorStudentTUniform{S, P} <: VgtkCore.AbstractOptimizationStrategy
    inner::S
    prior_update::P
end
function VgtkCore.locally_optimize!(L::VgtkEstimation.LoBuffers, lo::PriorStudentTUniform, problem, model,
                                    cs::VgtkCore.ConsensusSet, buf::VgtkCore.ScoreBuffers, scoring)
    return VgtkEstimation.posterior_irls!((rws, n) -> VgtkEstimation.PosteriorLikelihood(rws, n; prior_update = lo.prior_update),
                                          L, problem, model, cs, buf, scoring,
                                          VgtkEstimation.required_outlier_halfwidth(scoring); max_iter = lo.inner.max_iter)
end
for f in (:optimization_path, :scale_interface, :loss_family, :lo_marker, :lo_label)
    isdefined(VgtkCore, f) && @eval VgtkCore.$f(lo::PriorStudentTUniform) = VgtkCore.$f(lo.inner)
end
isdefined(VgtkEstimation, :strategy_loss) &&
    @eval VgtkEstimation.strategy_loss(::Type{<:PriorStudentTUniform{S}}) where {S} = VgtkEstimation.strategy_loss(S)

# Beta(m·κ, (1 − m)·κ)
_beta_parameters(arm, m) = (m * arm["strength"], (1 - m) * arm["strength"])

"The label prior of an arm's scorer (`nothing`: the configuration of record) at the inlier rate `m`."
function arm_label_prior(arm, m)
    arm["score"] == "flat" && return nothing
    arm["score"] == "profiled" && return VgtkCore.ProfiledLabelPrior()
    arm["score"] == "beta" && return VgtkCore.BetaLabelPrior(_beta_parameters(arm, m)...)
    error("arm $(arm["name"]): score = $(repr(arm["score"])) is not flat | profiled | beta")
end

"The LO strategy of an arm, `lo` the run config's (LODR's Stu-tU-IRLS), at the inlier rate `m`."
function arm_lo(arm, lo, m)
    arm["lo"] == "held" && return lo
    arm["lo"] == "estimated" && return PriorStudentTUniform(lo, VgtkCore.Estimated())
    arm["lo"] == "beta" && return PriorStudentTUniform(lo, BetaMapPrior(_beta_parameters(arm, m)...))
    error("arm $(arm["name"]): lo = $(repr(arm["lo"])) is not held | estimated | beta")
end

# =============================================================================
# The labelled residuals
# =============================================================================

# The centre offset, in units of the radius, at which two equal circles reach the overlap `min_iou` (Core's iou).
function gate_constant(min_iou)
    f(d) = VgtkCore.iou(VgtkCore.Ellipse([0.0, 0.0], 1.0, 1.0, 0.0), VgtkCore.Ellipse([d, 0.0], 1.0, 1.0, 0.0))
    lo, hi = 0.0, 2.0
    for _ in 1:80
        m = (lo + hi) / 2
        f(m) >= min_iou ? (lo = m) : (hi = m)
    end
    return lo
end

_centre(e) = (Float64(e.center[1]), Float64(e.center[2]))
_dist(p, q) = hypot(p[1] - q[1], p[2] - q[2])

# The projected pattern of a board at its pose: each blob's region centre and radius √(ab) (NaN where it does not transfer).
function projected_pattern(T, bundle)
    co = BlobBoards.cutoff(bundle)
    ep = [BlobBoards.transfer_ellipse(T, BlobBoards.region(d, co)) for d in BlobBoards.blobs(bundle, :world)]
    centres = [e === nothing ? (NaN, NaN) : _centre(e) for e in ep]
    rho = [e === nothing ? NaN : sqrt(Float64(e.a) * Float64(e.b)) for e in ep]
    return centres, rho
end

function nn_pitch(centres)
    ok = [c for c in centres if isfinite(c[1])]
    length(ok) < 2 && return NaN
    return median([minimum(_dist(c, q) for (j, q) in enumerate(ok) if j != i) for (i, c) in enumerate(ok)])
end

# One population's residuals (the pose problem's own) and labels.
# The detection's centre is read off the correspondence's own image side (the detection index of a phase-2
# board is into that phase's pool); the blob index `src_id` is into the board's world blobs.
function label_population!(rows, tbl, pop, cam, pose, centres; frame, uid, gate, pitch, weights = nothing)
    length(tbl) == 0 && return rows
    pts = BlobBoards.planar_lift(tbl)
    prob = VgtkEstimation.AbsolutePoseProblem(VgtkCore.model(cam), pts)
    r = zeros(length(pts))
    BlobBoards.mahalanobis_distances!(r, prob, pose)
    tg = VgtkCore.targets(tbl)
    for (k, j) in enumerate(tbl.src_id)
        cj = centres[j]
        isfinite(cj[1]) || continue
        x = _centre((center = BlobBoards.centroid(tg[k]),))
        own = _dist(x, cj)
        other = minimum((_dist(x, c) for (q, c) in enumerate(centres) if q != j && isfinite(c[1])); init = Inf)
        push!(rows.population, pop); push!(rows.frame, frame); push!(rows.uid, string(uid))
        push!(rows.residual, r[k]); push!(rows.own_px, own); push!(rows.other_px, other)
        push!(rows.inlier, own <= other); push!(rows.gate, gate); push!(rows.pitch, pitch)
        push!(rows.weight, weights === nothing ? NaN : Float64(weights[k]))
    end
    return rows
end

# ---- the truncated Student-t + uniform EM (Core's mixture blocks) ----------------------------------------------------

# A residual beyond the support disc (the uniform density's area, (2a)², as a disc) cannot be an outlier: its posterior is 1.
function _truncate!(pws, R2, n)
    m = VgtkCore.StudentTMixture(pws)
    sq = VgtkCore.sq_distances(pws)
    @inbounds for i in 1:n
        sq[i] > R2 || continue
        pws.inlier_posterior[i] = 1.0
        pws.w[i] = VgtkCore._student_t_weight(m.νdg, m.νs2, m.s2, sq[i])
    end
    return pws
end
function _truncated_nll(pws, R2, n)
    m = VgtkCore.StudentTMixture(pws)
    sq = VgtkCore.sq_distances(pws)
    lp = log(VgtkCore.inlier_prior(pws))
    s = 0.0
    @inbounds for i in 1:n
        s += sq[i] > R2 ? -(lp + VgtkCore._log_inlier_density(m, sq[i])) : VgtkCore.mixture_nll(m, sq[i])
    end
    return s
end

"""
    em_fit(r, a; nu0, max_iter, rtol) -> NamedTuple

The Student-t (scale s, dof ν) + uniform (half-width a, d = 2) mixture's ML
fit of the residual norms `r` with π estimated: Core's E-step and its ŝ², ν
and π blocks, the uniform truncated to its support disc.
"""
function em_fit(r::AbstractVector{Float64}, a::Real; nu0, max_iter, rtol)
    sq = r .^ 2
    n = length(sq)
    R2 = (2a)^2 / pi
    pws = VgtkCore.median_seeded_posterior(sq, n; outlier_halfwidth = a, ν = nu0, prior = VgtkCore.JeffreysPrior(), codimension = 2)
    _truncate!(pws, R2, n)
    prev = Inf; nll = Inf; it = 0
    for outer it in 1:max_iter
        VgtkCore.update_scale!(pws, n); VgtkCore.update_nu!(pws, n); VgtkCore.update_inlier_prior!(pws, n)
        VgtkCore.update_weights!(pws, n); _truncate!(pws, R2, n)
        nll = _truncated_nll(pws, R2, n)
        abs(prev - nll) <= rtol * abs(nll) && break
        prev = nll
    end
    return (; a = Float64(a), nll, s = sqrt(VgtkCore.s_hat_sq(pws)), nu = VgtkCore.student_t_dof(pws),
            inlier_prior = VgtkCore.inlier_prior(pws), iterations = it, n)
end

_logspace(lo, hi, per_decade) = collect(Tuning.grid(Tuning.SpacedGrid(lo, hi, Tuning.LogPerDecade(per_decade))))

function em_profile(r, as; em)
    fits = Vector{Any}(undef, length(as))
    Threads.@threads for k in eachindex(as)
        fits[k] = em_fit(r, as[k]; nu0 = em["nu0"], max_iter = em["max_iter"], rtol = em["rtol"])
    end
    return fits
end
_columns(fits) = Dict{String, Any}(String(k) => [getfield(f, k) for f in fits] for k in (:a, :nll, :s, :nu, :inlier_prior, :iterations, :n))

# The named and measured half-widths, px.
function named_halfwidths(gate, pitch, W, H)
    d = Dict{String, Float64}(k => Float64(v) for (k, v) in CFG["grid"]["named"])
    d["gate"] = gate; d["pitch"] = pitch; d["half_diagonal"] = hypot(W, H) / 2; d["diagonal"] = hypot(W, H)
    return d
end

# The CV's shared state, its frames capped by --max_frames.
function study_context()
    ctx = invariance_cv(CV, SPEC.scheme.cross_validation; device = CUDABackend())
    MAX_FRAMES > 0 || return ctx
    n = min(MAX_FRAMES, length(ctx.frames))
    return Main.BlobBoardsEval.InvarianceCV(ctx.image_dir, ctx.frames[1:n], ctx.stems[1:n], ctx.base, ctx.net, ctx.feats,
                                            ctx.bundles, ctx.uids, ctx.min_obs, ctx.device, ctx.run_config)
end

function run_residuals()
    ctx = study_context()
    cfg = ctx.base
    cam = BlobBoards._estimation_camera(cfg.geometry)
    min_iou = Float64(CFG["study"]["min_iou"])
    c_gate = gate_constant(min_iou)
    features = detection_run(ctx, cfg)
    res = invariance_errors(ctx, features, cfg)
    rows = (population = String[], frame = String[], uid = String[], residual = Float64[], own_px = Float64[],
            other_px = Float64[], inlier = Bool[], gate = Float64[], pitch = Float64[], weight = Float64[])
    boards = (frame = String[], uid = String[], gate = Float64[], pitch = Float64[], rho = Float64[], n_final = Int[])
    for s in ctx.stems
        feats = features[s]
        for (uid, pose, _, scored) in res.found[s]
            bundle = ctx.bundles[uid]
            T = BlobBoards._world_to_image(pose, cam)
            centres, rho = projected_pattern(T, bundle)
            rho_med = median(filter(isfinite, rho))
            gate = c_gate * rho_med; pitch = nn_pitch(centres)
            push!(boards.frame, s); push!(boards.uid, string(uid)); push!(boards.gate, gate); push!(boards.pitch, pitch)
            push!(boards.rho, rho_med); push!(boards.n_final, length(scored))
            kw = (; frame = s, uid, gate, pitch)
            label_population!(rows, scored, "final", cam, pose, centres; weights = scored.weights, kw...)
            g = BlobBoards.guided_search(T, bundle, feats; min_iou)
            label_population!(rows, g, "guided", cam, pose, centres; kw...)
            bf = ctx.feats[uid]
            mm = BlobBoards.match(bf, feats, BlobBoards.MutualNN(); min_similarity = cfg.min_similarity)
            d = BlobBoards._scored_with_ids(bf, feats, mm)
            label_population!(rows, d, "descriptor", cam, pose, centres; kw...)
        end
    end
    em = CFG["em"]
    pooled = Dict{String, Any}()
    per_frame = Dict{String, Any}()
    for pop in POPULATIONS
        sel = rows.population .== pop
        pooled[pop] = _columns(em_profile(rows.residual[sel], _logspace(em["lo"], em["hi"], em["per_decade"]); em))
        as = _logspace(em["lo"], em["hi"], em["per_decade_frame"])
        pf = Dict{String, Any}()
        for s in ctx.stems
            r = rows.residual[sel .& (rows.frame .== s)]
            length(r) >= em["min_n"] || continue
            pf[s] = _columns(em_profile(r, as; em))
        end
        per_frame[pop] = pf
    end
    H, W = size(load_image(joinpath(ctx.image_dir, first(ctx.frames))))   # Julia images are rows × columns
    gate = median(boards.gate); pitch = median(boards.pitch)
    named = named_halfwidths(gate, pitch, W, H)
    # The MAP decision boundary (Core's) at each listed half-width, at the pooled final population's ML fit
    # (its s and ν), the label prior 1/2 (the configuration of record) and the fitted π.
    fin = pooled["final"]; k = argmin(fin["nll"])
    s_ml, nu_ml, pi_ml = fin["s"][k], fin["nu"][k], fin["inlier_prior"][k]
    thresholds = Dict{String, Any}()
    for name in CFG["thresholds"]["at"]
        a = named[name]
        thresholds[name] = Dict{String, Any}("halfwidth" => a,
            "boundary_half" => VgtkCore.decision_boundary(VgtkCore.GauInlier(s_ml^2; nu = nu_ml, gamma = 0.5, d_g = 2), a),
            "boundary_fitted" => VgtkCore.decision_boundary(VgtkCore.GauInlier(s_ml^2; nu = nu_ml, gamma = pi_ml, d_g = 2), a))
    end
    sel = rows.population .== "final"
    rec = Dict{String, Any}(
        "rows" => Dict{String, Any}(String(k) => getfield(rows, k) for k in keys(rows)),
        "boards" => Dict{String, Any}(String(k) => getfield(boards, k) for k in keys(boards)),
        "em_pooled" => pooled, "em_per_frame" => per_frame, "named" => named, "thresholds" => thresholds,
        "ml_final" => Dict{String, Any}("halfwidth" => fin["a"][k], "s" => s_ml, "nu" => nu_ml, "inlier_prior" => pi_ml),
        "inlier_rate_final" => mean(rows.inlier[sel]), "gate_constant" => c_gate, "min_iou" => min_iou,
        "image_size" => [W, H], "frames" => ctx.stems, "reference_halfwidth" => cfg.geometry.outlier_halfwidth,
        "objective_reference" => -stats_objective(CRIT, unit_statistics(CRIT, res.translation)),
        "versions" => Dict{String, Any}(string(m) => string(pkgversion(m)) for m in (BlobBoards, VgtkCore, VgtkEstimation)))
    @info "residuals" n = length(rows.residual) inlier_rate_final = rec["inlier_rate_final"] gate pitch rec["ml_final"] rec["objective_reference"]
    return Produced(rec; inputs = run_inputs(CV, ctx))
end

# =============================================================================
# The CV curve of one arm over the extended grid
# =============================================================================

function arm_grid(res)
    named = res["named"]
    vals = vcat(_logspace(CFG["grid"]["lo"], CFG["grid"]["hi"], CFG["grid"]["per_decade"]),
                [named[k] for k in keys(CFG["grid"]["named"])], [named[k] for k in CFG["grid"]["measured"]])
    return sort!(unique!(vals))
end

function run_arm(name)
    arm = ARMS[name]
    res = load_jld2(product_output(residuals_product()), "record")
    m = res["inlier_rate_final"]
    ctx = study_context()
    base = ctx.base
    base = @set base.geometry.local_optimization_strategy = arm_lo(arm, base.geometry.local_optimization_strategy, m)
    features = detection_run(ctx, base)
    vals = arm_grid(res)
    stats = Matrix{Float64}[]; boards = Int[]; seconds = Float64[]
    LABEL_PRIOR[] = arm_label_prior(arm, m)
    try
        for a in vals
            t = @elapsed r = invariance_errors(ctx, features, @set base.geometry.outlier_halfwidth = a)
            push!(stats, unit_statistics(CRIT, r.translation)); push!(boards, sum(length(r.poses[s]) for s in ctx.stems))
            push!(seconds, t)
            @info "cell" arm = name halfwidth = a mAA = -stats_objective(CRIT, last(stats)) boards = last(boards) seconds = round(t; digits = 1)
        end
    finally
        LABEL_PRIOR[] = nothing
    end
    rec = Dict{String, Any}("arm" => arm, "inlier_rate" => m, "halfwidth" => vals, "unit_stats" => stats,
                            "objective" => [-stats_objective(CRIT, S) for S in stats], "boards" => boards, "seconds" => seconds,
                            "frames" => ctx.stems, "label_prior" => string(arm_label_prior(arm, m)),
                            "lo" => string(arm_lo(arm, ctx.base.geometry.local_optimization_strategy, m)))
    return Produced(rec; inputs = run_inputs(CV, ctx))
end

# =============================================================================
# The report: figures and summary
# =============================================================================

function run_report(dir)
    rp = residuals_product()
    produced(rp) || error("the residuals product is not produced: run --stage=residuals")
    res = load_jld2(product_output(rp), "record")
    arms = Dict{String, Any}()
    for n in ARM_NAMES
        p = arm_product(n)
        produced(p) ? (arms[n] = load_jld2(product_output(p), "record")) : @warn "arm not produced: left out" n
    end
    unc = CFG["uncertainty"]; lvl = (1 - unc["interval"]) / 2
    named = res["named"]
    nf = length(res["frames"])
    rng = Xoshiro(unc["seed"])
    draws = [rand(rng, 1:nf, nf) for _ in 1:unc["n_resamples"]]       # one set of frame resamples, shared by every arm: paired
    summary = Dict{String, Any}("named" => named, "ml_final" => res["ml_final"], "inlier_rate_final" => res["inlier_rate_final"],
                                "gate_constant" => res["gate_constant"], "thresholds" => res["thresholds"],
                                "objective_reference" => res["objective_reference"], "versions" => res["versions"])
    # ---- curves
    fig = Figure(size = (900, 560))
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "outlier half-width a [px]", ylabel = "mAA@10 mm (260521/bb)",
              title = "Half-width CV curve by scorer-prior arm (90% frame-bootstrap bands)")
    palette = Makie.wong_colors()
    arm_summary = Dict{String, Any}()
    for (c, n) in enumerate(n for n in ARM_NAMES if haskey(arms, n))
        A = arms[n]; a = A["halfwidth"]; S = A["unit_stats"]
        boot = [-stats_objective(CRIT, Sj[rows, :]) for rows in draws, Sj in S]      # resamples × cells
        lo = [quantile(boot[:, j], lvl) for j in eachindex(a)]; hi = [quantile(boot[:, j], 1 - lvl) for j in eachindex(a)]
        col = palette[mod1(c, length(palette))]
        band!(ax, a, lo, hi; color = (col, 0.15))
        lines!(ax, a, A["objective"]; color = col, linewidth = n == "deployed" ? 3 : 1.5, label = n)
        best = argmax(A["objective"])
        wins = bootstrap_argmin(CRIT, S, Xoshiro(unc["seed"]); n_resamples = unc["n_resamples"])
        ia = Dict{String, Any}()
        for (k, v) in named
            j = findfirst(x -> isapprox(x, v; rtol = 1e-9), a)
            j === nothing && continue
            # paired: P(mAA at k < mAA at the arm's optimum) over the resamples
            ia[k] = Dict{String, Any}("halfwidth" => v, "mAA" => A["objective"][j], "lo" => lo[j], "hi" => hi[j],
                                      "p_below_best" => mean(boot[:, j] .< boot[:, best]))
        end
        arm_summary[n] = Dict{String, Any}("best_halfwidth" => a[best], "best_mAA" => A["objective"][best],
            "argmin_interval" => [quantile(a[wins], lvl), quantile(a[wins], 1 - lvl)], "at" => ia,
            "inlier_rate" => A["inlier_rate"], "label_prior" => A["label_prior"], "lo" => A["lo"],
            "boards_at_best" => A["boards"][best], "halfwidth" => a, "mAA" => A["objective"], "boards" => A["boards"])
    end
    for (k, ls) in (("record", :solid), ("gate", :dash), ("pitch", :dashdot), ("half_diagonal", :dot), ("diagonal", :dot))
        vlines!(ax, [named[k]]; color = :gray40, linestyle = ls, linewidth = 1, label = @sprintf("%s %.3g px", k, named[k]))
    end
    axislegend(ax; position = :lb)
    autolimits!(ax)
    save(joinpath(dir, "fig_curves.png"), fig; px_per_unit = 2)
    summary["arms"] = arm_summary
    # ---- residual histograms by population and label, with the gates
    R = res["rows"]
    fig = Figure(size = (1100, 380))
    for (c, pop) in enumerate(POPULATIONS)
        ax = Axis(fig[1, c]; xlabel = "log10 residual [px]", ylabel = c == 1 ? "count" : "", yscale = log10,
                  title = "$pop (outlier share $(@sprintf("%.3f", 1 - mean(R["inlier"][R["population"] .== pop]))))")
        sel = R["population"] .== pop
        edges = range(-3, 4; length = 141)
        for (lab, col) in ((true, palette[1]), (false, palette[2]))
            r = R["residual"][sel .& (R["inlier"] .== lab)]
            isempty(r) && continue
            hist!(ax, log10.(max.(r, 1e-3)); bins = edges, color = (col, 0.55), label = lab ? "inlier" : "outlier",
                  fillto = 0.5)
        end
        for (k, ls) in (("record", :solid), ("gate", :dash), ("pitch", :dashdot), ("half_diagonal", :dot))
            vlines!(ax, [log10(named[k])]; color = :gray30, linestyle = ls)
        end
        c == 1 && axislegend(ax; position = :rt)
    end
    save(joinpath(dir, "fig_residuals.png"), fig; px_per_unit = 2)
    pops = Dict{String, Any}()
    for pop in POPULATIONS
        sel = R["population"] .== pop
        ro = R["residual"][sel .& .!R["inlier"]]; ri = R["residual"][sel .& R["inlier"]]
        q(v, p) = isempty(v) ? NaN : quantile(v, p)
        pops[pop] = Dict{String, Any}("n" => count(sel), "outlier_share" => 1 - mean(R["inlier"][sel]),
            "outlier_q" => [q(ro, p) for p in (0.1, 0.5, 0.9, 0.99, 1.0)], "inlier_q" => [q(ri, p) for p in (0.5, 0.9, 0.99, 0.999, 1.0)],
            "outlier_within_gate" => isempty(ro) ? NaN : mean(R["residual"][sel .& .!R["inlier"]] .<= R["gate"][sel .& .!R["inlier"]]),
            "outlier_within_pitch" => isempty(ro) ? NaN : mean(R["residual"][sel .& .!R["inlier"]] .<= R["pitch"][sel .& .!R["inlier"]]),
            "outlier_within_record" => isempty(ro) ? NaN : mean(ro .<= named["record"]))
        # the share of each label the MAP rule at each listed half-width keeps as an inlier (γ = 1/2: the configuration of record)
        th = res["thresholds"]
        pops[pop]["kept"] = Dict{String, Any}(k => Dict("boundary" => th[k]["boundary_half"],
                                                        "inliers_kept" => mean(ri .<= th[k]["boundary_half"]),
                                                        "outliers_kept" => isempty(ro) ? NaN : mean(ro .<= th[k]["boundary_half"]))
                                              for k in keys(th))
    end
    summary["populations"] = pops
    B = res["boards"]
    summary["boards"] = Dict{String, Any}("n" => length(B["gate"]), "gate_q" => [quantile(B["gate"], p) for p in (0.1, 0.5, 0.9)],
                                         "pitch_q" => [quantile(filter(isfinite, B["pitch"]), p) for p in (0.1, 0.5, 0.9)],
                                         "rho_q" => [quantile(B["rho"], p) for p in (0.1, 0.5, 0.9)])
    # ---- EM profiles
    fig = Figure(size = (1100, 380))
    em_summary = Dict{String, Any}()
    for (c, pop) in enumerate(POPULATIONS)
        P = res["em_pooled"][pop]
        ax = Axis(fig[1, c]; xscale = log10, xlabel = "a [px]", ylabel = c == 1 ? "NLL − min (pooled)" : "", title = pop)
        nll = P["nll"] .- minimum(P["nll"])
        lines!(ax, P["a"], nll; color = :black)
        k = argmin(P["nll"])
        vlines!(ax, [P["a"][k]]; color = palette[2])
        vlines!(ax, [named["record"]]; color = :gray40, linestyle = :dash)
        ylims!(ax, -1, max(10, quantile(nll, 0.5)))
        pf = res["em_per_frame"][pop]
        ahat = [F["a"][argmin(F["nll"])] for F in values(pf)]
        em_summary[pop] = Dict{String, Any}("pooled" => Dict("halfwidth" => P["a"][k], "s" => P["s"][k], "nu" => P["nu"][k],
                                                             "inlier_prior" => P["inlier_prior"][k], "n" => P["n"][k]),
                                            "per_frame_halfwidth_q" => isempty(ahat) ? Float64[] : [quantile(ahat, p) for p in (0.1, 0.5, 0.9)],
                                            "n_frames" => length(ahat),
                                            "profile" => Dict("a" => P["a"], "nll_minus_min" => nll, "s" => P["s"], "nu" => P["nu"],
                                                              "inlier_prior" => P["inlier_prior"]))
    end
    save(joinpath(dir, "fig_em.png"), fig; px_per_unit = 2)
    summary["em"] = em_summary
    # ---- the MAP decision boundary as a function of a
    ml = res["ml_final"]
    as = _logspace(0.1, 5096.0, 24)
    fig = Figure(size = (700, 420))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "a [px]", ylabel = "decision boundary r* [px]",
              title = @sprintf("MAP boundary at the pooled ML fit (s = %.3f px, ν = %.1f)", ml["s"], ml["nu"]))
    for (g, lab, col) in ((0.5, "γ = 1/2 (configuration of record)", palette[1]),
                          (ml["inlier_prior"], @sprintf("γ = π̂ = %.3f", ml["inlier_prior"]), palette[2]))
        lines!(ax, as, [max(VgtkCore.decision_boundary(VgtkCore.GauInlier(ml["s"]^2; nu = ml["nu"], gamma = g, d_g = 2), a), 1e-3)
                        for a in as]; color = col, label = lab)
    end
    for (k, t) in res["thresholds"]
        scatter!(ax, [t["halfwidth"]], [max(t["boundary_half"], 1e-3)]; color = palette[1])
        text!(ax, t["halfwidth"], max(t["boundary_half"], 1e-3); text = " " * k, fontsize = 9, align = (:left, :top))
    end
    axislegend(ax; position = :lt)
    save(joinpath(dir, "fig_boundary.png"), fig; px_per_unit = 2)
    open(io -> TOML.print(io, _tomlable(summary)), joinpath(dir, "summary.toml"), "w")
    return nothing
end

# TOML wants no Nothing and no NaN-free guarantees: NaN is written as a string.
_tomlable(x::AbstractDict) = Dict{String, Any}(string(k) => _tomlable(v) for (k, v) in x)
_tomlable(x::AbstractVector) = [_tomlable(v) for v in x]
_tomlable(x::AbstractFloat) = isfinite(x) ? Float64(x) : string(x)
_tomlable(x::Union{Integer, Bool, AbstractString}) = x
_tomlable(x) = string(x)

const ARM_CHOICE = cli_option(ARGS, "arm", "")
arm_items() = isempty(ARM_CHOICE) ? ARM_NAMES : (haskey(ARMS, ARM_CHOICE) ? [ARM_CHOICE] : error("no arm $(ARM_CHOICE) in [[arms]]"))

run_stages("halfwidth_study", [
    Stage(:residuals, "labelled residuals, gate, pitch, mixture EM, MAP boundaries (GPU)";
        items = () -> ["residuals"], product = _ -> residuals_product(),
        run = _ -> produce(_ -> run_residuals(), residuals_product(); driver = DRIVER, config_source = CFG_PATH)),
    Stage(:arm, "the CV curve of a scorer-prior arm over the extended grid (GPU; --arm=<name>)";
        items = arm_items, product = arm_product,
        run = n -> produce(_ -> run_arm(n), arm_product(n); driver = DRIVER, config_source = CFG_PATH)),
    Stage(:report, "figures and summary of the note";
        items = () -> ["report"], product = _ -> report_product(),
        run = _ -> produce(dir -> run_report(dir), report_product(); driver = DRIVER, config_source = CFG_PATH)),
])
