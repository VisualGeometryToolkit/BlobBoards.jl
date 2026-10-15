#=
branch_lrt_accuracy.jl — how often does the post-LO likelihood-ratio test
pick the true planar-pose branch?

Joins the branch telemetry emitted by _resolve_pose_ambiguity
(BB_BRANCH_LOG jsonl: both branches' rotations, the post-LO marginal-score
margin = the log likelihood ratio, the discriminability d) against the mocap
ground truth (hand-eye chain), and reports, overall and binned by d:

  P(LR picks the GT-closer branch)   — the empirical LRT accuracy
  the Φ(d/2) prediction              — what the separation says is possible

The retired scale-margin selector used to be reported alongside; it measured
52.6% (a coin) and its telemetry fields were removed 2026-08-17.

Each ROW IS ONE RESOLVER DECISION, not one committed board — telemetry
records outnumber committed poses ~30:1 because the agenda adjudicates
candidates it later rejects, and every one of those still made a branch
choice this script can score. Records are joined to GT by nearest translation
(no uid travels with the telemetry; board centers are >100 mm apart,
incumbent translations are within ~mm of truth — unmatched records are
dropped and counted). Do NOT reuse this join shape to attribute COMMITTED
poses: those come out of the guided fit, which moves them off the logged
translation, and every geometric proxy for that join has been measured to
fail (see the BRANCH_LOG comment in src/pipeline.jl).

Usage:
  julia --project=bmvc2026 bmvc2026/scripts/branch_lrt_accuracy.jl

Env knob: BB_BRANCH_LOG (required: the telemetry a sweep wrote with
BB_BRANCH_LOG set). The detections and the motion-capture records are the
evaluation's runs (configs/figures/evaluation.toml [runs], src/runs.jl).
=#

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
using .BlobBoardsEval
using .BlobBoardsEval: rig_sequence, load_detected_poses, eval_sources, load_config, poses_dir
using VisualGeometryCore: pose, ids
using JSON3
using LinearAlgebra
import VisualGeometryCore
import Rotations
using StaticArrays
using Statistics

SRC = eval_sources(load_config("figures/evaluation.toml"))
logpath = get(() -> error("set BB_BRANCH_LOG to a sweep's branch telemetry (jsonl)"), ENV, "BB_BRANCH_LOG")

_stem(f) = splitext(basename(f))[1]
# Rotations.jl's rotation_angle of the relative rotation, not a local
# guarded-trace copy: the invariance criterion (now VisualGeometryEval rigid_invariance_errors) already retired one such hand-roll, and
# a third spelling of the same formula is a third place for the clamp to drift.
_rotang(A, B) = Rotations.rotation_angle(Rotations.RotMatrix{3}(A' * B))

# GT board poses per frame from the mocap + hand-eye chain
frames = rig_sequence(SRC.mocap, "bb")
cf = rig_sequence(SRC.mocap, "calibration_bb")   # with the static-camera recovery's frames
cd_ = load_detected_poses(poses_dir(SRC, "bb"; calibration = true))
# The ONE hand-eye authority (src/calibration.jl): cached, keyed on the
# frames and markers that determine the fit, and tagged with the commit.
# This was an independent refit — the project had FOUR of them.
he = hand_eye(cf, cd_; dataset = SRC.dataset, subset = "calibration_bb")
Tcb = Dict(_stem(f.image) => f.T_cb for f in frames if f.T_cb !== nothing)

gt_by_frame = Dict{String, Vector{Tuple{SMatrix{3, 3, Float64, 9}, SVector{3, Float64}}}}()
for (stem, T) in Tcb
    gts = Tuple{SMatrix{3, 3, Float64, 9}, SVector{3, Float64}}[]
    for id in ids(he.targets)
        G = he.X ∘ T ∘ pose(he.targets, id)   # TargetFrame{id} → ViewFrame{:optical}, junction-checked
        push!(gts, (SMatrix{3, 3, Float64, 9}(G.R), SVector{3, Float64}(G.t)))
    end
    gt_by_frame[stem] = gts
end

records = [JSON3.read(l) for l in eachline(logpath) if !isempty(l)]
println("$(length(records)) telemetry records from $logpath")

rows = NamedTuple[]
unmatched = Ref(0)
for r in records
    stem = _stem(String(r.tag))
    gts = get(gt_by_frame, stem, nothing)
    gts === nothing && (unmatched[] += 1; continue)
    tinc = SVector{3, Float64}(r.t_inc...)
    dists = [norm(g[2] - tinc) for g in gts]
    k = argmin(dists)
    dists[k] < 60.0 || (unmatched[] += 1; continue)     # join gate: 60 mm
    Rgt = gts[k][1]
    Rinc = SMatrix{3, 3, Float64, 9}(reshape(collect(Float64, r.R_inc), 3, 3))
    Ralt = SMatrix{3, 3, Float64, 9}(reshape(collect(Float64, r.R_alt), 3, 3))
    einc, ealt = _rotang(Rinc, Rgt), _rotang(Ralt, Rgt)
    inc_true = einc < ealt
    push!(
        rows, (;
            d = Float64(r.d),
            sep = rad2deg(_rotang(Rinc, Ralt)),
            lr_correct = (r.q_inc > r.q_alt) == inc_true,
            einc = rad2deg(einc), ealt = rad2deg(ealt),
        )
    )
end
println("$(length(rows)) joined, $(unmatched[]) unmatched/dropped")

# logistic approximation of the normal CDF (max err < 1%) — avoids a dep
Φ(x) = 1 / (1 + exp(-1.702 * x))

println("\nP(correct) by discriminability d:")
edges = [0.0, 1.0, 3.0, 5.0, 10.0, 30.0, Inf]
for i in 1:(length(edges) - 1)
    sel = [r for r in rows if edges[i] <= r.d < edges[i + 1]]
    isempty(sel) && continue
    println(
        "  d ∈ [$(edges[i]), $(edges[i + 1])): n=$(length(sel))" *
            "  LRT=$(round(100 * count(r -> r.lr_correct, sel) / length(sel); digits = 1))%" *
            "  Φ(d/2) predicts $(round(100 * mean(Φ(r.d / 2) for r in sel); digits = 1))%"
    )
end
println(
    "\noverall: LRT $(round(100 * count(r -> r.lr_correct, rows) / length(rows); digits = 1))%" *
        "  over $(length(rows)) branch decisions"
)
