#=
calibration.jl — ONE hand-eye authority, config-keyed.

There used to be three independent fits of the same quantity:

  make_hand_eye.jl        -> results/hand_eye_<name>.json   read by NOBODY
  export_reconstruction   -> calibration/hand_eye.json      read by hand_eye_reprojection
  error_table.jl          -> in memory, never written       behind EVERY absolute-error
                                                            number in the paper

The third is the one that matters and the one that cannot be audited. All three
are deterministic, so they agree given identical inputs — but `poses/` is a
mutable slot each sweep clears and rewrites, so each fit captured whatever
snapshot existed when it ran. The two persisted artifacts already disagree
(ΔX = 0.0012° / 0.019 mm; worst record 0.016° / 0.031 mm) for exactly that
reason, and the gap is bounded by whatever the detector change was between runs,
not by any numerical tolerance.

`hand_eye` below is the single entry point. It is keyed on the inputs that
determine the fit — dataset, subset, the frame list, the marker set, the
gate — so a different input set is a different artifact rather than an
overwrite, and a re-run with unchanged inputs is a cache hit instead of a
fourth fit.

WHAT IS STORED. Two VisualGeometryToolkitIO files, the same shapes the published
`hand_eye_<name>.json` records take: a `vgio/transforms/v3` file holding X,
`body:camera_rig → camera_optical`, and beside it (`<name>_board_rig.json`) a
`vgio/rig/v6` file holding the marker mount — one target per printed board,
its pose `target:<id> → body:board_rig`. Every pose is UNCERTAIN: the fit states
its 6×6 covariance (rad, mm; the `RigidExpManifold6` chart, rotation first —
VisualGeometryToolkitCore conventions/persisted-geometry.md), an estimator's estimate
is never written without one. The fit's scalar diagnostics and the keying
config ride in the transforms file's provenance — never RVG's fit object and
never a private matrix layout: an artifact that embeds an upstream package's
internal types stops loading the day that package reorganises them, which is
the failure mode this cache exists to avoid.
=#

using LinearAlgebra: norm

"""
    hand_eye(src, frames, detected; subset, min_rel_angle = deg2rad(5), force = false)
        -> (X, targets, diagnostics, file)

The hand-eye (robot-world) calibration for one marker system of the
evaluation's sources `src`, cached by config.

`X` is the labeled camera-mount transform (`CameraRigFrame →
ViewFrame{:optical}`); `targets` is the marker mount as a
`Rig{BoardRigFrame}` of `PosedTarget` boards (marker id, as `Symbol` → a
pose-only `PlanarTarget` and its `TargetFrame{id} → BoardRigFrame` pose,
minted on access). `diagnostics`
carries the fit's scalar health numbers. `file` is the artifact path, for
provenance.

Supply `force = true` to refit and overwrite.
"""
# THE ARTIFACT IS a vgio transforms file (X) and a vgio rig file (the Y_m),
# written by VisualGeometryToolkitIO's `save` from the labeled poses and read
# back through `load`, the labels minted from the files' typed frames — never a
# JLD2, which would embed either a private layout (the 16-float 4×4 this
# replaced on 2026-09-14) or a package type. Keyed like every run: Eval's
# `cached_result` with `key = run_key` and a vgio codec, the file
# `handeye_<dataset>_<subset>_<key>.json` in the fit's `keyed_cell_dir`
# (`…/bmvc2026/hand_eye_fit-<run_key>`, the key over the evaluation's
# [experiment] layout and everything the fit reads, below); `force` keeps the
# outgoing file as Eval's `_#<n>`.
# The fit as Eval's `cached_result` codec of a vgio record (ext = ".json"):
# `save` writes the two VisualGeometryToolkitIO records, every other field of
# the cached Dict (the keying config, the scalar diagnostics, the code trees and
# DrWatson's tag) in the transforms file's provenance; `load` reads that
# provenance back (an incomplete pair — no mount beside it — reads as nothing,
# so the fit is computed again). The previous version is kept by Eval
# (safesave's `_#n`).
_save_hand_eye_record(file, data::AbstractDict) =
    save_hand_eye(file, data["fit"]; provenance = (; (Symbol(k) => v for (k, v) in data if k != "fit")...))
_load_hand_eye_record(file) = isfile(hand_eye_mount_file(file)) ?
    JSON.parse(read(file, String); dicttype = Dict{String, Any})["provenance"] : Dict{String, Any}()

function hand_eye(
        src::Sources, frames::Vector{RigFrame}, detected::Dict;
        subset::AbstractString, min_rel_angle::Float64 = deg2rad(5), force::Bool = false
    )
    dataset = src.dataset
    # Normalise the join key HERE so no caller has to know it. `_stem` was
    # being reimplemented at six call sites; it is the join key of the whole
    # evaluation and it is not exported, so every caller that needed it either
    # copied it or reached into the module.
    detected = Dict(_stem(k) => v for (k, v) in detected)
    # The key must name everything the fit depends on. Frame stems and marker
    # ids are listed rather than counted: two subsets of equal SIZE are
    # different calibrations, and a count would collide them into one artifact.
    usable = [
        fr for fr in frames
            if fr.T_cb !== nothing && haskey(detected, _stem(fr.image))
    ]
    isempty(usable) && error("hand-eye: no usable frames for $dataset/$subset")
    markers = sort(
        collect(
            reduce(
                union,
                (Set(p.id for p in detected[_stem(fr.image)]) for fr in usable)
            )
        )
    )

    cfg = Dict{String, Any}(
        "experiment" => copy(src.experiment),
        "subset" => String(subset),
        "min_rel_angle_deg" => round(rad2deg(min_rel_angle); digits = 9),
        "frames" => sort([_stem(fr.image) for fr in usable]),
        "markers" => markers,
        # The pose VALUES determine the fit — the frame LIST does not: a newly
        # promoted run re-detects the same calibration frames with different
        # poses while every other key component stays identical, which used to
        # be a silent cache hit on the PREVIOUS network's calibration
        # (2026-08-11 fix; the last mutable-slot staleness path). A content
        # key rather than the upstream run key, so the same mechanism covers a
        # caller's own detections: Eval's `run_key` (SHA-256 of the sorted
        # TOML) of every pose, stable across processes and Julia versions.
        "poses_content" => run_key(Dict{String, Any}("poses" => Dict{String, Any}(
            stem => [Dict{String, Any}("id" => p.id, "R" => collect(vec(p.R)), "t" => collect(p.t)) for p in ps]
            for (stem, ps) in detected))),
    )
    _, file = cached_result(
        c -> let he = hand_eye_calibrate(frames, detected; min_rel_angle)
            Dict{String, Any}(
                "fit" => he, "config" => c, "n_frames" => length(usable), "n_markers" => length(markers),
                # Scalars only — see the header on not embedding RVG's fit object.
                "sigma_rot_rad" => he.fit.metadata.σ_rot,      # the fit's per-block noise scales (its metadata)
                "sigma_trans_mm" => he.fit.metadata.σ_trans,
            )
        end,
        string("handeye_", dataset, "_", subset), cfg, keyed_cell_dir(cfg, EXPERIMENT, "hand_eye_fit");
        key = run_key, ext = ".json", load = _load_hand_eye_record, save = _save_hand_eye_record, force,
    )

    X, targets = load_hand_eye(file)
    # `load` hands back the records only; the diagnostics ride in the file's
    # `provenance`, which VgtkIO never reads.
    prov = JSON.parse(read(file, String)).provenance
    diagnostics = (
        n_frames = Int(prov.n_frames), n_markers = Int(prov.n_markers),
        sigma_rot_rad = prov.sigma_rot_rad, sigma_trans_mm = prov.sigma_trans_mm,
        gitcommit = get(prov, :gitcommit, nothing),
    )
    return (; X = geom(X), targets = _plain_targets(targets), uncertain = (; X, targets), diagnostics, file)
end

