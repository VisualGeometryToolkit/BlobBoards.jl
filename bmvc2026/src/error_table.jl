# =============================================================================
# The per-(image, marker) pose-error table — the paper's numbers.
#
# This file was called `aggregate.jl`, which said the opposite of what it does.
# Aggregation REDUCES; this EXPANDS to the full image x marker grid (56 x 30 =
# 1680 for blob) and keeps every row, `missing` where the marker was not
# detected, precisely so detection rate stays measurable. The reducing —
# medians, IQR, distance bins — happens downstream in scripts/plots/.
#
# It pairs with pose_error.jl: that file holds the METRICS
# (relative_errors, absolute_errors), this one applies them across the grid.
#
# Reproduces the shipped {april,aruco,blob}.csv layout (row grid = marker ids
# × images grouped by marker, rig_* columns, 10°/1 m bins, marker_scales,
# error_angles_deg). The error-column definitions are the canonical ones in
# pose_error.jl — the original pandas notebook that produced the shipped CSVs
# was never delivered, so they are defined cleanly rather than
# reverse-engineered (rig columns match the shipped files):
#   error_angles[_deg]     calibration-free relative rotation-angle error
#   error_distances        hand-eye ABSOLUTE translation error, mm
#   abs_error_angles[_deg] hand-eye absolute rotation error
# =============================================================================

const BOARD_TYPES = (
    (name = "apriltags", abbr = "apr", system = "april"),
    (name = "aruco", abbr = "uco", system = "aruco"),
    (name = "bb", abbr = "bb", system = "blob"),
)

# The board-definition subdirectory (`<inputs>/boards/<dir>`) of a marker
# system, by its image-subset abbreviation.
const BOARDS_DIRS = Dict("apr" => "apriltags", "uco" => "aruco", "bb" => "blobboards")

# The size (mm) of every target a run detected — its universe — from the
# targets' definitions.
function detected_marker_sizes(inputs_dir::AbstractString, abbr::AbstractString, detected::Dict)
    defs = marker_sizes(inputs_dir, BOARDS_DIRS[abbr])
    ids = Set(p.id for ps in values(detected) for p in ps)
    missing_ = setdiff(ids, keys(defs))
    isempty(missing_) || error("no board definition for detected target(s) $(sort!(collect(missing_))) in $(inputs_dir)/boards/$(BOARDS_DIRS[abbr])")
    return Dict(id => defs[id] for id in ids)
end

_stem(f) = splitext(basename(f))[1]

# Marker scale labels from the distinct size_mm values (rank order). The
# multiscale boards carry up to three sizes; fall back to tertiles if more.
function _scale_labels(sizes::Dict{String, Float64})
    isempty(sizes) && return Dict{String, String}()
    distinct = sort(unique(values(sizes)))
    labels = length(distinct) == 1 ? ["all"] :
        length(distinct) == 2 ? ["small", "large"] :
        length(distinct) == 3 ? ["small", "medium", "large"] : nothing
    if labels === nothing
        cuts = quantile(collect(values(sizes)), [1 / 3, 2 / 3])
        return Dict(
            id => (s <= cuts[1] ? "small" : s <= cuts[2] ? "medium" : "large")
                for (id, s) in sizes
        )
    end
    rank = Dict(s => labels[i] for (i, s) in enumerate(distinct))
    return Dict(id => rank[s] for (id, s) in sizes)
end

# Frontality diagnostic: |cos| between the optical ray to the marker center
# and the detected marker normal (z column).
_frontality(p) = abs(dot(normalize(p.t), p.R[:, 3]))

"""
    pose_error_table(frames, detected, X, targets; marker_sizes) -> DataFrame

Build the per-(image, marker) error table for one board type. `frames` is the
rig sequence of the evaluation sweep, `detected` its detected poses (keyed by
image name, joined on file stems).

`X` and `targets` are the hand-eye calibration (the labeled camera-mount
transform and the marker-mount target rig), taken as ARGUMENTS rather than fitted
here. This function used to fit its own — the third independent fit of the same
quantity in this project, and the only one never written to disk, yet the one
behind every absolute-error column in the paper. The layering was the bug: a
function that MEASURES error should consume a calibration, not produce one.
Resolve it once through `hand_eye` (src/calibration.jl) and forward it.

`marker_sizes` fixes the marker universe (id →
size, mm, from the targets' definitions: [`detected_marker_sizes`](@ref)); the
parent run's map for occlusion levels keeps the row grid comparable even when
detections collapse. Rows exist for every (marker, tracked image) pair; error
columns are `missing` where the marker was not detected (or absent from the
spot base).
"""
function pose_error_table(
        frames::Vector{RigFrame}, detected::Dict,
        X, targets::Rig;
        marker_sizes::Dict{String, Float64}
    )
    det = Dict(_stem(img) => ps for (img, ps) in detected)

    rel = relative_errors(frames, det)
    abs_err = absolute_errors(frames, det, X, targets)

    sizes = marker_sizes
    scales = _scale_labels(sizes)
    ids = sort(collect(keys(sizes)))

    rows = NamedTuple[]
    for id in ids, fr in frames
        fr.T_cb === nothing && continue
        stem = _stem(fr.image)
        p = findfirst(q -> q.id == id, get(det, stem, []))
        pose = p === nothing ? nothing : det[stem][p]
        r = get(rel, (stem, id), nothing)
        a = get(abs_err, (stem, id), nothing)
        push!(
            rows, (
                error_angles = r === nothing ? missing : r,
                error_distances = a === nothing ? missing : a.error_distance,
                marker_ids = id,
                rig_distances = fr.dist,
                rig_angles = fr.rig_angle,
                marker_scales = scales[id],
                dot_product = pose === nothing ? missing : _frontality(pose),
                rig_angles_bin = round(Int, rad2deg(fr.rig_angle) / 10) * 10,
                rig_distances_bin = round(Int, fr.dist / 1000) * 1000,
                error_angles_deg = r === nothing ? missing : rad2deg(r),
                abs_error_angles = a === nothing ? missing : a.error_angle,
                abs_error_angles_deg = a === nothing ? missing : rad2deg(a.error_angle),
            )
        )
    end
    return DataFrame(rows)
end

"""
    evaluate_dataset(src::Sources; occlusion_level = nothing, board_types = BOARD_TYPES,
                     hand_eye_fit = nothing) -> Dict{String,NamedTuple}

THE numbers: run the pose-error evaluation for each entry of `board_types`
(the three marker systems by default) of one dataset's [`Sources`](@ref) and
return the per-system error tables as a Dict. Persistence is
`write_pose_results` (src/results.jl): one keyed cell per (system, level).
Hand-eye calibration uses the `calibration_*` subsets. With
`occlusion_level` (e.g. `"50"`), detected poses are read from that level's
sweep (the rig join still comes from the parent subset, since the occluded
images are composited copies of the parent frames).

`board_types` is the comparison axis, one `(; name, abbr, system)` per series.
`abbr` selects the image subset and the motion-tracker sequence; `name` the
series of `src.poses` (its detection run). They are deliberately independent,
so a comparison can hold the captures and the ground truth fixed and vary
only the poses — contrasting two detection runs over the same frames instead
of contrasting marker systems.
"""
function evaluate_dataset(
        src::Sources;
        occlusion_level::Union{Nothing, AbstractString} = nothing,
        board_types = BOARD_TYPES,
        hand_eye_fit = nothing
    )
    out = Dict{String, Any}()
    for bt in board_types
        frames = rig_sequence(src.mocap, bt.abbr)
        parent_poses = poses_dir(src, bt.name)
        pdir = occlusion_level === nothing ? parent_poses : poses_dir(src, bt.name; level = occlusion_level)
        isdir(pdir) || error("no detections at $(pdir) — run the detection stage first")
        detected = load_detected_poses(pdir)

        # marker universe from the parent (unoccluded) run so occlusion levels
        # keep the full row grid even when detections collapse; the sizes are
        # the targets' definitions'
        marker_sizes = detected_marker_sizes(src.inputs, bt.abbr, load_detected_poses(parent_poses))

        # The calibration subset: the rig sequence carries the static-camera
        # recovery (recovery.jl: the provably-constant camera pose substituted
        # into camera-untracked frames, evidence-gated; 4 → 41 usable frames on
        # 260527 calibration_bb) as `recovered` frames of the camera-rig
        # trajectory. Whether the gate held changes the hand-eye fit and
        # therefore EVERY error column below, so it is recorded from the
        # trajectory's provenance.
        static_recovered = static_recovery_accepted(src.mocap, "calibration_" * bt.abbr)
        calib_frames = rig_sequence(src.mocap, "calibration_" * bt.abbr)
        calib_detected = load_detected_poses(poses_dir(src, bt.name; calibration = true))

        # ONE hand-eye authority (src/calibration.jl): cached, keyed on the
        # frames and markers that determine it, and tagged with the commit.
        # `hand_eye_fit` PINS the rigid transforms across every series instead of
        # refitting one per system. That matters whenever the series differ in
        # something upstream of the poses — an experiment that varies the
        # CAMERA, and a per-system refit would quietly absorb part of a camera
        # error into X, flattering the system with the worse camera. Pinning is
        # also physically right: X is rig → camera optical and Y_m is printed
        # marker → board rig, both rigid geometry that no intrinsic calibration
        # can change. Comparing marker SYSTEMS is the opposite case — they see
        # different printed targets, so each needs its own fit, which is why the
        # default stays `nothing`.
        he = hand_eye_fit === nothing ?
            hand_eye(calib_frames, calib_detected; dataset = src.dataset, subset = "calibration_" * bt.abbr) :
            hand_eye_fit
        df = pose_error_table(frames, detected, he.X, he.targets; marker_sizes)
        upstream = joinpath(pdir, "_config.toml")
        meta = Dict(
            "system" => bt.system,
            "poses_dir" => pdir,
            "poses_config" => isfile(upstream) ? read(upstream, String) : nothing,
            "occlusion_level" => something(occlusion_level, "none"),
            "hand_eye_source" => hand_eye_fit !== nothing ?
                "pinned by the caller (shared across series)" : "calibration subset",
            # the calibration ARTIFACT, so the CSV names the exact fit it
            # used instead of leaving it unauditable in a dead process
            "hand_eye_artifact" => basename(he.file),
            "static_recovery" => static_recovered
        )
        out[bt.system] = (; table = df, meta)
    end
    return out
end
