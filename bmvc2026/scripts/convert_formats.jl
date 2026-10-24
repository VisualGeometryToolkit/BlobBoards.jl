#=
convert_formats.jl — stage 0: the motion-capture records of the absolute-pose
evaluation, from the tracker's raw per-frame exports (inputs), onto
VisualGeometryToolkitIO's kinds, verified.

    julia --startup-file=no --project=bmvc2026 bmvc2026/scripts/convert_formats.jl [--config=<toml>]

Reads data_root()/absolute_pose_experiments/<session>/poses/motion_tracker
(the vendor's `marker_*.json`, READ ONLY) and the image listing of each subset
(the positional frame ↔ image join), and writes the records to
results_root()/convert_formats-<run key>/absolute_pose_experiments/<session>/
poses/motion_tracker/ (src/runs.jl `records_dir`), the run key that of the
config of record (configs/convert_formats.toml):

  <subset>/markers.json               vgio/target_points3d/v1 (the 14 markers)
  <subset>/board_rig.json             vgio/trajectory/v3, body:board_rig → world
  <subset>/camera_rig.json            vgio/trajectory/v3, body:camera_rig → world
                                      (+ the static-camera recovery of camera-untracked
                                      calibration frames, src/recovery.jl, with its evidence)
  <subset>/<body>.kabsch.json         the reference re-solve from the markers
  constellation_<body>.json           vgio/rig/v4, frame body:<id> (a PointTarget)

Every file carries its sources in `provenance`; the run writes `_config.toml`,
`_provenance.toml` (config, commit, inputs), `_manifest.jsonl` (one line per
source file: its sha256 and what it became) and `_report.json` at its root.
Every file is the direct serialization of a VisualGeometryToolkitCore value
(VisualGeometryToolkitIO `save`) and reads back to that value exactly. A mocap
pose's column-major `rot_mat` is quaternion-encoded by VisualGeometryToolkitIO's one
writer; translations, and every scalar, are written as the shortest decimal
that reads back to the same Float64.

The cameras of record and the detections are not converted here: the camera
is an input (vgio/cameras/v3, calibration/<camera>.json), and every detection
is produced in its own keyed run (src/runs.jl).
=#

using DrWatson
@quickactivate

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: data_root, figure_config, static_camera_recovery, run_dir, EVAL_TIERS
include(joinpath(@__DIR__, "..", "src", "eval_images.jl"))   # `_is_image`: the subsets' image predicate
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitIO: append_jsonl
using VisualGeometryToolkitCore: EuclideanMap, WorldFrame, BodyFrame, kabsch, geom, Rig,
    Mounted, PointTarget, BoundedCamera, Trajectory, TargetPoses, TargetPoints3D, ids
using Rotations: QuatRotation, params
using VisualGeometryToolkitEval.Experiments: run_key, write_run_records
using DrWatson: gitdescribe
import VisualGeometryToolkitCore
using StaticArrays: SMatrix, SVector
using Rotations: RotMatrix, rotation_angle
using LinearAlgebra: norm
using Statistics: mean, median
import JSON, SHA, TOML

const CFG, CFG_PATH = figure_config("convert_formats"; default = projectdir("configs", "convert_formats.toml"))
const KEY = run_key(CFG)
const SRC_ROOT = joinpath(data_root(; tiers = EVAL_TIERS), "absolute_pose_experiments")
const RUN_ROOT = run_dir("convert_formats", KEY)              # src/runs.jl records_dir
const OUT_ROOT = joinpath(RUN_ROOT, "absolute_pose_experiments")
const MANIFEST = joinpath(RUN_ROOT, "_manifest.jsonl")
const REPORT = Dict{String, Any}()

_sha(path) = bytes2hex(open(SHA.sha256, path))
_read(path) = JSON.parse(read(path, String); allownan = true)
_rel(path, root) = relpath(path, root)

# One manifest line per consumed source file.
function _record!(session, src::AbstractString, outs::AbstractVector{<:AbstractString}; note = nothing)
    row = (
        source = joinpath(session, _rel(src, joinpath(SRC_ROOT, session))), sha256 = _sha(src),
        outputs = [joinpath(session, _rel(o, joinpath(OUT_ROOT, session))) for o in outs],
    )
    append_jsonl(MANIFEST, note === nothing ? row : merge(row, (note = note,)))
    return nothing
end

# =============================================================================
# Motion tracker: vendor per-frame exports → markers, trajectories, constellations
# =============================================================================

const MOCAP = CFG["mocap"]
const BODIES = [(id = Symbol(b["id"]), pose = b["pose"], markers = b["markers"][1]:b["markers"][2]) for b in MOCAP["bodies"]]

_colmajor(v) = SMatrix{3, 3, Float64}(reshape(Float64.(collect(v)), 3, 3))

# The capture clock: `marker_<date>_<hh-mm-ss.ffffff>[_(<note>)].json` → (date,
# seconds since that midnight); an operator's note such as `_(moved)` is kept in
# the frame's place, as the positional join always has.
function _stamp(f)
    m = match(r"^marker_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2}(?:\.\d+)?)(?:_\([a-z]+\))?\.json$", f)
    m === nothing && error("mocap file name $(f) carries no timestamp")
    return String(m[1]), 3600 * parse(Int, m[2]) + 60 * parse(Int, m[3]) + parse(Float64, m[4])
end

# One subset's frames, parsed: t → (image, time, raw board/camera poses or nothing, markers).
function read_mocap_subset(session, subset)
    dir = joinpath(SRC_ROOT, session, "poses", "motion_tracker", subset)
    files = sort(filter(endswith(".json"), readdir(dir)))
    rejected = sort(filter(endswith(".json.del"), readdir(dir)))
    img_dir = joinpath(SRC_ROOT, session, "images", subset)
    images = sort(filter(_is_image, readdir(img_dir)))
    length(files) == length(images) || error("mocap/image count mismatch for $session/$subset: $(length(files)) vs $(length(images))")
    dates = Set{String}()
    frames = map(enumerate(zip(files, images))) do (t, (f, img))
        d = _read(joinpath(dir, f))
        date, sec = _stamp(f)
        push!(dates, date)
        poses = map(BODIES) do b
            p = d.poses[b.pose]
            v = Float64.(vcat(collect(p.rot_mat), collect(p.trans)))
            any(isnan, v) ? nothing :
                EuclideanMap(RotMatrix{3, Float64}(_colmajor(p.rot_mat)), SVector{3, Float64}(Float64.(collect(p.trans))); check = false)
        end
        mk = d.makers
        length(mk) == MOCAP["n_markers"] || error("$(dir)/$(f): $(length(mk)) markers, expected $(MOCAP["n_markers"])")
        markers = [any(isnan, Float64.(collect(x))) ? nothing : SVector{3, Float64}(Float64.(collect(x))) for x in mk]
        (; t, file = f, image = img, time = sec, poses, markers)
    end
    return (; dir, files, rejected, img_dir, images, date = only(dates), frames)
end

# A body's constellation: each marker's position in the body frame, the mean
# over every frame where the vendor measured the body and saw the marker.
function estimate_constellations(subsets)
    out = Dict{Symbol, Any}()
    for (k, b) in enumerate(BODIES)
        acc = [SVector{3, Float64}[] for _ in b.markers]
        for s in subsets, fr in s.frames
            T = fr.poses[k]
            T === nothing && continue
            Ti = inv(T)
            for (j, i) in enumerate(b.markers)
                fr.markers[i] === nothing || push!(acc[j], Ti(fr.markers[i]))
            end
        end
        any(isempty, acc) && error("body $(b.id): a marker is never observed with its body tracked")
        pts = [sum(a) / length(a) for a in acc]
        rms = [sqrt(mean(norm(p - c)^2 for p in a)) for (a, c) in zip(acc, pts)]
        out[b.id] = (; points = pts, n = length.(acc), rms, maxdev = [maximum(norm(p - c) for p in a) for (a, c) in zip(acc, pts)])
    end
    return out
end

# The reference re-solve: per frame, the rigid fit of the constellation to the
# observed markers (centred Kabsch, VGC `kabsch`), where ≥ 3 are seen.
function kabsch_pose(cons::Vector{SVector{3, Float64}}, obs)
    pairs = [(cons[j], obs[j]) for j in eachindex(cons) if obs[j] !== nothing]
    length(pairs) >= 3 || return nothing
    ca = sum(first, pairs) / length(pairs)
    cb = sum(last, pairs) / length(pairs)
    R = kabsch([(a - ca) => (b - cb) for (a, b) in pairs])
    return EuclideanMap(RotMatrix{3, Float64}(R), cb - R * ca; check = false)
end

function convert_mocap(session)
    root = joinpath(SRC_ROOT, session, "poses", "motion_tracker")
    out_root = joinpath(OUT_ROOT, session, "poses", "motion_tracker")
    mkpath(out_root)
    names = sort(filter(e -> isdir(joinpath(root, e)), readdir(root)))
    subsets = [read_mocap_subset(session, s) for s in names]

    # constellations first: the trajectories name them
    cons = estimate_constellations(subsets)
    cons_files = Dict{Symbol, String}()
    for b in BODIES
        c = cons[b.id]
        file = joinpath(out_root, "constellation_$(b.id).json")
        F = BodyFrame{b.id}
        rig = Rig{F}(Dict(b.id => Mounted(PointTarget(c.points), EuclideanMap(one(RotMatrix{3, Float64}), zero(SVector{3, Float64}); check = false))))
        VgtkIO.save(
            file, rig;
            provenance = (
                body = String(b.id), run_key = KEY,
                status = "estimated: the vendor's body definition is not in its export",
                method = "per marker, the mean over every frame (all subsets of the session) of the observed position " *
                    "mapped into the body frame by the vendor's pose of the body",
                markers = "point i is the body's i-th marker in the export's order (rows $(first(b.markers))–$(last(b.markers)) of `makers`)",
                n_observations = c.n, rms_residual_mm = round.(c.rms; digits = 4), max_residual_mm = round.(c.maxdev; digits = 4),
            ),
        )
        cons_files[b.id] = file
        REPORT["constellation/$(session)/$(b.id)"] = (; n = c.n, rms_mm = round.(c.rms; digits = 3), max_mm = round.(c.maxdev; digits = 3))
    end

    for s in subsets
        sub = basename(s.dir)
        odir = joinpath(out_root, sub)
        mkpath(odir)
        origin = s.date * "T00:00:00"
        images = Dict(fr.t => fr.image for fr in s.frames)
        times = Dict(fr.t => fr.time for fr in s.frames)

        # the markers: the sensor's measurements, the source of every trajectory
        obs = Dict(
            fr.t => [
                (target = b.id, index = j, xyz = fr.markers[i])
                    for b in BODIES for (j, i) in enumerate(b.markers) if fr.markers[i] !== nothing
            ] for fr in s.frames
        )
        mfile = joinpath(odir, "markers.json")
        VgtkIO.save(
            mfile, TargetPoints3D{WorldFrame}(obs; sensors = MOCAP["sensor"], images, times, time_origin = origin);
            provenance = (
                source = joinpath("poses", "motion_tracker", sub), n_files = length(s.files), run_key = KEY,
                source_format = "vendor per-frame export {poses: [board rig, camera rig] (rot_mat column-major, trans mm), makers: 14×3 (mm)}",
                markers = "target = the body a marker is mounted on, index = its position in that body's constellation " *
                    "(constellation_<body>.json); an unseen marker (NaN in the export) is absent",
                rejected_frames = s.rejected,
                time = "seconds since time_origin, from each export's file name (the capture PC's local clock)",
            ),
        )
        mhash = _sha(mfile)

        # the vendor's trajectories, and the recovery of camera-untracked calibration frames
        recovery = (status = "not needed",)
        recovered = Dict{Int, EuclideanMap}()
        if occursin(r"^calibration_", sub)
            cam = [fr.poses[2] for fr in s.frames]
            board_ok = [fr.poses[1] !== nothing for fr in s.frames]
            r = try
                rc = MOCAP["recovery"]
                static_camera_recovery(cam, board_ok, s.images, s.img_dir; t_tol_mm = rc["t_tol_mm"], ang_tol_deg = rc["ang_tol_deg"], bg_tol = rc["bg_tol"])
            catch e
                e isa ErrorException || rethrow()
                recovery = (status = "refused", reason = e.msg)
                nothing
            end
            if r !== nothing
                for i in r.untracked
                    recovered[s.frames[i].t] = r.pose
                end
                recovery = (
                    status = "applied", method = "static-camera chordal-mean substitution (src/recovery.jl)",
                    frames = sort!(collect(keys(recovered))), evidence = r.evidence,
                )
            end
        end
        for (k, b) in enumerate(BODIES)
            poses = Dict{Int, EuclideanMap}(fr.t => fr.poses[k] for fr in s.frames if fr.poses[k] !== nothing)
            rec = Set{Int}()
            if b.id === :camera_rig
                for (t, X) in recovered
                    poses[t] = X
                    push!(rec, t)
                end
            end
            tr = Trajectory(
                BodyFrame{b.id}, WorldFrame, poses;
                images = Dict(t => images[t] for t in keys(poses)), times = Dict(t => times[t] for t in keys(poses)),
                time_origin = origin, recovered = rec,
            )
            file = joinpath(odir, "$(b.id).json")
            VgtkIO.save(
                file, tr;
                provenance = merge(
                    (
                        body = String(b.id), run_key = KEY,
                        source = (
                            markers = "markers.json", markers_sha256 = mhash,
                            constellation = "../" * basename(cons_files[b.id]), constellation_sha256 = _sha(cons_files[b.id]),
                            vendor_export = joinpath("poses", "motion_tracker", sub),
                        ),
                        method = MOCAP["method"],
                        untracked = "an instant the vendor did not track (NaN in the export) is absent",
                    ),
                    b.id === :camera_rig ? (recovery = recovery,) : (;),
                ),
            )
            # the reader returns exactly the quaternion written for each pose
            back = VgtkIO.load(Trajectory, file)
            for (t, X) in poses
                Y = geom(back[t])
                (Y.R isa QuatRotation && SVector{4, Float64}(params(Y.R)) == _written_quaternion(X) && Y.t == X.t) ||
                    error("$file: frame $t does not read back")
            end

            # the reference re-solve from the markers, and its agreement with the vendor
            kab = Dict{Int, EuclideanMap}()
            for fr in s.frames
                X = kabsch_pose(cons[b.id].points, fr.markers[b.markers])
                X === nothing || (kab[fr.t] = X)
            end
            kfile = joinpath(odir, "$(b.id).kabsch.json")
            VgtkIO.save(
                kfile, Trajectory(BodyFrame{b.id}, WorldFrame, kab; images = Dict(t => images[t] for t in keys(kab)), times = Dict(t => times[t] for t in keys(kab)), time_origin = origin);
                provenance = (
                    body = String(b.id), run_key = KEY,
                    source = (markers = "markers.json", markers_sha256 = mhash, constellation = "../" * basename(cons_files[b.id]), constellation_sha256 = _sha(cons_files[b.id])),
                    method = "reference re-solve: per frame, centred Kabsch (VisualGeometryToolkitCore `kabsch`) of the constellation onto the frame's observed markers (≥ 3 seen)",
                ),
            )
            both = [t for t in keys(kab) if haskey(poses, t) && !(t in rec)]
            if !isempty(both)
                dθ = [rad2deg(rotation_angle(RotMatrix{3}(SMatrix{3, 3}(kab[t].R)' * SMatrix{3, 3}(poses[t].R)))) for t in both]
                dt = [norm(kab[t].t - poses[t].t) for t in both]
                REPORT["kabsch/$(session)/$(sub)/$(b.id)"] = (n = length(both), deg_median = round(median(dθ); sigdigits = 3), deg_max = round(maximum(dθ); sigdigits = 3), mm_median = round(median(dt); sigdigits = 3), mm_max = round(maximum(dt); sigdigits = 3))
            end
        end
        REPORT["recovery/$(session)/$(sub)"] = recovery.status
        for (f, fr) in zip(s.files, s.frames)
            _record!(session, joinpath(s.dir, f), [mfile, joinpath(odir, "board_rig.json"), joinpath(odir, "camera_rig.json")])
        end
        for f in s.rejected
            _record!(session, joinpath(s.dir, f), String[]; note = "operator-rejected frame (.json.del): listed in markers.json provenance, not converted")
        end
    end
    return nothing
end

# The quaternion a pose computed as a rotation MATRIX is written as (w ≥ 0) —
# the one encoding VisualGeometryToolkitIO's writer applies; the reader returns it as is.
function _written_quaternion(X::EuclideanMap)
    q = SVector{4, Float64}(params(QuatRotation(RotMatrix{3, Float64}(X.R))))
    return q[1] < 0 ? -q : q
end

# =============================================================================
# Run
# =============================================================================

# Every converted vgio file is the serialization of a VisualGeometryToolkitCore value:
# load → save → load gives the same value (rotations bit for bit). The kinds a
# conversion writes, and the type each reads back as.
const KIND_TYPES = Dict(
    "cameras" => Dict{Symbol, BoundedCamera}, "transforms" => Vector{EuclideanMap}, "rig" => Rig,
    "trajectory" => Trajectory, "target_poses" => TargetPoses, "target_points3d" => TargetPoints3D,
)
_same(a, b) = a == b
_same(a::Rig, b::Rig) = typeof(a) == typeof(b) && ids(a) == ids(b) && all(id -> a[id] == b[id], ids(a))
function roundtrip_all(root)
    counts = Dict{String, Int}()
    for (dir, _, files) in walkdir(root; follow_symlinks = false), f in files
        p = joinpath(dir, f)
        (islink(p) || !endswith(f, ".json") || startswith(f, "_")) && continue
        kind = try
            VgtkIO.validate(p)
        catch e
            e isa ArgumentError || rethrow()
            continue                                     # not a vgio file (a copied sidecar)
        end
        T = KIND_TYPES[kind]
        v = VgtkIO.load(T, p)
        tmp = tempname() * ".json"
        VgtkIO.save(tmp, v)
        _same(VgtkIO.load(T, tmp), v) || error("$p: does not round-trip as a $(T)")
        rm(tmp)
        counts[kind] = get(counts, kind, 0) + 1
    end
    return counts
end
# THE RUN — produce-or-skip: a run whose records exist is complete.
function main()
    isdir(RUN_ROOT) && error("$(RUN_ROOT) exists without its run records — an interrupted run; move it aside to redo")
    mkpath(OUT_ROOT)
    for session in CFG["sessions"]["names"]
        convert_mocap(session)
    end
    REPORT["roundtrip"] = roundtrip_all(OUT_ROOT)

    # sizes: the payload threshold (VisualGeometryToolkitCore conventions/persisted-geometry.md, ~10 MB)
    sizes = Dict{String, Int}()
    for (root, _, files) in walkdir(RUN_ROOT; follow_symlinks = false), f in files
        p = joinpath(root, f)
        islink(p) || (endswith(f, ".json") || endswith(f, ".jsonl")) && (sizes[relpath(p, RUN_ROOT)] = filesize(p))
    end
    REPORT["largest_files"] = first(sort(collect(sizes); by = last, rev = true), 10)
    REPORT["over_10MB"] = [k for (k, v) in sizes if v > 10_000_000]

    write_run_records(
        RUN_ROOT, CFG; driver = "bmvc2026/scripts/convert_formats.jl",
        config_source = CFG_PATH, key = KEY, inputs = [joinpath(SRC_ROOT, s) for s in CFG["sessions"]["names"]],
    )
    # The code that converted: the commit of every repository whose source ran
    # (the project's own tree, and the packages that own the kinds).
    REPORT["code"] = Dict(
        "bmvc2026" => something(gitdescribe(realpath(@__DIR__)), "?"),
        "VisualGeometryToolkitIO" => something(gitdescribe(pkgdir(VgtkIO)), "?"),
        "VisualGeometryToolkitCore" => something(gitdescribe(pkgdir(VisualGeometryToolkitCore)), "?"),
    )
    open(joinpath(RUN_ROOT, "_report.json"), "w") do io
        JSON.json(io, REPORT; pretty = 4, sort_keys = false)
    end
    println("converted -> ", RUN_ROOT)
    return nothing
end

isfile(joinpath(RUN_ROOT, "_provenance.toml")) ?
    println("convert_formats: run $(KEY) complete at $(RUN_ROOT)") : main()
