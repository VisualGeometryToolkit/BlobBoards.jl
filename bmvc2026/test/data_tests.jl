# =============================================================================
# Inputs only. data_root() holds the INPUTS (captures, raw motion capture, the
# cameras of record, board definitions, the synthetic occlusion images, the
# networks); every RESULT is produced by a codified stage into a keyed run
# directory under results_root() and found through its config of record
# (src/runs.jl). These tests enforce it:
#   1. the cameras of record are read from calibration/ (never the images tier);
#   2. every run resolver points into results_root(), never into data_root();
#   3. no driver or library file joins a data_root()-derived path with a
#      result kind, and the promotion / primary-copy machinery is gone;
#   4. the inputs tier, where it is installed, holds input kinds only.
# =============================================================================

@testitem "cameras of record: calibration/<camera>.json, never the images tier" setup = [BBEval] begin
    using DrWatson: projectdir
    using .BBEval.BlobBoardsEval: camera_of_record
    mktempdir() do dir
        session = mkpath(joinpath(dir, "260527"))
        cam = "sony-fx3-iso100_tamron-20mm-f2.8-(set-to-f22).json"
        store(v) = """{"schema": "vgio/cameras/$(v)", "pixel_frame": "colmap", "cameras": []}"""
        mkpath(joinpath(session, "images"))
        write(joinpath(session, "images", cam), store("v1"))                                   # the images tier's
        @test_throws ErrorException camera_of_record(session)                                  # never read
        mkpath(joinpath(session, "calibration"))
        write(joinpath(session, "calibration", cam), store("v3"))
        @test camera_of_record(session) == joinpath(session, "calibration", cam)
        write(joinpath(session, "calibration", "second.json"), store("v3"))
        @test_throws ErrorException camera_of_record(session)                                  # one camera of record
    end
    # No code path names a camera store under images/.
    raw_image_camera = r"images/[^\"']*\.json\"|\"images\"\s*,\s*\"[^\"]*\.json\""
    hits = String[]
    for top in ("src", "scripts"), (root, _, files) in walkdir(projectdir(top)), f in files
        endswith(f, ".jl") || continue
        for (i, line) in enumerate(eachline(joinpath(root, f)))
            code = strip(first(split(line, '#'; limit = 2)))
            isempty(code) || match(raw_image_camera, code) === nothing || push!(hits, "$(relpath(joinpath(root, f), projectdir())):$(i)")
        end
    end
    @test isempty(hits)
end

@testitem "every run resolves into results_root(), never data_root()" setup = [BBEval] begin
    using DrWatson: projectdir
    using .BBEval.BlobBoardsEval: eval_sources, poses_dir, val_dir, hand_eye_dir, records_dir, mocap_dir,
        sweep_dir, baselines_dir, load_config, network_path
    mktempdir() do tmp
        data, results = mkpath(joinpath(tmp, "data")), mkpath(joinpath(tmp, "results"))
        # an inputs-only data root: the network a run config names (the sweep
        # keys hash its content) and the session
        rc = load_config("bmvc2026_260527.toml")
        net = joinpath(data, rc["inputs"]["network"]["path"])
        mkpath(dirname(net)); write(net, "not a network")
        mkpath(joinpath(data, "absolute_pose_experiments", "260527"))
        withenv("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => results) do
            # Eval's input_path checks the stated sha256: another content is refused
            @test_throws ArgumentError network_path(rc)
            src = eval_sources(load_config("figures/evaluation.toml"))
            under(p, root) = startswith(abspath(p), abspath(root) * "/")
            @test under(src.inputs, data)
            dirs = [src.mocap, records_dir("260527"), mocap_dir("260527"), hand_eye_dir(src),
                sweep_dir("bmvc2026_260527.toml"), sweep_dir("bmvc2026_260527.toml"; kind = "occlusion"),
                sweep_dir("bmvc2026_260527_noflip.toml"), baselines_dir(),
                (poses_dir(src, s) for s in ("apriltags", "aruco", "bb", "bb_noflip"))...,
                (poses_dir(src, s; calibration = true) for s in ("apriltags", "aruco", "bb"))...,
                (poses_dir(src, s; level = "50") for s in ("apriltags", "aruco", "bb"))...,
                (val_dir(src, s) for s in ("apriltags", "aruco", "bb"))...]
            @test all(d -> under(d, results) && !under(d, data), dirs)
            # keyed: a run's directory is its config's, so two configs never share one
            @test sweep_dir("bmvc2026_260527.toml") != sweep_dir("bmvc2026_260527_noflip.toml")
            @test sweep_dir("bmvc2026_260527.toml") != sweep_dir("bmvc2026_260527.toml"; kind = "occlusion")
            # the occlusion levels every system is evaluated on are one list
            @test rc["experiment"]["occlusion_levels"] == load_config("baselines.toml")["occlusion"]["levels"]
        end
    end
end

@testitem "no driver reads a result from data_root()" begin
    using DrWatson: projectdir
    # A result kind: what only our code or a detector produces.
    const RESULT_SEGMENTS = ["poses", "val", "results", "pose_results", "figures", "reconstruction",
        "target_poses.json", "telemetry.jsonl", "_promoted.toml", "hand_eye.json", "markers.json",
        "board_rig.json", "camera_rig.json"]
    # The one producer that reads a raw input under one of those names: the
    # motion-capture stage reads the tracker's exports at poses/motion_tracker.
    const RAW_INPUT_READERS = Dict("scripts/convert_formats.jl" => ["poses"])

    seg = join((replace(s, "." => "\\.") for s in RESULT_SEGMENTS), "|")
    hits = String[]
    for top in ("src", "scripts"), (root, _, files) in walkdir(projectdir(top)), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        rel = relpath(path, projectdir())
        src = read(path, String)
        # the names a file binds to data_root()-derived paths (one level of
        # derivation: `X = joinpath(data_root(…), …)` and `Y = joinpath(X, …)`)
        names = Set(String(m[1]) for m in eachmatch(r"\b([A-Za-z_]\w*)\s*=\s*(?:joinpath\(\s*)?(?:\w+\.)?data_root\(", src))
        for _ in 1:2, m in eachmatch(r"\b([A-Za-z_]\w*)\s*=\s*joinpath\(\s*([A-Za-z_]\w*)\b", src)
            m[2] in names && push!(names, String(m[1]))
        end
        start = "(?:(?:\\w+\\.)?data_root\\([^)]*\\)" * (isempty(names) ? "" : "|\\b(?:$(join(sort!(collect(names)), "|")))\\b") * "|\\b\\w+\\.inputs\\b)"
        re = Regex("joinpath\\(\\s*$(start)[^\\n]*?\"($(seg))\"")
        allowed = get(RAW_INPUT_READERS, rel, String[])
        for (i, line) in enumerate(split(src, '\n'))
            code = strip(first(split(line, '#'; limit = 2)))
            isempty(code) && continue
            for m in eachmatch(re, code)
                m[1] in allowed || push!(hits, "$(rel):$(i): $(code)")
            end
        end
    end
    foreach(h -> @info(h), hits)
    @test isempty(hits)
    # the promotion and primary-copy machinery is gone: nothing is written into
    # the input tree, and nothing is selected by hand
    for top in ("src", "scripts"), (root, _, files) in walkdir(projectdir(top)), f in files
        endswith(f, ".jl") || continue
        code = join((first(split(l, '#'; limit = 2)) for l in eachline(joinpath(root, f))), "\n")
        @test !occursin(r"\b(promoted_run|assert_primary_copy|is_primary_copy)\b", code)
    end
    @test !isfile(projectdir("scripts", "promote_run.jl"))
end

@testitem "the inputs tier holds inputs only" tags = [:data] begin
    using DrWatson: projectdir
    import TOML
    import Pkg
    toml = TOML.parsefile(projectdir("Artifacts.toml"))
    @test !haskey(toml, "bb_absolute_pose_metadata") && !haskey(toml, "bb_absolute_pose_val_260527")
    @test haskey(toml, "bb_absolute_pose_inputs")
    hash = Base.SHA1(toml["bb_absolute_pose_inputs"]["git-tree-sha1"])
    if Pkg.Artifacts.artifact_exists(hash)
        root = Pkg.Artifacts.artifact_path(hash)
        input_kind = [
            r"^\d{6}/poses/motion_tracker/[a-z_]+/marker_[^/]+\.json(\.del)?$",     # raw motion capture
            r"^\d{6}/calibration/[^/]+\.json$",                                     # the camera of record
            r"^\d{6}/boards/(apriltags|aruco|blobboards)/[^/]+\.(json|pdf|png)$",   # board definitions
            r"^teaser/[^/]+\.(png|jpg)$",                                           # photos
            r"^provenance/",                                                        # the provenance material
        ]
        files = [relpath(joinpath(r, f), root) for (r, _, fs) in walkdir(root) for f in fs]
        @test !isempty(files)
        bad = filter(f -> !any(p -> occursin(p, f), input_kind), files)
        foreach(f -> @info("not an input kind: $f"), bad)
        @test isempty(bad)
        @test !any(f -> endswith(f, "_tmp.png"), files)
        for c in filter(f -> occursin(r"^\d{6}/calibration/", f), files)
            @test occursin("\"vgio/cameras/v4\"", read(joinpath(root, c), String))
        end
    else
        @info "bb_absolute_pose_inputs is not installed — the tier-content check is skipped"
        @test_skip false
    end
end

# A pose-error cell already in the results cache is REUSED: `write_pose_results`
# asks the cache for every cell before it evaluates anything, so a rerun with
# every cell present evaluates nothing (the sources below have no detections:
# an evaluation throws). It once ran `evaluate_dataset` first and only then
# asked the cache, so every rerun recomputed every cell.
@testitem "write_pose_results reuses cached cells without evaluating" setup = [BBEval] begin
    using .BBEval.BlobBoardsEval: Sources, PoseRun, write_pose_results, BOARD_TYPES
    using VisualGeometryToolkitEval.Experiments: cached_result, run_key
    const BBE = BBEval.BlobBoardsEval
    mktempdir() do tmp
        withenv("VGTK_RESULTS_ROOT" => mkpath(joinpath(tmp, "results"))) do
            none = joinpath(tmp, "nothing-here")
            layout = Dict{String, Any}("dataset" => "260527", "layout" => ["260527", "evaluation"])
            src = Sources(layout, "260527", none, none, Dict(bt.name => PoseRun(none, none, bt.name) for bt in BOARD_TYPES),
                Dict("records" => "r", "baselines" => "b", "bb" => "k", "occlusion" => "o", "bb_noflip" => "n"))
            # nothing cached: the cells are evaluated, and the evaluation reads detections that are not there
            @test_throws Exception write_pose_results(src)
            # every cell cached (through the cache itself, under the cell's key): nothing is evaluated
            cache(bt) = last(cached_result(c -> Dict{String, Any}("cached" => true), "pose",
                BBE._cell_config(src, bt.system, "none"), BBE.results_dir(src); key = run_key))
            files = [cache(bt) for bt in BOARD_TYPES]
            @test allunique(files)
            @test sort(write_pose_results(src)) == sort(files)
            # one cell missing: it is evaluated again (and the evaluation throws here)
            rm(first(files))
            @test_throws Exception write_pose_results(src)
        end
    end
end
