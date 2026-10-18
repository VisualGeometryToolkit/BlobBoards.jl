import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIo as VgtkIo
using Test
using BlobBoards
using BlobBoards: annotate_blobs, canonicalize, FindBoardsConfig,
    DescriptorNetwork, BoardGallery, load_bundle, uid_from_hash,
    DEFAULT_CUTOFF,
    plot_boards, plot_board_poses, plot_tracks, plot_annotations, load_cspond_sets
using CairoMakie: CairoMakie   # the Makie viewers render headless here
using Colors: Gray
using CUDA: CUDA, CUDABackend

# Base swallows extension load failures (it only logs them), and every CUDA
# hook has a generic KA.Backend fallback — so a broken BlobBoardsCUDAExt would
# leave this whole file green. Assert the extension actually loaded.
@testset "BlobBoardsCUDAExt loads" begin
    @test Base.get_extension(BlobBoards, :BlobBoardsCUDAExt) !== nothing
end
import cuDNN
import VideoIO
import VisualGeometryToolkitIo
using FileIO: load as load_image

using VisualGeometryToolkitCore: CameraModel, EllipticFrame, Point2, Size2
import VisualGeometryToolkitFeatures   # tilt-schedule guard reaches _imas_tilts_25

# =============================================================================
# Data — the package's OWN example artifacts (Artifacts.toml, Zenodo):
# the 50 board definitions, a calibrated still with its vgio/cameras/v3 store,
# and a calibrated handheld clip with its own. Fetched lazily on first use, so
# these tests run wherever the artifacts can be downloaded — no local dataset,
# no LocalPreferences. Until 2026-09-07 the inputs were `data_dir()` paths that
# existed only on the author's primary dataset copies; on every other machine
# (and in every fresh-clone gate) the testsets reported 0 tests and a green summary.
# =============================================================================

const _EXAMPLE = try
    (;
        boards = BlobBoards.example_data(:boards),
        still = BlobBoards.example_data(:still),
        video = BlobBoards.example_data(:video),
    )
catch err
    @warn "example artifacts unavailable (offline?) — the data-driven smoke tests SKIP; that is a hole, not a success" exception = err
    nothing
end
_only(dir, pred) = (fs = filter(pred, readdir(dir; join = true)); length(fs) == 1 ? only(fs) : nothing)
const _BOARDS_DIR = _EXAMPLE === nothing ? nothing : _EXAMPLE.boards
# Network from configs/default.toml — the one place a default entry point
# learns which descriptor to load: the iter4 log-polar net, resolved by
# artifact name through the package's Artifacts.toml catalog.
const _MODEL_PATH = descriptor_network_path()
# The gallery is a CACHE (keyed on network hash, patch size, extent, cutoff);
# it is built fresh into a temp dir, never read from a dataset.
const _GALLERY = joinpath(mktempdir(), "board_features.jld2")
const _IMAGE_PATH = _EXAMPLE === nothing ? nothing : _only(_EXAMPLE.still, p -> endswith(uppercase(p), ".JPG"))
const _CAMERA_JSON = _EXAMPLE === nothing ? nothing : joinpath(_EXAMPLE.still, "cameras.json")
const _VIDEO_PATH = _EXAMPLE === nothing ? nothing : _only(_EXAMPLE.video, p -> endswith(lowercase(p), ".mov") || endswith(lowercase(p), ".mp4"))
const _VIDEO_CAMERA_JSON = _EXAMPLE === nothing ? nothing : joinpath(_EXAMPLE.video, "cameras.json")

_all_present(paths...) = all(p !== nothing && ispath(p) for p in paths)

@testset "media_resolution / group_by_resolution" begin
    if !_all_present(_VIDEO_PATH, _IMAGE_PATH)
        @warn "Skipping media_resolution — example artifacts unavailable"
    else
        # BOTH branches must agree on the frame size of the SAME medium — that
        # is what makes the grouping key match the frame the detector, and
        # hence its plan cache, actually sees. Compare `Size2` to `Size2`:
        # VideoIO's `out_frame_size` is (width, height) while `Base.size` is
        # (rows, cols), and `Size2(::AbstractMatrix)` is the VGC intrinsic that
        # performs exactly that reversal. Asserting against a raw `size()`
        # tuple — as this test used to — reintroduces the very ambiguity
        # `media_resolution` returns a `Size2` to eliminate, and fails on a
        # non-square frame for reasons that have nothing to do with the code.
        reader = VideoIO.openvideo(_VIDEO_PATH)
        frame = try
            VideoIO.read(reader)
        finally
            close(reader)
        end
        @test BlobBoards.media_resolution(_VIDEO_PATH) == Size2(frame)
        @test BlobBoards.media_resolution(_IMAGE_PATH) == Size2(load_image(_IMAGE_PATH))

        # groups partition the input, largest group first, order preserved
        paths = [_VIDEO_PATH, _IMAGE_PATH, _VIDEO_PATH]
        g = BlobBoards.group_by_resolution(paths)
        @test sum(length(ps) for (_, ps) in g) == length(paths)
        @test issorted([-length(ps) for (_, ps) in g])
        @test first(g).second == [_VIDEO_PATH, _VIDEO_PATH]   # the 2-member group leads
    end
end

@testset "descriptor arena is reserved at construction" begin
    if !_all_present(_MODEL_PATH) || !CUDA.functional()
        @info "Skipping arena reservation — model or CUDA unavailable"
    else
        GC.gc(); CUDA.reclaim()
        before = CUDA.free_memory()
        net = DescriptorNetwork(_MODEL_PATH)
        after_construct = CUDA.free_memory()

        if net.session.execution_provider !== :cuda
            @info "ONNX CPU execution provider — no device arena to reserve"
        else
            # ORT's `gpu_mem_limit` is a CAP, not a reservation. Without the
            # warm-up in `reserve_workspace!`, constructing a session claims
            # only weights (~20 MiB measured) and the arena does not grow until
            # the first `describe` — by which time VGF's `_chunk_warps` has
            # already budgeted 80% of "free" memory believing the arena was
            # accounted for. Construction must move the needle by the whole
            # working set (~526 MiB at max_batch=64 on this network).
            @test before - after_construct > 100 * 2^20

            # ...and having reserved, real inference must not grow it further.
            # That is precisely what makes `device_free_bytes` truthful for the
            # two elastic consumers. 4x max_batch exercises the chunked path.
            P = BlobBoards.patch_size(net)
            BlobBoards.describe(net, [zeros(Float32, P, P) for _ in 1:(4 * net.max_batch)])
            GC.gc(); CUDA.reclaim()
            @test after_construct - CUDA.free_memory() < 32 * 2^20
        end
    end
end

# Keep the test fast: single image, 2 video frames, default cutoff/patch_size.
const _N_FRAMES = 2

# FULL IMAS-25 tilt schedule (empty tilts → VGF `_imas_tilts_25()`).
# Do NOT reintroduce a reduced schedule here: the historical 17-tilt
# workaround dated from before the ASIFT driver chunked the warp schedule
# to fit device memory (VGF asift/driver.jl `_chunk_warps`) — the full 25
# tilts run on any card now, just in more chunks. The smoke MUST exercise
# the production schedule; the guard testset below fails if the default
# drifts.
_detect_cfg() = DetectConfig(
    polarity = :dark, peak_threshold = 0.02,
    edge_threshold = 5.0,
    nms = EllipseNms(cutoff = Float32(DEFAULT_CUTOFF))
)

@testset "detect config uses the full IMAS-25 tilt schedule" begin
    # An EMPTY schedule routes to VGF's IMAS-25 default (BB spells it as the
    # empty vector, VGF as `nothing`; `detect_blobs` translates at the
    # boundary) — assert both halves so a reduced schedule can't sneak back in
    # via either layer.
    @test isempty(_detect_cfg().tilts)
    @test length(VgtkFeatures._imas_tilts_25()) == 25
end

@testset "board rasterization dpi (bare-ustrip density regression)" begin
    # A BARE ustrip of a density yields its base-unit magnitude (px/m now;
    # px/mm ≈ dpi/25.4 under the old (mm, px) basis, when boards rendered at
    # ~47 dpi with off-image canvas
    # coordinates → blank gallery patches → indiscriminate descriptors
    # (2026-07-29, masked for months by the gallery feature cache).
    # `_render_dpi` must use the target-unit form.
    using VisualGeometryToolkitCore: dpi
    @test BlobBoards._render_dpi(1200dpi) == 1200
    @test BlobBoards._render_dpi(300dpi) == 300
    @test_throws Exception BlobBoards._render_dpi(3.0)   # dimensionless is a bug
end

# Cameras are vgio/cameras/v3 stores (the example artifacts ship them that
# way); the store owns the reader.
_load_camera(path) = VgtkIo.load(CameraModel, path)

function _setup_gallery()
    model = DescriptorNetwork(_MODEL_PATH)
    gallery = BoardGallery(_GALLERY, model)
    bpaths = filter(endswith(".json"), readdir(_BOARDS_DIR; join = true))
    dict = Dict(
        uid => feats for (uid, feats) in
            insert!(gallery, bpaths, model; device = CUDABackend(), progress = false)
    )
    bundles = Dict(
        let (b, _, ch) = load_bundle(bp)
                uid_from_hash(ch) => b
        end for bp in bpaths
    )
    return (dict = dict, bundles = bundles, paths = bpaths, model = model)
end

@testset "canonicalize — CPU/GPU interpolant parity" begin
    # Guards the gallery↔query contract: reference patches (CPU path) and
    # detection patches (CUDA path) MUST share one interpolant. A silent
    # cubic/bilinear split here degraded matching for every shipped gallery
    # before 2026-07-29. Tolerance-based (Float64 vs Float32 coordinate
    # arithmetic); bit-parity across devices is the unified-warp goal.
    img = [
        Gray{Float32}(0.5f0 + 0.4f0 * sin(0.11f0 * i) * cos(0.07f0 * j))
            for i in 1:256, j in 1:256
    ]
    frames = [
        EllipticFrame(Point2(96.0, 112.0), 6.0),
        EllipticFrame(Point2(180.0, 64.0), 9.0),
    ]
    cpu = canonicalize(
        frames, img; patch_size = 32, inner_extent = 2.0,
        outer_extent = 24.0
    )
    if CUDA.functional()
        gpu = canonicalize(
            frames, img; patch_size = 32, inner_extent = 2.0,
            outer_extent = 24.0, device = CUDABackend()
        )
        for (pc, pg) in zip(cpu, gpu)
            d = abs.(Float32.(pc) .- Float32.(pg))
            @test maximum(d) < 2.0f-2
            @test sum(d) / length(d) < 1.0f-3
        end
    else
        @info "CUDA unavailable — CPU/GPU canonicalize parity not exercised"
    end
end

@testset "annotate_blobs — smoke" begin
    if !_all_present(
            _BOARDS_DIR, _MODEL_PATH, _IMAGE_PATH,
            _CAMERA_JSON, _VIDEO_PATH, _VIDEO_CAMERA_JSON
        )
        @warn "Skipping annotate_blobs smoke test — example artifacts unavailable"
        return
    end

    g = _setup_gallery()
    img = Gray.(load_image(_IMAGE_PATH))
    cam = _load_camera(_CAMERA_JSON)          # the still's camera
    vcam = _load_camera(_VIDEO_CAMERA_JSON)   # the clip's — a different sensor/lens
    # Adaptation is GPU-only — the annotate pipeline needs the CUDA backend.
    backend = CUDABackend()
    fb_cfg = FindBoardsConfig(detect = _detect_cfg())
    patch_size = 64

    @testset "image — uncalibrated (homography)" begin
        r = annotate_blobs(
            g.dict, g.model, img;
            board_bundles = g.bundles,
            config = fb_cfg, patch_size, device = backend
        )
        @test keys(r) === (:board_uids, :blob_ids, :patches, :confuser_patches)
        @test length(r.board_uids) == length(r.blob_ids) == length(r.patches)
        # Per-patch shape check when any were produced:
        if !isempty(r.patches)
            @test all(size(p) == (patch_size, patch_size) for p in r.patches)
        end
    end

    @testset "image — calibrated (P3P)" begin
        r = annotate_blobs(
            g.dict, g.model, img;
            board_bundles = g.bundles,
            config = FindBoardsConfig(geometry = PoseGeometry(cam), detect = _detect_cfg()), patch_size, device = backend
        )
        @test keys(r) === (:board_uids, :blob_ids, :patches, :confuser_patches)
        @test length(r.board_uids) == length(r.blob_ids) == length(r.patches)
        if !isempty(r.patches)
            @test all(size(p) == (patch_size, patch_size) for p in r.patches)
        end
        # The calibrated path should find at least as many labeled patches on
        # this image as the uncalibrated path (same data, stronger estimator).
        # Don't assert equality — outlier halfwidths differ and the bound is
        # directional, not tight.

        # The viewers, on real find_boards output (they had no test at all
        # until 2026-09): the accepted-board grid and the pose overlay.
        results = find_boards(
            g.dict, g.model, img;
            board_bundles = g.bundles,
            config = FindBoardsConfig(geometry = PoseGeometry(cam), detect = _detect_cfg()),
            device = backend
        )
        @test !isempty(results)
        poses = [(r[1], r[2]) for r in results]
        @test plot_boards(cam, results, g.bundles, img) isa CairoMakie.Makie.Figure
        @test plot_board_poses(cam, poses, g.bundles, img) isa CairoMakie.Makie.Figure
        # the retired orders bind no method: image first, and the image second
        @test !applicable(plot_boards, img, results, g.bundles, cam)
        @test !applicable(plot_board_poses, img, poses, g.bundles, cam)
        @test !applicable(plot_boards, cam, img, results, g.bundles)
        @test !applicable(plot_board_poses, cam, img, poses, g.bundles)
        @test !applicable(plot_board_poses, cam, img, Dict(poses), g.bundles)
    end

    # (The former BLOBBOARDS_RUN_VIDEO_SMOKE gate is gone: the VGF ASIFT
    # boundary-mirror fault at small pyramid levels was fixed in VGF 0.3.3 —
    # single-fold `_mirror_idx` returned negative indices for levels with a
    # dimension in (32, 38].)
    for (label, geom) in (("uncalibrated", HomographyGeometry()), ("calibrated", PoseGeometry(vcam)))
        @testset "video — $label (n_frames = $_N_FRAMES)" begin
            outdir = mktempdir()
            try
                annotate_blobs(
                    g.dict, g.model, _VIDEO_PATH;
                    output_dir = outdir,
                    board_bundles = g.bundles,
                    board_paths = g.paths,
                    config = FindBoardsConfig(geometry = geom, detect = _detect_cfg()),
                    patch_size,
                    device = backend,
                    n_frames = _N_FRAMES
                )
                # Per-frame JLD2 files land directly in `outdir`. Smoke
                # tolerates one fewer than requested (stride arithmetic
                # can off-by-one on short videos).
                frame_files = filter(
                    f -> endswith(f, ".jld2") && f != "_config.jld2",
                    readdir(outdir)
                )
                @test 1 <= length(frame_files) <= _N_FRAMES
                # the annotation viewer, on the directory just written
                sets = load_cspond_sets(outdir)
                @test !isempty(sets)
                @test plot_annotations(sets; max_blobs = 3, max_obs = 2) isa CairoMakie.Makie.Figure
            finally
                isdir(outdir) && rm(outdir; recursive = true)
            end
        end
    end
end

@testset "track_boards — smoke (warm-start tracking)" begin
    if !_all_present(_BOARDS_DIR, _MODEL_PATH, _VIDEO_PATH)
        @warn "Skipping track_boards smoke — example artifacts unavailable"
        return
    end
    g = _setup_gallery()
    v = VideoIO.openvideo(_VIDEO_PATH)
    frames = [Gray{Float32}.(read(v)) for _ in 1:6]
    close(v)
    seen = Int[]
    trace = TrackTrace(source = _VIDEO_PATH)
    trks = track_boards(
        g.dict, g.model, frames;
        board_bundles = g.bundles,
        config = FindBoardsConfig(
            geometry = HomographyGeometry(),
            detect = _detect_cfg()
        ),
        device = CUDABackend(),
        # one sink feeds both the live view and the trace
        on_frame = f -> (push!(seen, f.record.frame_idx); push!(trace, f))
    )
    @test !isempty(trks)
    ds = first(values(trks))
    @test first(ds).source === :detect
    # consecutive frames of a slow handheld clip: warm bets must win
    @test count(d -> d.source === :track, ds) >= 3
    @test all(d -> d.n_inliers >= 4, ds)

    # ---- capture + live sink + bundle round-trip -------------------------
    @test seen == collect(1:length(frames))          # live sink saw every frame
    @test length(trace) == length(frames)
    @test !isempty(trace.config) && !isempty(trace.network_hash)   # configuration of record stamped
    warm_recs = [r for r in trace if any(b -> b.source === :track, r.boards)]
    @test !isempty(warm_recs)
    wb = first(first(warm_recs).boards)
    @test !isempty(wb.associations) && !isempty(wb.predicted)   # bet diagnostics captured
    @test !isempty(first(trace.records).detections)
    # the trace viewer (first test of it): one triptych frame, headless
    @test plot_tracks(trace; frame = 2) isa CairoMakie.Makie.Figure
    tdir = mktempdir()
    try
        save_trace(joinpath(tdir, "t.trace"), trace)
        tr2 = load_trace(joinpath(tdir, "t.trace"))
        @test tr2 isa TrackTrace
        @test length(tr2) == length(trace)
        @test [r.frame_idx for r in tr2] == [r.frame_idx for r in trace]
        r1, r2 = trace.records[end], tr2.records[end]
        @test [b.uid for b in r1.boards] == [b.uid for b in r2.boards]
        @test r1.detections == r2.detections
        @test [b.associations for b in r1.boards] == [b.associations for b in r2.boards]
    finally
        rm(tdir; recursive = true)
    end
end

# The pose arm under `undistortion = :none` keeps the FULL camera as its
# estimation camera, which has no plane homography: `track_boards` projects the
# blob centres through the `Camera` itself. Until the `Camera` method of
# `_project_blob_centers` existed this died in a MethodError on the first
# accepted board.
@testset "track_boards — pose arm, full camera (undistortion = :none)" begin
    if !_all_present(_BOARDS_DIR, _MODEL_PATH, _VIDEO_PATH, _VIDEO_CAMERA_JSON)
        @warn "Skipping track_boards pose-arm smoke — example artifacts unavailable"
        return
    end
    g = _setup_gallery()
    v = VideoIO.openvideo(_VIDEO_PATH)
    frames = [Gray{Float32}.(read(v)) for _ in 1:4]
    close(v)
    cam = _load_camera(_VIDEO_CAMERA_JSON)
    trks = track_boards(
        g.dict, g.model, frames;
        board_bundles = g.bundles,
        config = FindBoardsConfig(geometry = PoseGeometry(cam), detect = _detect_cfg()),
        device = CUDABackend(),
    )
    @test !isempty(trks)
    ds = first(values(trks))
    @test all(d -> d.model isa BlobBoards.Extrinsics3, ds)
    @test count(d -> d.source === :track, ds) >= 1
end
