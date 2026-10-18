#=
import VisualGeometryToolkitIo as VgtkIo
make_pipeline_panels.jl — Fig. `fig:pipeline` of the BMVC paper: the four
stages of the detection-to-pose pipeline on ONE detected board.

# reproduce: writes papers/images/bb_pipe_detect.pdf
# reproduce: writes papers/images/bb_pipe_adapt.pdf
# reproduce: writes papers/images/bb_pipe_logpolar.pdf
# reproduce: writes papers/images/bb_pipe_register.pdf

The frame, board and crop are those of the submitted panels (identified
2026-10-08 by registering board 6b0 in the submitted panel rasters and in
DSC02800: a similarity, see configs/figures/pipeline_panels.toml); the
overlays are today's detector, adaptation and registration.

  (a) detect    raw affine detections from the ASIFT scale space (cyan)
  (b) adapt     the IRLS-adapted ellipses (magenta), except the twelve selected
                pose inliers, each in its own colour (`:Paired_12`)
  (c) logpolar  those twelve inliers canonicalized to 64x64 log-polar patches
                (the descriptor network's input), each bordered in its colour
  (d) register  the board's reference pattern projected under the recovered
                pose (yellow), reference blob centres in green at opacity equal
                to their posterior inlier weight

Everything runs through `BlobBoards.trace_pipeline` — detect -> adapt ->
filter once, and registration with `find_boards` under ONE `FindBoardsConfig`,
the evaluation's run config of record (configs/bmvc2026_260527.toml, read by
reference) — so the four panels show one pipeline run of the evaluated method.

Source frame (all in configs/figures/pipeline_panels.toml): DSC02800 of the
Sony FX3 + Tamron 20 mm f/8 corridor session (a descriptor-training still,
primary copy `blobboard_training/images/`), board 6b0, shown on the image
UNDISTORTED with the session calibration, cropped to rows 612-1111 / cols
126-500 — the submitted panels' frame, board and crop.

Selection rules (deterministic given the features):
  * the twelve highlighted inliers: each MAP inlier's reference blob
    projected through the pinhole pose and paired with the nearest adapted
    frame (within `highlight_match_px`); the twelve with the largest ellipse
    area sqrt(det S);
  * colours: `Makie.to_colormap(:Paired_12)` in that order;
  * RANSAC is seeded by the config (`ransac_seed`), so the registration is
    bit-reproducible given the features; CUDA adaptation is not bitwise
    deterministic, so ellipse parameters may differ in the last digits.

Data tiers: `bb_pipeline_figure` (the frame and its camera) and `bb_assets`
(boards, network); the gallery is a cache (tempdir, or results_root()/galleries/).

Run (GPU; CairoMakie, headless):
    CUDA_VISIBLE_DEVICES=0 julia --project=bmvc2026 bmvc2026/scripts/make_pipeline_panels.jl

Every parameter — the run config, frame, board, camera, gallery (a cache: point
it at a writable copy when the dataset is read-only), panel size, crop padding,
highlight count, patch size, palette, output names — is in the config of
record, configs/figures/pipeline_panels.toml (`--config=<path>` overrides).
Env: BB_PAPER_IMAGES (output root; see src/paper_assets.jl). Each run writes
its record under plotsdir("records").
=#
# reproduce: reads configs/figures/pipeline_panels.toml

using DrWatson
@quickactivate
using DrWatson: projectdir
using BlobBoards
using BlobBoards: trace_pipeline, PoseGeometry, DetectConfig, EllipseNms, descriptor_network_path, load_bundle, cutoff, blobs, is_map_inlier,
    canonicalize, DEFAULT_OUTER_EXTENT
using CairoMakie; CairoMakie.activate!()
using CUDA
import cuDNN
using CUDA: CUDABackend
using Accessors: @set
using FileIO: load as load_image
using Colors: Gray, RGB, RGBA
using StaticArrays: SMatrix, SVector
using GeometryBasics: Point2, Point2f, Rect
using LinearAlgebra: det, norm, eigvals, Symmetric
using Statistics: mean
using CoordinateTransformations: PerspectiveMap
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, imshow, ellipses!, HomographyMap, feature_frame,
    centroid, ustrip, targets, sources, EllipticFrame, CameraModel, Pinhole
import VisualGeometryToolkitIo
using VisualGeometryToolkitEval.Experiments: Stage, run_stages

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: data_root, paper_images_dir, figure_config, write_figure_records, camera_of_record, gallery_path
const BlobBoardsEval = Main.BlobBoardsEval

const CFG, CFG_PATH = figure_config("pipeline_panels"; default = projectdir("configs", "figures", "pipeline_panels.toml"))
const INP, RND = CFG["inputs"], CFG["render"]
const CONFIG_PATH = projectdir("configs", INP["run_config"])
const RUN = read_run_config(CONFIG_PATH)
# Two sources: "data_root" — a still anywhere under data_root() (the
# descriptor-training stills live there) — or "dataset", a frame of
# <dataset>/images/bb. The descriptor network is the run config of record's.
const SOURCE = INP["source"]
if SOURCE == "data_root"
    # a still anywhere under data_root(): image, camera and boards are
    # data_root()-relative (the descriptor-training stills live here)
    const ROOT = data_root(; tiers = String.(CFG["experiment"]["tiers"]))
    const IMAGE = joinpath(ROOT, INP["image"])
    const CAMERA_JSON = joinpath(ROOT, INP["camera"])
    const BOARDS_DIR = joinpath(ROOT, INP["boards_dir"])
    const NETWORK = joinpath(ROOT, RUN.descriptor["model"])
    const GALLERY = isempty(INP["gallery"]) ? joinpath(tempdir(), "blobboards_pipeline_gallery.jld2") : INP["gallery"]
elseif SOURCE == "dataset"
    const ROOT = data_root(; tiers = String.(CFG["experiment"]["tiers"]))
    const DATASET = joinpath(ROOT, "absolute_pose_experiments", RUN.experiment["dataset"])
    # the config names the network data-root-relative (the frozen form)
    const NETWORK = joinpath(ROOT, RUN.descriptor["model"])
    # the session's camera of record (src/data.jl), never an images-tier file
    const CAMERA_JSON = camera_of_record(DATASET)
    basename(CAMERA_JSON) == INP["camera"] || error("the session camera of record is $(basename(CAMERA_JSON)), the config names $(INP["camera"])")
    const BOARDS_DIR = joinpath(DATASET, INP["boards_dir"])
    const GALLERY = isempty(INP["gallery"]) ? gallery_path(RUN.config_toml) : INP["gallery"]
    const IMAGE = joinpath(DATASET, "images", "bb", INP["image"])
else
    error("pipeline_panels: unknown inputs.source $(repr(SOURCE))")
end
const BOARD = INP["board"]

const PANEL_SIZE = Tuple(RND["panel_size"])   # pt; the paper sets each at 0.245\textwidth
const CROP_PAD = RND["crop_pad"]               # crop = outline bbox grown by this per side
const N_HIGHLIGHT = RND["n_highlight"]
const PATCH = RND["patch"]

# Crop window (rows, cols) around a pixel bbox: padded, grown to 3:4 (w:h), clamped.
function crop_window(xmin, xmax, ymin, ymax, H, W)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w, h = (xmax - xmin) * (1 + 2CROP_PAD), (ymax - ymin) * (1 + 2CROP_PAD)
    h = max(h, w * 4 / 3); w = h * 3 / 4
    x0 = clamp(round(Int, cx - w / 2), 1, W); x1 = clamp(round(Int, cx + w / 2), 1, W)
    y0 = clamp(round(Int, cy - h / 2), 1, H); y1 = clamp(round(Int, cy + h / 2), 1, H)
    return y0:y1, x0:x1
end

shift(f::EllipticFrame, dx, dy) = EllipticFrame(Point2(centroid(f)[1] - dx, centroid(f)[2] - dy), f.S)
area(f) = sqrt(abs(det(f.S)))

function panel_axis(img_crop)
    fig = Figure(size = PANEL_SIZE, figure_padding = RND["figure_padding"])
    ax, _ = imshow(fig[1, 1], img_crop)
    hidedecorations!(ax); hidespines!(ax)
    return fig, ax
end
# Pin the axis to the photo: overlays reaching past the crop must not widen it.
# imshow shows the image with y reversed (row 1 at the top); pin in that
# orientation, or limits! would flip it back.
pin!(ax, lims) = (limits!(ax, lims[1], lims[2], lims[4], lims[3]); ax)

function main()
    device = CUDABackend()
    camera = VgtkIo.load(CameraModel, CAMERA_JSON)
    cam_pinhole = CameraModel(camera.intrinsics, Pinhole())
    # Detection, adaptation, matching and verification: the run config of
    # record of the evaluation, BY REFERENCE (the figure and the evaluation
    # cannot diverge); only the camera is the figure's.
    cfg = FindBoardsConfig(CONFIG_PATH; camera)
    net = DescriptorNetwork(NETWORK)
    gallery = BoardGallery(GALLERY, net)
    board_paths = filter(endswith(".json"), readdir(BOARDS_DIR; join = true))
    feats = Dict(uid => f for (uid, f) in insert!(gallery, board_paths, net; device, progress = false))
    bundles = Dict(let (b, _, ch) = load_bundle(bp)
        uid_from_hash(ch) => b
    end for bp in board_paths)
    cuts = unique(Float64(cutoff(b)) for b in values(bundles))
    # one printed cutoff, or (a mixed board set, as the example's) their mean —
    # the same rule find_boards uses to describe the image
    cut = length(cuts) == 1 ? only(cuts) : sum(Float64(cutoff(b)) for b in values(bundles)) / length(bundles)

    img = Gray.(load_image(IMAGE))
    acfg = (@set cfg.adapt.cutoff = cut).adapt
    trace = trace_pipeline(
        camera, img; detect_config = cfg.detect, cutoff = cut, device,
        adapt_arms = [:default => acfg], board_feature_dict = feats,
        board_bundles = bundles, descriptor_network = net, find_config = cfg
    )
    accepted = trace.registration.accepted
    isempty(accepted) && error("no board registered in $(IMAGE)")
    ninl(r) = count(is_map_inlier, r[4].weights)
    @info "registered boards" boards = [(r[1], ninl(r), round.(Int, mean(centroid.(feature_frame.(targets(r[4])))))) for r in accepted]
    pick = isempty(BOARD) ? argmax(ninl, accepted) : only(r for r in accepted if r[1] == BOARD)
    uid, pose, _, scored = pick
    bundle = bundles[uid]

    # pinhole plane homography (detect_img is undistorted) and the board outline
    K = SMatrix{3, 3, Float64}(cam_pinhole.intrinsics.K)
    R = SMatrix{3, 3, Float64}(pose.R); t = SVector{3, Float64}(ustrip.(pose.t))
    Hw = K * hcat(R[:, 1], R[:, 2], t)
    rc = Rect(bundle, :world)
    o = ustrip.(rc.origin); wd = ustrip.(rc.widths)
    corners = [Hw * SVector(o[1] + a * wd[1], o[2] + b * wd[2], 1.0) for (a, b) in ((0, 0), (1, 0), (1, 1), (0, 1))]
    px = [c[1] / c[3] for c in corners]; py = [c[2] / c[3] for c in corners]
    detect_img = Array(trace.detect_img)
    Himg, Wimg = size(detect_img)
    # an explicit crop [row0, col0, rows, cols] (1-based origin) from the config,
    # else the padded 3:4 window around the board outline
    rows, cols = if isempty(RND["crop"])
        crop_window(minimum(px), maximum(px), minimum(py), maximum(py), Himg, Wimg)
    else
        r0, c0, nr, nc = RND["crop"]
        r0:(r0 + nr - 1), c0:(c0 + nc - 1)
    end
    dx, dy = first(cols) - 1, first(rows) - 1
    crop = Gray.(reinterpret(Gray{BlobBoards.N0f8}, detect_img[rows, cols]))
    # drawn: every frame the pipeline produced whose centre lies in the crop
    inside(f) = (c = centroid(f); first(cols) <= c[1] <= last(cols) && first(rows) <= c[2] <= last(rows))

    (_, arm) = only(trace.arms)
    dets = [feature_frame(d) for d in Array(trace.detections)]
    adapted = [feature_frame(f) for f in Array(arm.filtered)]

    # The highlighted inliers, in the DISPLAYED (undistorted) frame. The
    # registration's targets are in the raw image (find_boards measures on the
    # raw pixels and verifies through the full camera), so each MAP inlier's
    # reference blob is projected through the pinhole pose and paired with the
    # nearest adapted frame; the `n_highlight` largest are shown.
    w = scored.weights
    inl = [i for i in eachindex(w) if is_map_inlier(w[i])]
    ref = [Hw * SVector(ustrip.(centroid(feature_frame(sources(scored)[i])))..., 1.0) for i in inl]
    ref = [SVector(r[1] / r[3], r[2] / r[3]) for r in ref]
    actr = [SVector{2, Float64}(ustrip.(centroid(f))...) for f in adapted]
    pairs = Tuple{Int, Float64}[]
    for r in ref
        d, j = findmin(a -> norm(a - r), actr)
        d <= RND["highlight_match_px"] && push!(pairs, (j, w[inl[1]]))
    end
    cand = unique(first.(pairs))
    sort!(cand; by = j -> -area(adapted[j]))
    sel = adapted[cand[1:min(N_HIGHLIGHT, end)]]
    pal = Makie.to_colormap(Symbol(RND["palette"]))

    out = paper_images_dir(); mkpath(out)
    files = String[]

    # (a) detect
    fig, ax = panel_axis(crop); lims = (0.5, size(crop, 2) + 0.5, 0.5, size(crop, 1) + 0.5)
    ellipses!(ax, [shift(f, dx, dy) for f in dets if inside(f)]; cutoff = cut, strokecolor = :cyan, strokewidth = 1.0)
    pin!(ax, lims)
    push!(files, joinpath(out, CFG["output"]["names"][1])); Makie.save(files[end], fig; pt_per_unit = RND["pt_per_unit"])

    # (b) adapt
    fig, ax = panel_axis(crop); lims = (0.5, size(crop, 2) + 0.5, 0.5, size(crop, 1) + 0.5)
    ellipses!(ax, [shift(f, dx, dy) for f in adapted if inside(f)]; cutoff = cut, strokecolor = :magenta, strokewidth = 1.0)
    for (k, f) in enumerate(sel)
        ellipses!(ax, [shift(f, dx, dy)]; cutoff = cut, strokecolor = pal[k], strokewidth = 3.0)
    end
    pin!(ax, lims)
    push!(files, joinpath(out, CFG["output"]["names"][2])); Makie.save(files[end], fig; pt_per_unit = RND["pt_per_unit"])

    # (c) logpolar — the network's input patches, on the adapt image
    adapt_img = Gray{Float32}.(reinterpret(Gray{BlobBoards.N0f8}, detect_img))
    patches = canonicalize(sel, adapt_img; mode = :logpolar, patch_size = PATCH,
        inner_extent = cut, outer_extent = DEFAULT_OUTER_EXTENT, device = BlobBoards.KA.CPU())
    fig = Figure(size = PANEL_SIZE, figure_padding = RND["logpolar_padding"])
    for (k, p) in enumerate(patches)
        r, c = fldmod1(k, 3)
        a = Axis(fig[r, c]; aspect = DataAspect(), spinewidth = 6, leftspinecolor = pal[k],
            rightspinecolor = pal[k], topspinecolor = pal[k], bottomspinecolor = pal[k])
        image!(a, rotr90(Array(p)); interpolate = false, colormap = :grays)
        hidedecorations!(a)
    end
    colgap!(fig.layout, 4); rowgap!(fig.layout, 4)
    push!(files, joinpath(out, CFG["output"]["names"][3])); Makie.save(files[end], fig; pt_per_unit = RND["pt_per_unit"])

    # (d) register — pattern under the recovered pose; reference centres in green
    T = SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, -dx, -dy, 1)
    proj = PerspectiveMap() ∘ HomographyMap(T * Hw)
    fig, ax = panel_axis(crop); lims = (0.5, size(crop, 2) + 0.5, 0.5, size(crop, 1) + 0.5)
    board!(ax, proj, rc, blobs(bundle, :world); cutoff = cutoff(bundle),
        outline_color = :yellow, blob_color = :yellow, outline_width = 3.0, blob_width = 1.0, show_frame = false)
    src = [ustrip.(centroid(feature_frame(s))) for s in sources(scored)]
    pts = [Point2f(proj(Point2(s[1], s[2]))...) for s in src]
    scatter!(ax, pts; marker = :+, markersize = 9, color = [RGBA(0, 1, 0.3, clamp(Float64(x), 0, 1)) for x in w])
    pin!(ax, lims)
    push!(files, joinpath(out, CFG["output"]["names"][4])); Makie.save(files[end], fig; pt_per_unit = RND["pt_per_unit"])

    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_pipeline_panels.jl",
        inputs = [IMAGE, CAMERA_JSON, GALLERY, BOARDS_DIR, CONFIG_PATH], outputs = files)
    @info "pipeline panels" image = basename(IMAGE) uid n_detections = length(dets) n_adapted = length(adapted) n_inliers = length(inl) crop = (rows, cols) files
    return (; image = basename(IMAGE), uid, n_detections = length(dets), n_adapted = length(adapted),
        n_inliers = length(inl), n_accepted = length(accepted), rows, cols, files)
end

const RESULT = Ref{Any}(nothing)
run_stages("make_pipeline_panels", [Stage(:render, "the four pipeline panels (GPU)", () -> false, () -> (RESULT[] = main(); nothing))])
