#=
make_adapted_panels.jl — one figure per detected board: the adapted inlier
features overlaid on the FULL image.

Writes ONE FIGURE PER BOARD (not a grid), so the paper can arrange them as
subfigures and caption them individually.

What each panel shows, and why both layers are needed:

Styling follows Fig. 1 (make_bmvc2026_intro.jl): the COLOR image, the same
`distinguishable_colors` palette seeded against white and black, and the same
2.0/1.0 stroke weights. The yellow overlay is drawn LIGHTER than the board
outline so it supports the registration rather than competing with it.

  * yellow ellipses + a `+` at each centre — the ADAPTED features that were
    matched to this board, at their fitted elliptical shape. Not raw
    detections: these are the frames after affine adaptation, so they elongate
    with the board's foreshortening.
  * the board's world pattern projected through the recovered homography, in
    the board's own color.

Agreement between the two IS the registration, per correspondence. A
misregistration shows as the projected pattern drifting off the yellow
ellipses, which a reader can check blob by blob rather than take on trust.

The image is shown IN FULL rather than cropped to each board: how much of the
frame a board occupies is part of what the figure demonstrates, and cropping
normalises exactly that away.

Inliers come from `targets(scored)` — the image side of each correspondence —
NOT from a second `extract_image_features` call indexed by `det_id`. Adapt is
nondeterministic on CUDA, so a separate extraction need not index-align with the
features `find_boards` actually matched, and the ellipses could be silently
paired with the wrong detections.

CairoMakie, not GLMakie: this only writes files, and a CPU renderer avoids the
OpenGL context that aborts inside libnvidia-glcore when `find_boards` is holding
~18 GB of device memory in the same process.

OUTPUT = WHAT THE PAPER AND POSTER INCLUDE. Fig. occexamples (top row)
\includegraphics `images/adapted/original.jpg` and
`images/adapted/adapted_<uid>.pdf` for the four boards in PAPER_UIDS, so that is
what this writes: the source photo (byte copy) and, per paper board, the
layers of scripts/overlaid_figure.jl — `adapted_<uid>_photo.jpg` (the photo),
`adapted_<uid>.pdf` (outlines, blob circles, adapted ellipses and `+` markers
only, a transparent page the size of the photo; composited with `\overlaid`,
papers/overlaid.tex) and `adapted_<uid>.jpg` (the full panel, a plain backup).

Run: pixi run julia --project=bmvc2026 bmvc2026/scripts/make_adapted_panels.jl
(GPU; the gallery cache lives under results_root()/galleries/.)

Every parameter — source photo, gallery, detector settings, render sizes and
strokes, the paper's boards, output names — is in the config of record,
configs/figures/adapted_panels.toml (`--config=<path>` overrides; `uids = []`
writes every detected board). Env:
BB_PAPER_IMAGES (images root, default papers/images — see src/paper_assets.jl;
set it to review a regeneration before it replaces the submitted panels).
Each run writes its record under plotsdir("records").
=#
# reproduce: writes papers/images/adapted/adapted_*{.pdf,.jpg,_photo.jpg}
# reproduce: writes papers/images/adapted/original.jpg
# reproduce: reads configs/figures/adapted_panels.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

using BlobBoards
using CairoMakie; CairoMakie.activate!()
# using CUDA
# import cuDNN
# using CUDA: CUDABackend
using FileIO: load as load_image
using Colors: Gray, distinguishable_colors, RGB
using StaticArrays: SMatrix
using GeometryBasics: Point2, Point2f, Rect
using CoordinateTransformations: PerspectiveMap
using VisualGeometryToolkitCore: is_map_inlier, ellipses!, HomographyMap, feature_frame, centroid,
    ustrip, targets, sources, EllipticFrame

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: data_root, results_root, paper_images_dir, figure_config, write_figure_records, network_path, load_config
include(joinpath(@__DIR__, "overlaid_figure.jl"))
using VisualGeometryToolkitEval.Experiments: Stage, run_stages
using DrWatson: projectdir

const BB = BlobBoards
const DEVICE = BB.KA.CPU()
const CFG, CFG_PATH = figure_config("adapted_panels"; default = projectdir("configs", "figures", "adapted_panels.toml"))
const INP, FB, RND, OUT = CFG["inputs"], CFG["find_boards"], CFG["render"], CFG["output"]
const ROOT = data_root(; tiers = String.(CFG["experiment"]["tiers"]))
const IMG_PATH = joinpath(ROOT, INP["image"])
const PHOTO_PX = RND["photo_px"]
# papers/images (not plotsdir()) because the PAPER reads it here, through
# paper_images_dir (src/paper_assets.jl), whose BB_PAPER_IMAGES override is how
# a regeneration is reviewed before it replaces the submitted panels.
const OUT_DIR = paper_images_dir(OUT["dir"])
# The boards Fig. occexamples shows, in its column order; empty = every board.
const PANEL_UIDS = isempty(OUT["uids"]) ? nothing : String.(OUT["uids"])
_cutoff(v) = v == "default" ? Float32(BB.DEFAULT_CUTOFF) : Float32(v)
# "package" = the package's default network (configs/default.toml), else a run
# config under configs/ naming it.
_network_path(v) = v == "package" ? descriptor_network_path() : network_path(load_config(v))
const GALLERY = joinpath(results_root(), "galleries", INP["gallery"])   # a cache, never in the input tree

# One panel: the photo layer (resized to `photo_px` wide when > 0, drawn over
# the full-resolution extent), the board's inlier frames `fr0` and its
# homography `model` (full-resolution pixels), its pattern and colour.
function _panel(img_color, fr0, model, bundle, col)
    fig, ax, photo = photo_figure(img_color; width = RND["figure_width"], photo_px = PHOTO_PX)
    ctr = [Point2f(Float32.(ustrip.(centroid(f)))...) for f in fr0]
    # Thick strokes: at FULL-frame scale a small distant board's ellipses
    # are only a few pixels across, and a hairline outline disappears in
    # print. The `+` marks the feature point itself, which stays visible
    # even where the ellipse is too small to read.
    ellipses!(
        ax, fr0; cutoff = Float64(BB.DEFAULT_CUTOFF),
        strokecolor = Symbol(RND["inlier_color"]), strokewidth = RND["inlier_strokewidth"]
    )
    Makie.scatter!(
        ax, ctr; marker = :+, color = Symbol(RND["inlier_color"]),
        markersize = RND["marker_size"], strokewidth = 0
    )
    proj = PerspectiveMap() ∘ HomographyMap(SMatrix{3, 3, Float64}(model))
    # FULL pattern, as in Fig. 1: the dense tracery is what makes a board
    # read as a board. Drawing only the matched subset was tried and left
    # the partly-occluded boards looking empty.
    board!(
        ax, proj, Rect(bundle, :world), BB.blobs(bundle, :world);
        cutoff = BB.cutoff(bundle), outline_color = col,
        blob_color = col, outline_width = RND["outline_width"], blob_width = RND["blob_width"],
        show_frame = false
    )
    return fig, photo
end

function main()
    # Detect on gray (find_boards expects it), DRAW on color -- the same split
    # make_bmvc2026_intro.jl uses. Rendering the gray copy is what made these
    # panels look washed out next to Fig. 1.
    img_color = load_image(IMG_PATH)
    img = Gray.(img_color)
    net = DescriptorNetwork(_network_path(INP["network"]), allow_cpu_fallback = true)
    mkpath(dirname(GALLERY))
    gallery = BoardGallery(GALLERY, net)
    board_paths = filter(endswith(".json"), readdir(joinpath(ROOT, INP["boards_dir"]); join = true))
    bfd = Dict(uid => f for (uid, f) in insert!(gallery, board_paths, net; device = DEVICE))
    bundles = Dict(
        let (b, _, ch) = BB.load_bundle(bp)
                uid_from_hash(ch) => b
        end for bp in board_paths
    )
    FB["geometry"] == "homography" || error("adapted panels: only the homography path is implemented")
    cfg = FindBoardsConfig(
        geometry = HomographyGeometry(),
        detect = DetectConfig(
            polarity = Symbol(FB["polarity"]), peak_threshold = FB["peak_threshold"],
            edge_threshold = FB["edge_threshold"], nms = EllipseNms(cutoff = _cutoff(FB["nms_cutoff"]))
        )
    )
    acc = find_boards(bfd, net, img; config = cfg, device = DEVICE, board_bundles = bundles)


    mkpath(OUT_DIR)
    # The source frame is the first cell of the figure's top row, so it has to
    # be produced here too -- it was a hand copy, which meant the figure could
    # not be rebuilt from the script alone.
    original = joinpath(OUT_DIR, OUT["original"] * last(splitext(IMG_PATH)))
    # follow_symlinks: the data view links into the artifact store; copy the file, not the link
    cp(IMG_PATH, original; force = true, follow_symlinks = true)
    chmod(original, 0o644)   # not the artifact store's read-only, executable mode
    outputs = [original]
    cols = distinguishable_colors(length(acc), [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed = true)
    found = [first(a) for a in acc]
    if PANEL_UIDS !== nothing
        missing_uids = setdiff(PANEL_UIDS, found)
        isempty(missing_uids) || error("boards $(missing_uids) of the paper's panel set were not detected (found $(found))")
    end
    # Colors are assigned over ALL detections (in detection order), so a board
    # keeps its color whether or not every panel is written.
    for (i, (uid, model, _, scored)) in enumerate(acc)
        PANEL_UIDS === nothing || uid in PANEL_UIDS || continue
        keep = is_map_inlier.(scored.weights)
        fr0 = feature_frame.(targets(scored))[keep]
        bundle = bundles[uid]

        fig, photo = _panel(img_color, fr0, model, bundle, cols[i])
        base = joinpath(OUT_DIR, replace(OUT["name"], "{uid}" => uid))
        written = save_overlaid(base, fig, photo; source = IMG_PATH, quality = CFG["jpeg"]["quality"], pt_per_unit = RND["pt_per_unit"])
        append!(outputs, written)
        @info "panel" uid n_inliers = count(keep) written mb = [round(filesize(o) / 2^20; digits = 1) for o in written]
    end
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_adapted_panels.jl", outputs,
        inputs = [IMG_PATH, GALLERY, joinpath(ROOT, INP["boards_dir"])])
    return nothing
end

run_stages("make_adapted_panels", [Stage(:render, "original.jpg + one panel per paper board (GPU)", () -> false, main)])
