#=
make_adapted_panels.jl — one figure per detected board: the adapted inlier
features overlaid on the FULL image.

Writes ONE PNG PER BOARD (not a grid), so the paper can arrange them as
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

OUTPUT = WHAT THE PAPER INCLUDES. Fig. occexamples (top row) \includegraphics
`images/adapted/original.jpg` and `images/adapted/adapted_<uid>.png` for the
four boards in PAPER_UIDS, so that is what this writes: the source photo
(byte copy) and one 2400x3200 PNG per paper board (the 1600-unit-wide figure
at px_per_unit = 1.5). The vector PDF (~23 MB each, gitignored) was the old
output and is now opt-in (BB_PANEL_VECTOR=1); the PNGs used to be converted
from it by hand, which is how the script's output and the paper's input came
to differ.

Run: pixi run julia --project=bmvc2026 bmvc2026/scripts/make_adapted_panels.jl
(GPU; the gallery cache lives under results_root()/galleries/.)

Every parameter — source photo, gallery, detector settings, render sizes and
strokes, the paper's boards, output names — is in the config of record,
configs/figures/adapted_panels.toml (`--config=<path>` overrides; `uids = []`
writes every detected board, `write_vector = true` adds the PDF masters). Env:
BB_PAPER_IMAGES (images root, default papers/images — see src/paper_assets.jl;
set it to review a regeneration before it replaces the submitted panels).
Each run writes its record under plotsdir("records").
=#
# reproduce: writes papers/images/adapted/adapted_*.png
# reproduce: writes papers/images/adapted/original.jpg
# reproduce: reads configs/figures/adapted_panels.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

using BlobBoards
using CairoMakie; CairoMakie.activate!()
using CUDA
import cuDNN
using CUDA: CUDABackend
using FileIO: load as load_image
using Colors: Gray, distinguishable_colors, RGB
using Images: imresize
using StaticArrays: SMatrix
using GeometryBasics: Point2, Point2f, Rect
using CoordinateTransformations: PerspectiveMap
using VisualGeometryToolkitCore: is_map_inlier, imshow, ellipses!, HomographyMap, feature_frame, centroid,
    ustrip, targets, sources, EllipticFrame

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: data_root, results_root, paper_images_dir, figure_config, write_figure_records, network_path, load_config
using VisualGeometryToolkitEval.Experiments: Stage, run_stages
using DrWatson: projectdir

const BB = BlobBoards
const DEVICE = CUDABackend()
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
const WRITE_VECTOR = RND["write_vector"]
const PX_PER_UNIT = RND["px_per_unit"]
_cutoff(v) = v == "default" ? Float32(BB.DEFAULT_CUTOFF) : Float32(v)
# "package" = the package's default network (configs/default.toml), else a run
# config under configs/ naming it.
_network_path(v) = v == "package" ? descriptor_network_path() : network_path(load_config(v))
const GALLERY = joinpath(results_root(), "galleries", INP["gallery"])   # a cache, never in the input tree

function main()
    # Detect on gray (find_boards expects it), DRAW on color -- the same split
    # make_bmvc2026_intro.jl uses. Rendering the gray copy is what made these
    # panels look washed out next to Fig. 1.
    img_color = load_image(IMG_PATH)
    img = Gray.(img_color)
    net = DescriptorNetwork(_network_path(INP["network"]))
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

    # VECTOR output with a DOWNSAMPLED photo. The overlay must stay vector --
    # rasterising it is what produced the pixelation -- but CairoMakie embeds
    # the background losslessly in every panel, and at full resolution that is
    # 37 MB per panel (669 MB for 25). A subfigure at a third of \textwidth
    # needs ~1350 px at 600 dpi against a 3816 px source, so the photo can
    # shrink ~2.8x with nothing visible lost while the paths stay exact.
    #
    # Scaling the image by `sc` requires scaling the geometry with it:
    # world->image homography by diag(sc, sc, 1), and each adapted frame by
    # centre -> sc*centre, S -> sc*S (verified: doubling S doubles the rendered
    # radii, S*sc^2 quadruples them).
    sc = min(1.0, PHOTO_PX / size(img_color, 2))
    img_plot = sc < 1 ? imresize(img_color; ratio = sc) : img_color
    Ssc = SMatrix{3, 3, Float64}(sc, 0, 0, 0, sc, 0, 0, 0, 1)

    mkpath(OUT_DIR)
    # The source frame is the first cell of the figure's top row, so it has to
    # be produced here too -- it was a hand copy, which meant the figure could
    # not be rebuilt from the script alone.
    original = joinpath(OUT_DIR, OUT["original"] * last(splitext(IMG_PATH)))
    cp(IMG_PATH, original; force = true)
    outputs = [original]
    cols = distinguishable_colors(length(acc), [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed = true)
    h, w = size(img_plot)
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
        fr = [
            EllipticFrame(
                    Point2(sc .* Float64.(ustrip.(centroid(f)))...),
                    SMatrix{2, 2, Float64}(sc .* f.S)
                ) for f in fr0
        ]
        ctr = [Point2f(Float32.(ustrip.(centroid(f)))...) for f in fr]

        # Aspect of the ORIGINAL photo, not of the resized copy: the resize
        # rounds its height, which put the panel one pixel off the submitted
        # 2400x3200 (2026-10-08 review).
        fw = RND["figure_width"]
        fig = Makie.Figure(size = (fw, round(Int, fw * size(img_color, 1) / size(img_color, 2))))
        ax, _ = imshow(fig[1, 1], img_plot)
        # Thick strokes: at FULL-frame scale a small distant board's ellipses
        # are only a few pixels across, and a hairline outline disappears in
        # print. The `+` marks the feature point itself, which stays visible
        # even where the ellipse is too small to read.
        ellipses!(
            ax, fr; cutoff = Float64(BB.DEFAULT_CUTOFF),
            strokecolor = Symbol(RND["inlier_color"]), strokewidth = RND["inlier_strokewidth"]
        )
        Makie.scatter!(
            ax, ctr; marker = :+, color = Symbol(RND["inlier_color"]),
            markersize = RND["marker_size"], strokewidth = 0
        )
        bundle = bundles[uid]
        proj = PerspectiveMap() ∘ HomographyMap(Ssc * SMatrix{3, 3, Float64}(model))
        # FULL pattern, as in Fig. 1: the dense tracery is what makes a board
        # read as a board. Drawing only the matched subset was tried and left
        # the partly-occluded boards looking empty.
        board!(
            ax, proj, Rect(bundle, :world), BB.blobs(bundle, :world);
            cutoff = BB.cutoff(bundle), outline_color = cols[i],
            blob_color = cols[i], outline_width = RND["outline_width"], blob_width = RND["blob_width"],
            show_frame = false
        )
        Makie.hidedecorations!(ax); Makie.hidespines!(ax)

        out = joinpath(OUT_DIR, replace(OUT["name"], "{uid}" => uid))
        Makie.save(out, fig; px_per_unit = PX_PER_UNIT)
        push!(outputs, out)
        if WRITE_VECTOR
            # Opt-in VECTOR master: only the photograph is raster; outlines,
            # blob circles, adapted ellipses and `+` markers stay paths.
            Makie.save(joinpath(OUT_DIR, "adapted_$(uid).pdf"), fig)
        end
        @info "panel" uid n_inliers = count(keep) path = out
    end
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_adapted_panels.jl", outputs,
        inputs = [IMG_PATH, GALLERY, joinpath(ROOT, INP["boards_dir"])])
    return nothing
end

run_stages("make_adapted_panels", [Stage(:render, "original.jpg + one panel per paper board (GPU)", () -> false, main)])
