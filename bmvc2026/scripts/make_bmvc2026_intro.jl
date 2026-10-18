#=
make_bmvc2026_intro.jl — BMVC 2026 intro/teaser figure.

Detect every BlobBoard in the intro image and overlay each one in a distinct
color (outline + blob pattern, transparent fill so the photo shows through) on
the ORIGINAL COLOR image, then save a vector PDF for the paper.

Detection runs on a grayscale copy (find_boards expects Gray); the overlay is
drawn on the color image. Uses the homography path (the montage isn't a single
calibrated camera view), the ONNX descriptor model, and the
`blobboards_original` gallery.

Run: pixi run julia --project=bmvc2026 bmvc2026/scripts/make_bmvc2026_intro.jl

Reproducible: the source photo lives under the dataset at
`absolute_pose_experiments/teaser/`, which is inside the tree
make_zenodo_artifacts.jl walks -- so it ships in the inputs tier and a
consumer can rebuild this figure. (`evaluation/` is a SIBLING of that tree and
is never packaged, which is where it first landed.) The PDF
is written where the paper reads it (`bmvc2026/papers/images/`). Both used to be
per-machine paths -- ~/Downloads for the input and the RETIRED ~/Documents
Overleaf copy for the output -- so the teaser could not be rebuilt by anyone,
including its author. Every parameter — photo, gallery, network, detector
settings, render sizes, background, output name — is in the config of record,
configs/figures/intro.toml; a variant (the project site's transparent
background) is a second config, named by `--config=<path>`. The PNG goes to
paper_images_dir() (src/paper_assets.jl, honouring BB_PAPER_IMAGES — how a
regeneration is reviewed before it replaces the submitted figure). Each run
writes its record under plotsdir("records").
=#
# reproduce: writes papers/images/intro_overlay.png
# reproduce: reads configs/figures/intro.toml

using DrWatson
@quickactivate  # activates bmvc2026 from this file's location (DrWatson layout)

using BlobBoards
using CairoMakie; CairoMakie.activate!()              # vector PDF backend
using FileIO: load as load_image
using Colors: Gray, distinguishable_colors, RGB
using CUDA            # GPU extensions
using CUDA: CUDABackend
import cuDNN          # ONNXRunTime :cuda EP
using VisualGeometryToolkitCore: imshow, HomographyMap, Rect
using CoordinateTransformations: PerspectiveMap

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using DrWatson: projectdir
using Main.BlobBoardsEval: data_root, results_root, paper_images_dir, figure_config, write_figure_records, network_path, load_config
using VisualGeometryToolkitEval.Experiments: Stage, run_stages

# ---- config of record --------------------------------------------------------
const CFG, CFG_PATH = figure_config("intro"; default = projectdir("configs", "figures", "intro.toml"))
const INP, FB, RND = CFG["inputs"], CFG["find_boards"], CFG["render"]
const ROOT = data_root(; tiers = String.(CFG["experiment"]["tiers"]))
IMG_PATH = joinpath(ROOT, INP["image"])
# RASTER, not vector. The same figure as a PDF is ~48 MB and Ghostscript
# cannot reduce it -- the size is the overlay geometry (25 boards' outlines plus
# every pattern blob), not a recompressible image stream, so /prepress,
# /printer and /ebook all return ~48 MB. `sweep.jl` reached this conclusion
# already for its val overlays: "crisp raster; ~MB not ~100 MB vector PDF".
# px_per_unit = 2 keeps blob edges sharp at print size.
OUT_PNG = paper_images_dir(CFG["output"]["name"])
# Figure + axis background. The paper's figure of record stays white; the
# project site (blobboards.github.io) renders the same figure with a
# transparent background so the boards that overhang the photo float on the page.
BG = Symbol(RND["background"])
const BOARDS_DIR = joinpath(ROOT, INP["boards_dir"])
const GALLERY = joinpath(results_root(), "galleries", INP["gallery"])   # a cache, never in the input tree
CUTOFF = FB["nms_cutoff"] == "default" ? BlobBoards.DEFAULT_CUTOFF : FB["nms_cutoff"]
# CUDABackend from CUDA — VGF's CudaBackend is from the retired
# ComputeBackend hierarchy and no longer resolves.
BACKEND = CUDABackend()
MIN_INLIERS = FB["min_inliers"]

function main()
    # ---- load image (color for display, gray for detection) ---------------------
    img_color = load_image(IMG_PATH)
    img_gray = Gray.(img_color)

    # ---- gallery + boards + descriptor model (ONNX) -----------------------------
    # Same network as the main sweep — this figure used to resolve a package
    # default that pointed at the OLD SupCon model, so the paper mixed networks.
    descriptor_network = DescriptorNetwork(
        INP["network"] == "package" ? descriptor_network_path() : network_path(load_config(INP["network"]))
    )
    mkpath(dirname(GALLERY))
    gallery = BoardGallery(GALLERY, descriptor_network)
    board_paths = filter(endswith(".json"), readdir(BOARDS_DIR; join = true))
    board_feature_dict = Dict(
        uid => f for (uid, f) in [insert!(gallery, bp, descriptor_network) for bp in board_paths]
    )
    board_bundles = Dict(
        let (b, _, ch) = BlobBoards.load_bundle(bp)
                uid_from_hash(ch) => b
        end for bp in board_paths
    )

    # Hand-built DELIBERATELY (uncalibrated intro figure, not the method): the
    # config of record is type="pose" and needs a camera. Was broken twice —
    # `n_slots` is not a DetectConfig field (threw at construction) and
    # find_boards takes `device`, not `backend`.
    FB["geometry"] == "homography" || error("intro: only the homography path is implemented")
    config = FindBoardsConfig(
        geometry = HomographyGeometry(min_inliers = MIN_INLIERS),
        detect = DetectConfig(
            polarity = Symbol(FB["polarity"]), peak_threshold = FB["peak_threshold"],
            edge_threshold = FB["edge_threshold"], nms = EllipseNms(cutoff = Float32(CUTOFF))
        )
    )

    accepted = find_boards(
        board_feature_dict, descriptor_network, img_gray;
        config, device = BACKEND, board_bundles
    )
    @info "detected boards" n = length(accepted)

    # ---- figure: distinct color per board id, overlaid on the COLOR image -------
    cols = distinguishable_colors(length(accepted), [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed = true)
    h, w = size(img_color)
    fw = RND["figure_width"]
    fig = Makie.Figure(size = (fw, round(Int, fw * h / w)), backgroundcolor = BG)
    ax, _ = imshow(fig[1, 1], img_color)
    ax.backgroundcolor = BG
    for (i, (uid, model, _, _)) in enumerate(accepted)
        bundle = board_bundles[uid]
        proj = PerspectiveMap() ∘ HomographyMap(model)
        board!(
            ax, proj, Rect(bundle, :world), blobs(bundle, :world);
            cutoff = cutoff(bundle), outline_color = cols[i], blob_color = cols[i],
            outline_width = RND["outline_width"], blob_width = RND["blob_width"]
        )
    end
    Makie.hidedecorations!(ax); Makie.hidespines!(ax)     # clean, axis-free figure

    mkpath(dirname(OUT_PNG))
    Makie.save(OUT_PNG, fig; px_per_unit = RND["px_per_unit"])
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_bmvc2026_intro.jl",
        inputs = [IMG_PATH, GALLERY, BOARDS_DIR], outputs = [OUT_PNG])
    @info "saved figure" path = OUT_PNG n = length(accepted) mb = round(filesize(OUT_PNG) / 2^20; digits = 1)
    return nothing
end

run_stages("make_bmvc2026_intro", [Stage(:render, "intro_overlay.png (GPU)", () -> false, main)])
