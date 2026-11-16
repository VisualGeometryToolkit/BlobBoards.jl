#=
make_asift_detections.jl — the RAW ASIFT detections on a blob-board photo, and
the same detections on a Gaussian-blurred copy of the photo.

"Raw" = what `detect_blobs` returns: the affine scale-space extrema of every
ASIFT warp, back-projected into the photo (S = σ·A⁻¹) and merged by the
cross-warp NMS — BEFORE adaptation, filtering or matching. Each is drawn as its
`cutoff`·σ ellipse, coloured by the warp (tilt, angle) it was found in, so the
tilted warps' elongated frames can be told apart from the untilted ones.

The blurred panel shows the scale the detector responds to: at σ_blur of a few
pixels the printed discs smear into the blobs the DoG pyramid sees, and the
detections sit on them. By default it carries the SAME detections as the photo
panel (`[blur] redetect = false`); `redetect = true` runs the detector again on
the blurred copy, to show what survives the blur.

Outputs, per panel (scripts/overlaid_figure.jl): `<name>_photo.jpg` (the photo
layer), `<name>.pdf` (the ellipses only, on a transparent page the size of the
photo; the paper composites the two with `\overlaid`, papers/overlaid.tex) and
`<name>.jpg` (the full panel, a plain backup); plus a side-by-side of the two
backups as a JPEG. All under plotsdir(), and copied to papers/images/<[output] paper_dir> for the
paper/poster sources (paper_images_dir: BB_PAPER_IMAGES overrides the root).

Run (CPU on macOS, CUDA on Linux by default — `[detect] device`):
    julia --project=bmvc2026 bmvc2026/scripts/make_asift_detections.jl

Every parameter — photo, detector settings, blur σ, crop, colours, output
names — is in the config of record, configs/figures/asift_detections.toml
(`--config=<path>` overrides). Each run writes its record under
plotsdir("records").
=#
# reproduce: writes plots/asift_detections/asift_original{.pdf,.jpg,_photo.jpg}
# reproduce: writes plots/asift_detections/asift_blurred{.pdf,.jpg,_photo.jpg}
# reproduce: writes plots/asift_detections/asift_side_by_side.jpg
# reproduce: writes papers/images/asift_detections/asift_*
# reproduce: reads configs/figures/asift_detections.toml

using DrWatson
@quickactivate

using BlobBoards
using BlobBoards: DetectConfig, EllipseNms, MxmNms, detect_blobs
using CairoMakie; CairoMakie.activate!()
using FileIO: load as load_image, save as save_image
using Colors: Gray, RGB, distinguishable_colors
using Images: imfilter, Kernel, N0f8
using GeometryBasics: Point2
using VisualGeometryToolkitCore: ellipses!, feature_frame, centroid, EllipticFrame
using VisualGeometryToolkitEval.Experiments: Stage, run_stages

if !isdefined(Main, :BlobBoardsEval)
    include(joinpath(@__DIR__, "..", "src", "BlobBoardsEval.jl"))
end
using Main.BlobBoardsEval: data_root, figure_config, write_figure_records, paper_images_dir
include(joinpath(@__DIR__, "overlaid_figure.jl"))

const BB = BlobBoards
const CFG, CFG_PATH = figure_config("asift_detections"; default = projectdir("configs", "figures", "asift_detections.toml"))
const INP, DET, BLUR, RND, OUT, JPG = CFG["inputs"], CFG["detect"], CFG["blur"], CFG["render"], CFG["output"], CFG["jpeg"]
const ROOT = data_root(; tiers = String.(CFG["experiment"]["tiers"]))
const IMG_PATH = joinpath(ROOT, INP["image"])

const USE_CUDA = DET["device"] == "cuda" || (DET["device"] == "auto" && Sys.islinux())
DET["device"] in ("auto", "cuda", "cpu") || error("asift_detections: unknown detect.device $(repr(DET["device"]))")
if USE_CUDA
    using CUDA
    import cuDNN
end
_device() = USE_CUDA ? CUDA.CUDABackend() : BB.KA.CPU()

_cutoff(v) = v == "default" ? Float32(BB.DEFAULT_CUTOFF) : Float32(v)

function _detect_config()
    nms = DET["nms"] == "ellipse" ? EllipseNms(cutoff = _cutoff(DET["nms_cutoff"])) :
        DET["nms"] == "mxm" ? MxmNms() :
        DET["nms"] == "none" ? nothing :
        error("asift_detections: unknown detect.nms $(repr(DET["nms"]))")
    return DetectConfig(
        polarity = Symbol(DET["polarity"]), peak_threshold = DET["peak_threshold"],
        edge_threshold = DET["edge_threshold"], nms = nms, upsample = DET["upsample"]
    )
end

# The detections as (frame, warp index) — the warp is the ASIFT warp the
# extremum was found in (1 = the untilted identity warp).
function _detect(img_gray)
    dets = detect_blobs(img_gray; config = _detect_config(), device = _device())
    return feature_frame.(dets), [Int(d.metadata.warp_idx) for d in dets]
end

# The displayed window (rows, cols) of an H×W image: the configured crop, or all of it.
function _window(H, W)
    isempty(RND["crop"]) && return 1:H, 1:W
    r0, c0, nr, nc = RND["crop"]
    return r0:min(H, r0 + nr - 1), c0:min(W, c0 + nc - 1)
end

# Frames whose centre lies in the window, moved into its coordinates (the
# convention of make_pipeline_panels.jl's crops).
function _in_window(frames, warps, rows, cols)
    dx, dy = first(cols) - 1, first(rows) - 1
    keep = [first(cols) <= centroid(f)[1] <= last(cols) && first(rows) <= centroid(f)[2] <= last(rows) for f in frames]
    moved = [EllipticFrame(Point2(centroid(f)[1] - dx, centroid(f)[2] - dy), f.S) for f in frames[keep]]
    return moved, warps[keep]
end

# Draw `frames` on the photo axis `ax`; one stroke colour per warp
# (`palette[w]`), or the single configured colour.
function _overlay!(ax, frames, warps, palette)
    cut = Float64(_cutoff(RND["cutoff"]))
    kw = (; cutoff = cut, strokewidth = RND["strokewidth"])
    if RND["color_by"] == "warp"
        for w in sort!(unique(warps))
            ellipses!(ax, frames[warps .== w]; kw..., strokecolor = (palette[w], RND["alpha"]))
        end
    elseif RND["color_by"] == "single"
        ellipses!(ax, frames; kw..., strokecolor = (Symbol(RND["color"]), RND["alpha"]))
    else
        error("asift_detections: unknown render.color_by $(repr(RND["color_by"]))")
    end
    return ax
end

function main()
    img_color = load_image(IMG_PATH)
    # blur the FULL photo, then crop: a crop blurred on its own gets edge artefacts
    blurred = imfilter(img_color, Kernel.gaussian(BLUR["sigma"]))

    frames, warps = _detect(Gray.(img_color))
    frames_b, warps_b = BLUR["redetect"] ? _detect(Gray.(blurred)) : (frames, warps)
    @info "raw ASIFT detections" photo = length(frames) blurred = length(frames_b) redetect = BLUR["redetect"] per_warp = sort([w => count(==(w), warps) for w in unique(warps)])

    rows, cols = _window(size(img_color)...)
    fr, wr = _in_window(frames, warps, rows, cols)
    fr_b, wr_b = _in_window(frames_b, warps_b, rows, cols)
    crop, crop_b = img_color[rows, cols], blurred[rows, cols]

    nwarp = max(maximum(warps; init = 1), maximum(warps_b; init = 1))
    palette = distinguishable_colors(nwarp, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed = true)

    out_dir = plotsdir(OUT["dir"]); mkpath(out_dir)
    outputs = String[]
    # the original photo is its source file's bytes when shown uncropped and unresized
    uncropped = (rows, cols) == (axes(img_color, 1), axes(img_color, 2))
    for (name, img, f, w, src) in ((OUT["names"][1], crop, fr, wr, uncropped ? IMG_PATH : nothing),
            (OUT["names"][2], crop_b, fr_b, wr_b, nothing))
        fig, ax, photo = photo_figure(img; width = RND["figure_width"], photo_px = RND["photo_px"])
        _overlay!(ax, f, w, palette)
        append!(outputs, save_overlaid(joinpath(out_dir, name), fig, photo; source = src,
            quality = JPG["quality"], pt_per_unit = RND["pt_per_unit"]))
    end
    # side by side: the two backups with a white gap, a JPEG only
    a, b = (load_image(joinpath(out_dir, n * ".jpg")) for n in OUT["names"][1:2])
    gap = fill(RGB{N0f8}(1, 1, 1), size(a, 1), OUT["side_by_side_gap"])
    sbs = joinpath(out_dir, OUT["names"][3] * ".jpg")
    save_image(sbs, hcat(a, gap, b[axes(a, 1), :]); quality = JPG["quality"])
    push!(outputs, sbs)

    # The same files, byte for byte, where the paper/poster sources include them.
    if !isempty(OUT["paper_dir"])
        paper_dir = paper_images_dir(OUT["paper_dir"]); mkpath(paper_dir)
        for o in copy(outputs)
            push!(outputs, cp(o, joinpath(paper_dir, basename(o)); force = true))
        end
    end

    @info "written" outputs
    write_figure_records(CFG, CFG_PATH; driver = "bmvc2026/scripts/make_asift_detections.jl", outputs, inputs = [IMG_PATH])
    return nothing
end

run_stages("make_asift_detections", [Stage(:render, "raw ASIFT detections on the photo and its blurred copy", () -> false, main)])
