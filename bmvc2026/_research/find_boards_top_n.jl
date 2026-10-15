#=
Run find_board for every board in the gallery, sort by inlier count,
and plot the top-N candidates in a grid (before spatial winner-take-all).

Useful for inspecting "runner-up" hypotheses that get suppressed by
spatial_verify but would still be valid matches in a multi-board image.
=#

using BlobBoards
using BlobBoards: find_board   # advanced API (unexported)
using BlobBoards: data_dir, detect_blobs, DescriptorNetwork, BoardGallery, uid_from_hash,
                  cutoff, find_board, FindBoardsConfig, DetectConfig,
                  DEFAULT_CUTOFF, load_bundle, blobs, board!
using BlobBoards: _ensure_network, canonicalize, describe, described_features
using VisualGeometryFeatures: nn_match_stacked, MutualNN
using VisualGeometryFeatures: Detection
using VisualGeometryCore: is_map_inlier, imshow, HomographyMap, matches!, window!, ustrip, centroid, Rect
using CoordinateTransformations: PerspectiveMap
using GLMakie; GLMakie.activate!()
using GeometryBasics: Point2, Point2f
using FileIO: load as load_image
using Colors: Gray
using CUDA
using CUDA: CUDABackend

const N_TOP        = 4
const CUTOFF       = DEFAULT_CUTOFF
const DEVICE = CUDABackend()
const BOARDS_DIR   = joinpath(data_dir(), "evaluation/blobboards_original")
const IMG_PATH     = joinpath(data_dir(), "dataset_sonyfx3_tamron20mm_f8/jpeg/DSC02847.JPG")
# Network from configs/default.toml — the one place a default entry point
# learns which descriptor to load. Same log-polar _cuda net as the paper.
const MODEL_PATH   = descriptor_network_path()
const GALLERY_PATH = joinpath(data_dir(), "models/board_features.jld2")

img = Gray.(load_image(IMG_PATH))
desc_model = DescriptorNetwork(MODEL_PATH)
gallery    = BoardGallery(GALLERY_PATH, desc_model)
board_paths = filter(endswith(".json"), readdir(BOARDS_DIR; join = true))
board_features = Dict(uid => f for (uid, f) in [insert!(gallery, bp, desc_model) for bp in board_paths])
board_bundles  = Dict(let (b, _, ch) = load_bundle(bp); uid_from_hash(ch) => b end for bp in board_paths)
@info "boards" n=length(board_paths)

detections = detect_blobs(img;
    device = CUDABackend(),
    config = DetectConfig(polarity = :dark, peak_threshold = 0.01),
    max_detections = 2048)
frames = BlobBoards.filter_blobs(
    [Detection(d.frame, (; id = i)) for (i, d) in enumerate(detections)],
    detections; cutoff = CUTOFF, image = img)
patches = canonicalize(frames, img; inner_extent = CUTOFF, outer_extent = 96.0, device = DEVICE)
descriptors = describe(_ensure_network(desc_model), patches)
image_features = described_features(frames, descriptors)
@info "features" n=length(image_features)

cfg = FindBoardsConfig()
# MutualNN explicitly: VGF's strategy argument is positional, required and
# leads, unlike the BB-local matcher it replaced, which defaulted to it.
match_dict = nn_match_stacked(MutualNN(), board_features, image_features;
                              backend = DEVICE, min_similarity = cfg.min_similarity)

# Store full result tuples per board.
results = NamedTuple[]
for (uid, bf) in board_features
    r = find_board(nothing, bf, image_features;
                   min_similarity = cfg.min_similarity, matches = match_dict[uid],
                   fit_kw = (; outlier_halfwidth = cfg.geometry.outlier_halfwidth))
    isnothing(r) && continue
    model, quality, csponds = r
    push!(results, (; uid, model, quality, csponds,
                      n_inliers = count(is_map_inlier, csponds.weights)))
end
sort!(results; by = r -> -r.n_inliers)

top = results[1:min(N_TOP, length(results))]
@info "top N" n=length(top) inlier_counts=[r.n_inliers for r in top]

ncols = min(2, length(top))
nrows = cld(length(top), ncols)
fig = Figure(size = (ncols * 700, nrows * 700))
for (i, r) in enumerate(top)
    rr, cc = fldmod1(i, ncols)
    ax, _ = imshow(fig[rr, cc], img;
        title = "rank $i  uid=$(r.uid)  inliers=$(r.n_inliers)  q=$(round(r.quality; digits=1))")
    proj = PerspectiveMap() ∘ HomographyMap(r.model)
    bundle = board_bundles[r.uid]
    board!(ax, proj, Rect(bundle, :world), blobs(bundle, :world);
           cutoff = cutoff(bundle))
    proj_pts = Point2f.(proj.(Point2.(centroid.(sources(r.csponds)))))
    det_pts  = Point2f.(centroid.(targets(r.csponds)))
    matches!(ax, proj_pts, det_pts; weights = r.csponds.weights)
end

screen = window!(fig, "Top $(length(top)) board candidates")
isinteractive() || wait(screen)
nothing
