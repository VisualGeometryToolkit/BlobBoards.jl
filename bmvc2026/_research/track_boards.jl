#=
Board Tracking (Homography — no intrinsics needed)

Track boards across video frames using descriptor matching and homography
estimation. Displays results in a GLMakie window with projected blob ellipses.

Usage:
  include("examples/pipeline/track_boards.jl")
=#

using BlobBoards
using GLMakie; GLMakie.activate!()
using VideoIO
using Colors: Gray
using GeometryBasics: Point2f, Polygon, coordinates
using VisualGeometryToolkitCore: imshow, imshow!, HomographyMap, Ellipse, Circle,
    boundary, centroid, is_ellipse, ustrip, Disc
using CUDA  # triggers GPU extensions

# =============================================================================
# Paths
# =============================================================================

boards_dir = joinpath(data_dir(), "evaluation/blobboards_original")
# Network from configs/default.toml — the one place a default entry point
# learns which descriptor to load. Same log-polar _cuda net as the paper.
model_path = descriptor_network_path()
video_path = joinpath(data_dir(), "blobboard_videos/C0012.MP4")
gallery_path = joinpath(data_dir(), "models/board_gallery.jld2")

# =============================================================================
# Setup
# =============================================================================

println("Loading ONNX model...")
descriptor_network = DescriptorNetwork(model_path)

println("Building gallery...")
gallery = BoardGallery(gallery_path, descriptor_network)
board_paths = filter(endswith(".json"), readdir(boards_dir; join=true))
board_feature_dict = Dict(
    uid => features for (uid, features) in [insert!(gallery, bp, descriptor_network) for bp in board_paths])
board_bundles = Dict(
    let (b, _, ch) = BlobBoards.load_bundle(bp); uid_from_hash(ch) => b end for bp in board_paths)
println("  $(length(board_feature_dict)) boards ready")

# =============================================================================
# Create figure with Observables
# =============================================================================

println("Opening video...")
reader = VideoIO.openvideo(video_path)
first_frame = Gray.(first(reader))
close(reader)

img_obs = Observable(first_frame)
polys_obs = Observable(Polygon{2,Float32}[])

quit = Ref(false)

fig = Figure(size=(1200, 800))
ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Tracking")
imshow!(ax, img_obs)
Makie.poly!(ax, polys_obs; color=:transparent, strokecolor=:yellow, strokewidth=1.5)
btn = Button(fig[2,1]; label="Stop", tellwidth=false)
on(btn.clicks) do _
    quit[] = true
end
display(fig)

# =============================================================================
# Tracking loop
# =============================================================================

function run_tracking(img_obs, polys_obs, ax, quit, board_feature_dict,
                      board_bundles, descriptor_network, video_path;
                      n_frames=50, stride=10)
    quit[] = false
    reader = VideoIO.openvideo(video_path)
    frame_count = 0
    processed = 0

    for raw_frame in reader
        quit[] && (@info "Stopped by user"; break)
        frame_count += 1
        (frame_count % stride == 0) || continue
        processed += 1
        processed > n_frames && break

        img = Gray.(raw_frame)

        t = @elapsed results = find_boards(board_feature_dict, descriptor_network, img;
            board_bundles, device = CUDABackend())

        # Collect projected ellipses as polygons
        polys = Polygon{2,Float32}[]
        for (uid, H, quality, scored, weights) in results
            bundle = board_bundles[uid]
            Hmap = HomographyMap(H)
            for wd in blobs(bundle, :world)
                Q = Hmap(Circle(wd, cutoff(bundle)))
                is_ellipse(Q) || continue
                push!(polys, Polygon(coordinates(Ellipse(Q), 64)))
            end
        end

        # Update observables
        img_obs[] = img
        polys_obs[] = polys

        n_boards = length(results)
        uids = join([r[1] for r in results], ", ")
        ax.title = "Frame $frame_count: $n_boards board(s) [$uids]"

        @info "Frame $frame_count ($processed/$n_frames): $n_boards boards [$uids] $(round(t; digits=2))s"
    end

    close(reader)
    @info "Done: $processed frames"
end

run_tracking(img_obs, polys_obs, ax, quit, board_feature_dict, board_bundles,
             descriptor_network, video_path; n_frames=200, stride=1)
