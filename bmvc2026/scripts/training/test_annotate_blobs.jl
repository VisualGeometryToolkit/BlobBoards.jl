#=
import VisualGeometryToolkitIO as VgtkIO
Test script for annotate_blobs — visualize extracted patches and blob IDs

Goal: Check if blob transformation is correct by visually inspecting
canonical patches extracted from a single image.

INTERACTIVE: needs GLMakie (a display), which the paper environment does not
carry — bmvc2026 is headless-reproducible (test/paper_project_tests.jl). Run
it from an interactive environment that adds GLMakie beside this project, e.g.
`julia --project=bmvc2026 -i` then `]activate --temp`, `]dev .` (the
repository root) and `]add GLMakie`, on a machine with a display.
=#

using BlobBoards
using FileIO, Colors
using GLMakie
using VideoIO

import VisualGeometryToolkitIO   # canonical vgio/cameras/v3 camera stores
using VisualGeometryToolkitCore: CameraModel   # dispatch token for VisualGeometryToolkitIO.load

# =============================================================================
# Setup — load gallery, boards, descriptor model
# =============================================================================

root = data_dir()
boards_dir = joinpath(root, "boards")
# Network from configs/default.toml — the one place a default entry point
# learns which descriptor to load. Same log-polar _cuda net as the paper.
model_path = descriptor_network_path()
gallery_path = joinpath(root, "models/board_gallery.jld2")

@info "Loading descriptor model..."
descriptor_network = DescriptorNetwork(model_path)

@info "Loading gallery..."
gallery = BoardGallery(gallery_path, descriptor_network)

@info "Loading boards..."
board_paths = filter(endswith(".json"), readdir(boards_dir; join = true))
board_feature_dict = Dict(
    uid => features for (uid, features) in
        [insert!(gallery, bp, descriptor_network) for bp in board_paths]
)

board_bundles = Dict(
    let (b, _, ch) = BlobBoards.load_bundle(bp)
            uid_from_hash(ch) => b
    end
        for bp in board_paths
)

@info "$(length(board_paths)) boards loaded"

# Build UID → board path mapping
uid_to_path = Dict(
    let (b, _, ch) = BlobBoards.load_bundle(bp)
            uid_from_hash(ch) => bp
    end
        for bp in board_paths
)

# =============================================================================
# Load camera calibration
# =============================================================================

# @info "Loading camera calibration..."
# camera = VisualGeometryToolkitIO.load(CameraModel,
#     joinpath(root, "images", "apple-iphone-13-mini_f1.6-1x_till.json"))
# @info "PosedCamera loaded: $(camera.intrinsics)"

# =============================================================================
# Load image
# =============================================================================

# img_path = joinpath(root, "images/iphone/IMG_2169.jpeg")
# @info "Loading image: $img_path"
# img = Gray{Float32}.(FileIO.load(img_path))
# @info "Image size: $(size(img))"

# =============================================================================
# Load video
# =============================================================================
video_path = joinpath(root, "videos/iphone_till/26-04-23 15-36-18 2200.mov")
video_camera = VgtkIO.load(
    CameraModel,
    joinpath(root, "videos", "apple-iphone-13-mini-till_f1.6-1x-video-fullhd.json")
)

# =============================================================================
# Find boards for test if there exist some
# =============================================================================
# Note: Skipping standalone find_boards test to avoid long computation time.
# Boards are detected and processed inside annotate_blobs.
# @info "Test run to find boards..."
# boards_test = find_boards(board_feature_dict, descriptor_network, img;
#     board_bundles = board_bundles,
#     camera = camera,
#     device = KA.CPU(),
# )
# @info "Found $(length(boards_test)) board(s) in the image."

#
# Extract frame in video
#
vid = VideoIO.openvideo(video_path)
frame_number = 240  # Extract the 10th frame
img = nothing
for (i, frame) in enumerate(vid)
    if i == frame_number
        global img
        img = Gray{Float32}.(frame)
        break
    end
end
close(vid)
isnothing(img) && error("Frame $frame_number not found in video")
@info "Extracted frame $frame_number: size=$(size(img))"

# =============================================================================
# Annotate — extract patches and blob IDs
# =============================================================================

@info "Running annotate_blobs..."
result = annotate_blobs(
    board_feature_dict,
    descriptor_network,
    img;
    board_bundles = board_bundles,
    camera = video_camera,
    device = KA.CPU(),
)

# annot_out_path = joinpath(root, "training/test_annotation.h5")
# result = annotate_blobs(board_feature_dict, descriptor_network, video_path; board_bundles=board_bundles, output_path=annot_out_path, board_paths=board_paths,
#     camera=video_camera, n_frames=3)

n_obs = length(result.blob_ids)
n_conf = length(result.confuser_patches)

@info "Annotation complete:"
@info "  Observations: $n_obs patches"
@info "  Confusers: $n_conf unmatched detections"

if n_obs == 0
    @warn "No patches extracted! Check if boards are detected in the image."
    exit(1)
end

# =============================================================================
# Extract reference patches from board PDFs
# =============================================================================

@info "Extracting reference patches from board PDFs..."
unique_uids = unique(result.board_uids)
ref_patches_by_uid = Dict()

for uid in unique_uids
    if haskey(uid_to_path, uid)
        ref_result = BlobBoards._extract_reference_patches(
            uid_to_path[uid];
            patch_size = 64,
            outer_extent = 96.0
        )
        ref_patches_by_uid[uid] = ref_result
        @info "  Extracted $(length(ref_result.blob_ids)) reference patches for board $uid"
    else
        @warn "  Board path not found for UID: $uid"
    end
end

# =============================================================================
# Summary
# =============================================================================

@info "Extracted patches by board and blob ID:"
for (uid, bid) in zip(result.board_uids, result.blob_ids)
    print("($uid, $bid), ")
end

# =============================================================================
# Visualization — GLMakie grid: reference patches vs. image patches
# =============================================================================

@info "Creating GLMakie visualization with reference patches..."

P = size(result.patches[1], 1)  # patch size
max_blobs = min(length(unique_uids), 20)  # limit number of boards to show
max_images_per_blob = 8  # show up to 8 image patches per reference

# Count image patches for each reference patch
image_counts = []
match_counts = Dict()  # track how many image patches per reference patch
for uid in unique_uids
    for bid in 1:length(ref_patches_by_uid[uid].blob_ids)
        count = sum((result.board_uids .== uid) .& (result.blob_ids .== bid))
        push!(image_counts, min(count, max_images_per_blob))
    end
end
max_images = isempty(image_counts) ? 1 : maximum(image_counts)

n_cols = 1 + max_images  # ref + image patches
uid_bid_pairs = []

for uid in unique_uids
    haskey(ref_patches_by_uid, uid) || continue
    ref = ref_patches_by_uid[uid]
    for bid in ref.blob_ids
        push!(uid_bid_pairs, (uid, bid))
    end
end

n_rows = min(length(uid_bid_pairs), 24)  # limit total rows
uid_bid_pairs = uid_bid_pairs[1:n_rows]

fig = Figure(size = (200 + n_cols * 90, 100 + n_rows * 90), backgroundcolor = :white)

for (row, (uid, bid)) in enumerate(uid_bid_pairs)
    ref = ref_patches_by_uid[uid]
    ref_idx = findfirst(==(bid), ref.blob_ids)

    # Column 1: Reference patch
    ref_patch = ref.patches[ref_idx]
    ax_ref = Axis(fig[row, 1], aspect = DataAspect(), title = "Ref: BID=$uid, BID=$bid")
    image!(ax_ref, rotr90(Float32.(ref_patch)), colormap = :grays)
    hidedecorations!(ax_ref)
    hidespines!(ax_ref)

    # Columns 2+: Image patches matching this (uid, bid)
    img_indices = findall((result.board_uids .== uid) .& (result.blob_ids .== bid))
    img_indices = img_indices[1:min(length(img_indices), max_images_per_blob)]

    for (col_offset, img_idx) in enumerate(img_indices)
        img_patch = result.patches[img_idx]
        img_uid = result.board_uids[img_idx]
        img_bid = result.blob_ids[img_idx]
        ax_img = Axis(fig[row, col_offset + 1], aspect = DataAspect(), title = "Img: UID=$img_uid, BID=$img_bid")
        image!(ax_img, rotr90(Float32.(img_patch)), colormap = :grays)
        hidedecorations!(ax_img)
        hidespines!(ax_img)
    end
end

FileIO.save("/tmp/patches_comparison.png", fig)
@info "Saved comparison visualization to /tmp/patches_comparison.png"

@info "Displaying figure..."
display(fig)

# =============================================================================
# Detailed inspection — select a patch to examine
# =============================================================================

if n_obs > 0
    @info "\n=== First patch details ==="
    @info "Board UID: $(result.board_uids[1])"
    @info "Blob ID: $(result.blob_ids[1])"
    @info "Patch size: $(size(result.patches[1]))"
    @info "Patch min/max intensity: $(minimum(result.patches[1])), $(maximum(result.patches[1]))"
end

@info "Test complete!"
