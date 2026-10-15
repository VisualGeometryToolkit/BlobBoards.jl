#=
scan_media_directory.jl — process a mixed corpus of stills and videos.

Deliberately an EXAMPLE, not a library function: every interesting line here is
policy the package should not choose for you — which entry point a medium gets,
how stills are batched, the processing order, and when detector plans are
released. BlobBoards supplies the primitives (`walk_media`,
`media_resolution`, `group_by_resolution`); the orchestration is yours.

Four things it demonstrates, each of which was measured rather than guessed:

0. BUILD THE DESCRIPTOR NETWORK FIRST. The three GPU consumers here are not
   peers. ASIFT's pyramids and the stacked matcher both size themselves against
   free memory at the moment they run, so they adapt to whatever is left. The
   ONNX descriptor arena does not: it is fixed at construction and never
   shrinks. Being the only inelastic consumer, it must claim its memory before
   the elastic ones measure — `DescriptorNetwork` does that itself (see
   `reserve_workspace!`; ~526 MiB at the default `max_batch = 64`, per the
   measured `arena_MiB = 14 + 8 * max_batch` in src/descriptor.jl), but only
   when you construct it before the detector runs, as below.

1. STAGE THE GALLERY ONCE. `BlobBoards.GalleryIndex` is independent of frame
   size and of medium, so building it per call re-uploads the whole descriptor
   set (~31 MB for 60k descriptors) for nothing. Supply it as `gpu_index`.

2. GROUP BY RESOLUTION, AND CLEAR PLANS AT THE BOUNDARY. The ASIFT detector
   caches one scale-space plan per distinct frame size — 11 GiB for a 1600x1200
   frame with the 25-tilt schedule — and the cache is bounded to one plan by
   default. Interleaved resolutions therefore rebuild constantly: measured
   32 s/frame on a 13-resolution still corpus versus ~2 s/frame grouped.

3. PICK THE ENTRY POINT BY MEDIA TYPE, AND DO IT EXPLICITLY.
     video  -> track_boards : consecutive frames, temporal association is the point
     stills -> find_boards  : unrelated images, each an independent search
   Getting this wrong FAILS SILENTLY. `track_boards` only searches the full
   gallery on frame 1 and every `keyframe_every`-th frame; between those it
   looks only for boards it is already tracking. Fed unrelated stills it does
   not error — it simply stops finding new boards, and the run looks fine.

Run:
    julia --project=examples examples/pipeline/scan_media_directory.jl <media-dir>
=#

using BlobBoards
using BlobBoards: walk_media, media_resolution, group_by_resolution,
                  DescriptorNetwork, BoardGallery, load_bundle, uid_from_hash,
                  descriptor_network_path
using VisualGeometryFeatures: clear_asift_plans!
using Colors: Gray
using FileIO: load as load_image
using CUDA: CUDABackend            # any KA backend; KA.CPU() also works
import cuDNN                       # ONNXRunTime :cuda execution provider

const MEDIA_ROOT = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(BlobBoards.data_dir(), "..", "BlobBoardTrainingFootage", "Nextcloud")

const DEVICE = CUDABackend()
const CONFIG = FindBoardsConfig(joinpath(pkgdir(BlobBoards), "configs", "default.toml"))

# ---- gallery: the board set this corpus is searched against ------------------
const BOARDS_DIR = joinpath(BlobBoards.data_dir(), "evaluation", "blobboards_original")
const GALLERY_JLD2 = joinpath(tempdir(), "scan_media_gallery.jld2")

net = DescriptorNetwork(descriptor_network_path())   # (0) fixed claim, made first
gallery = BoardGallery(GALLERY_JLD2, net)
board_paths = filter(endswith(".json"), readdir(BOARDS_DIR; join = true))
board_features = Dict(uid => f for (uid, f) in
                      [insert!(gallery, bp, net; device = DEVICE) for bp in board_paths])
board_bundles = Dict(let (b, _, ch) = load_bundle(bp); uid_from_hash(ch) => b end
                     for bp in board_paths)
@info "gallery" boards=length(board_features) descriptors=sum(length, values(board_features))

# (1) stage the stacked gallery ONCE, reused by every call below
gpu_index = BlobBoards.GalleryIndex(DEVICE, board_features)

# ---- walk, then order the work ----------------------------------------------
media = walk_media(MEDIA_ROOT)
isempty(media) && error("no image or video media under $MEDIA_ROOT")
groups = group_by_resolution(media)
@info "corpus" media=length(media) resolutions=length(groups) root=MEDIA_ROOT

results = Dict{String, Any}()

for (res, paths) in groups                          # (2) like-sized media together
    videos = filter(BlobBoards.is_video, paths)
    stills = filter(!BlobBoards.is_video, paths)
    @info "group" resolution=res n_videos=length(videos) n_stills=length(stills)

    # (3a) video: temporal — one call per clip, frames in capture order
    for v in videos
        results[v] = track_boards(board_features, net, v;
                                  board_bundles, config = CONFIG,
                                  device = DEVICE, gpu_index)
    end

    # (3b) stills: independent — ONE batched call amortises extraction setup
    #      across the group. Loaded eagerly here for clarity; a generator
    #      keeps host memory flat on large corpora.
    if !isempty(stills)
        imgs = [Gray{Float32}.(load_image(p)) for p in stills]
        per_frame = find_boards(board_features, net, imgs;
                                board_bundles, config = CONFIG,
                                device = DEVICE, gpu_index)
        for (p, r) in zip(stills, per_frame)
            results[p] = r
        end
    end

    # (2) release this resolution's plan before the next group's is built
    clear_asift_plans!()
end

# ---- report ------------------------------------------------------------------
n_boards(r) = r isa AbstractDict ? sum(length, values(r); init = 0) : length(r)
total = sum(n_boards(r) for r in values(results); init = 0)
@info "done" media=length(results) detections=total
for (res, paths) in groups
    got = sum(n_boards(results[p]) for p in paths; init = 0)
    println("  ", res, "  ", lpad(length(paths), 4), " media  ", lpad(got, 6), " detections")
end
