#=
Track Boards — temporal tracking through a video, both geometries, traced.

`track_boards` mirrors `find_boards`: one entry point, `config.geometry`
picks the path.
  • HomographyGeometry()   → no camera, 2D↔2D homography per frame
  • PoseGeometry(camera)   → calibrated, 3D(z=0)↔2D P3P pose per frame

This script runs BOTH on the same clip, so the two variants are directly
comparable, and captures a `TrackTrace` for each — the replayable record of
every frame: detections, the warm bet's predicted↔associated ellipses, and
the committed models' board projections.

Replay a captured trace afterwards (interactive; needs GLMakie on thread 1,
so run it in a plain `julia -i --project=examples`, not an agent gate):

    using BlobBoards, GLMakie
    trace = load_trace("<trace dir printed below>")
    display(GLMakie.Screen(), BlobBoards.replay(trace))

Usage:
    julia --project=examples examples/pipeline/track_boards.jl [video] [n_frames]

`n_frames = 0` (default 150) tracks the whole clip.
=#

using BlobBoards
# render_on_demand=false: in an agent gate session the on-demand renderer gets
# no damage events and the window stays black; continuous rendering fixes it.
using GLMakie; GLMakie.activate!(render_on_demand = false)
using Colors: Gray
import VideoIO
using CUDA: CUDABackend            # any KA backend; KA.CPU() also works
import cuDNN                       # ONNXRunTime :cuda execution provider
using Statistics: mean
import VisualGeometryToolkitIo as VGIo   # canonical vgio/cameras/v3 camera stores
using VisualGeometryToolkitCore: BoundedCamera, model, extent

const VIDEO = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(BlobBoards.example_data(:video), "blobboards_handheld.mov")
const N_FRAMES = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 150
const DEVICE = CUDABackend()
const TRACE_ROOT = joinpath(tempdir(), "blobboards_traces")

const BOARDS_DIR = BlobBoards.example_data(:boards)
# The calibration must match the VIDEO's resolution — the guard below checks.
# (The default pair does: iPhone clip and calibration are both 1920x1080.
# C0012.MP4 was the original candidate and is 1080p too, but the only FX3
# calibration is for 4240x2832 stills — the mismatch cost its pose arm 95
# re-inits in the 2026-08-18 run, which is why the guard exists.)
const CAMERA_JSON = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(BlobBoards.example_data(:video), "cameras.json")

# Network first: the ONNX arena is the one inelastic GPU consumer, so it must
# claim its memory before the elastic ones (ASIFT, matcher) measure free space
# — see scan_media_directory.jl for the measured story.
net = DescriptorNetwork(descriptor_network_path())
gallery = BoardGallery(joinpath(tempdir(), "track_example_gallery.jld2"), net)
board_paths = filter(endswith(".json"), readdir(BOARDS_DIR; join = true))
board_features = Dict(
    uid => f for (uid, f) in
        [insert!(gallery, bp, net; device = DEVICE) for bp in board_paths]
)
board_bundles = Dict(
    let (b, _, ch) = BlobBoards.load_bundle(bp)
        uid_from_hash(ch) => b
    end for bp in board_paths
)
@info "gallery" boards = length(board_features)

cam_bounded = VGIo.load(BoundedCamera, CAMERA_JSON)
cam = model(cam_bounded)

# Sanity: intrinsics are resolution-specific. A mismatch here means the pose
# variant runs with the wrong K — track it down before believing any pose.
let r = VideoIO.openvideo(VIDEO)
    h, w = size(read(r))
    close(r)
    e = extent(cam_bounded)
    @info "video" VIDEO size = (w, h) calibrated_for = (e.width, e.height)
    (w, h) == (e.width, e.height) ||
        @warn "video resolution differs from the calibration's — pose output is suspect"
end

traces = Dict{String, String}()
for (label, geometry) in (
        ("homography", HomographyGeometry()),
        ("pose", PoseGeometry(cam)),
    )
    trace_dir = joinpath(TRACE_ROOT, "$(splitext(basename(VIDEO))[1])_$label.trace")
    isdir(trace_dir) && rm(trace_dir; recursive = true)
    # :thumbs keeps quarter-res frame images in the bundle so the replay
    # panels have pixels to draw on; :none records artifacts only.
    trace = TrackTrace(keep_frames = :thumbs, dir = trace_dir, source = VIDEO)

    reader = VideoIO.openvideo(VIDEO)
    frames = (Gray.(f) for f in reader)
    tracks = try
        track_boards(
            board_features, net,
            N_FRAMES > 0 ? Iterators.take(frames, N_FRAMES) : frames;
            board_bundles, config = FindBoardsConfig(; geometry),
            device = DEVICE, on_frame = f -> push!(trace, f)
        )
    finally
        close(reader)
    end
    save_trace(trace_dir, trace)
    traces[label] = trace_dir

    println("\n== $label ==  ($(length(trace)) frames traced -> $trace_dir)")
    for uid in sort(collect(keys(tracks)))
        dets = tracks[uid]
        srcs = Dict{Symbol, Int}()
        for d in dets
            srcs[d.source] = get(srcs, d.source, 0) + 1
        end
        println(
            "  $uid  frames=$(length(dets))  " *
                "sources=$(join(("$k:$v" for (k, v) in sort(collect(srcs))), " "))  " *
                "mean inliers=$(round(mean(d.n_inliers for d in dets); digits = 1))"
        )
    end
end

# Replay both variants side by side — the interactive triptych (detections |
# association | boards) with a play button and scrubber, one window per arm.
screen = nothing
for (label, dir) in sort(collect(traces))
    global screen = display(
        GLMakie.Screen(; title = "track_boards — $label"),
        BlobBoards.replay(load_trace(dir))
    )
end
# Block only when run as a script — under include() the windows stay up.
abspath(PROGRAM_FILE) == (@__FILE__) && screen !== nothing && wait(screen)
