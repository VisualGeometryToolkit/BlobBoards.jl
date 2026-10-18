# =============================================================================
# Media enumeration — turning a directory tree into an ordered work list
# =============================================================================
#
# The pipeline's entry points take frames, not paths: `find_boards` takes images
# (independent), `track_boards` takes one video or one temporally-ordered
# sequence, `build_annotations` takes a flat list of sources. Getting from a
# corpus to those calls is the caller's job, and the ORDER matters more than it
# looks:
#
#   The ASIFT detector caches one scale-space plan per distinct frame size, and
#   the cache is bounded (LRU, one plan by default — see
#   `VisualGeometryToolkitFeatures.set_max_cached_plans!`). Interleaved resolutions
#   therefore rebuild a plan per medium; grouping makes it one rebuild per
#   group. Measured on a 13-resolution corpus of stills: 32 s/frame interleaved
#   versus ~2 s/frame when every frame shares a plan.
#
# These helpers supply the work list and the ordering. They deliberately do NOT
# decide which entry point to call — that is policy (a burst of stills may want
# tracking; a long video sampled at stride may not), and choosing it silently
# is how a corpus gets processed by the wrong path without erroring.
# =============================================================================

using VideoIO: VideoIO

# One authority for what counts as media. `build_annotations` dispatches on
# `is_video` too — it took a flat list of mixed sources long before the
# walkers existed.
const _VIDEO_EXTS = Set([".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".mts"])
const _IMAGE_EXTS = Set([".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".pgm"])

_ext(path) = lowercase(splitext(path)[2])
"""
    is_video(path) -> Bool

Whether `path` names a video source (by extension) — the classification
`walk_media` and the annotation drivers share.
"""
is_video(path) = _ext(path) in _VIDEO_EXTS
_is_image(path) = _ext(path) in _IMAGE_EXTS

"""
    media_resolution(path) -> Size2

Frame size of the image or video at `path`, as a `Size2` (`width`, `height`).

Videos report it from stream metadata without decoding; images fall back to a
decode, since Julia's image stack exposes no general header-only probe. Costs
~0.6 s for a video and ~0.8 s for a 24 MP still, so it is a preliminary probe, not
something to call per frame.

`Size2` rather than a tuple ON PURPOSE: `VideoIO.out_frame_size` is
(width, height) while `Base.size` is (rows, cols), and a bare tuple cannot say
which it holds — that ambiguity produced a real transposition bug here. The
named fields make both branches state the same thing.
"""
function media_resolution(path::AbstractString)
    if is_video(path)
        reader = VideoIO.openvideo(path)
        try
            w, h = VideoIO.out_frame_size(reader)   # VideoIO speaks (width, height)
            return Size2(width = Int(w), height = Int(h))
        finally
            close(reader)
        end
    end
    # `Size2(::AbstractMatrix)` reverses (rows, cols) for us — VGC intrinsic.
    return Size2(FileIO.load(path))
end

# `n_frames` → frame stride spanning a clip. ONE spelling: the annotation driver
# derived the count from duration × fps (≠ the container's frame count on VFR
# or truncated streams) and had no `n_frames ≥ 2` guard (div by zero at 1);
# the track builder had the guard and the honest count. Both call this.
function _frame_stride(video_path::AbstractString, n_frames::Integer)
    n_frames >= 2 || throw(ArgumentError("n_frames must be ≥ 2, got $n_frames"))
    n_video = VideoIO.get_number_frames(video_path)
    if n_video === nothing
        # ffprobe reports no nb_frames for Matroska/WebM: fall back to the
        # duration × frame-rate estimate the annotation driver always used.
        vid = VideoIO.openvideo(video_path)
        n_video = try
            round(Int, VideoIO.get_duration(video_path) * VideoIO.framerate(vid))
        finally
            close(vid)
        end
    end
    n_video > 1 || throw(
        ArgumentError(
            "cannot sample n_frames = $n_frames from $video_path: frame count unknown or ≤ 1 — supply `stride` instead"
        )
    )
    return max(1, div(n_video - 1, n_frames - 1))
end

"""
    walk_media(root; videos = true, images = true) -> Vector{String}

Every image and video under `root`, recursively, sorted for reproducible order.

`root` may also be a single file (returned as a one-element vector if it is
media) so callers need not special-case it. Supply `videos = false` or
`images = false` to restrict the kind — a still corpus belongs in `find_boards`
and a video corpus in `track_boards`, so they are usually walked separately.
"""
function walk_media(root::AbstractString; videos::Bool = true, images::Bool = true)
    keep(p) = is_video(p) ? videos : (images && _is_image(p))
    isfile(root) && return keep(root) ? [String(root)] : String[]
    out = String[]
    for (dir, _, files) in walkdir(root)
        for f in files
            p = joinpath(dir, f)
            keep(p) && push!(out, p)
        end
    end
    return sort!(out)
end

"""
    group_by_resolution(paths; probe = media_resolution)
        -> Vector{Pair{Size2, Vector{String}}}

Group media by frame resolution, largest group first, order preserved within
each group.

Process each group together and call
`VisualGeometryToolkitFeatures.clear_asift_plans!()` at the boundary: one plan build
per group instead of one per resolution change. Also stage the gallery once
(`GalleryIndex`) outside the loop and supply it as `gpu_index` — it is
independent of resolution, so rebuilding it per medium re-uploads the whole
descriptor set for nothing.

ORDER IS SAFE AT MEDIA GRANULARITY ONLY. Never reorder frames within one
medium: `track_boards` predicts each frame from the previous two, so shuffling
a sequence destroys the temporal association. Every frame of a video shares one
resolution anyway, so a video is intrinsically one group.
"""
function group_by_resolution(paths; probe = media_resolution)
    groups = Dict{Size2{Int}, Vector{String}}()
    for p in paths
        push!(get!(groups, probe(p), String[]), String(p))
    end
    # Largest group first: the dominant resolution amortises its plan over the
    # most media. Ties break on size for a deterministic order.
    return sort!(
        collect(groups);
        by = g -> (-length(g.second), g.first.width, g.first.height)
    )
end
