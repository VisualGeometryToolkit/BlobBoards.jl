# =============================================================================
# The detector's pixel convention, end to end: COLMAP
# =============================================================================
#
# Every camera in the stack is a VGC `CameraModel`, canonical (COLMAP: the
# top-left pixel's centre at (0.5, 0.5)); detections meet cameras in the
# bundles, the pose fits and the recovery with no conversion between them. That
# is only right if the detections are COLMAP too, so it is pinned here through
# the real geometric path — the undistort pair, VGF's ASIFT detection, the
# printed-profile rescale, the adaptation — on dark isotropic blobs centred
# EXACTLY on known pixels. (Until 2026-09-22 the generic-grid bundle shifted
# detections by −0.5 on the belief that they were Julia-index, and BB's writers
# labeled them `PixelChart{:julia}` (now `PixelFrame`); measured here: COLMAP
# to 0.001 px.)

using Colors: Gray
using StaticArrays: SVector
using VisualGeometryToolkitCore: centroid
using BlobBoards: FindBoardsConfig, HomographyGeometry, detect_blobs, rescale_detections,
    adapt_detections, DEFAULT_CUTOFF, _undistorted_pair
import KernelAbstractions as KA

@testset "detector pixel convention: COLMAP, end to end" begin
    H, W, σ = 240, 320, 4.0
    # blob centres as JULIA array indices (x = column, y = row): two on pixel
    # centres, one at a quarter pixel; their COLMAP coordinates are 0.5 less
    centres = [(101.0, 81.0), (201.0, 161.0), (241.25, 71.75)]
    img = [
        Gray{Float32}(clamp(1.0 - 0.8 * sum(exp(-((c - x)^2 + (r - y)^2) / (2σ^2)) for (x, y) in centres), 0, 1))
            for r in 1:H, c in 1:W
    ]
    cfg = FindBoardsConfig(; geometry = HomographyGeometry())
    device = KA.CPU()
    detect_img, adapt_img = _undistorted_pair(img, cfg.geometry, device)
    dets = detect_blobs(detect_img; config = cfg.detect, device)
    frames = adapt_detections(
        rescale_detections(dets, DEFAULT_CUTOFF), adapt_img;
        cutoff = DEFAULT_CUTOFF, adapt_config = cfg.adapt, device,
    )
    @test length(dets) == length(centres) && length(frames) == length(centres)
    nearest(pts, p) = pts[argmin([hypot((q - p)...) for q in pts])]
    raw = [SVector{2, Float64}(centroid(d)...) for d in dets]
    adapted = [SVector{2, Float64}(centroid(f)...) for f in frames]
    for (x, y) in centres
        colmap = SVector(x - 0.5, y - 0.5)
        @test maximum(abs, nearest(raw, colmap) - colmap) < 0.03       # the detector (measured ≤ 0.018)
        @test maximum(abs, nearest(adapted, colmap) - colmap) < 0.005  # after adaptation (measured ≤ 0.001)
    end
end

# ONE detection cap: find_boards' (applied per polarity group, `_describe_for_boards`)
# is detect_blobs' — Features' `spread_strongest`, about `max_detections` of the
# strongest SPREAD over the image, then ordered by response. Until 0.7.0 the board path
# kept the top-N by |response| (a copy of Features' retired `strongest`), so on a frame
# whose strong blobs crowd one side it kept none of the other side.
@testset "one detection cap: find_boards' is detect_blobs'" begin
    H, W = 240, 480
    strong = [(40.0 + 40i, 40.0 + 40j) for i in 0:4 for j in 0:4]              # left half, contrast 0.8
    weak = [(280.0 + 40i, 40.0 + 40j) for i in 0:4 for j in 0:4]               # right half, contrast 0.3
    img = [Gray{Float32}(clamp(1.0 -
                0.8 * sum(exp(-((c - x)^2 + (r - y)^2) / (2 * 3.0^2)) for (x, y) in strong) -
                0.3 * sum(exp(-((c - x)^2 + (r - y)^2) / (2 * 3.0^2)) for (x, y) in weak), 0, 1))
           for r in 1:H, c in 1:W]
    dets = detect_blobs(img; device = KA.CPU())
    N = 20                                                                    # < the 25 strong blobs: top-N by response is all left
    @test length(dets) > N
    capped = BlobBoards._cap_detections(dets, size(img), N; device = KA.CPU())
    @test [centroid(d) for d in capped] == [centroid(d) for d in detect_blobs(img; max_detections = N, device = KA.CPU())]
    @test any(d -> centroid(d)[1] > W / 2, capped)                             # the weak side is represented
end
