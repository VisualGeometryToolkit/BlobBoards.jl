# Board Training Tracks

```@meta
CurrentModule = BlobBoards
```

`build_tracks_for_sequences` is the current driver for generating descriptor
training data from **calibration boards**. It takes a directory of media (videos
and/or photos), a directory of board `.json` files, and an ONNX descriptor model,
and accumulates multi-modal, multi-scale patch tracks into a single `.tracks`
(HDF5) file.

!!! note "How observations are keyed"
    `build_tracks_for_sequences` detects the board in each frame independently
    and transfers the board's ground-truth blobs through the estimated
    **world→image** homography, so every observation of a blob is keyed by its
    board-relative identity rather than by a chain of pairwise matches. Tracks
    therefore survive occlusion and frame gaps.

    An earlier `build_tracks` chained correspondences frame-to-frame from
    *relative* GT homographies in a `.npy` file. It was removed in 2026-08 along
    with `merge_tracks`, the `Vector{Track}` HDF5 layout and its reader; this is
    now the only track builder.

## Quick Start

```julia
using BlobBoards

out = BlobBoards.build_tracks_for_sequences(
    "/data/boards",          # board .json files
    "/data/models/blob_matcher.onnx",
    "/data/videos";          # videos and/or photos
    stride = 10,
    max_detections = 5000,   # see "Detection cap" below — set this on 4K+ footage
    output_dir = "/data/out",
    output_name = "tracks",
)
# => "/data/out/tracks_logpolar_s32.tracks" (the name gets _<patch_type>_s<scale>)
```

Run with `julia --threads=N` (e.g. 16). Two stages use the extra threads: the
per-frame correspondence search, and the async HDF5 writer.

## What It Produces

For each media item × board that is successfully detected, one sequence named
`<media>_<uid>`, plus one **anchor sequence** per board named by the bare board
`uid`.

| Sequence | Name | Contents |
|----------|------|----------|
| Tracked | `<media>_<uid>` | Observations of each GT blob found in the media, 1 obs per frame it was matched in |
| Anchor | `<uid>` | Every GT reference blob rendered from the board image, exactly 1 obs each |

Anchor and tracked sequences share track ids (`board_offset + i`, where
`board_offset` is derived from the board preamble hash), so a track's ground-truth
reference patch is recoverable by id. Anchors are **not** seeded into the tracked
sequences.

Each observation stores one patch — of the run's `patch_type` (`:logpolar`,
`:cartesian` or `:original`) at its `scale` — plus a mask and the observation's
`affine_shape`; a file holds one patch type at one scale (run again for
another). Cartesian and log-polar patches are sampled off the **adapted**
elliptical frame; `original` is sampled off the raw **SIFT-detection** disc, so
its scale is relative to the detector's σ rather than the fitted σ.

## Pipeline

Per media item:

1. **Read** — videos are sampled by `n_frames`/`stride`; photos are a single frame.
   Frames over `max_image_pixels` are downscaled at read time (see below).
2. **Track boards** — [`track_boards`](@ref) detects and tracks every board across
   the frames, yielding a per-frame world→image homography and a fit `quality`.
3. **Augment** — optional resolution-loss resample and/or perspective warp.
4. **Detect + extract** — one `detect_blobs` → `adapt` → `filter_blobs` →
   `canonicalize` call per frame, shared across all boards in that frame.
5. **Correspond** — the board's GT world blobs are transferred through the
   homography and validated by symmetric IoU (`iou_threshold`).
6. **Save** — tracks with ≥1 observation are kept; unmatched detections become
   confusers, capped and randomly subsampled.

Step 5 runs **once per frame**, not once per (frame, board): a frame containing
four boards is extracted once and matched against all four.

## Augmentation

Two augmentations may fire independently on each frame, both before extraction.

### Resolution loss

With probability `p_downsample` (default `0.75`) the frame is down- then
up-sampled (0.5 / 0.25 / 0.125×) to simulate blur and scale loss. The resample
preserves image coordinates, so the homography stays valid. Conditional on
firing, the three tiers (0.5× / 0.25× / 0.125×) are equally likely.

### Perspective

With probability `p_perspective` the frame is warped by a homography that
**strengthens the board's real foreshortening**, rather than applying a
perspective-agnostic random warp.

A board's world→image homography already encodes the plane's foreshortening: its
projective (bottom) row is the board plane's vanishing line in board coordinates.
Scaling that row by `1+γ` amplifies the existing foreshortening along the board's
own tilt direction; a small random-direction tilt adds pose diversity on top.
Magnitudes are applied in a board-normalized frame (world extent → `[-1,1]²`), so
they are unit- and board-size-invariant.

The board centroid is a **fixed point** of the warp — the world center maps to the
normalized origin, whose image is the third column of the normalized homography,
untouched by both the row-scaling and the jitter — so the board stays put in the
frame while only its foreshortening changes.

One warp applies to the whole frame, so it is anchored to the **primary**
(highest-quality) board; the other boards ride along, and every board's
world→image homography is composed with the same `H_aug`, keeping all
correspondences valid.

**Warp-void clip filter.** A warp pulls image content in from outside the frame,
leaving a fill void *inside* the array. `filter_blobs` checks each core ellipse
against the full array and is blind to that interior boundary, so when a warp
fires each ellipse is additionally pulled back through `inv(H_aug)` and checked
against the original rectangle with the same `max_clip_ratio=0.2` — treating the
void exactly like the real image edge. It is a no-op on unwarped frames.

## Anti-Aliasing

`supersample` is forwarded to [`canonicalize`](@ref) for every view and scale.
At the large scales (`64`/`96`/`128`) one output texel spans many source pixels,
so the default single-tap sampling aliases badly at the outer radii — and because
the resulting moiré depends on subpixel position, it is not repeatable across
views of the same blob. `supersample > 1` samples on a finer grid and box-averages
down.

Cost scales as `supersample²` and is paid at every scale, including the small ones
that do not alias. `3`–`4` is the quality sweet spot.

## Keyword Arguments

### Sampling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_frames` | `nothing` | Uniformly sample N frames (stride computed) |
| `stride` | `nothing` | Process every `stride`-th frame |
| `max_image_pixels` | `nothing` | Cap on total pixels per frame; `nothing` = native |

`n_frames` and `stride` are mutually exclusive; neither = every frame. Both are
ignored for photos.

`max_image_pixels` is a **pixel budget**, not a longest-side cap, so square frames
are tamed too. It is required for high-resolution stills: the Vulkan SIFT detector
allocates a full image pyramid sized to the frame and OOMs on e.g. a 24 MP phone
photo. 4K footage is unaffected. Applied at read time, so detect / track / adapt /
canonicalize all share one coordinate system.

### Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iou_threshold` | `0.6` | Minimum symmetric IoU for a valid correspondence |
| `max_detections` | `typemax(Int)` | Per-frame detection cap |

**Quality filter.** A spurious board's fitted homography does not explain the
image, so its quality goes negative, whereas genuine detections score positive —
`0.0` sits on the sign boundary. Transferring a spurious board's GT blobs would
inject wrong correspondences into the training set. Note that inlier and matched
counts do **not** separate the two cases (a weak genuine detection can have fewer
inliers than a spurious one), which is why the filter is on quality.

**Detection cap.** Set `max_detections` (e.g. `5000`) on high-resolution footage.
Without it the GPU descriptor matcher runs a detections × board-features GEMM that
can exhaust GPU memory — a Sony FX3 frame yields ~50k detections.

### Patches

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_type` | `:logpolar` | `:logpolar`, `:cartesian` or `:original`; one per file |
| `scale` | `32` | Patch scale = half patch width/height in σ; one per file |
| `supersample` | `1` | Anti-aliasing supersample factor (see above) |
| `confuser_factor` | `1.0` | Cap confusers at `factor × tracked patches` |

Each confuser gets **one** randomly chosen scale; its log-polar and cartesian
views are stored under `untracked/`. All confusers are collected in memory per
board-sequence before the random subsample — lower `max_detections` if memory is
tight.

### Augmentation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_downsample` | `0.75` | Probability of the resolution-loss down/up-resample; `0` disables |
| `p_perspective` | `0.5` | Probability of the perspective warp; `0` disables |
| `perspective_strengthen_max` | `1.0` | `γ ~ U[0, max]`; vanishing line scaled by `1+γ` |
| `perspective_jitter_max` | `0.1` | Extra random-direction tilt, board-normalized |

`perspective_strengthen_max = 0` reproduces the frame's real perspective; larger
values exaggerate it along the board's own tilt.

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | `"."` | Directory for `$output_name.tracks` |
| `output_name` | `"tracks"` | Output file basename |
| `write_anchors` | `true` | Write the per-board GT anchor sequences |
| `async_writer` | `true` | Overlap HDF5 compression with GPU compute |
| `writer_buffer` | `1` | Depth of the async-writer queue |
| `backend` | `default_device()` | Compute backend for adapt/canonicalize/describe |

## Resume

Sequences already present in the output file are skipped — both `<media>_<uid>`
tracked sequences and `<uid>` anchor sequences — so an interrupted run can simply
be re-invoked with the same arguments.

## Async Writer

zlib compression is single-threaded and dominates wall-clock on anchor galleries,
leaving the GPU idle. With `async_writer=true` a single background task owns the
output file and performs every HDF5 write, fed by a bounded channel; the GPU
thread hands off each finished sequence and proceeds to the next while compression
runs concurrently.

One writer = one file owner, which is required: concurrent writers to a single
HDF5 file corrupt it. Real overlap needs `julia -t ≥2`; under `-t 1` it degrades
to serial (still correct). Set `async_writer=false` for synchronous writes on the
GPU thread — an escape hatch for `-t 1` and memory-tight runs.

`writer_buffer` trades memory for slack: larger values let the GPU thread run
further ahead but hold more finished (large) sequences in memory. `0` is a
rendezvous.

Since the workload is compression-bound, the writer overlaps GPU time behind zlib
but cannot speed zlib itself — a faster codec (LZ4/Blosc) is the complementary
lever.

## Performance Notes

- **Threading**: only `symmetric_iou_match` parallelizes (pure-CPU ellipse/IoU
  over the frame's detections). HDF5 compression and the media loop do not.
- **Media-level parallelism is not available via threads.** VulkanSift holds a
  single global Vulkan instance, and its handle is evicted and destroyed on any
  image-size change, so concurrent detection from multiple threads races on that
  global. Per-thread ONNX sessions do not unlock it. True concurrent detection
  needs separate *processes*, each of which replicates the board gallery and CUDA
  context.
- **`write_anchors=false`** is a large speedup for diagnostics that only need the
  tracked sequences — anchor galleries are big and compression-bound.
- **CUDA/Vulkan memory**: the driver calls `GC.gc(); CUDA.reclaim()` between media
  items so the growing CUDA pool leaves physical GPU memory for Vulkan's pyramid
  init.
- **Progress output**: Julia buffers `println` to redirected files, so progress
  prints only flush on exit. Watch the output `.tracks` file (HDF5 flushes on
  write) as the live success signal.

## HDF5 Layout

One file holds a single patch type at a single scale (both are recorded as
`tracks`-group attributes and encoded in the file name — e.g.
`tracks_logpolar_s32.tracks`).

```
/sequences/
    <media>_<uid>/                  # tracked sequence
        tracks/                     # @attr patch_type (String), scale (Float64)
            patches               — (P, P, N) Float32, chunked + compressed
            masks                 — (P, P, N) Float32  # board ∩ in-frame, per pixel
            affine_shapes         — (2, 2, N) Float32
            frame_id              — (N,) Int32   # source frame of each observation
            homographies          — (3, 3, F) Float64  # one H_wi per detected frame
            view_angles           — (F,) Float64        # per-frame obliquity, radians
            homography_frame_ids  — (F,) Int32   # frame index of each H_wi column
            n_visible             — (F,) Int32   # board GT blobs inside the frame
            n_detections          — (F,) Int32   # blobs detected in that frame
            track_id              — (N,) Int32
            n_tracks              — Int32 scalar
            track_lengths         — (T,) Int32
        untracked/                  # confusers
            patches               — (P, P, M) Float32
            masks                 — (P, P, M) Float32
    <uid>/                          # anchor sequence — no untracked/, no frame_id,
        tracks/...                  # and no homographies/view_angles (GT reference)
```

Detection geometry is stored **per frame, not per blob**: `homographies` holds
one estimated world→image homography `H_wi` (including any perspective
augmentation) for each of the `F` frames the board was detected in, and
`view_angles` the matching viewing obliquity in radians — derived intrinsics-free
from the local affine tilt `t = σ_max/σ_min` of `H_wi` at the board centroid
(`acos(1/t)`; `0` = fronto-parallel, `π/2` = edge-on), metric because the board is
physically square. `homography_frame_ids` gives the frame index of each column, so
an observation joins to its pose via `frame_id[k]` → the matching entry of
`homography_frame_ids`. `n_visible` counts the board's own GT blobs that project
inside the frame — the honest denominator for how many it matched — and
`n_detections` how many blobs the detector kept anywhere in the frame. All six are
written for tracked sequences only; GT-anchor sequences omit them. (One `H_wi` per
board per frame assumes at most one detection of a board per frame, which
`track_boards`' spatial suppression enforces.)

HDF5.jl reverses dimensions on write (Julia column-major → HDF5 row-major), so
h5py reads `patches` as `(N, P, P)`, `homographies` as `(F, 3, 3)` and
`view_angles`/`homography_frame_ids`/`n_visible`/`n_detections` as `(F,)` directly.

## Detection Yield vs. Viewing Angle

Track construction charts how the per-board blob yield falls off as a board tilts
away from fronto-parallel. `build_tracks_for_sequences` writes
`<output_name>_view_angles.png` next to the `.tracks` file at the end of a run, and
`build_absolute_tracks` writes a per-media `<media>_view_angles.png` into
`debug_dir` when one is set. Rendering needs a Makie backend — `using CairoMakie`
is the headless choice, and `examples/training/build_tracks.jl` loads it for
you. Without a backend the same binning prints as a terminal bar chart instead.

Three panels share one obliquity axis:

| Panel | Shows | Read it for |
|---|---|---|
| matched blobs / board-frame | mean ±1 sd | the raw yield — what a training batch gets |
| recall | `Σ matched / Σ visible GT` | yield with board-in-frame size divided out |
| board-frames | samples per bin | which bars carry enough support to believe |

One sample is one *(board, frame)* detection. **matched** is the board's GT blobs
that won a bidirectional-IoU correspondence. **visible** is its GT blobs whose
centers project inside the frame — this is what recall divides by, and it falls as
the board leaves the frame, so a partly-out-of-frame board doesn't read as a
detector failure. The separately recorded `n_detections` is frame-level and
board-agnostic (every board in a frame shares it), so it is context, not a
denominator.

Nothing extra is stored for the chart — it is rebuilt from `view_angles`,
`n_visible` and per-observation `frame_id` counts, so it is recomputable offline:

```julia
stats = BlobBoards.view_angle_profile("tracks_logpolar_s32.tracks")     # records, no plotting
BlobBoards.save_view_angle_figure("profile.png", stats)                 # needs CairoMakie
BlobBoards.save_view_angle_figure("fine.svg", stats; edges = 0:5:90)    # finer bins, vector out
BlobBoards.save_view_angle_figure("one_board.png", filter(s -> s.uid == uid, stats))
BlobBoards.print_view_angle_profile(stats)                              # text, no backend needed
```

Files built before `n_visible` existed still profile; the recall panel says so
rather than drawing an empty one.


## Known Issues

- Vulkan SIFT logs `~25-30k feature(s) lost because the SIFT buffer was full` on
  8 MP photos — the detector's internal buffer caps below the detected count
  (before `max_detections` applies). This biases which blobs survive; raise the
  SIFT buffer size if completeness matters. Non-fatal.
- `DomainError: sqrt(-1.5e-11)` from `RANSACScoringDoneRight._mahalanobis` during
  RANSAC is caught per-board by `find_board` ("skipping this board for this
  frame"). Rare, upstream, non-fatal.
