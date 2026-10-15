# `.tracks` file structure

Training tracks are HDF5 files written by
[`build_tracks_for_sequences`](../../src/training.jl) (single-mode format).

## One file = one patch type at one scale

`build_tracks_for_sequences(...; patch_type=:logpolar, scale=16)` writes every
patch as the **single** chosen `patch_type` (`:logpolar` | `:cartesian` |
`:original`) at the **single** chosen `scale` (patch half-width in units of σ).
The `output_name` is auto-suffixed `_<patch_type>_s<scale>` so different types /
scales never collide, e.g.:

```
tracks_logpolar_s16.tracks
tracks_cartesian_s16.tracks
```

## Sequences

Every media item × board produces one **tracked** sequence named
`<media>_<uid>`; every board also produces one **anchor** sequence named `<uid>`
holding its GT reference blobs. Anchors have their own sequences, so the
`is_anchor` flag lives on the sequence group (one flag, not one per patch).

Track ids are shared: a tracked sequence's blob `board_offset + i` has its GT
reference at the same id in the board's anchor sequence.

## HDF5 layout

Julia writes `P×P×N` column-major; h5py reads it back transposed as `(N, P, P)`.

```
/sequences/<name>                        group
    @is_anchor        uint8              1 = GT anchor sequence, 0 = tracked
  /tracks                                group
      @patch_type     string             "logpolar" | "cartesian" | "original"
      @scale          float64            patch half-width in σ
    patches           (P, P, N) float32  intensity, values in [0, 1]
    masks             (P, P, N) float32  board∩in-frame mask (see below)
    affine_shapes     (2, 2, N) float32  per-observation ellipse shape matrix
    track_id          (N,)      int32    blob identity per observation
    n_tracks          scalar    int32    number of distinct tracks
    track_lengths     (T,)      int32    observations per track
  /untracked                             group (tracked sequences only, optional)
    patches           (P, P, M) float32  confuser (unmatched-detection) patches
    masks             (P, P, M) float32  confuser masks (in-frame only)
```

## Masks

Each patch has a companion `P×P` mask marking which pixels genuinely belong to a
board:

- **tracked patches** — `1` where the pixel back-projects onto the board's world
  rectangle (`Rect(bundle, :world)`) **AND** onto a real image pixel; `0` over
  off-board background, image edges, and perspective-warp gray-fill voids.
- **confuser patches** — in-frame validity only (no board association).
- **anchor patches** — in-frame validity against the rendered board image.

The mask is computed by evaluating the same region map `canonicalize` uses on the
`P×P` output grid, recovering each pixel's source sample location, then testing
its in-frame validity and (for tracked patches) board-rectangle membership via
the inverse world→image homography.

## Reading (Python)

`blob_dataset.py` lives in `bmvc2026/scripts/training/` — put it on `sys.path`,
or run from that directory.

```python
from blob_dataset import BlobTrackDataset
ds = BlobTrackDataset("tracks_logpolar_s16.tracks")                # (patch, label)
ds = BlobTrackDataset("tracks_logpolar_s16.tracks", with_mask=True) # 2-ch patch+mask
```

## Building

```bash
julia --project=gpu -t auto examples/training/build_tracks.jl \
    VIDEO_DIR BOARD_DIR MODEL.onnx \
    --patch-type logpolar --scale 16 --n-frames 30 --output-dir OUT
```
