"""
PyTorch Dataset for BlobBoards training tracks (.tracks files, HDF5 format).

Loads tracks written by BlobBoards.build_tracks_for_sequences and exposes them
as (patch, label) pairs for metric learning.

Single-mode layout: one `.tracks` file holds exactly ONE patch type at ONE scale
(the filename is auto-suffixed `_<patch_type>_s<scale>`). Each patch carries a
companion board∩in-frame mask, and every sequence is flagged as a GT anchor or a
tracked sequence.

HDF5 layout (per-sequence groups; dimensions reversed by HDF5.jl, so a Julia
`P×P×N` array reads back as `(N, P, P)` in h5py):
    /sequences/<name>                      — attr is_anchor  (uint8: 1 anchor, 0 tracked)
    /sequences/<name>/tracks               — attrs patch_type (str), scale (float64)
    /sequences/<name>/tracks/patches       — (N, P, P) float32, values in [0, 1]
    /sequences/<name>/tracks/masks         — (N, P, P) float32; 1 where the pixel
                                             lands on the board sheet AND on a real
                                             image pixel, 0 over off-board / edges /
                                             perspective-warp voids
    /sequences/<name>/tracks/affine_shapes — (N, 2, 2) float32
    /sequences/<name>/tracks/track_id      — (N,) int32; GLOBALLY unique, stable
                                             per (board, blob): the same physical
                                             blob has the SAME id in every media
                                             sequence and in the board's anchor
                                             sequence. Use these ids directly —
                                             do NOT re-offset per sequence (see the
                                             DeprecationWarning in
                                             `_load_all_sequences`).
    /sequences/<name>/tracks/n_tracks      — int32 scalar
    /sequences/<name>/tracks/track_lengths — (T,) int32
    /sequences/<name>/untracked/patches    — (M, P, P) float32  [optional confusers]
    /sequences/<name>/untracked/masks      — (M, P, P) float32  [optional confusers]

Usage:
    dataset = BlobTrackDataset("tracks_logpolar_s16.tracks")   # all sequences
    dataset = BlobTrackDataset("tracks_logpolar_s16.tracks",
                               sequence="video1")              # single sequence
    dataset = BlobTrackDataset(path, with_mask=True)           # 2-channel patches
    sampler = MPerClassSampler(dataset.labels, m=3, batch_size=96)
    loader  = DataLoader(dataset, batch_size=96, sampler=sampler)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_all_sequences(f, sequence=None, with_mask=False):
    """Load and concatenate track data from per-sequence HDF5 groups.

    Returns (patches, masks, track_ids, track_lengths). `masks` is None when
    with_mask is False (or when a sequence predates the masks dataset).

    .. deprecated::
        The per-sequence ``tid_offset`` below is legacy behavior from when
        ``track_id`` was only unique *within* a sequence. Since the blob-id
        refactor (`blob_track_id` in ``src/training.jl``), ``track_id`` is already
        GLOBALLY unique and stable per (board, blob) — the same physical blob
        carries the same id across every media sequence and its anchor sequence.
        Re-offsetting therefore BREAKS cross-media identity: it hands the same
        blob a different label in each sequence. This loader keeps the offset only
        for backward compatibility with old `.tracks` files (1-based per-sequence
        ids); for files written by the current pipeline, use ``track_id`` directly
        and drop the offset. `BlobTrackDataset`'s ``valid_set`` logic assumes the
        old 1-based-per-sequence ids and must be revisited alongside this.
    """
    import warnings

    seqs = f["sequences"]
    seq_names = [sequence] if sequence is not None else list(seqs.keys())

    all_patches = []
    all_masks = []
    all_track_ids = []
    all_track_lengths = []
    tid_offset = 0

    if len(seq_names) > 1:
        warnings.warn(
            "blob_dataset re-offsets track_id per sequence (tid_offset); this is "
            "legacy behavior. Current .tracks files write globally-unique, "
            "stable-per-(board,blob) ids — re-offsetting breaks cross-media blob "
            "identity. Consume track_id directly for pipeline-current files.",
            DeprecationWarning,
            stacklevel=2,
        )

    for name in seq_names:
        sg = seqs[name]
        g = sg["tracks"]
        patches = g["patches"][:]
        track_ids = g["track_id"][:] + tid_offset
        track_lengths = g["track_lengths"][:]
        n_tracks = int(g["n_tracks"][()])

        all_patches.append(patches)
        if with_mask:
            all_masks.append(g["masks"][:] if "masks" in g
                             else np.ones_like(patches))
        all_track_ids.append(track_ids)
        all_track_lengths.append(track_lengths)
        tid_offset += n_tracks

    patches = np.concatenate(all_patches) if all_patches else np.zeros((0, 64, 64), dtype=np.float32)
    masks = (np.concatenate(all_masks) if all_masks else np.zeros((0, 64, 64), dtype=np.float32)) if with_mask else None
    track_ids = np.concatenate(all_track_ids) if all_track_ids else np.zeros(0, dtype=np.int32)
    track_lengths = np.concatenate(all_track_lengths) if all_track_lengths else np.zeros(0, dtype=np.int32)
    return patches, masks, track_ids, track_lengths


class BlobTrackDataset(Dataset):
    """Dataset of canonicalized blob patches labeled by track identity.

    Each item returns (patch, label) where:
      - patch: (1, P, P) float32 tensor (grayscale, values in [0, 1]), or
               (2, P, P) with the board∩in-frame mask as channel 2 if with_mask
      - label: int, track identity (contiguous 0-based)

    Only tracks with >= `min_track_length` observations are included.
    Loads all sequences by default; supply `sequence="name"` for one.
    """

    def __init__(self, h5_path, min_track_length=2, load_into_memory=True,
                 sequence=None, with_mask=False):
        self.with_mask = with_mask
        with h5py.File(h5_path, "r") as f:
            all_patches, all_masks, track_ids, track_lengths = \
                _load_all_sequences(f, sequence, with_mask=with_mask)

            # Identify valid tracks (long enough for positive pairs)
            valid_set = set(
                (np.where(track_lengths >= min_track_length)[0] + 1).tolist()
            )  # track_ids are 1-based per sequence (offset applied)

            # Build selection of patches belonging to valid tracks
            sel = np.isin(track_ids, list(valid_set))

            if load_into_memory:
                self.patches = all_patches[sel]
                self.masks = all_masks[sel] if with_mask else None
            else:
                self._h5_path = h5_path
                self._sequence = sequence
                self._indices = np.where(sel)[0]
                self.patches = None
                self.masks = None

            self._raw_labels = track_ids[sel]

        # Remap to contiguous 0-based labels
        unique, inverse = np.unique(self._raw_labels, return_inverse=True)
        self.labels = inverse.astype(np.int64)
        self.n_classes = len(unique)

        n = len(self.labels)
        print(
            f"BlobTrackDataset: {n} patches, {self.n_classes} tracks "
            f"(filtered {sel.sum()}/{len(sel)}, min_length={min_track_length}, "
            f"with_mask={with_mask})"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.patches is not None:
            patch = self.patches[idx]  # (P, P) float32
            mask = self.masks[idx] if self.with_mask else None
        else:
            with h5py.File(self._h5_path, "r") as f:
                patches, masks, _, _ = _load_all_sequences(
                    f, self._sequence, with_mask=self.with_mask)
                patch = patches[self._indices[idx]]
                mask = masks[self._indices[idx]] if self.with_mask else None

        patch = torch.from_numpy(patch).unsqueeze(0)  # (1, P, P)
        if self.with_mask:
            mask = torch.from_numpy(mask).unsqueeze(0)  # (1, P, P)
            patch = torch.cat([patch, mask], dim=0)     # (2, P, P)
        label = self.labels[idx]
        return patch, label


class BlobUntracked(Dataset):
    """Optional dataset of untracked patches for distractor injection.

    Each item returns a (1, P, P) float32 tensor with no label.
    Use these as additional negatives during training.
    Loads all sequences by default; supply `sequence="name"` for one.
    """

    def __init__(self, h5_path, sequence=None):
        with h5py.File(h5_path, "r") as f:
            seqs = f["sequences"]
            seq_names = [sequence] if sequence is not None else list(seqs.keys())
            parts = []
            for name in seq_names:
                sg = seqs[name]
                if "untracked" in sg:
                    parts.append(sg["untracked/patches"][:])
            if parts:
                self.patches = np.concatenate(parts)
            else:
                self.patches = np.empty((0, 64, 64), dtype=np.float32)
        print(f"BlobUntracked: {len(self.patches)} patches")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.from_numpy(self.patches[idx]).unsqueeze(0)
