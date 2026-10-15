# Single-view planar-pose flip disambiguation: a measured negative result

Drivers: `bmvc2026/scripts/go_bmvc2026.jl` with `BB_DISAMB=1`
(PoseGeometry `resolve_pose_ambiguity = true`), `bmvc2026/scripts/flip_stats.jl`;
forensics run interactively on frame DSC04560 / board `0d9`
(3 m ring, station −20.7°, certified flip: 35.6° rotation error).
Data: `absolute_pose_experiments/260527`, log-polar s96 detections.

## The mechanism (kept, off by default)

After RANSAC converges, reflect the winner about the viewing ray
(`VisualGeometryCore.mirror` — the planar two-fold ambiguity's second
branch, a PROPER rotation of the same board, separated by twice the
viewing tilt), refine the reflection with the same seeded LO, select by
a per-blob scale-field margin. The machinery verifiably works: on the
certified flip the mirror branch refined to 1.2° from mocap ground truth
— the correct pose was constructed and available.

## The result

Zero of 40 certified 3 m flips corrected; zero healthy detections
disturbed (sweep statistics identical with the flag on/off: rotation
p90 per ring 0.95°/1.18°/25.8°, flips 1/6/40).

## Why: every in-view cue is branch-invariant at this operating point

Measured on the certified case (4 cm board at 3 m ≈ 33 px, 7-inlier
fit from 52 descriptor matches):

| channel | branch difference | noise floor | verdict |
|---|---|---|---|
| reprojection residuals | 0.023 px mean, 0.13 px max | ~0.4 px | tie |
| marginal score | margin 0.8, FLAT across halfwidths 0.5–5 px | — | tie (band-independent) |
| scale field (log RSS) | 0.10 | 55 (45/52 matches are cross-scale mismatches) | tie |
| predicted frames (216 blobs, exact conic projection) | 0.038 px center, 0.3–0.8% axes, 0.03° orientation | — | canonicalized patches — hence descriptors/embeddings — identical |

Root cause is geometric even-ness: tilting a plane by ±θ about an
in-plane axis yields the SAME orthographic image of ANY pattern
(foreshortening cos θ is even; blob placement — random or structured —
is irrelevant, and the mirror branch is a rotation, never a chirality
flip). Only perspective breaks the tie, at O(board angular size), ≈1%
per blob here — below every measurement noise in the system. Position,
scale, and appearance are all functions of the predicted image, and
both branches predict the same image.

## The likelihood-ratio selector: promising, then measured, then retracted

The post-LO paired LRT (compare the branches' marginal scores at their
optima) initially measured 97.4% correct over 5028 GT-joined telemetry
records (82.5% in the d<1 core) — but the telemetry population is the raw
candidate soup (12k agenda candidates, translation-join-biased), and the
implementation compared qualities from DIFFERENT scoring contexts
(RANSAC's marginal scorer vs locally_optimize's Jeffreys default). The
validation sweep exposed it: with that selector live, healthy small
boards flipped at EVERY ring (rot medians 0.40/0.44/0.60° →
5.5/5.3/4.8°; flips 1/6/40 → 61/55/55). Poses were restored from backup;
`resolve_pose_ambiguity` stays off.

The SYMMETRIC LRT — both branches through one identical seeded-LO call,
on the SAME correspondence set (fix the measurements, fix the scorer,
vary only the hypothesis) — is the correct form, and it WORKS:

  1 m: 0.40°/0.95°, flips 1 (unchanged — do-no-harm succeeds)
  2 m: 0.44°/1.15°, flips 6 → 4
  3 m: 0.58°/10.28°, flips 40 → 32; p90 25.6° → 10.3°

Ten flips corrected, zero created, medians untouched. The remaining 32
are the d≈0 core where the LR correctly ties — those need context
(panel vote / temporal) or the ambiguity flag.

## Disposition

- The symmetric-LRT selector ships behind `resolve_pose_ambiguity` (still
  default-false pending validation beyond this dataset); the 260527
  poses/bb of record are the symmetric-LRT sweep. The branch margin is
  additionally the per-detection AMBIGUITY FLAG (near-zero ⇒ orientation
  unreliable), which four-corner tags cannot even express.
- Single-view resolution below this operating point is information-
  starved — for tags as well (they flip 2–3× more; blob boards defer
  the wall by roughly one distance ring but do not escape it).
- Real fixes are out-of-view: (1) PANEL VOTE — the 30 boards share one
  rigid rig; a flipped board disagrees with 29 neighbors by ~2×tilt;
  a post-hoc consensus would deterministically repair all 40 flips;
  (2) temporal consistency (track.jl) for video; (3) the flag for
  single-shot consumers.
