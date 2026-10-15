# results/

Write-ups of the experiments in `../../scripts/`, checked in so a result
and the code that produced it travel together (stack convention:
cau-tooling `EVAL-CONVENTIONS.md`).

| File | Driver | Data read from |
|---|---|---|
| `hand_eye.md` | `bmvc2026/scripts/make_hand_eye.jl` | `absolute_pose_experiments/260527/results/hand_eye_*.json` |
| `flip_disambiguation.md` | `bmvc2026/scripts/go_bmvc2026.jl` (BB_DISAMB=1) + `bmvc2026/scripts/flip_stats.jl` | `absolute_pose_experiments/260527/{poses/bb,blob.csv}` |
| `descriptor_inference_speedup.md` | `bmvc2026/scripts/onnx_depad.py` + `bmvc2026/scripts/go_bmvc2026.jl` | `models/track_logpolar_circ_s96{,_cuda,_src}` |

Each write-up names the driver that produced it and the data-root-relative
file it was read from. The data itself is distributed via the Julia
Artifacts system (`bmvc2026/Artifacts.toml`, Zenodo-hosted tiers) and
does not belong in the repo.
