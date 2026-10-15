# The descriptor was 93% of the frame because ORT ran it on the CPU

**Outcome: fixed by rewriting ten `Pad` nodes in the ONNX graph. No Lux port.**
Descriptors are bit-identical to the old model; `find_boards` went from
6.88 s to 2.07 s median per frame on the 260527 sweep, and the descriptor
dropped from 93% of the frame to 18%.

Driver: `bmvc2026/scripts/onnx_depad.py` (graph rewrite), `bmvc2026/scripts/go_bmvc2026.jl`
(sweep). Model: `models/track_logpolar_circ_s96.onnx` →
`models/track_logpolar_circ_s96_cuda.onnx`.

## The measurement that started it

Profiled 2026-08-08 on a 12 MP frame (260521/DSC03859), RTX 4090, CUDA runtime
12.9, ONNX CUDA EP confirmed active:

| stage | time |
|---|---|
| undistort | 0.009 s |
| ASIFT detect (25 tilts, upsampling on) | 0.35 s |
| adapt | 0.15 s |
| canonicalize | 0.14 s |
| **describe (ONNX)** | **5.19 s** |
| **total** | **5.6 s / frame** |

The ONNX path was pathological, not merely suboptimal: per-patch cost was flat
at ~0.85 ms from batch 1 to 512 (zero batch efficiency), and the CUDA EP was
only 2.17× faster than the pure-CPU EP. Both were CPU-class throughput for a
1.06 M-parameter CNN on 64×64 patches.

## Diagnosis: ORT node placement

ONNXRunTime.jl's `logging_level` sets the *env* logger, which ORT makes a
process-wide singleton — by the time a model is loaded it is already fixed at
`:warning`, so `load_inference(...; logging_level = :verbose)` silently does
nothing. The session-level severity is what matters, and it is not exposed by
the high-level API; reach into the C API instead:

```julia
const CAPI = BlobBoards.ONNXRunTime.CAPI
api  = CAPI.GetApi(; execution_provider = :cuda)
opts = CAPI.CreateSessionOptions(api)
CAPI.check_and_release(api,
    @ccall $(api.SetSessionLogSeverityLevel)(opts::Ptr{Cvoid}, Cint(0)::Cint)::CAPI.OrtStatusPtr)
CAPI.check_and_release(api,
    @ccall $(api.SetSessionLogVerbosityLevel)(opts::Ptr{Cvoid}, Cint(1)::Cint)::CAPI.OrtStatusPtr)
```

ORT logs from C++, so the output has to be captured at the file-descriptor
level (`dup`/`dup2` onto a temp file). Do **not** use `redirect_stdout` inside
a REPL that already captures output: it has swapped `stdout` for a capture object that
`Base.RedirectStdStream` cannot restore, and the restore throws, leaving fd 1
pointing at your log file.

What it said, on the original model:

```
16 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider
Node(s) placed on [CPUExecutionProvider]. Number of nodes: 12
  Concat (node_Concat_5), Concat (node_Concat_28)      <- shape arithmetic, harmless
  Pad (node_pad, _2, _4, _6, _7, _8, _10, _12, _13, _14)
Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 64
```

Every `Pad` in the network fell to the CPU. The CUDA EP has no kernel for
`Pad` in `wrap` or `edge` mode, so the graph was cut into ten CPU islands with
a device round-trip at each boundary — at batch 128 that is ~2 GB of PCIe
traffic and ten CPU-side copies of a 134 MB tensor per forward evaluation. That is
the flat 0.85 ms/patch.

## The fix

`Slice` and `Concat` *are* CUDA-supported, and circular/edge padding is
exactly a slice-and-concatenate. `bmvc2026/scripts/onnx_depad.py` rewrites each
`Pad` accordingly — wrap becomes `Concat(x[-b:], x, x[:e])`, edge becomes
`Concat(x[:1]×b, x, x[-1:]×e)` — leaving the weights untouched:

```
uv venv /tmp/onnxenv && uv pip install --python /tmp/onnxenv/bin/python onnx
/tmp/onnxenv/bin/python bmvc2026/scripts/onnx_depad.py \
    "$DATA/models/track_logpolar_circ_s96.onnx" \
    "$DATA/models/track_logpolar_circ_s96_cuda.onnx"
```

60 nodes → 80 nodes, 3 dead initializers dropped. Re-running the placement
probe on the result: **zero Memcpy nodes**, 78 nodes on CUDA, and the only two
left on the CPU are the same harmless int64 shape `Concat`s that feed
`Reshape`/`Expand`.

`onnx` is in no pixi env (it is declared under the `train` feature, which is
not installed, and neither `pip` nor conda-forge onnx is available in `default`
or `py`); a throwaway `uv venv` is the cheapest way in.

## Verification

**Descriptors are bit-identical.** 256 real canonicalized patches from the
profiled frame through both sessions:

```
max |Δ| over 256×128 descriptor entries = 0.0f0
min cosine similarity                    = 0.99999964   (Float32 dot rounding)
```

Not "≥ 0.999" — exactly equal. The rewrite is a pure op decomposition.

**Throughput** (min of 7, RTX 4090, µs per patch):

| batch | original | `_cuda` | speedup |
|---|---|---|---|
| 1 | 1103.3 | 420.3 | 2.6× |
| 32 | 791.7 | 48.2 | 16.4× |
| 64 | 827.0 | 50.2 | 16.5× |
| 128 | 863.5 | 63.2 | 13.7× |
| 256 | 884.7 | 73.1 | 12.1× |

Batching now does something.

`describe`'s `max_batch` default of 128 is no longer the right operating point.
Re-measured on the pipeline's own 3247 patches (DSC04540), host path:

| `max_batch` | 32 | 64 | 128 (default) | 256 | 512 | 0 (all) |
|---|---|---|---|---|---|---|
| µs/patch | 61.7 | 63.1 | 76.8 | 91.7 | 442.4 | 472.0 |

32–64 is ~20% better than the default, and past 256 the ORT arena thrashes and
throughput falls off a cliff. Worth lowering `DescriptorNetwork`'s default.

**End-to-end**, `find_boards` wall time per frame over the 56-image 260527 `bb`
subset, read from `elapsed_time_ms` in the pose JSONs:

| | median | mean | min | max |
|---|---|---|---|---|
| baseline (`poses/bb.upsample`) | 6.88 s | 7.54 s | 4.62 s | 23.70 s |
| `_cuda` model | 2.07 s | 1.97 s | 0.68 s | 3.12 s |

3.3× on the median, 3.8× on the mean, and the tail is gone — the worst frame
went 23.7 s → 3.1 s.

## Where the frame time goes now

Stage breakdown on DSC04540 (12 MP, 3247 blobs), each stage CUDA-synchronized:

| stage | time | share |
|---|---|---|
| undistort | 0.021 s | 2% |
| **ASIFT detect (25 tilts)** | **0.360 s** | **31%** |
| adapt | 0.137 s | 12% |
| canonicalize + describe (fused device path, `BlobBoardsCUDAExt`) | 0.213 s | 18% |
| match + RANSAC + pose (remainder) | ~0.42 s | 36% |
| **find_boards total** | **1.152 s** | |

The descriptor is no longer the bottleneck — detection is.

**And the 65.7 µs/patch that remains is not framework overhead.** The network
does ~1.49 GFLOP per 64×64 patch (the first three 5×5 convs run at full 64×64
with no downsampling), so 3247 patches is **4.84 TFLOP per frame**. At 0.213 s
that is **22.7 effective TFLOPS** — a sane fraction of a 4090's realistic
FP32/TF32 conv throughput. It is compute-bound in cuDNN. A Lux port would
dispatch to the same kernels and land in the same place; the only real levers
left are FP16/TF32 (~2×, an export-side change) or an architecture that
downsamples earlier.

**Pose statistics are unchanged.** Full 56 + 41-image sweep, 241 s end to end.
Same 56 files, **1272 detections, P(det) 0.757**, and the same board set in
every single image (zero added, zero dropped):

| ring | detections | rot median | rot p90 | trans median | flips (>10°) |
|---|---|---|---|---|---|
| 1 m | 476 / 476 | 0.41 / 0.41 | 0.93 / 0.93 | 3.06 / 3.06 mm | 1 / 1 |
| 2 m | 429 / 429 | 0.44 / 0.44 | 1.15 / 1.16 | 4.16 / 4.14 mm | 6 / 7 |
| 3 m | 367 / 367 | 0.60 / 0.59 | 27.16 / 25.71 | 5.43 / 5.43 mm | 45 / 44 |

(baseline / `_cuda`; degrees). Total flip count 52 both ways — one moved from
the 3 m ring to the 2 m ring.

Pose-by-pose, 1270 of 1272 agree to ≤ 0.08° and ≤ 0.05 mm. Two do not:
DSC04546/`51b` (138°, 2.79 mm) and DSC04569/`d68` (51°, 0.16 mm) — both
flips, i.e. the planar two-fold ambiguity resolving to the other branch. That
is RANSAC sampling, not the descriptors, which are bit-identical; the same ±1
flip jitter is visible between the runs recorded in
`flip_disambiguation.md` (1/6/40 there vs 1/6/45 here).

Since the descriptors are bit-identical, the rebuilt gallery holds identical
features to the old one — only the stored `network_hash` differs.

## The re-run was adopted as the dataset's pose set

`poses/bb` and `poses/calibration/bb` now hold the re-run, and everything
downstream was regenerated from them: `{april,aruco,blob}.csv` plus all ten
`occlusion/<NN>/` levels (`make_pose_csvs.jl`), the floor-plan/histogram TikZ
panels and `bb_stats.tex` (`plots/error_distributions.jl`), and
`flip_table.tex` (`flip_table.jl`, `BB_BASE=bb.noflip.up` — that is the base
that reproduces the shipped `0/10/48`). Pre-rewrite CSVs are kept beside them
as `*.csv.pre-cudaonnx.bak`, so the old figures are reproducible.

Two poses moving produced exactly two changes in the paper's derived content:

- `\bbRotPninetyMedium` 1.04° → 1.03° (the only one of the 48 `bb_stats.tex`
  macros that moved)
- flip table, `BlobBoards + mirror LRT` row: 1/6/45 → 1/7/44. Total 52 and
  corrected 6 are unchanged, as is the un-disambiguated 0/10/48 row.

`~/Documents/bmvc2026` rebuilds clean at 18 pages with the regenerated
fragments.

**`tab:timing` was NOT touched and is now stale.** It is hardcoded in
`blob_boards_bmvc.tex` (BlobBoards 3426.41 ± 788.47 ms per board) with no
generator anywhere in the repo, and that figure does not reconcile with the
pose JSONs either before or after this work — 7.54 s/frame over ~22.7
boards/frame is ~332 ms/board, an order of magnitude below it. Its provenance
needs to be established before it is re-measured; the prose around it ("about
3.4 s per board", "we expect substantial speed-ups from straightforward
engineering") depends on the same number.

## What changed in the repo

- `bmvc2026/scripts/onnx_depad.py` — the rewrite, so the artifact is reproducible
  from the original at any time.
- `bmvc2026/scripts/go_bmvc2026.jl`, `bmvc2026/scripts/tune_invariance.jl` — `ONNX_PATH`
  now defaults to `track_logpolar_circ_s96_cuda.onnx`. Both still honour
  `BB_ONNX`, so the original is one env var away.
- Data root: `models/track_logpolar_circ_s96_cuda.onnx` added beside the
  original (which stays — it is the provenance link to the training export).
  The gallery `260527/board_features_logpolar_s96.jld2` was rebuilt, since
  `BoardGallery` keys its cache on `network_hash = sha256(model file)`; the
  pre-rewrite copy is at `…jld2.pre-cudaonnx.bak`.

## If a Lux port is ever wanted anyway

It is no longer on the critical path, but the graph is now fully recovered, so
the expensive part of that work is done. `HardNetLogPolar`, 1,059,431 params,
input `(B, 1, 64, 64)`, output `(B, 128)`, opset 20. **H is the angular axis**
(circularly padded, then max-pooled away — that is where rotation invariance
comes from); **W is log-radius** (zero/edge padded).

| # | op | detail |
|---|---|---|
| 0–22 | instance norm | per-patch, over all 4096 pixels; `var` uses the Bessel factor 4096/4095; `x ← (x − µ)/(σ + 1e-7)` |
| 23–31 | 3× (wrap-pad H by 2, Conv 5×5 with zero-pad W by 2, ReLU) | 32, 32, 64 channels |
| 32–36 | blur-pool | wrap-pad H by 1, edge-pad W by 1, depthwise Conv 3×3 (`group=64`), stride 2 via two `Slice`s |
| 37–42 | 2× (wrap-pad H by 2, Conv 5×5, ReLU) | 64, 128 channels |
| 43–47 | blur-pool | as above, `group=128` |
| 48–50 | wrap-pad H by 2, Conv 5×5, ReLU | 128 channels |
| 51 | MaxPool | a 16×1 pooling window, stride 16×1 — collapses the angular axis |
| 52 | Conv | `(128, 128, 1, 16)`, the angular/radial readout |
| 53–59 | L2 normalize | with a `Greater`/`Where` zero guard at 1e-10 |

Spatial trace: `64×64 → 64×64 → 32×32 → 16×16 → 1×16 → 1×1`.

**Build it from the ONNX initializers, not from `best.pth`.** The export folded
BatchNorm into the convolutions — the ONNX graph has no BN nodes, and every
`Conv` carries a `features.N.weight` plus a synthetic `features.N.weight_bias`
that the `.pth` does not have. (In the checkpoint the convolutions are
bias-free and the BNs are `affine=False`, mean/var only, eps 1e-5.) Mixing the
two sources yields plausible-looking descriptors that quietly destroy matching.
Also: ONNX/PyTorch `Conv` is cross-correlation, so a Lux `Conv` needs
`cross_correlation = true`, and weights go `(O,I,kH,kW)` → `permutedims(W,
(4,3,2,1))` with data NCHW → WHCN.

Artifacts for that path are durable at
`models/track_logpolar_circ_s96_src/`: `best.pth`, `cfg.yaml`, and
`hardnet_logpolar.npz` (all 30 tensors, readable without torch). Source of
record: the training run's log directory,
`logs/2026_07_25_PreliminaryExperimentRealFirstIteration/track_logpolar_circ_s96/`.

## Still open

- **`max_batch`**: `DescriptorNetwork`'s default of 128 costs ~20% against 32–64
  now that the graph is GPU-resident. Not changed here — it is a src/ default
  that affects every caller, and the measurement above is one frame on one
  card.
- **Detection is the new bottleneck** (0.36 s, 31%), followed by the ~0.42 s of
  matching/RANSAC. That is where the next factor has to come from.
- The 3×3×3 tuning grid (`bmvc2026/scripts/tune_invariance.jl`) is still un-run — it
  was cancelled to free the GPU for the profiling. It tunes on session 260521
  (f/9.0, P(det) = 0.426) and reports held out on 260527. The estimate was
  ~37 min for 40 frames at the old descriptor speed; at ~3.5× it should now be
  ~11 min.
- The paper builds clean at 18 pages with all figures and Table 1 current.
