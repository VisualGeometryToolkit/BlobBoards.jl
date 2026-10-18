# SLURM annotation pipeline

Distributed annotation across multiple GPUs via SLURM. Submits one
`sbatch` allocation with `--ntasks=N+1 --gpus-per-task=1` (driver +
N workers, each in its own cgroup → `CUDA_VISIBLE_DEVICES=0` inside
each task — no manual GPU pinning).

## Files

- `submit_blobboard_training_scm.sh` — entry point. Builds the source
  manifest, exports `BB_*` env vars, `sbatch`s the wrapper.
- `sbatch_scm.sh` — SLURM wrapper with `#SBATCH` directives. Runs the
  driver inside the allocation.
- `driver_scm.jl` — Julia driver. Uses `SlurmClusterManager` to spawn
  workers across the allocated tasks; distributes the source list.
- `build_source_list.jl` — builds a flat `(source|camera)` manifest
  from per-group glob patterns.

## Quick start (on the cluster's login node)

```bash
cd BlobBoards.jl
BB_DATA=/path/to/blobboard_training BB_OUT=/local/disk/bb_annot_output \
BB_PROJECT=/path/to/bb-slurm-env \
    bash bmvc2026/scripts/training/slurm/submit_blobboard_training_scm.sh
```

Output lands at `$BB_OUT/{truth_full_run,_sources.txt}` (timestamped).
SLURM log appears as `bb-annot-scm-<jobid>.log` in the cwd.

## Tunable env vars

| Var | Default | Notes |
|-----|---------|-------|
| `BB_DATA` | required | Must contain `boards/`, `images/`, `videos/`, `models/` |
| `BB_OUT` | required | A timestamped subdirectory is created. Must be on LOCAL disk (SMB hangs JLD2 writes) |
| `BB_PROJECT` | required | Julia project — registered deps only, no dev |
| `N_WORKERS` | `2` | GPU workers. Total tasks = N_WORKERS + 1 (driver). |
| `CPUS_PER_TASK` | `28` | 1/8 of an 8-GPU, 224-cpu node. Lower if other jobs are holding CPUs. |
| `BB_N_FRAMES` | `30` | Frames per video (stride decimated). |

## Monitoring

```bash
squeue -u $USER                     # job queue
tail -F bb-annot-scm-<jobid>.log    # SLURM log (errors here mean the job died)
```

The driver prints per-source `[N/total] Processing: …` lines and
per-frame `Frame X: Y blobs` summaries.

## Setting up the Julia env on a new machine

The env at `$BB_PROJECT` lives outside this repo by convention
(see comment in `submit_*.sh`). To create it from scratch:

```bash
mkdir -p /path/to/bb-slurm-env && cd /path/to/bb-slurm-env
julia --project=. -e 'using Pkg; Pkg.add(["BlobBoards", "VisualGeometryToolkitCore", "VisualGeometryToolkitFeatures",
    "CUDA", "cuDNN", "AcceleratedKernels", "NNlib", "Lux", "Optimisers", "Enzyme", "Reactant", "Zygote",
    "TensorBoardLogger", "SlurmClusterManager", "ClusterManagers", "CairoMakie", "FileIO", "ImageIO",
    "ImageMagick", "Images", "ImageDraw", "StaticArrays"]); Pkg.precompile()'
```

Registered packages only — no `Pkg.develop` calls, no `[sources]`
overrides. BlobBoards needs no forked package: the QR encoder is vendored
and SpatialIndexing is General's 0.1.6.

## SLURM partition expectations

- Partition: `gpu`
- Node: 8× B200, 224 cpus, 1.8 TiB RAM (what the defaults were sized for)
- Per-task share (default): 28 cpus, 8 GB/cpu = 224 GB
- Time limit (in `sbatch_scm.sh`): 4 hours

If another tenant is holding CPUs, lower `CPUS_PER_TASK` to fit
remaining capacity rather than queueing behind their job.
