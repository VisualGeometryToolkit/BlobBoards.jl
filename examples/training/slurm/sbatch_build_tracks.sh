#!/bin/bash
# sbatch wrapper for track generation (build_tracks_for_sequences via the
# examples/training/build_tracks.jl CLI). Single task, single GPU — the builder
# runs serially on the GPU thread and overlaps HDF5 compression on a background
# writer task, so request >=2 threads (`-t auto` below).
#
# Submit:
#   BB_VIDEO_DIR=/data/footage BB_BOARD_DIR=/data/boards \
#   BB_MODEL_PATH=/data/model.onnx BB_OUTPUT_DIR=/data/out \
#   mkdir -p logs && sbatch examples/training/slurm/sbatch_build_tracks.sh   # log -> logs/bb-tracks-<jobid>.log
#
# BB_REPO is the BlobBoards.jl checkout (it holds the CLI). Set it explicitly
# when submitting from outside the repo; otherwise it is inferred — see below.
#
# BB_PROJECT defaults to $BB_REPO/gpu, the GPU env: CUDA and cuDNN are
# BlobBoards [weakdeps] and cannot be loaded from `--project=<the checkout>`
# itself, so pointing this at the package dir silently demotes the run to the
# CPU backend + ONNX CPU EP. Override only with another env that has BlobBoards
# + CUDA + cuDNN (e.g. the shared bb-slurm-env).
#
# One file = one patch type at one scale; override via BB_PATCH_TYPE / BB_SCALE.
# On a big GPU set BB_UNCAP=1 to drop the detection / pixel / confuser caps.
#SBATCH --job-name=bb-tracks
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bb-tracks-%j.log
#SBATCH --error=logs/bb-tracks-%j.log

set -euo pipefail

# --- Locate the checkout ------------------------------------------------------
# The CLI ships in the repo, but ${BASH_SOURCE[0]} does NOT point there under
# sbatch: SLURM stages the script into the job spool dir, so it resolves to
# /var/spool/slurmd/job<id> and ../../.. lands on /var. Derive the repo from
# BB_REPO, else from BB_PROJECT (whichever of the checkout or <checkout>/gpu it
# names), else from the submit dir; the script's own dir is the last resort and
# is only right for a direct `bash examples/training/slurm/...` run.
_cli_rel="examples/training/build_tracks.jl"
# Resolved up front, not inside the candidate list: the list is expanded in full
# before the loop runs, so an unguarded command substitution here would fire
# (and, with BASH_SOURCE unset, trip `set -u`) even when an earlier candidate wins.
_src="${BASH_SOURCE[0]:-}"
_script_repo=""
[[ -n "$_src" ]] && _script_repo="$(cd "$(dirname "$_src")/../../.." 2>/dev/null && pwd || true)"
if [[ -z "${BB_REPO:-}" ]]; then
  for _cand in \
      "${BB_PROJECT:-}" "${BB_PROJECT:+$BB_PROJECT/..}" \
      "${SLURM_SUBMIT_DIR:-}" \
      "$_script_repo"; do
    [[ -n "$_cand" && -f "$_cand/$_cli_rel" ]] || continue
    BB_REPO="$(cd "$_cand" && pwd)"
    break
  done
fi
[[ -n "${BB_REPO:-}" && -f "$BB_REPO/$_cli_rel" ]] || {
  echo "cannot locate the BlobBoards checkout — set BB_REPO=/path/to/BlobBoards.jl" >&2
  exit 1
}

# --- Required paths -----------------------------------------------------------
BB_PROJECT="${BB_PROJECT:-$BB_REPO/gpu}"
# The checkout itself cannot load CUDA (both it and cuDNN are [weakdeps]), so a
# BB_PROJECT pointing at it runs the whole job on the CPU backend + ONNX CPU EP
# — correct, but hours slower. Warn rather than override an explicit setting.
[[ "$(cd "$BB_PROJECT" 2>/dev/null && pwd)" == "$BB_REPO" ]] &&
  echo "WARNING: BB_PROJECT=$BB_PROJECT is the checkout — CUDA is a weakdep there and cannot load; this job will run CPU-only. Use BB_PROJECT=$BB_REPO/gpu." >&2
: "${BB_VIDEO_DIR:?must be set (dir of videos/photos)}"
: "${BB_BOARD_DIR:?must be set (dir of board .json files)}"
: "${BB_MODEL_PATH:?must be set (descriptor .onnx)}"
: "${BB_OUTPUT_DIR:?must be set (output dir)}"

# --- Optional knobs (CLI defaults in build_tracks.jl) -------------------------
BB_JULIA="${BB_JULIA:-julia}"                       # or "pixi run julia"
BB_THREADS="${BB_THREADS:-auto}"                    # >=2 enables writer overlap
BB_PATCH_TYPE="${BB_PATCH_TYPE:-logpolar}"          # logpolar|cartesian|original
BB_SCALE="${BB_SCALE:-96}"
BB_MIN_INLIERS="${BB_MIN_INLIERS:-20}"
BB_OUTPUT_NAME="${BB_OUTPUT_NAME:-tracks}"
# Detection cap. Default 15000 (NOT uncapped): boards are accepted only when
# >=BB_MIN_INLIERS of their blobs survive the top-N-by-response cut, so the cap
# gates DISTANT/OBLIQUE boards whose low-contrast blobs rank below scene clutter.
# Profiled on real footage: 5000 misses distant boards that 15000 recovers; the
# per-frame cost is ~linear in the cap and, with the guided_search spatial-grid
# fix, dominated by the GPU match/describe (cheap on a big GPU). Memory is not the
# binding constraint on a large card (the match GEMM is #board_features x N_det x
# 4B; 15000 is a few GB), so raise this for maximum board coverage; lower it only
# if throughput-bound. Uncapped OOMs a 12GB GPU and is not worth it — extra
# detections past board saturation are just clutter. BB_UNCAP=1 still overrides.
# Optional extras: BB_N_FRAMES / BB_STRIDE (mutually exclusive), BB_CONFUSER_FACTOR,
# BB_SUPERSAMPLE, BB_MAX_IMAGE_PIXELS, BB_DEBUG_DIR, BB_UNCAP=1.

CLI="$BB_REPO/examples/training/build_tracks.jl"

# --- Assemble CLI flags -------------------------------------------------------
args=(
  "$BB_VIDEO_DIR" "$BB_BOARD_DIR" "$BB_MODEL_PATH"
  --patch-type "$BB_PATCH_TYPE" --scale "$BB_SCALE"
  --min-inliers "$BB_MIN_INLIERS"
  --output-dir "$BB_OUTPUT_DIR" --output-name "$BB_OUTPUT_NAME"
)
[[ -n "${BB_N_FRAMES:-}"        ]] && args+=(--n-frames "$BB_N_FRAMES")
[[ -n "${BB_STRIDE:-}"          ]] && args+=(--stride "$BB_STRIDE")
[[ -n "${BB_CONFUSER_FACTOR:-}" ]] && args+=(--confuser-factor "$BB_CONFUSER_FACTOR")
[[ -n "${BB_SUPERSAMPLE:-}"     ]] && args+=(--supersample "$BB_SUPERSAMPLE")
[[ -n "${BB_MAX_DETECTIONS:-}"  ]] && args+=(--max-detections "$BB_MAX_DETECTIONS")
[[ -n "${BB_MAX_IMAGE_PIXELS:-}" ]] && args+=(--max-image-pixels "$BB_MAX_IMAGE_PIXELS")
[[ -n "${BB_DEBUG_DIR:-}"       ]] && args+=(--debug-dir "$BB_DEBUG_DIR")
[[ "${BB_UNCAP:-0}" == "1"      ]] && args+=(--uncap)
[[ -n "${BB_IOU:-}"  ]] && args+=(--iou-threshold "$BB_IOU")

mkdir -p "$BB_OUTPUT_DIR"
echo "=== sbatch job ${SLURM_JOB_ID:-local} on $(hostname) — track generation ==="
echo "    video=$BB_VIDEO_DIR  boards=$BB_BOARD_DIR"
echo "    out  =$BB_OUTPUT_DIR/${BB_OUTPUT_NAME}_${BB_PATCH_TYPE}_s${BB_SCALE}.tracks"
echo "    ${BB_JULIA} -t ${BB_THREADS}  args: ${args[*]}"

exec $BB_JULIA --startup-file=no --project="$BB_PROJECT" -t "$BB_THREADS" "$CLI" "${args[@]}"
