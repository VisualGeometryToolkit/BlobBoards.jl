# =============================================================================
# Data root — the EVAL-CONVENTIONS.md (cau-tooling) contract, bound through
# VisualGeometryEval.Experiments (the ladder is spelled out at the
# `set_project_root!` call below). This file keeps only what is THIS
# project's: its tiers, their mounts, and the derived-artifact layout.
# =============================================================================

using DrWatson: projectdir
using VisualGeometryEval.Experiments: set_project_root!, ArtifactView, data_root, results_root
import TOML

# projectdir(), not joinpath(@__DIR__, ".."): @__DIR__ is relative to THIS
# FILE, so every such path breaks silently when a directory is renamed —
# which is exactly how make_zenodo_artifacts.jl (and the since-retired freeze.jl) broke when
# experiments/ became paper/ became bmvc2026/. projectdir() resolves from the
# ACTIVE PROJECT and survives any file move.
# projectdir() is right only when bmvc2026 is the active project — that is the
# trade against @__DIR__, which is right relative to the FILE. Assert it, so
# running a driver under the parent package's project fails here with a clear
# message instead of silently resolving paths into the wrong tree.
function _assert_project()
    isfile(projectdir("src", "BlobBoardsEval.jl")) && return nothing
    error(
        """
        DrWatson's projectdir() resolves to $(projectdir()), which is not the
        bmvc2026 subproject (no src/BlobBoardsEval.jl there). Run drivers with
        `--project=bmvc2026`; the active project is $(Base.active_project()).
        """
    )
end
_assert_project()

const _ARTIFACTS_TOML = projectdir("Artifacts.toml")
# DEFERRED: the shared convention puts the view AT datadir("exp_raw"); this
# project's stays at output/data_view until its next full regeneration, so no
# consumer path moves now.
const _VIEW_DIR = projectdir("output", "data_view")

# Tiers needed per use; images/occlusion stay lazy until a driver asks. Every
# tier holds INPUTS only (configs/data_artifacts.toml): the results are
# regenerated under results_root() by the stages src/runs.jl names.
const _DEFAULT_TIERS = ["bb_absolute_pose_inputs", "bb_assets"]
# Drivers that walk rig sequences (`rig_sequence` pairs mocap files with the
# image listing of images/<subset>, positionally) need the images tier too —
# for the FILE NAMES, even though they never decode a pixel. Without it a
# consumer of the Zenodo record hits ENOENT on images/apr (tier-1 test,
# 2026-08-28). ~2.1 GB, lazy like the rest.
const EVAL_TIERS = [_DEFAULT_TIERS..., "bb_absolute_pose_images_260527"]

# Mount point per tier: the absolute-pose tiers hold trees RELATIVE TO
# absolute_pose_experiments/; the tiers in `_ROOT_TIERS` hold trees relative
# to the DATASET ROOT, so they mount at the view root and
# `data_root()/evaluation/...`, `data_root()/descriptor_training/...` resolve
# identically on both paths. `bb_assets` = evaluation/ board gallery +
# models/ networks; `bb_descriptor_training_*` = the frozen descriptor
# training set (sources, boards, bootstrap network chain) — see
# make_zenodo_artifacts.jl. `bb_pipeline_figure` = the source frame of
# fig:pipeline and its camera record, under blobboard_training/.
const _ROOT_TIERS = (
    "bb_assets",
    "bb_descriptor_training_metadata",
    "bb_descriptor_training_sources",
    "bb_pipeline_figure",
)
_tier_mount(t) = t in _ROOT_TIERS ? "" : "absolute_pose_experiments"

# THE data binding (VisualGeometryEval.Experiments, the stack's one
# implementation). `data_root(; tiers)`:
#
#   1. ENV["CAU_DATA_ROOT"]        explicit override
#   2. datadir("exp_raw")          the project's own symlink, RESOLVED
#                                  (`resolve_links`: run keys downstream hash
#                                  paths under it) — per machine, gitignored
#   3. BlobBoards.data_dir()       the author's copy of the inputs
#                                  (BLOBBOARDS_DATA / LocalPreferences.toml)
#   4. the artifact view           consumers: lazy per-tier download,
#                                  symlink-merged, READ-ONLY
#
# Rungs 2-4 resolve to the same tree on the author's machine. Invariant:
# staging-tree layout ≡ packed-tier layout ≡ composed view — artifact tarballs
# store dataset-root-relative paths, so drivers use only data_root()-relative
# paths and cannot tell whose machine they are on. `results_root()`:
# ENV["CAU_RESULTS_ROOT"] → datadir("exp_pro"), created on demand — the
# add-only outputs tree, where EVERY result lives (src/runs.jl). Nothing is
# ever written into data_root(): it is the input set.
set_project_root!(
    projectdir();
    resolve_links = true,
    data_fallback = () -> BlobBoards.data_dir(),
    artifacts = ArtifactView(_ARTIFACTS_TOML; dir = _VIEW_DIR, mount = _tier_mount, default = _DEFAULT_TIERS),
    create_results = true,
)

# THE CAMERA OF RECORD. A session's camera — the intrinsics every detection
# of it is estimated with — is `<session>/calibration/<camera>.json`
# (vgio/cameras/v3, with its calibration block), an INPUT of the inputs tier:
# the native port, by scripts/import_cameras.jl, of the upstream calibration
# export (published beside it as the conversion's source). The images tier's
# own camera files (`images/<camera>.json`, `…_old.json`: record 1.0.0's
# tarball, reused unchanged) are superseded and never read.
_is_camera_store_file(p) = isfile(p) && endswith(p, ".json") &&
    occursin(r"\"schema\"\s*:\s*\"vgio/cameras/", read(p, String))

"""
    camera_of_record(session_dir) -> String

The path of the session's camera store, `calibration/<camera>.json` — exactly
one.
"""
function camera_of_record(session_dir::AbstractString)
    cal = joinpath(session_dir, "calibration")
    cs = isdir(cal) ? sort(filter(_is_camera_store_file, readdir(cal; join = true))) : String[]
    length(cs) == 1 || error("$(cal): $(length(cs)) camera stores; a session has exactly one camera of record")
    return only(cs)
end
