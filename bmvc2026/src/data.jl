# =============================================================================
# Data root — the EVAL-CONVENTIONS.md (cau-tooling) contract, bound through
# VisualGeometryToolkitEval.Experiments (the ladder is spelled out at the
# `set_project_root!` call below). This file keeps only what is THIS
# project's: its tiers, their mounts, and the derived-artifact layout.
# =============================================================================

using DrWatson: projectdir
using VisualGeometryToolkitEval.Experiments: set_project_root!, data_root, results_root, assert_active_project
using VisualGeometryToolkitEval.DataArtifacts: project_catalog, catalog_view, required_artifacts
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
assert_active_project(joinpath("src", "BlobBoardsEval.jl"))   # VisualGeometryToolkitEval's check: projectdir() is this paper project

# THE catalog of the published inputs (configs/data_artifacts.toml): which tiers
# exist, what each holds, and where each mounts in the artifact view.
const CATALOG = project_catalog(projectdir())

# Tiers needed per use; images/occlusion stay lazy until a driver asks. The
# view's default is the catalog's `required` tiers (inputs, assets). Every tier
# holds INPUTS only: the results are regenerated under results_root() by the
# stages src/runs.jl names. Drivers that walk rig sequences (`rig_sequence`
# pairs mocap files with the image listing of images/<subset>, positionally)
# need the images tier too — for the FILE NAMES, even though they never decode
# a pixel. Without it a consumer of the Zenodo record hits ENOENT on images/apr
# (tier-1 test, 2026-08-28). ~2.1 GB, lazy like the rest.
const EVAL_TIERS = [required_artifacts(CATALOG)..., "bb_absolute_pose_images_260527"]

# THE data binding (VisualGeometryToolkitEval.Experiments, the stack's one
# implementation). `data_root(; tiers)`:
#
#   1. ENV["VGTK_DATA_ROOT"]        explicit override
#   2. datadir("exp_raw")          the project's own link at the author's copy,
#                                  RESOLVED (`resolve_links`: run keys
#                                  downstream hash paths under it) — per
#                                  machine, gitignored
#   3. the site's data_root        the site profile VGTK_SITE names
#                                  ([storage] data_root, once it exists)
#   4. the artifact view           consumers: the catalog's tiers installed
#                                  into the artifact store (scripts/fetch_data.jl,
#                                  or lazily per tier on read), per-file links,
#                                  READ-ONLY, AT datadir("exp_raw")
#
# Invariant: author's tree layout ≡ packed-tier layout ≡ composed view — the
# tiers hold mount-root-relative paths (configs/data_artifacts.toml), so drivers
# use only data_root()-relative paths and cannot tell whose machine they are on.
# `results_root()`: ENV["VGTK_RESULTS_ROOT"] → datadir("exp_pro") → the site's
# results_root, created on demand — the add-only outputs tree, where EVERY
# result lives (src/runs.jl). Nothing is ever written into data_root(): it is
# the input set.
set_project_root!(
    projectdir();
    resolve_links = true,
    artifacts = catalog_view(CATALOG),
    create_results = true,
)

# THE CAMERA OF RECORD. A session's camera — the intrinsics every detection
# of it is estimated with — is `<session>/calibration/<camera>.json`
# (vgio/cameras/v4, with its calibration block), an INPUT of the inputs tier:
# the native port, by scripts/import_cameras.jl, of the upstream calibration
# export (published beside it as the conversion's source). The session's own
# camera files beside its captures (`images/<camera>.json`, `…_old.json`, in
# the dataset's primary copy and record 1.0.0-1.2.0's images tarball; the
# images record leaves them out) are superseded and never read.
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
