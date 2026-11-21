#!/usr/bin/env julia
# =============================================================================
# scripts/fetch_data.jl — the new-machine setup step: the published inputs of
# bmvc2026 (configs/data_artifacts.toml, bound in Artifacts.toml) installed into
# the artifact store (VisualGeometryToolkitCore's `artifact_store`) and linked
# into data/exp_raw, the artifact view the drivers read through data_root(). It is the one way the paper's data reach
# a machine (VisualGeometryToolkitEval paper_project_violations, L5).
#
#   julia --project=bmvc2026 bmvc2026/scripts/fetch_data.jl              # the required tiers (inputs, assets)
#   julia --project=bmvc2026 bmvc2026/scripts/fetch_data.jl --all        # every tier (≈ 60 GB with the occlusion images)
#   julia --project=bmvc2026 bmvc2026/scripts/fetch_data.jl --only=a,b   # the named tiers
#   julia --project=bmvc2026 bmvc2026/scripts/fetch_data.jl --list       # the catalog, nothing fetched
#
# An author whose data root is bound otherwise (VGTK_DATA_ROOT, or data/exp_raw a
# link at the dataset) gets the trees installed and nothing linked.
# =============================================================================
using DrWatson
@quickactivate
using VisualGeometryToolkitEval.DataArtifacts: fetch_data!, project_catalog

fetch_data!(project_catalog())
