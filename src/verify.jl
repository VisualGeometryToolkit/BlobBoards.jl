# =============================================================================
# Board Pose Verification
# =============================================================================
#
# board_quad — the ONE model→image board-outline projection (claim masks,
# feature claiming, agenda claim quads all share it).
#
# appearance_support — descriptor-based false-alarm
# rejection. MANDATORY in the accept path since 2026-08-13:
# `_claim_boards` gates every candidate on
# `appearance_support >= config.min_appearance_support`, and the warm-track
# verifier (`_warm_track_frame`) applies the same test to its association
# sims.
#
# WHY APPEARANCE, NOT GEOMETRY. `guided_search` associates board blobs to
# detections by ellipse IoU and proximity, never consulting descriptors. On a
# cluttered frame that is nearly free: at 13,844 detections/megapixel and a
# 5 px inlier gate, 66% of arbitrary positions have SOME detection in range,
# so once RANSAC seeds any plausible model the inlier set fills in
# geometrically. Inlier counts and fractions therefore cannot distinguish a
# real board from a manufactured one — but the appearance of those
# correspondences can, because a false board's inliers pair blobs with
# unrelated image content. Measured 2026-08-12 over 27 accepted boards on 8
# images, using the inliers themselves:
#
#     real  boards: median similarity 0.950 - 0.970, frac(>=0.8) = 0.993 - 1.0
#     false boards: median similarity 0.071 - 0.367, frac(>=0.8) = 0.0
#
# Zero overlap, and no re-description needed — the descriptors are already in
# the gallery and the image features.
#
# HISTORY — two earlier verification mechanisms are gone:
#
# * A `verify_board` that REPROJECTED a board's largest GT blobs through the
#   fitted model, re-described those image patches, and thresholded the
#   median cosine similarity. Removed 2026-08 as "not discriminative
#   enough". That diagnosis was wrong: re-describing at reprojected
#   positions is a different measurement from judging the correspondences
#   the fit was actually built on (the numbers above).
#
# * `spatial_verify` — greedy winner-take-all suppression over FINISHED
#   results. Superseded by the claim-commit loop inside `_claim_boards`,
#   which applies the same semantics DURING the claiming loop, on claim-cleaned
#   refits rather than post hoc. The standalone carried its own
#   `min_inliers = 9` default that had silently drifted from the pipeline's
#   7/6 — an accidental second threshold — and had no callers left. Removed
#   2026-08-13 alongside the min_inlier_fraction retirement.
#
# =============================================================================

using CoordinateTransformations: PerspectiveMap

"""
    board_quad(camera_or_nothing, model, bundle::BoardBundle) -> GeometryBasics.Polygon

Project a board's world-coordinate bounding rectangle into the image through
the estimated `model` — the ONE place the model→image quad is built (spatial
suppression, feature claiming, agenda claim masks all share it). Homography
path (`camera === nothing`): `model` IS the world→image H. Pose path: the
VGC `Camera` composes pose + intrinsics. `bundle` is typed so a call in the
retired `(camera, bundle, model)` order is a MethodError.
"""
board_quad(::Nothing, H, bundle::BoardBundle) =
    (PerspectiveMap() ∘ HomographyMap(H))(Rect(bundle, :world))
board_quad(camera_model::CameraModel, model, bundle::BoardBundle) =
    project(Camera(camera_model, model), Rect(bundle, :world))

# Radial arm: the model is NOT a collineation — the distortion curves the
# board's edges — so sample the rectangle boundary (as the calibrated path
# does via `project(::Camera, rect, n)`) and transfer every sample through
# the model. Samples whose ray leaves the representable field are dropped.
board_quad(::Nothing, m::VisualGeometryCore.RadialHomography, bundle::BoardBundle) =
    GeometryBasics.Polygon(radial_outline(m, Rect(bundle, :world)))

# `radial_outline` (the rect boundary sampler through the radial model) is
# VisualGeometryCore's since 0.21.1, beside `radial_transfer`.
using VisualGeometryCore: radial_outline


# =============================================================================
# appearance verification — do the inliers LOOK like the board they claim?
# =============================================================================

using VisualGeometryFeatures: descriptor_matrix


# Core loop over prebuilt descriptor matrices (rows = features) — the form
# `_claim_boards` calls with per-phase hoisted matrices, so the mandatory
# gate costs one dot product per inlier and zero matrix rebuilds.
#
# `skip` excludes correspondences by `(src_id, det_id)` — used to isolate the
# GUIDED subset (the inliers descriptor matching did NOT supply) for the
# `Telemetry.appearance_log` rows. It is deliberately NOT wired into the accept gate:
# judging that subset alone was measured on 2026-08-16 and reverted, because a
# distant real board's evidence is mostly guided and those sims sit at
# 0.66-0.71 against a 0.6 bar — 109 boards lost, detection rate 76.5 -> 70.4 pp.
# Re-attempting it requires its own threshold, derived from the logged
# distribution rather than inherited from `verify_similarity`.
function _inlier_sims(Dg::AbstractMatrix, Di::AbstractMatrix, scored; skip = nothing)
    sims = Float32[]
    hasproperty(scored, :src_id) || return sims
    sizehint!(sims, length(scored))
    for k in eachindex(scored)
        is_map_inlier(scored.weights[k]) || continue
        gi = scored.src_id[k]
        di = scored.det_id[k]
        (1 <= gi <= size(Dg, 1) && 1 <= di <= size(Di, 1)) || continue
        skip !== nothing && (gi, di) in skip && continue
        push!(sims, dot(view(Dg, gi, :), view(Di, di, :)))
    end
    return sims
end

"""
    appearance_support(sims; verify_similarity = 0.6) -> Float64

Fraction of inliers whose descriptors agree — "how much of this board's
evidence is corroborated by appearance rather than by geometry alone". The
accept gate compares it against `FindBoardsConfig.min_appearance_support`.

`verify_similarity`, deliberately NOT `min_similarity`: that name belongs to
VisualGeometryFeatures, which takes it as the descriptor MATCHING gate on six
public functions, and `FindBoardsConfig` forwards it at 0.6. This is a
different quantity, at a different stage, with a different value — sharing the
name is how two independent knobs silently become one.

Named for the quantity rather than `verify_threshold` because cosine
similarity is higher-is-better, so the name carries the comparison direction;
"threshold" alone would leave a reader to guess whether it gates a similarity
or a distance, and guessing wrong inverts the test.

Measured separation on real imagery is total: 0.0 for every false board,
>= 0.993 for every real one (close boards, 0.8 sim gate). The default was
calibrated DOWN twice on the 260527 corpus (2026-08-14) — 0.8 → 0.7 on the
strided slice (distant real medians 0.73–0.80), then 0.7 → 0.6 on the full
corpus (the inter-slice 2–3 m frames dip to 0.66–0.71) — while false
boards stay at median 0.07–0.37 throughout, so the separation survives.
See FindBoardsConfig's docstring for the full record. Returns 0.0 for an
empty set, so a board with no judgeable inliers fails rather than succeeds.
"""
appearance_support(sims::AbstractVector; verify_similarity = 0.6) =
    isempty(sims) ? 0.0 : count(>=(Float32(verify_similarity)), sims) / length(sims)
