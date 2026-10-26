# =============================================================================
# track_boards — track board detections across video frames
# =============================================================================
#
# Tier-2 tracking (2026-07-28): temporal priors replace per-frame re-detection.
# Per live track and frame, tracking places a VERIFIED BET:
#
#   1. PREDICT  — per-blob point-field extrapolation (x̂ = 2x_t − x_{t−1} per
#      associated blob, pure image-space — no projective extrapolation of H,
#      which is not a linear quantity). Ellipse SHAPES come from the current
#      model's projection, translated to the predicted centers.
#   2. ASSOCIATE — predicted ellipses vs detected ellipses by IoU (VGC's
#      optimized `iou`, R-tree bbox prefilter — the annotate.jl idiom).
#      IoU is scale-aware where a pixel radius is not: a detection at the
#      right position but the wrong scale is NOT the same blob.
#   3. GATE     — spatial support: ≥ `support_fraction` of the predicted
#      blobs must associate, or the bet is off (no fit attempted).
#   4. ESTIMATE — RANSAC every frame over the guided associations
#      (`_find_model`, association IoUs as PROSAC scores). The track's warm
#      information is confined to data association; the MODEL never
#      conditions on the previous fit — a seeded refit collapses when
#      inter-frame motion exceeds the outlier halfwidth and inherits the
#      association gate's selection bias. On a ~95 %-pure set RANSAC's
#      adaptive stop fires in a handful of iterations, so this costs
#      essentially nothing.
#   5. VERIFY   — the same min_inliers + appearance-support gates
#      `find_boards` uses (the association sims double as the inlier
#      appearance, so the gate is free here). The verified margins are wide
#      (real boards ≈ 85–95 % posterior inlier fraction, accidental fits
#      1–3 %), so warm-path failures are LOUD, never silent.
#
# Fallback ladder (no smooth-motion assumption anywhere):
#   • Failed / missing tracks within `patience` go into the frame's REINIT
#     set — full find_boards with matching restricted to just those boards'
#     gallery entries (targeted, cheap).
#   • Every `keyframe_every` frames a DISCOVERY search runs the full gallery
#     to admit new boards entering the scene.
#   • Frame 1 (or an empty track table) is a discovery search by definition.
#
# The gallery GPU index and per-size detection plans amortize across the
# whole sequence.
# =============================================================================

using VisualGeometryToolkitCore: iou, HomographyMat, HomographyMap, PosedCamera, transfer_ellipse,
    Extrinsics3, Ellipse, region, centroid, feature_frame,
    EllipticFrame, planar_lift,  # #21 lazy endpoint lifts
    Point3   # the full-camera pose arm projects blob centres (`_project_blob_centers`)

# `_model_to_homography` is defined once in pipeline.jl (included earlier) and
# shared by every world→image projection site.


# Live per-board tracking state. `xs_curr`/`xs_prev` hold the imaged
# positions of the board's world blobs in the last two accepted frames —
# the point field the predictor extrapolates. An EMPTY `xs_prev` is "no
# previous frame yet" (the first accepted frame of a track): a sentinel rather
# than `nothing`, so the field is concrete and the per-frame predictor below
# reads it without a union to split.
mutable struct _BoardTrack{M}
    uid::String
    last::BoardDetection{M}
    xs_curr::Vector{Point2{Float64}}
    xs_prev::Vector{Point2{Float64}}
    misses::Int
end

"""
    track_boards(board_feature_dict, descriptor_network, video_path::AbstractString; kwargs...)

Track boards through a video file. Opens `video_path` and streams it frame by
frame — nothing is materialized, since the sequence method only ever
`enumerate`s its input.

This mirrors `annotate_blobs` and `save_trace`, which already take video paths
directly; VideoIO is a hard dependency, so accepting one here costs nothing and
saves every caller an `openvideo` + `close` dance.
"""
function track_boards(
        board_feature_dict::AbstractDict, descriptor_network,
        video_path::AbstractString; kwargs...
    )
    reader = VideoIO.openvideo(video_path)
    try
        return track_boards(
            board_feature_dict, descriptor_network,
            (Gray.(f) for f in reader); kwargs...
        )
    finally
        close(reader)
    end
end

"""
    track_boards(board_feature_dict, descriptor_network, images;
                 board_bundles, config=FindBoardsConfig(),
                 iou_threshold=config.track.iou_threshold,
                 support_fraction=config.track.support_fraction,
                 patience=config.track.patience,
                 keyframe_every=config.track.keyframe_every, device=default_device(),
                 gpu_index=nothing, on_frame=nothing)
        -> Dict{String, Vector{BoardDetection}}

Track boards across a sequence of images: motion-guided association plus a
per-frame RANSAC over the associated correspondences (see the file header
for the algorithm). Calibration (if any) lives in `config.geometry`.

- `iou_threshold`: per-blob predicted↔detected ellipse IoU gate (`[find_boards.track]`;
  tolerates roughly ½σ of center prediction error but rejects wrong-octave scale).
- `support_fraction`: fraction of predicted blobs that must associate for
  the warm bet to proceed to estimation.
- `patience`: consecutive failed frames before a track stops being
  re-targeted every frame (it stays eligible for discovery keyframes).
- `keyframe_every`: discovery cadence for boards not currently tracked
  (`0` disables periodic discovery after frame 1). NB between keyframes only
  boards ALREADY tracked are searched for, so a board first appearing at frame
  `k` is not looked for until the next keyframe — correct for video, and the
  reason unrelated stills belong in `find_boards` instead.
- `gpu_index`: prebuilt [`GalleryIndex`](@ref), as on `find_boards`. Built
  here when omitted; supply one to reuse the staged gallery across several media
  instead of re-uploading it per call. When supplied, the INDEX defines which
  boards are matched — an index from a different gallery matches the wrong
  boards silently.
- `on_frame`: a sink called `on_frame(frame::TrackFrame)` after every frame
  with that frame's record, pixels and configuration of record (a failing sink
  warns and tracking continues). Capture a replayable trace with
  `trace = TrackTrace(); track_boards(…; on_frame = f -> push!(trace, f))`.

Returns `Dict{uid => Vector{BoardDetection}}`; each detection's `source`
records whether the frame was won by the `:track` association path, a
targeted `:init`, or `:detect`.
"""
function track_boards(
        # `board_feature_dict::AbstractDict` makes a call in the retired order
        # (`images, board_feature_dict, descriptor_network`) a MethodError.
        board_feature_dict::AbstractDict,
        descriptor_network,
        images;
        # BoardBundle specifically: the warm path needs :world-frame blobs and
        # the board extent, which PatternBundle does not carry — untyped, a
        # pattern bundle would die in a MethodError mid-sequence.
        # `<:BoardBundle`, not `BoardBundle`: the latter is a UnionAll and Dict's
        # value parameter is INVARIANT, so a concrete `Dict{String, BoardBundle{B,P,D}}`
        # would not match.
        board_bundles::AbstractDict{<:AbstractString, <:BoardBundle},
        config::FindBoardsConfig = FindBoardsConfig(),
        # The four tracker knobs default from the config of record
        # ([find_boards.track]) and override per call.
        iou_threshold::Real = config.track.iou_threshold,
        support_fraction::Real = config.track.support_fraction,
        patience::Int = config.track.patience,
        keyframe_every::Int = config.track.keyframe_every,
        device = default_device(),
        # Prebuilt device-resident gallery (`GalleryIndex`),
        # as on `find_boards`. Reuse it across media to skip re-staging the whole
        # gallery per call — ~31 MB for a 60k-descriptor set. Built here when
        # `nothing`.
        #
        # CONTRACT: when an index is supplied its rows are split back to boards
        # through the INDEX's own `uids`/`offsets` (`match_boards`), keeping only
        # the boards of `board_feature_dict`. An index built from a DIFFERENT
        # gallery therefore matches the wrong boards silently. Supply one whose
        # boards cover this dict.
        gpu_index::Union{Nothing, GalleryIndex} = nothing,
        on_frame = nothing,
    )
    # Polarity comes from the BOARDS, exactly as in find_boards. Tracking
    # describes features once per frame and shares them across every track, so
    # a mixed-polarity set detects at the :none superset; each board's own
    # polarity still governs which blobs it can match. The image-side extent
    # is the gallery's shared cutoff (`prepare_image_features` derives it from
    # the bundles); each board's own cutoff stays a per-board geometric parameter.
    config = _detect_config_for_boards(
        config, _detection_polarity(board_bundles, keys(board_feature_dict))
    )

    config.geometry isa RadialHomographyGeometry && throw(
        ArgumentError(
            "track_boards: the radial arm is not wired into tracking (its fit endpoints and lift differ); run find_boards per frame"
        )
    )
    camera = _camera(config.geometry)
    cam_est = _estimation_camera(config.geometry)
    desc_network = _ensure_network(descriptor_network)

    # The configuration of record every emitted `TrackFrame` carries — resolved
    # AFTER the board-derived polarity is folded in, so a trace records what
    # actually ran, not what was asked. Built only when there is a sink.
    record_config = on_frame === nothing ? Dict{String, Any}() : config_dict(config)
    record_hash = on_frame === nothing ? "" : network_hash(desc_network)
    gpu_index = @something gpu_index GalleryIndex(device, board_feature_dict)

    # World blob geometry per board (z = 0 plane), fixed for the sequence.
    world_discs = Dict(
        uid => blobs(board_bundles[uid], :world)
            for uid in keys(board_feature_dict)
    )

    tracks = Dict{String, _BoardTrack}()
    history = Dict{String, Vector{BoardDetection}}()
    oh = config.geometry.outlier_halfwidth

    collect_diag = on_frame !== nothing

    for (k, img) in enumerate(images)
        t0 = time_ns()
        (; adapt_img, image_features) = prepare_image_features(
            desc_network, img;
            board_bundles, uids = keys(board_feature_dict), config, device
        )
        t_extract = (time_ns() - t0) / 1.0e9
        t0 = time_ns()
        detc = [centroid(d) for d in image_features]
        det_rtree = detection_rtree(detc)

        won = Dict{String, BoardDetection}()
        diags = Dict{String, Tuple{Vector{NTuple{3, Float64}}, Vector{NTuple{5, Float64}}}}()

        # ---- 1. Warm bets for live tracks — strongest first, claim-guarded.
        # Tracks bet SERIALLY in descending last-support order and each winner
        # claims the detections inside its committed quad, so a weaker track
        # cannot re-consume another board's blobs (measured on C0012: track
        # b9f re-locked onto 336's detections once 336's support collapsed —
        # two independent warm bets are otherwise blind to each other).
        claimed = falses(length(image_features))
        live = sort!(
            [p for p in tracks if p.second.misses <= patience];
            by = p -> p.second.last.n_inliers, rev = true
        )
        for (uid, tr) in live
            det, diag = _warm_track_frame(
                tr, world_discs[uid], board_bundles[uid],
                board_feature_dict[uid],
                image_features, det_rtree, cam_est, camera, k,
                size(adapt_img);
                iou_threshold, support_fraction, oh, config,
                claimed, collect_diag
            )
            diag === nothing || (diags[uid] = diag)
            det === nothing && continue
            won[uid] = det
            # ONE claim authority (`_mark_inside!`, pipeline.jl): flag the
            # still-unclaimed features inside the won board's quad.
            _mark_inside!(claimed, detc, board_quad(cam_est, det.model, board_bundles[uid]))
        end

        # ---- 2. Reinit + discovery ----------------------------------------
        reinit_uids = [
            uid for (uid, tr) in tracks
                if !haskey(won, uid) && tr.misses <= patience
        ]
        discovery = k == 1 || (keyframe_every > 0 && k % keyframe_every == 0) ||
            isempty(tracks)

        target_dict = discovery ? board_feature_dict :
            Dict(uid => board_feature_dict[uid] for uid in reinit_uids)
        if !isempty(target_dict)
            # Warm winners' quads are invisible to find_boards' intra-call
            # claim guard — exclude them explicitly or a discovery keyframe
            # can lock a second gallery board onto already-owned detections.
            won_polys = [
                board_quad(cam_est, det.model, board_bundles[uid])
                    for (uid, det) in won
            ]
            results = find_boards(
                target_dict, image_features;
                board_bundles,
                config, device, gpu_index, exclude_polys = won_polys
            )
            src = discovery ? :detect : :init
            for (uid, model, quality, scored) in results
                haskey(won, uid) && continue
                H = _model_to_homography(model, cam_est)
                won[uid] = BoardDetection(
                    uid, k, model, Float64(quality), count(is_map_inlier, scored.weights), src
                )
            end
        end

        # ---- 3. Track bookkeeping ----------------------------------------
        for (uid, det) in won
            xs = _project_blob_centers(_model_to_homography(det.model, cam_est), world_discs[uid])
            if haskey(tracks, uid)
                tr = tracks[uid]
                tr.xs_prev = tr.xs_curr
                tr.xs_curr = xs
                tr.last = det
                tr.misses = 0
            else
                tracks[uid] = _BoardTrack(uid, det, xs, Point2{Float64}[], 0)
            end
            push!(get!(history, uid, BoardDetection[]), det)
        end
        for (uid, tr) in tracks
            haskey(won, uid) || (tr.misses += 1)
        end

        n_warm = count(p -> p.second.source === :track, won)
        @info "Frame $k: $(length(won)) board(s) ($(n_warm) warm), $(length(tracks)) track(s)"

        if collect_diag
            t_estimate = (time_ns() - t0) / 1.0e9
            dets = _detection_ellipses(image_features, DEFAULT_CUTOFF)
            entries = BoardFrameEntry[
                _frame_entry(
                        uid, det, diags,
                        _project_board_ellipses(
                            _model_to_homography(det.model, cam_est), world_discs[uid],
                            BlobBoards.cutoff(board_bundles[uid]),
                            size(adapt_img)
                        )
                    )
                    for (uid, det) in won
            ]
            _emit_track_frame(
                on_frame,
                TrackFrame(
                    TrackFrameRecord(k, entries, dets, t_extract, t_estimate), img,
                    record_config, record_hash
                )
            )
        end
    end

    filter!(p -> !isempty(p.second), history)
    return history
end

function _project_blob_centers(H::HomographyMap, wdiscs)
    P = PerspectiveMap() ∘ H     # euclidean image points (H alone is homogeneous)
    return [Point2{Float64}(P(centroid(d))...) for d in wdiscs]
end
# The pose arm under `undistortion = :none` keeps the FULL camera as its
# estimation camera, for which no plane homography exists: `_model_to_homography`
# hands back the `PosedCamera` itself, and the blob centres go through it — the z = 0
# lift of each world disc's centre, projected through pose and lens.
function _project_blob_centers(cam::PosedCamera, wdiscs)
    return [
        begin
                c = centroid(d)
                Point2{Float64}(project(cam, Point3{Float64}(c[1], c[2], 0.0))...)
            end
            for d in wdiscs
    ]
end

# One warm bet: predict → associate (ellipse-overlap IoU, guided by the
# track's motion) → support gate → per-frame RANSAC on the associations →
# verify. Returns a BoardDetection or `nothing` (bet failed — the caller
# adds the board to the frame's reinit set).
function _warm_track_frame(
        tr::_BoardTrack, wdiscs, bundle, board_features,
        image_features, det_rtree, cam_est, camera,
        frame_idx, imsize;
        iou_threshold, support_fraction, oh, config,
        claimed::BitVector, min_sigma_px = 1.5,
        collect_diag::Bool = false
    )
    min_similarity = config.min_similarity
    diag = collect_diag ? _WarmDiag(NTuple{3, Float64}[], NTuple{5, Float64}[]) : nothing
    board_cutoff = BlobBoards.cutoff(bundle)
    Himg, Wimg = imsize

    # ---- predict: current-model ellipse shapes at velocity-extrapolated centers
    xs_pred = isempty(tr.xs_prev) ? tr.xs_curr :                       # constant position
        [2 .* c .- p for (c, p) in zip(tr.xs_curr, tr.xs_prev)]           # constant velocity

    # ---- associate: predicted ellipse ↔ detections, gated INDEPENDENTLY on
    # geometry (ellipse-overlap IoU ≥ iou_threshold) and identity (descriptor
    # similarity ≥ config.min_similarity). The similarity gate is what breaks
    # the lattice alias: a one-blob-period shift of the repetitive grid is a
    # geometrically self-consistent homography that RANSAC cannot reject —
    # only appearance can. Among gated candidates the winner per blob and the
    # greedy 1:1 conflict resolution are LEXICOGRAPHIC (similarity first,
    # IoU tiebreak): identity evidence is unconditioned on the track, while
    # IoU is conditioned on our own prediction and saturates after the gate.
    bdesc = board_features.descriptor
    idesc = image_features.descriptor
    cands = Tuple{Float32, Float64, Int, Int}[]   # (−sim, −iou, world_idx, det_idx)
    # Detection ellipses hoisted out of the association loops: one conversion
    # per detection instead of one SVD per (predicted blob × candidate) cspond
    # inside the R-tree callback.
    det_ells = [Ellipse(region(feature_frame(f), board_cutoff)) for f in image_features]
    n_pred = 0
    for (wi, wd) in enumerate(wdiscs)
        ell = transfer_ellipse(_model_to_homography(tr.last.model, cam_est), region(wd, board_cutoff))
        ell === nothing && continue
        # Visibility filter: only blobs the detector COULD return count as
        # predictions — in-bounds centers and a detectable projected scale.
        # Without this the support denominator counts the board's full blob
        # census (1118 on the A4 boards) against ~300 detectable-at-1080p
        # detections, and no bet can ever succeed.
        c = xs_pred[wi]
        (1.0 <= c[1] <= Wimg && 1.0 <= c[2] <= Himg) || continue
        sqrt(Float64(ell.a) * Float64(ell.b)) / board_cutoff >= min_sigma_px || continue
        ell_pred = Ellipse(Point2{Float64}(c...), ell.a, ell.b, ell.θ)
        diag === nothing || push!(
            diag.pred, (
                Float64(c[1]), Float64(c[2]),
                Float64(ell.a), Float64(ell.b), Float64(ell.θ),
            )
        )
        n_pred += 1
        r = 2 * max(ell_pred.a, ell_pred.b)
        cx, cy = Float64(ell_pred.center[1]), Float64(ell_pred.center[2])
        d_wi = bdesc[wi]
        best_sim = Ref(0.0f0); best_iou = Ref(0.0); best_j = Ref(0)
        foreach_intersecting(
            det_rtree,
            SpatialIndexing.Rect((cx - r, cy - r), (cx + r, cy + r))
        ) do elem
            j = SpatialIndexing.id(elem)
            claimed[j] && return false
            sim = dot(d_wi, idesc[j])
            sim >= min_similarity || return false
            s = iou(ell_pred, det_ells[j])
            s >= iou_threshold || return false
            if (sim, s) > (best_sim[], best_iou[])
                best_sim[] = sim; best_iou[] = s; best_j[] = j
            end
            false
        end
        best_j[] == 0 || push!(cands, (-best_sim[], -best_iou[], wi, best_j[]))
    end
    n_pred == 0 && return (nothing, _finish_diag(diag))

    sort!(cands)                          # lexicographic: sim desc, iou desc
    used_w = Set{Int}(); used_d = Set{Int}()
    assoc = Tuple{Int, Int}[]
    sims = Float32[]
    for (nsim, niou, wi, di) in cands
        (wi in used_w || di in used_d) && continue
        push!(used_w, wi); push!(used_d, di)
        push!(assoc, (wi, di))
        push!(sims, -nsim)
        diag === nothing || push!(diag.assoc, (Float64(wi), Float64(di), -niou))
    end
    # PROSAC scores: assoc is already in lexicographic (sim, IoU) order, so
    # positional ranks ARE the ordering — exact, no ε-blend scale traps.
    scores = collect(Float32, length(assoc):-1:1)

    # ---- spatial-support gate (bet viability — refit not even attempted).
    # Denominator = the smaller of the visible-predicted count and the
    # track's own established inlier support: the track knows what level of
    # support this board actually achieves at this scale.
    n_ref = max(4, min(n_pred, tr.last.n_inliers))
    length(assoc) >= support_fraction * n_ref || return (nothing, _finish_diag(diag))
    length(assoc) >= 4 || return (nothing, _finish_diag(diag))

    # ---- RANSAC every frame on the guided associations. The track's warm
    # information lives ONLY in data association (predicted motion → IoU
    # gate); the MODEL never conditions on the previous fit. A seeded warm
    # refit was measured to collapse whenever inter-frame motion exceeded
    # the outlier halfwidth (8 px pan vs oh = 5: 139/140 associated, 0
    # inliers — IRLS born outside its own basin), and any seed fit on the
    # gated set inherits the gate's selection bias. RANSAC hypotheses don't:
    # on a ~95%-pure set it terminates in a handful of trials, sampling
    # in the lexicographic association order. Same estimator path as discovery.
    matches = NNMatch[NNMatch(wi, di, sims[k], 0.0f0) for (k, (wi, di)) in enumerate(assoc)]
    scored = csponds(wdiscs, image_features, matches)
    pt = _fit_endpoints(camera, scored)
    # Rank scores REPLACE the :score column here — the warm-track prior is
    # the lexicographic association order, not the descriptor similarity (the
    # prior is always the column: the correspondence standard).
    res = _find_model(
        camera, withattrs(pt; score = scores); outlier_halfwidth = oh,
        local_optimization = config.geometry.local_optimization,
        config = config.ransac_config,
        extent = Rect(bundle, :world),
        rng = fit_rng(config.ransac_seed, tr.uid)
    )
    (res !== nothing && res.converged) || return (nothing, _finish_diag(diag))

    # ---- verify with find_boards' gates
    w = res.weights
    nin = count(is_map_inlier, w)
    # >= : inclusive, in lockstep with find_boards' gates and the bumped
    # defaults (2026-08-11) — with `>` this site would have silently
    # tightened warm-track verification by one inlier.
    nin >= config.geometry.min_inliers || return (nothing, _finish_diag(diag))
    # Inlier FRACTION: the share of associated correspondences the fit
    # actually explains. A warm bet that associates 140 blobs and explains 9
    # is not a track, whatever its absolute inlier count.
    #
    # 0.2 is the value `geometry.min_inlier_fraction` carried at this site
    # before that field was removed (2026-08-13); it lives here as a local
    # constant because the removal was about the DISCOVERY path, where the
    # mandatory appearance gate replaced it. It has no analogue there and no
    # reason to be configurable here, so it is not a config key.
    nin >= 0.2 * length(w) || return (nothing, _finish_diag(diag))
    # Appearance, in lockstep with `_claim_boards`'s mandatory gate.
    # `sims[k]` IS the inlier-appearance dot product — warm association
    # computed it — so this needs no re-description.
    #
    # NB it is a NO-OP whenever `verify_similarity <= min_similarity`, which
    # includes the shipped configs (both 0.6 since the 2026-08-14
    # calibration): warm association already refuses any cspond below
    # `min_similarity` (see the `sim >= min_similarity` gate above), so every
    # inlier here is descriptor-certified by construction. That is the honest
    # difference from discovery, where `guided_search` is descriptor-BLIND and
    # the gate is the only appearance evidence. Kept because it stops being a
    # no-op the moment `verify_similarity` is raised above the matching gate —
    # which is exactly how it shipped this morning.
    vs = Float32(config.verify_similarity)
    n_app = count(k -> is_map_inlier(w[k]) && sims[k] >= vs, eachindex(w))
    n_app >= config.min_appearance_support * nin || return (nothing, _finish_diag(diag))

    return (
        BoardDetection(
            tr.uid, frame_idx, res.estimate, Float64(res.quality), nin, :track
        ),
        _finish_diag(diag),
    )
end


# Warm-bet diagnostics carrier. One value (not parallel Union locals) so the
# collecting/non-collecting split is a single JET-narrowable union.
struct _WarmDiag
    assoc::Vector{NTuple{3, Float64}}
    pred::Vector{NTuple{5, Float64}}
end
_finish_diag(::Nothing) = nothing
_finish_diag(d::_WarmDiag) = (d.assoc, d.pred)


_ellipse5(e::Ellipse) = (
    Float64(e.center[1]), Float64(e.center[2]),
    Float64(e.a), Float64(e.b), Float64(e.θ),
)

# Function barrier: `image_features`' container type is not inferrable at
# the `track_boards` entry, and iterating the abstract union inside the
# frame loop trips JET's union-split reports on `feature_frame(f)`.
_detection_ellipses(feats, cutoff) =
    NTuple{5, Float64}[
    _ellipse5(Ellipse(region(feature_frame(f), cutoff)))
        for f in feats
]

# Committed-model board projection: every world blob mapped through the won
# H, kept when its projected center lands in-frame. This is the "boards"
# panel's playback artifact — recorded so the triptych replays without the
# bundles or the model.
function _project_board_ellipses(H, wdiscs, board_cutoff, imsize)
    Himg, Wimg = imsize
    out = NTuple{5, Float64}[]
    for wd in wdiscs
        e = transfer_ellipse(H, region(wd, board_cutoff))
        e === nothing && continue
        (1.0 <= e.center[1] <= Wimg && 1.0 <= e.center[2] <= Himg) || continue
        push!(out, _ellipse5(e))
    end
    return out
end

function _frame_entry(uid, det::BoardDetection, diags, proj)
    a, p = get(diags, uid, (NTuple{3, Float64}[], NTuple{5, Float64}[]))
    return BoardFrameEntry(uid, det.source, det.model, det.quality, det.n_inliers, a, p, proj)
end
