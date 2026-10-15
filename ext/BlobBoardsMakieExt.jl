# =============================================================================
# BlobBoardsMakieExt — backend-agnostic INSPECTION recipes
# =============================================================================
#
# Triggered by Makie (any backend — CairoMakie for static export, GLMakie for
# interactive windows). Board overlays for find_boards results, the pipeline
# inspection grid and the TrackTrace viewers.

module BlobBoardsMakieExt

using Makie: Makie
using BlobBoards
using BlobBoards: PipelineTrace
# `board` looks unused here — it is NOT. The `@recipe` in
# makie/board_overlay_recipe.jl defines methods on BOTH `board` and `board!`,
# and the import must precede the include or the macro mints an ext-local
# binding and the core `BlobBoards.board` stub never gets its method.
import BlobBoards: board, board!, inspect_pipeline, plot_annotations, plot_boards, plot_board_poses, plot_tracks, replay
import BlobBoards: board_overlay!, inliers!, board_panel!, plot_comparison
# `_clamp_patch` (track_trace.jl) — plot_annotations called it unqualified and
# nothing exercised the viewer until the 2026-09 smoke test caught the UndefVarError.
using BlobBoards: _clamp_patch
import BlobBoards: view_angle_figure, save_view_angle_figure

using VisualGeometryCore: VisualGeometryCore, imshow, ellipses!, matches!,
    HomographyMap, Camera, project
# `Colors` is a hard dep of the package, so an extension may use it directly.
using Colors: Colors
using StaticArrays: SMatrix, SVector
using FlexUnits: Quantity
using GeometryBasics: Point2, Point2f, Point3, Rect
using VisualGeometryCore: EllipticFrame, Ellipse
using CoordinateTransformations: PerspectiveMap

include("makie/board_overlay_recipe.jl")
include("makie/plot_annotations.jl")
include("makie/view_angle_profile.jl")

# =============================================================================
# The composable painter layer
# =============================================================================
#
# Makie's convention: `f!` paints into a target the CALLER owns, `f` wraps it
# in a Figure. Until 2026-09-17 every public entry point here built its own
# Figure and fixed its own layout, so two overlays could not share one figure
# and a comparison had to reach into `_board_panel!`. These painters are the
# supported seam; the `plot_*` wrappers below are a Figure plus one call.
#
# Docstrings live on the stubs in src/api.jl, not here: a docstring on a method
# of an imported function breaks extension precompilation.

# A result is (uid, model, quality, scored); a pose is (uid, model). Both
# answer the only two questions a board overlay asks.
_uid_of(t) = String(t[1])
_model_of(t) = t[2]
_scored_of(t) = length(t) >= 4 ? t[4] : nothing
_n_inliers(t, threshold = 0.5) =
    (sc = _scored_of(t)) === nothing ? 0 : count(>(threshold), sc.weights)



function board_overlay!(
        ax::Makie.Axis, camera::Union{Nothing, VisualGeometryCore.CameraModel},
        results, board_bundles::AbstractDict;
        colors = :auto, labels = true, outline_width = 3.0, blob_width = 1.5,
        show_frame = false, lens::Bool = true,
    )
    items = [(_uid_of(t), _model_of(t)) for t in results]
    # One palette authority (stable per board id, sorted): BlobBoards.board_palette.
    # Supply a Dict to SHARE one palette across panels — a comparison figure must
    # colour a board the same way in every panel, which per-call derivation cannot do.
    palette = colors === :auto ? BlobBoards.board_palette([u for (u, _) in items]) : colors
    label_pos = Point2f[]
    label_txt = String[]
    for (uid, model) in items
        proj = _board_proj(camera, model; lens)
        b = board_bundles[uid]
        col = palette === nothing ? nothing : get(palette, uid, nothing)
        kw = col === nothing ? (;) : (;
                outline_color = col, blob_color = col, outline_width, blob_width,
            )
        _draw_board!(
            ax, proj, Rect(b, :world), BlobBoards.blobs(b, :world);
            cutoff = BlobBoards.cutoff(b), show_frame, kw...
        )
        if labels
            rc = Rect(b, :world)
            # Defensive identity-or-strip (audited): the :world frame is
            # unitless mm by contract. Anchor at the board frame's TOP-RIGHT.
            tr = VisualGeometryCore.ustrip.(rc.origin .+ (rc.widths[1], zero(rc.widths[2])))
            # `CameraBoardProj` returns `nothing` for a corner the lens cannot
            # image (past the fold, behind the camera) — the same guard every
            # other call site has. A board with one off-field corner loses its
            # label, not the whole figure.
            anchor = proj(Point2(tr[1], tr[2]))
            if anchor !== nothing
                push!(label_pos, Point2f(anchor))
                push!(label_txt, uid)
            end
        end
    end
    # align + screen-pixel offset put the text just ABOVE the corner (the
    # image axis is y-reversed; positive screen-y offset moves visually up)
    isempty(label_pos) || Makie.text!(
        ax, label_pos; text = label_txt,
        color = :red, fontsize = 22, align = (:right, :bottom), offset = (0, 6)
    )
    return ax
end

# HOIST CANDIDATE (VisualGeometryCore): `_inlier_frames` and the `Csponds`
# method below touch NO BlobBoards type — they are weights + targets + frame +
# ellipses!, all of which VGC owns. They belong beside VGC's `matches!`, which
# is already the correspondence painter. Only the result-set method is ours,
# because only the (uid, model, quality, scored) tuple is.
_inlier_frames(sc; threshold = 0.5) =
    [VisualGeometryCore.feature_frame(sc.targets[i]) for (i, w) in enumerate(sc.weights) if w > threshold]

function inliers!(
        ax::Makie.Axis, sc::VisualGeometryCore.Csponds;
        cutoff = DEFAULT_CUTOFF, color = :cyan, threshold = 0.5, kwargs...
    )
    ells = _inlier_frames(sc; threshold)
    isempty(ells) && return ax
    ellipses!(ax, ells; cutoff, strokecolor = color, kwargs...)
    return ax
end

# Batched on purpose: ONE ellipses! call for the whole result set rather than
# one per board, which is what keeps a 17-board 12 MP frame interactive.
function inliers!(
        ax::Makie.Axis, results;
        cutoff = DEFAULT_CUTOFF, color = :cyan, threshold = 0.5, kwargs...
    )
    ells = EllipticFrame{Float64}[]
    for t in results
        sc = _scored_of(t)
        sc === nothing && continue
        append!(ells, _inlier_frames(sc; threshold))
    end
    isempty(ells) && return ax
    ellipses!(ax, ells; cutoff, strokecolor = color, kwargs...)
    return ax
end

board_panel!(figpos, camera, img::AbstractMatrix, result, bundle; show_frame = true, lens::Bool = true) =
    _board_panel!(
        figpos, camera, img, _uid_of(result), _model_of(result), result[3],
        _scored_of(result).weights, bundle; show_frame, lens
    )

# =============================================================================
# plot_boards — grid of accepted find_boards results with board overlays
# =============================================================================

# One panel of a registered-boards grid: image + "uid q n" title + board
# overlay. Shared by plot_boards and the trace viewer's _registered_panels!
# (character-identical twins that had drifted on ncols/show_frame).
function _board_panel!(
        figpos, camera::Union{Nothing, VisualGeometryCore.CameraModel}, img::AbstractMatrix,
        uid, model, quality, weights, bundle; show_frame = true, lens::Bool = true,
    )
    n_inl = count(VisualGeometryCore.is_map_inlier, weights)
    ax, _ = imshow(figpos, img; title = "$uid  q=$(round(quality; digits = 1))  n=$n_inl")
    _draw_board!(
        ax, _board_proj(camera, model; lens), Rect(bundle, :world), BlobBoards.blobs(bundle, :world);
        cutoff = BlobBoards.cutoff(bundle), show_frame
    )
    return ax
end

"""
    plot_boards(camera, results, board_bundles, img; n = min(10, length(results)))

Figure with one panel per accepted `find_boards` result: the image with the
board's world rect + blobs projected through the estimated model (`board!`
recipe). `camera === nothing` for the homography path; for the pose path supply
the PINHOLE camera and the UNDISTORTED image (poses live in undistorted-image
coordinates). Backend-agnostic — display via GLMakie or export via CairoMakie.
"""
function plot_boards(
        # The camera, the results and bundles drawn, the image they are drawn
        # on last. Typed so a call in either retired order —
        # `(img, results, board_bundles, camera)` or
        # `(camera, img, results, board_bundles)` — is a MethodError.
        camera::Union{Nothing, VisualGeometryCore.CameraModel},
        results::AbstractVector, board_bundles::AbstractDict, img::AbstractMatrix;
        n = min(10, length(results)),
        # `lens = true` (the default) draws through the FULL camera, for a RAW
        # frame. Supply `lens = false` when `img` has already been undistorted —
        # the pose then reduces to the pinhole plane homography, and applying
        # the lens on top of it is a second undistortion (measured on board
        # "336": pinhole 0.03 px from its detection, the camera 72 px).
        lens::Bool = true,
    )
    top = results[1:n]
    ncols = min(5, n)
    nrows = cld(n, ncols)
    fig = Makie.Figure(size = (ncols * 500, nrows * 500))
    for (i, result) in enumerate(top)
        r, c = fldmod1(i, ncols)
        board_panel!(fig[r, c], camera, img, result, board_bundles[_uid_of(result)]; lens)
    end
    return fig
end

"""
    plot_board_poses(camera, poses, board_bundles, img;
                     detections = nothing, colors = :auto, labels = true)

Overlay board poses on ONE image axis — the post-hoc counterpart of
`plot_boards` for poses reloaded from disk (no correspondences required).
`poses` iterates `(uid, pose)` pairs with object→camera `pose.R` / `pose.t`;
supply the PINHOLE camera and the UNDISTORTED image (fitted poses live in
undistorted-image coordinates), or `camera === nothing` with world→image
homographies. Backend-agnostic.

`detections` optionally overlays measured image features (e.g. adapted
elliptical frames from `adapt`, or `feature_frame.(detect_blobs(img))`) in a
contrasting color, so model-projected blobs and actual detections can be
compared in the same axis.

`colors` gives each board its own color, applied to BOTH its outline and its
blobs so a board reads as one object rather than an outline with unrelated dots
inside it. `:auto` (the default) derives a palette with `distinguishable_colors`
seeded against white and black, the same construction the sweep's `val/`
overlays use — supply a `Dict(uid => color)` to fix specific boards instead, or
`nothing` for the recipe's own defaults. `labels = false` drops the uid text,
which is usually what you want once color already identifies each board.
"""
function plot_board_poses(
        # The camera, the poses and bundles drawn, the image they are drawn on
        # last. Typed so a call in either retired order —
        # `(img, poses, board_bundles, camera)` or
        # `(camera, img, poses, board_bundles)` — is a MethodError.
        camera::Union{Nothing, VisualGeometryCore.CameraModel},
        poses, board_bundles::AbstractDict, img::AbstractMatrix;
        detections = nothing, detection_color = :cyan,
        detection_cutoff = DEFAULT_CUTOFF,
        colors = :auto, outline_width = 3.0, blob_width = 1.5,
        labels = true, lens::Bool = true, size = (1400, 1000)
    )
    poses = collect(poses)
    fig = Makie.Figure(; size)
    ax, _ = imshow(fig[1, 1], img; title = "$(length(poses)) boards")
    detections === nothing || ellipses!(
        ax, detections;
        cutoff = detection_cutoff, strokecolor = detection_color
    )
    board_overlay!(
        ax, camera, poses, board_bundles;
        colors, labels, outline_width, blob_width, lens
    )
    return fig
end

# =============================================================================
# plot_comparison — labelled result sets side by side, ONE shared palette
# =============================================================================

_labelled(e) = e isa Pair ? (String(e.first), e.second) : (String(e[1]), e[2])

function plot_comparison(
        camera::Union{Nothing, VisualGeometryCore.CameraModel},
        labelled, board_bundles::AbstractDict, img::AbstractMatrix;
        colors = :auto, labels = false, show_inliers = true,
        inlier_color = :cyan, inlier_cutoff = DEFAULT_CUTOFF, panel_size = 700,
    )
    entries = [_labelled(e) for e in labelled]
    # ONE palette over the UNION of every panel's boards. This is the whole
    # reason the painters take `colors`: derived per panel, a board present in
    # two panels would get two different colours and the figure would lie.
    uids = unique!([_uid_of(t) for (_, res) in entries for t in res])
    palette = colors === :auto ? BlobBoards.board_palette(uids) : colors
    fig = Makie.Figure(size = (length(entries) * panel_size, panel_size))
    for (i, (label, res)) in enumerate(entries)
        nin = sum(_n_inliers, res; init = 0)
        ax, _ = imshow(fig[1, i], img; title = "$label — $(length(res)) boards, $nin inliers")
        show_inliers && inliers!(ax, res; cutoff = inlier_cutoff, color = inlier_color)
        board_overlay!(ax, camera, res, board_bundles; colors = palette, labels)
    end
    return fig
end

# =============================================================================
# inspect_pipeline — linked-axes stage grid (columns = steps, rows = arms)
# =============================================================================

# Crop image + shapes to a region. This crops the DATA, not just the view, so
# a close-up doesn't carry the whole 12 MP frame into every panel — full-frame
# ellipses are unreadable at that scale anyway. (An earlier comment here
# blamed GPU texture memory for window-creation failures; that was wrong. The
# actual cause was GLFW's thread-1 requirement.)
_in_region(c, xr, yr) = xr[1] <= c[1] <= xr[2] && yr[1] <= c[2] <= yr[2]

function _crop(img, dets, xr, yr)
    x0 = max(1, floor(Int, xr[1])); x1 = min(size(img, 2), ceil(Int, xr[2]))
    y0 = max(1, floor(Int, yr[1])); y1 = min(size(img, 1), ceil(Int, yr[2]))
    sub = img[y0:y1, x0:x1]
    keep = [
        d for d in dets if _in_region(
                VisualGeometryCore.centroid(
                    VisualGeometryCore.feature_frame(d)
                ), (x0, x1), (y0, y1)
            )
    ]
    # Shift frames into crop coordinates. EllipticFrame carries its shape
    # matrix; only the center moves.
    dx, dy = x0 - 1, y0 - 1
    shifted = map(keep) do d
        f = VisualGeometryCore.feature_frame(d)
        c = VisualGeometryCore.centroid(f)
        EllipticFrame(Point2(c[1] - dx, c[2] - dy), f.S)
    end
    return sub, shifted
end

"""
    inspect_pipeline(trace; region=nothing) -> Figure

Linked-axes grid: raw detections, then adapted/filtered per adaptation arm.
Pan/zoom is synchronized across all panels. `region = ((x0,x1), (y0,y1))`
crops the image DATA (not just the view) when you want a close-up.

Full 12 MP × 5 panels renders fine — but NOT in a process holding a large
CUDA allocation (GL context creation fails against a nearly-full card).
Capture the trace on the GPU, then plot it in a CUDA-free process; see
`save_trace` and the dev env's `inspect_ab.jl` (vg-dev-env; not shipped).
"""
function inspect_pipeline(trace::PipelineTrace; region = nothing)
    n_arms = length(trace.arms)
    # Rows = arms, columns = steps. Scales to a hyperparameter sweep; a
    # single row would shrink each panel past usefulness past ~2 arms.
    fig = Makie.Figure(size = (1000, 380 * (1 + n_arms)))
    axes = Makie.Axis[]

    function panel(row, col, dets, title, color)
        im, shapes = region === nothing ?
            (trace.detect_img, [VisualGeometryCore.feature_frame(d) for d in dets]) :
            _crop(trace.detect_img, dets, region[1], region[2])
        ax, _ = imshow(fig[row, col], im; title = "$title (n=$(length(shapes)))")
        ellipses!(ax, shapes; cutoff = trace.cutoff, strokecolor = color)
        return push!(axes, ax)
    end

    panel(1, 1, trace.detections, "detections", :yellow)
    for (i, (name, arm)) in enumerate(trace.arms)
        panel(i + 1, 1, arm.frames, "$name: adapted", :cyan)
        panel(i + 1, 2, arm.filtered, "$name: filtered", :lime)
    end
    # One panel PER verified board, each over the full frame — a board occupies
    # a small part of a 12 MP image, so overlaying all of them in one axis is
    # unreadable. Every axis keeps the image's extent (no autoscale to the
    # board) so the boards' relative positions stay comparable.
    if BlobBoards._has_attr(trace, :registration)
        append!(axes, _registered_panels!(fig, trace, 1 + length(trace.arms)))
    end
    Makie.linkaxes!(axes...)
    # Pin every axis to the IMAGE extent. Without this the linked limits grow to
    # the union of all plotted content — a reprojected board rect or a match
    # line reaching outside the frame drags every panel off the image.
    if !isempty(axes)
        H, W = size(trace.detect_img)
        Makie.limits!(first(axes), 0, W, 0, H)   # linked ⇒ applies to all
    end
    return fig
end

# Verified boards: each accepted model reprojects its world Rect + blobs into
# the image via `board!`, with inlier correspondences drawn by `matches!` —
# the same primitives examples/pipeline/find_boards.jl uses, so this panel and
# the example agree by construction.
#
# Both paths hand `board!` a 2D->2D homography composition, the mapping its
# recipe is actually exercised with. `_project_shape` nominally also accepts a
# `Camera`, but supplying one blows the stack inside Makie's argument pipeline —
# that path appears never to have been used.
#
# Pose path: the board is planar (z = 0), so a pinhole pose reduces exactly to
# the plane homography K*[r1 | r2 | t]. PINHOLE, not the calibrated lens: the
# trace's `detect_img` is undistorted and fitted poses live in undistorted
# coordinates. Measured on board "336" — pinhole reprojects a matched blob to
# 0.03 px of its detection, the OpenCV-model camera to 72 px.
# Deliberately NOT routed through BlobBoards._model_to_homography (its
# documented sixth spelling): plotting receives quantity-carrying poses that
# the pipeline path never sees, and this defensive strip is the contract
# difference (2026-09 audit).
function _board_proj(camera_model::VisualGeometryCore.CameraModel, pose; lens::Bool = true)
    # `lens = true` (the default): the FULL camera, raw pixels — what a board
    # overlaid on a captured frame needs. `lens = false` keeps the pinhole
    # plane homography below, for an image that has already been undistorted
    # (the trace viewer's `detect_img`).
    lens && return CameraBoardProj(VisualGeometryCore.Camera(camera_model, pose))
    K = SMatrix{3, 3, Float64}(camera_model.intrinsics.K)
    R = SMatrix{3, 3, Float64}(pose.R)
    # Extrinsics3.t is bare mm by convention; a quantity-carrying pose strips in mm.
    t = SVector{3, Float64}(VisualGeometryCore.strip_or_bare.(VisualGeometryCore.mm, pose.t))
    return PerspectiveMap() ∘ HomographyMap(K * hcat(R[:, 1], R[:, 2], t))
end
_board_proj(::Nothing, H; lens::Bool = true) = PerspectiveMap() ∘ HomographyMap(SMatrix{3, 3, Float64}(H))

# Radial arm: the estimated model is a RadialHomography — world plane →
# DISTORTED image through the division lift (`radial_transfer`). Under the
# radial map a circle is no longer a conic and a rect edge no longer a line,
# so shapes are transferred by boundary sampling; the recipe's `poly!` takes
# the resulting polygons (one for the outline, one per blob).
struct RadialBoardProj{M <: VisualGeometryCore.RadialHomography}
    m::M
end
(r::RadialBoardProj)(p::AbstractVector) = VisualGeometryCore.radial_transfer(r.m, Point2(p[1], p[2]))
# The rect outline is BlobBoards' one radial sampler (`radial_outline`, the
# same one `board_quad` uses); samples beyond the representable field are
# dropped there and here — `radial_transfer` returns `nothing` for them.
function (r::RadialBoardProj)(rect::Rect)
    pts = Point2f.(BlobBoards.radial_outline(r.m, rect))
    return length(pts) < 3 ? nothing : pts       # nothing representable (board past the fold): no polygon
end
(r::RadialBoardProj)(c::VisualGeometryCore.Circle) = _sampled_polygon(r, VisualGeometryCore.boundary_points(c, 24))

# A shape transferred by boundary SAMPLING, for the maps under which a rect
# edge is not a line and a circle not a conic: every sample through `proj`,
# the ones it cannot image (`nothing`) dropped, no polygon under three.
function _sampled_polygon(proj, samples)
    pts = Point2f[]
    for p in samples
        q = proj(p)
        q === nothing || push!(pts, Point2f(q[1], q[2]))
    end
    return length(pts) < 3 ? nothing : pts
end
_board_proj(::Nothing, m::VisualGeometryCore.RadialHomography; lens::Bool = true) = RadialBoardProj(m)

# The CALIBRATED camera and the board's pose — the board as the lens actually
# images it, in RAW pixels. `_board_proj(::CameraModel, pose)` above is the
# PINHOLE plane homography, which is right only on an undistorted image; on a
# raw frame a wide lens puts the outline tens of pixels off (72 px measured on
# board "336"). Under a distorting lens a rect edge is not a line and a circle
# is not a conic, so shapes are transferred by boundary SAMPLING, exactly as
# `RadialBoardProj` does — and samples the lens cannot image (past the fold,
# behind the camera) are dropped rather than drawn at a fabricated pixel.
struct CameraBoardProj{C}
    cam::C          # VisualGeometryCore.Camera(model, pose): world plane → raw pixels
end

function _cam_pt(p::CameraBoardProj, x, y)
    u = VisualGeometryCore.project(p.cam, Point3{Float64}(Float64(x), Float64(y), 0.0))
    return (isfinite(u[1]) && isfinite(u[2])) ? Point2f(u[1], u[2]) : nothing
end
(p::CameraBoardProj)(q::AbstractVector) = _cam_pt(p, q[1], q[2])

# One rect walk for every outline: VGC's `boundary_points` (constant arc
# length along the perimeter). Samples the lens cannot image are dropped.
(p::CameraBoardProj)(rect::Rect; samples::Int = 80) =
    _sampled_polygon(p, VisualGeometryCore.boundary_points(rect, samples))
(p::CameraBoardProj)(c::VisualGeometryCore.Circle) =
    _sampled_polygon(p, VisualGeometryCore.boundary_points(c, 24))

# A radial camera (VisualGeometryCore's 1-D `RadialCamera`) is rows 1-2 of the
# plane homography: it constrains the DIRECTION to a blob, not its position
# along that ray, so there is no conic to draw and no quad to place. Refuse
# loudly rather than return something wrong.
_board_proj(::Nothing, m::VisualGeometryCore.RadialCamera; lens::Bool = true) = throw(
    ArgumentError(
        "a RadialCamera fixes DIRECTIONS, not positions — it cannot place a board. " *
            "Plot through a camera and the board's pose instead " *
            "(`plot_board_poses(camera, poses, bundles, img)`)."
    )
)

# The `board!` recipe takes its mapping as a PLOT ARGUMENT, and Makie's argument
# pipeline recurses into it — anything holding a `Camera` blows the stack there
# (the note at `_board_proj` above records the same for a bare `Camera`, and
# `CameraBoardProj` hits it identically). So the camera path draws the two
# polygons the recipe would have drawn, itself, with the same batching: ONE
# `poly!` for the outline and ONE for all the blobs, never a call per blob.
function _draw_board!(
        ax, proj::CameraBoardProj, rect, blobs;
        cutoff = DEFAULT_CUTOFF, show_frame = false,
        outline_color = :red, blob_color = :red, outline_width = 3.0, blob_width = 1.5,
    )
    o = proj(rect)
    o === nothing || Makie.poly!(
        ax, [o]; color = :transparent, strokecolor = outline_color, strokewidth = outline_width
    )
    if blob_width > 0
        polys = Vector{Point2f}[]
        for c in VisualGeometryCore.Circle.(blobs, cutoff)
            q = proj(c)
            q === nothing || push!(polys, q)
        end
        isempty(polys) || Makie.poly!(
            ax, polys; color = :transparent, strokecolor = blob_color, strokewidth = blob_width
        )
    end
    return ax
end
# every other mapping is a plain 2D→2D composition the recipe handles
_draw_board!(ax, proj, rect, blobs; kwargs...) = board!(ax, proj, rect, blobs; kwargs...)

# Board geometry is planar: lift each world POSITION onto z = 0 as a `Point3`
# (positions are Points, displacements are Vecs) before projecting.
_project_pts(proj, pts) = [proj(Point2(p[1], p[2])) for p in pts]

function _registered_panels!(fig, trace, row0; ncols = 4)
    reg = trace.registration
    out = Makie.Axis[]
    for (i, (uid, model, quality, csponds)) in enumerate(reg.accepted)
        r, c = fldmod1(i, ncols)
        # show_frame = false: the pose-axes overlay issues a nested `arrows2d!`
        # from inside the recipe's `plot!`; it adds nothing at frame scale.
        ax = _board_panel!(
            fig[row0 + r, c], reg.cam, trace.detect_img, uid, model, quality,
            csponds.weights, reg.bundles[uid]; show_frame = false,
            # `detect_img` is UNDISTORTED ("what the detector saw",
            # src/inspect.jl) and the trace's poses live in that frame, so the
            # pose reduces to the pinhole plane homography. Drawing through the
            # full camera here would undistort a second time.
            lens = false
        )
        # Source centroids are world mm, target centroids image px — both
        # arrive bare or as quantities depending on the trace's provenance, and
        # a quantity strips in its named unit.
        pp = Point2f.(
            _project_pts(
                proj, map(c -> VisualGeometryCore.strip_or_bare.(VisualGeometryCore.mm, c), VisualGeometryCore.centroid.(VisualGeometryCore.sources(csponds)))
            )
        )
        dp = Point2f.(
            map(c -> VisualGeometryCore.strip_or_bare.(VisualGeometryCore.px, c), VisualGeometryCore.centroid.(VisualGeometryCore.targets(csponds)))
        )
        matches!(ax, pp, dp; weights = csponds.weights)
        push!(out, ax)
    end
    return out
end

# =============================================================================
# TrackTrace viewers — plot_tracks / replay: the tracking TRIPTYCH
# =============================================================================
#
# Same paradigm as `inspect_pipeline`: linked-axes panels built on VGC's
# imshow/imshow! + one batched `ellipses!` per overlay group. Every panel
# plays back from RECORDED artifacts (detections, warm-bet evidence,
# committed board projections) — no pipeline state needed at view time.

using BlobBoards: TrackTrace, TrackFrameRecord

# Records store ellipses as (cx, cy, a, b, θ) tuples; rebuild the VGC
# primitive so drawing goes through the canonical batched overlay.
_rec_ellipse(t::NTuple{5, Float64}) = Ellipse(Point2(t[1], t[2]), t[3], t[4], t[5])

_SOURCE_COLOR = Dict(:track => :limegreen, :init => :orange, :detect => :dodgerblue)

# ---- panel painters (one `ellipses!` per overlay group, never per-shape) ----

# Panel 1 — every detection the extractor returned.
_paint_detections!(ax, rec::TrackFrameRecord) =
    ellipses!(
    ax, _rec_ellipse.(rec.detections);
    strokecolor = (:yellow, 0.6), strokewidth = 1
)

# Panel 2 — the warm bet's evidence: predicted ellipses (dashed, colored by
# how the board's frame was won) over dimmed detections, associated
# detections highlighted.
function _paint_association!(ax, rec::TrackFrameRecord)
    ellipses!(
        ax, _rec_ellipse.(rec.detections);
        strokecolor = (:gray, 0.3), strokewidth = 1
    )
    for be in rec.boards
        col = get(_SOURCE_COLOR, be.source, :magenta)
        ellipses!(
            ax, _rec_ellipse.(be.predicted);
            strokecolor = (col, 0.35), strokewidth = 1, linestyle = :dash
        )
        hits = [
            _rec_ellipse(rec.detections[Int(di)]) for (_, di, _) in be.associations
                if 1 <= Int(di) <= length(rec.detections)
        ]
        ellipses!(ax, hits; strokecolor = col, strokewidth = 2)
    end
    return ax
end

# Panel 3 — the committed models: each board's world blobs projected through
# the won homography (recorded artifact, not recomputed).
function _paint_boards!(ax, rec::TrackFrameRecord)
    for be in rec.boards
        col = get(_SOURCE_COLOR, be.source, :magenta)
        ellipses!(ax, _rec_ellipse.(be.projected); strokecolor = col, strokewidth = 1)
    end
    return ax
end

const _TRACK_PANELS = (
    ("detections", _paint_detections!),
    ("association", _paint_association!),
    ("boards", _paint_boards!),
)

_panel_title(name, rec::TrackFrameRecord) =
    name == "detections" ? "detections (n=$(length(rec.detections)))" :
    name == "association" ?
    "association (" * join(
        [
            "$(b.source): $(length(b.associations))"
            for b in rec.boards
        ], ", "
    ) * ")" :
    "boards (" * join(
        [
            "$(b.uid[1:min(3, end)]) q=$(round(b.quality; digits = 1)) n=$(b.n_inliers)"
            for b in rec.boards
        ], ", "
    ) * ")"

_record_title(rec::TrackFrameRecord) =
    "frame $(rec.frame_idx)  |  " * join(
    ["$(b.uid[1:min(3, end)]):$(b.source) n=$(b.n_inliers)" for b in rec.boards], "  "
) *
    "  |  $(round(1000 * (rec.t_extract + rec.t_estimate); digits = 0)) ms"

# (Re)paint the triptych's axes for one record — shared by the static view
# and the scrubber. Pixels resolve via the trace's `keep_frames` policy;
# blank canvas when unavailable.
function _paint_track_panels!(axes, label, trace::TrackTrace, i::Int)
    rec = trace.records[i]
    img = BlobBoards._trace_frame_image(trace, i)
    # Recorded artifacts live in FULL-resolution pixel coordinates while a
    # :thumbs trace stores the frame at 1:_THUMB_STRIDE:end — so the image is
    # drawn stretched to the true frame extent, and the limits are pinned to
    # it: `empty!` + repaint would otherwise let autolimits refit the moving
    # content every frame, dragging all three linked axes around.
    extent = trace.frame_size
    if iszero(extent.width) && img !== nothing
        # pre-frame_size trace: exact only when the frame divides evenly
        s = trace.keep_frames === :thumbs ? BlobBoards._THUMB_STRIDE : 1
        extent = s * Size2(img)
    end
    for (ax, (name, paint!)) in zip(axes, _TRACK_PANELS)
        Makie.empty!(ax)
        if img !== nothing
            e = @something extent Size2(img)
            Makie.image!(
                ax, (0, e.width), (0, e.height), transpose(img);
                interpolate = false
            )
        end
        paint!(ax, rec)
        ax.title = _panel_title(name, rec)
    end
    extent === nothing ||   # linked ⇒ applies to all panels
        Makie.limits!(first(axes), 0, extent.width, 0, extent.height)
    label.text = _record_title(rec)
    return nothing
end

function _track_triptych!(fig, trace::TrackTrace)
    axes = [
        Makie.Axis(fig[1, c]; yreversed = true, aspect = Makie.DataAspect())
            for c in 1:length(_TRACK_PANELS)
    ]
    label = Makie.Label(fig[0, 1:length(_TRACK_PANELS)], ""; font = :bold)
    Makie.linkaxes!(axes...)
    return axes, label
end

"""
    plot_tracks(trace::TrackTrace; frame = 1) -> Figure

One frame of a track trace as the tracking triptych — pan/zoom-linked panels
in the `inspect_pipeline` style, all drawn from recorded artifacts:

1. **detections** — every ellipse the extractor returned;
2. **association** — the warm bet's predicted ellipses (dashed, colored by
   how the board's frame was won: green `:track`, orange `:init`, blue
   `:detect`) with associated detections highlighted;
3. **boards** — the committed models' board projections.
"""
function plot_tracks(trace::TrackTrace; frame::Int = 1, figsize = (1920, 760))
    fig = Makie.Figure(size = figsize)
    axes, label = _track_triptych!(fig, trace)
    _paint_track_panels!(axes, label, trace, frame)
    return fig
end

"""
    replay(trace::TrackTrace; frame = 1, fps = 8) -> Figure

Playback of the `plot_tracks` triptych: a play/pause button advancing the
frames at `fps` (looping), plus a slider for scrubbing. Interactive with
GLMakie; with a static backend it renders the initial frame.
"""
function replay(
        trace::TrackTrace; frame::Int = 1, fps::Real = 8,
        figsize = (1920, 820)
    )
    fig = Makie.Figure(size = figsize)
    axes, label = _track_triptych!(fig, trace)
    nP = length(_TRACK_PANELS)
    playing = Makie.Observable(false)
    # ASCII labels only — Makie's default font has no glyph for ▶/⏸ and an
    # unrepresentable character KILLS the GLMakie renderloop.
    btn = Makie.Button(
        fig[2, 1]; label = Makie.lift(p -> p ? "pause" : "play", playing),
        width = 80, tellwidth = false, tellheight = false
    )
    sl = Makie.Slider(
        fig[2, 2:nP];
        range = 1:length(trace.records), startvalue = frame
    )
    Makie.on(btn.clicks) do _
        playing[] = !playing[]
    end
    # Advance on the backend's OWN tick events — mutating the slider from an
    # @async task races the GLMakie renderloop thread and segfaults.
    last_tick = Base.RefValue(-Inf)
    Makie.on(Makie.events(fig).tick) do tick
        playing[] || return
        tick.time - last_tick[] >= 1 / fps || return
        last_tick[] = tick.time
        Makie.set_close_to!(sl, sl.value[] % length(trace.records) + 1)
    end
    Makie.on(i -> _paint_track_panels!(axes, label, trace, i), sl.value; update = true)
    return fig
end

end # module BlobBoardsMakieExt
