# ==============================================================================
# Projected Board Recipe (2D)
# ==============================================================================
#
# Lives in BlobBoardsMakieExt (trigger: Makie alone — loading ANY Makie
# backend activates it) because the recipe must be available
# to ANY Makie backend — CairoMakie for PDF/PNG export, GLMakie for interactive
# display — without re-triggering on a specific backend.

using VisualGeometryToolkitCore: PosedCamera, Circle, project
import LinearAlgebra: I
using StaticArrays: SMatrix
using GeometryBasics: Point2f, Point3f

"""
    board(mapping, rect, blobs; kwargs...)
    board!(ax, mapping, rect, blobs; kwargs...)

Overlay projected board geometry on a 2D image axis.

`mapping` is any callable that projects world geometry to image coordinates:
- `PosedCamera(model, pose)` — 3D→2D, supports points/circles/rects/etc. via
  `VisualGeometryToolkitCore.project(camera, x)`. Used for the calibrated path.
- `PerspectiveMap() ∘ HomographyMap(H)` — 2D→2D projective. Used for the
  uncalibrated (homography) path.

# Example
```julia
board!(ax, PosedCamera(cam_model, pose), rect, discs; cutoff=2.0)
board!(ax, PerspectiveMap() ∘ HomographyMap(H), rect, discs; cutoff=2.0)
```
"""
Makie.@recipe(Board, mapping, rect, blobs) do scene
    Makie.Attributes(
        cutoff = BlobBoards.DEFAULT_CUTOFF,
        show_outline = true,
        show_blobs = true,
        show_frame = true,
        outline_color = :red,
        outline_width = 3.0,
        blob_color = :yellow,
        blob_width = 1.5,
        frame_scale = 30.0,
        frame_width = 3.0,
    )
end

# Dispatch the projection step on the mapping type. The default path uses
# the mapping as a callable (works for `HomographyMap`, `PerspectiveMap ∘
# HomographyMap`, and any user-supplied transformation that overloads
# `(::Self)(x)`). `PosedCamera` doesn't have a global callable in VGC — some
# lens models (OpenCV's, fisheye) are project-only, others are
# backproject-only — so we route through VGC's `project(camera, x)`
# explicitly for the PosedCamera case.
_project_shape(m, x) = m(x)
_project_shape(cam::PosedCamera, x) = project(cam, x)

function Makie.plot!(p::Board)
    m = p.mapping[]

    # Outline: single shape, one poly! call. `nothing` when no part of the
    # rect is representable (radial arm, board past the fold) — a 0-point
    # polygon is a BoundsError inside Makie.
    outline = _project_shape(m, p.rect[])
    outline === nothing || Makie.poly!(
        p, outline;
        strokecolor = p.outline_color, strokewidth = p.outline_width,
        color = :transparent, visible = p.show_outline
    )

    # Blobs: batch ALL projected shapes into ONE poly! call. Per-blob
    # poly!() loops are catastrophically slow under CairoMakie (~30s for
    # 28 boards × ~30 blobs/board on PDF save); batching all ellipses
    # into one vector cuts that to ~ms — Makie/CairoMakie amortize their
    # per-call overhead across the whole vector. See
    # `feedback_no_per_object_makie_loops` memory.
    # `filter(!isnothing, …)` keeps the Union eltype; `identity.()` narrows the
    # surviving vector to `Vector{Ellipse}` so Makie's poly recipe matches.
    proj_blobs = identity.(
        filter(
            !isnothing,
            [_project_shape(m, c) for c in Circle.(p.blobs[], p.cutoff[])]
        )
    )
    if !isempty(proj_blobs)
        Makie.poly!(
            p, proj_blobs;
            strokecolor = p.blob_color, strokewidth = p.blob_width,
            color = :transparent, visible = p.show_blobs
        )
    end

    _board_frame!(p, m)
    return p
end

_board_frame!(::Board, ::Any) = nothing
function _board_frame!(p::Board, cam::PosedCamera)
    p.show_frame[] || return
    origin_2d = Point2f(project(cam, Point3f(0, 0, 0)))
    scale = p.frame_scale[]
    for (axis_vec, color) in zip(
            eachcol(SMatrix{3, 3, Float32}(scale * I)),
            [:red, :green, :blue]
        )
        tip_2d = Point2f(project(cam, Point3f(axis_vec...)))
        d = tip_2d - origin_2d
        Makie.arrows2d!(
            p, [origin_2d[1]], [origin_2d[2]], [d[1]], [d[2]];
            color, shaftwidth = p.frame_width, visible = p.show_frame
        )
    end
    return
end
