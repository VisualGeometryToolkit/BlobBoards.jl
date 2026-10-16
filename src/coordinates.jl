# =============================================================================
# Coordinate Maps, Reference Frames, and Unit Conversion
# =============================================================================

# =============================================================================
# Unit Conversion Helpers
# =============================================================================

"""
    px_per_mm(density) -> Float64

Convert a density to pixels per millimeter.
"""
px_per_mm(density::Quantity) = ustrip(px / mm, density)   # a ConversionError unless a density

"""
    mm_per_px(density) -> VgcQuantity

The pixel pitch of a density, as a length-per-pixel quantity — the inverse
of the density (`ustrip(mm / px, mm_per_px(300dpi)) ≈ 0.0847`). Until 2026-09
it returned a bare length in mm, dropping the `px` dimension its callers then
had to supply by convention.
"""
function mm_per_px(density::Quantity)
    dimension(density) == dimension(dpi) ||
        throw(ArgumentError("density must be a density (px per length), got $(density)"))
    return ubase(1 / density)
end

# A physical length at a density as plain px: mm × px/mm, the arithmetic the
# rendered sheets were made with (a length quantity is held in metres, and its
# product with a density differs from this in the last ulp — enough to move a
# floor at a whole pixel).
_length_to_px(q::Quantity, density::Quantity) = ustrip(mm, q) * px_per_mm(density)

# =============================================================================
# Reference Frame Coordinate Maps
# =============================================================================

# Reference frame positions in pattern coordinates (pixels)
const REFERENCE_FRAMES = (:pattern, :board, :canvas)

"""
    Translation(bc::BoardConfig, from::Symbol, to::Symbol) -> Translation

Translation between board reference frames, in the board's STORED units:
points on a point-grid (v4) board, its print density's pixels on a stored v3
one. It is not a pixel map — for pixel coordinates in a frame use
[`blobs`](@ref)`(bundle, frame; density)`.

# Reference Frames
- `:pattern`: Origin at pattern corner (excludes border; on a v4 board the
  pattern is the whole board, so this is `:board`)
- `:board`: Origin at board corner (includes border)
- `:canvas`: Origin at canvas corner (includes border + margins)

# Returns
`Translation` from CoordinateTransformations.jl (composable with other transforms).

# Examples
```julia
# Pattern → canvas frame, stored blobs (σ and centres in the stored units)
T = Translation(board_config, :pattern, :canvas)
shifted = T.(centroid.(bundle.blobs))
```
"""
function CoordinateTransformations.Translation(bc::BoardConfig, from::Symbol, to::Symbol)
    from ∈ REFERENCE_FRAMES || throw(ArgumentError("Unknown reference_frame :$from. Valid: $REFERENCE_FRAMES"))
    to ∈ REFERENCE_FRAMES || throw(ArgumentError("Unknown reference_frame :$to. Valid: $REFERENCE_FRAMES"))

    # The layout's own integral sizes — the margins the figure is laid out with
    # (`BoardLayout.margin`: left = floor, top = CEIL of half the slack).
    # Recomputing them as `(canvas − board) ÷ 2` put the pattern 1 px too high
    # whenever the vertical slack was odd (fixed 2026-10-10; 4 of 136 boards).
    sizes = BoardLayout(bc)
    inset = Int(ustrip(px, sizes.pattern_inset))   # the frame width; 0 when the pattern fills the board
    left, _, _, top = sizes.margin

    # Frame positions in pattern coordinates, in whole grid units
    positions = (
        pattern = (0, 0),
        board = (inset, inset),
        canvas = (inset + left, inset + top),
    )

    # in the board's stored unit (px; pt on a point grid): k/100, exact
    offset = positions[to] .- positions[from]
    return Translation(_stored.(Ref(bc.layout_grid), offset))
end

# The canvas the layout draws, in mm: a point-grid board's page is its canvas
# rounded to the grid (what the print master's MediaBox holds); a v3 board's
# world frame keeps the nominal canvas it was always centred on.
_canvas_mm(bc::BoardConfig) = ustrip.(mm, _frame_size(bc.layout_grid, bc, :canvas))

# =============================================================================
# Bundle Accessor Functions (blobs)
# =============================================================================

# Internal: compute pixel offset for reference frame translation
function _frame_offset(bundle::BoardBundle, reference_frame::Symbol)
    T = Translation(board_config(bundle), :pattern, reference_frame)
    return (T.translation[1] * px, T.translation[2] * px)
end

# Internal dispatch helpers for unit conversion (used by blobs). `items` are in
# the board's stored unit, whose density is `stored`. Dimensions are runtime
# values: a plain Real unit means "strip" (the stored numbers:
# points on a v4 board, pixels on a v3), px means printer pixels at
# `density` (no pixel number exists without its density), and a length
# unit converts through the stored unit's density. `units::Real` is the API
# spelling for "give me bare magnitudes": the stored numbers, stripped in px.
_to_units(items, units::Real, _, _) = ustrip.(px, items)
function _to_units(items, units::Units, stored, density)
    if dimension(units) == dimension(px)
        density === nothing && throw(
            ArgumentError(
                "blobs in px are pixels at a density, and this board (schema v4) has no density of its own: " *
                    "give the density they are at, `density = 300dpi` (or ask for units = mm)"
            )
        )
        s = ustrip(dpi, density) / ustrip(dpi, stored)   # a ConversionError unless a density
        return isone(s) ? items : [Disc(b.center .* s, b.σ * s) for b in items]
    end
    dimension(units) == dimension(mm) ||
        throw(ArgumentError("units must be a Real (strip), px, or a length unit; got $(units)"))
    return _to_length_units(items, units, stored)
end
# px ÷ (px/length) is a length: a quantity holds no display unit, so `units`
# only selects the length dimension here.
_to_length_units(discs::AbstractVector{<:Disc}, units, density) =
    [Disc(Point2(d.center[1] / density, d.center[2] / density), d.σ / density) for d in discs]
function _to_length_units(pts::AbstractVector{<:Point2}, units, density)
    scale = ustrip(units / px, mm_per_px(density))
    return [Point2(ustrip.(px, p) .* scale) for p in pts]
end

"""
    blobs(bundle::BoardBundle, reference_frame::Symbol; units=px, density=nothing) -> Vector{Disc}

Get blobs from a board bundle with specified coordinate origin and units.

Coordinates are always returned in COLMAP convention (the frame's corner at 0;
in pixels, the first pixel's centre at 0.5, 0.5).
Use `EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})` to convert to other conventions if needed.

# Arguments
- `bundle::BoardBundle`: The board bundle containing blobs
- `reference_frame::Symbol`: Coordinate system origin
  - `:pattern`, `:board`, `:canvas` - 2D image frames (y-down)
  - `:world` - 3D world frame (centered, y-up, unitless mm; ignores `units` parameter)
- `units`: Target unit for coordinates (default: `px`)
  - `px` - pixels at `density`
  - Physical: `mm`, `cm`, `inch`, etc. - length units
- `density`: the density pixel coordinates are at (e.g. `300dpi`). A
  stored v3 board's is its own by default; a v4 board's layout has no density,
  so `units = px` requires one

# Examples
```julia
# Pixels at 300 dpi in the canvas frame (the page of a 300 dpi conversion)
b = blobs(bundle, :canvas; density = 300dpi)

# Physical units in pattern frame
b = blobs(bundle, :pattern; units=mm)

# World coordinates for 3D/P3P (centered, y-up, unitless mm)
b = blobs(bundle, :world)
X = Point3.(vcat.(centroid.(b), 0.0))  # 3D points on z=0 plane

# Julia's pixel origin: pixel coordinates first, then the origin map, which
# acts on plain numbers (strip the pixel unit)
b = blobs(bundle, :canvas; density = 300dpi)
to_julia = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
u = [to_julia(ustrip.(px, centroid(d))) for d in b]
```
"""
function blobs(bundle::BoardBundle, reference_frame::Symbol; units::Union{Real, Units} = px, density::Union{Nothing, Quantity} = nothing)
    # :world is special: centered, y-up, unitless mm (ignores units parameter)
    if reference_frame === :world
        bc = board_config(bundle)
        half_extents = _canvas_mm(bc) ./ 2
        # the stored canvas blobs ÷ the stored density's px per mm, as plain
        # numbers: a length quantity is held in metres, and the mm round trip
        # through it would perturb the board geometry every pose is solved
        # against by an ulp.
        ppm = px_per_mm(stored_density(bc))
        canvas = bundle.blobs .+ Ref(_frame_offset(bundle, :canvas))
        return map(canvas) do blob
            c = ustrip.(px, blob.center) ./ ppm
            σ = ustrip(px, blob.σ) / ppm
            Disc(Point2(c[1] - half_extents[1], half_extents[2] - c[2]), σ)
        end
    end
    offset = _frame_offset(bundle, reference_frame)
    shifted = bundle.blobs .+ Ref(offset)
    bc = board_config(bundle)
    return _to_units(shifted, units, stored_density(bc), _given_or_own_density(bc.layout_grid, density))
end


# The density a board's pixels are at: the one given, else a v3 board's own
# (its grid IS its printer's pixels); a v4 board has none to fall back on.
_given_or_own_density(_, d::Quantity) = d
_given_or_own_density(g::PixelGrid, ::Nothing) = g.density
_given_or_own_density(::PointGrid, ::Nothing) = nothing

# =============================================================================
# Coordinate System Maps (pixel/physical/world)
# =============================================================================

"""
    AffineMap(bundle::BoardBundle, from::Symbol, to::Symbol; image_origin=:colmap, density=nothing) -> AffineMap

Affine transformation between board coordinate systems.

# Coordinate Systems
- `:pixel` - Pixel coordinates (uses `image_origin` for convention)
- `:physical` - Physical units in mm (y-down, origin at canvas top-left)
- `:world` - World coordinates (y-up, centered at canvas center, unitless mm)

# Arguments
- `bundle::BoardBundle`: Board bundle containing geometry information
- `from::Symbol`: Source coordinate system (one of `:pixel`, `:physical`, `:world`)
- `to::Symbol`: Target coordinate system (one of `:pixel`, `:physical`, `:world`)
- `image_origin::Symbol=:colmap`: Image origin convention when `:pixel` is involved
  (`:julia`, `:colmap`, or `:opencv`). Default is `:colmap` since blobs are stored in
  COLMAP convention (pixel centers at 0.5, 0.5).
- `density`: the density of the `:pixel` coordinates (e.g. `300dpi`), whenever
  `:pixel` is involved: a stored v3 board's own by default; required for a v4
  board, whose layout has no density.

For reference-frame translations (`:pattern`/`:board`/`:canvas`), use
`Translation(bc::BoardConfig, from, to)` instead.

# Example
```julia
bundle, image = BlobBoards.load("board.png")

# Pixels of a 300 dpi conversion → world coordinates (COLMAP, the default origin)
H = AffineMap(bundle, :pixel, :world; density = 300dpi)
world_pt = H(Point2(100.0, 200.0))
```
"""
function CoordinateTransformations.AffineMap(
        bundle::BoardBundle, from::Symbol, to::Symbol;
        image_origin::Symbol = :colmap, density::Union{Nothing, Quantity} = nothing
    )
    for s in (from, to)
        s in (:pixel, :physical, :world) || throw(
            ArgumentError(
                "Unknown coordinate system: $(repr(s)). " *
                    (s === :logical ? "(`:logical` was renamed `:pixel`: pixels at `density`.) " : "") *
                    "Valid systems are :pixel, :physical, :world. " *
                    "For reference-frame translations (:pattern/:board/:canvas), use Translation(::BoardConfig, …)."
            )
        )
    end
    if from == to
        return AffineMap(SMatrix{2, 2, Float64}(1, 0, 0, 1), SVector(0.0, 0.0))
    end
    density = _given_or_own_density(board_config(bundle).layout_grid, density)

    if from == :pixel && to == :physical
        return _pixel_to_physical(_pixel_density(density, from, to), image_origin)
    elseif from == :physical && to == :pixel
        return inv(_pixel_to_physical(_pixel_density(density, from, to), image_origin))
    elseif from == :physical && to == :world
        return _physical_to_world(bundle)
    elseif from == :world && to == :physical
        return inv(_physical_to_world(bundle))
    elseif from == :pixel && to == :world
        return _physical_to_world(bundle) ∘ _pixel_to_physical(_pixel_density(density, from, to), image_origin)
    elseif from == :world && to == :pixel
        return inv(_pixel_to_physical(_pixel_density(density, from, to), image_origin)) ∘ inv(_physical_to_world(bundle))
    end
end

# The density of `:pixel` pixels: required — a pixel coordinate means nothing without one.
_pixel_density(density::Quantity, _, _) = density
_pixel_density(::Nothing, from, to) = throw(
    ArgumentError("AffineMap(bundle, $(repr(from)), $(repr(to))): :pixel coordinates need `density`, the density they are at (a v4 board has none of its own)")
)

"""
    _pixel_to_physical(density, image_origin::Symbol) -> AffineMap

Internal: Create transform from pixel coordinates at `density` (the
canvas frame) to physical (mm) coordinates. Composes origin convention mapping
with scaling.
"""
function _pixel_to_physical(density::Quantity, image_origin::Symbol)

    # Scale factor: pixels to mm
    s = Float64(ustrip(px / mm, density))

    # Compose: scale ∘ inv(origin_map)
    # physical = (pixel - offset) / s
    scale_map = LinearMap(@SMatrix [1 / s 0.0; 0.0 1 / s])
    # The frame map's offset rewrapped as a CT `Translation`, ON PURPOSE (the
    # map itself can act on primitives since VGC 0.21.1): this keeps the eager
    # `LinearMap ∘ Translation → AffineMap` fusion (the documented
    # `-> AffineMap` contract) and a concrete return type — a bare
    # `EuclideanMap{2, …, PixelFrame{image_origin}}` would compose lazily and
    # carry the runtime symbol into the type domain.
    origin_map = Translation(EuclideanMap(PixelFrame{:colmap}, PixelFrame{image_origin}).t)

    return scale_map ∘ inv(origin_map)
end

"""
    _physical_to_world(bundle::BoardBundle) -> AffineMap

Internal: Create transform from physical (mm, y-down) to world (centered, y-up) coordinates.
"""
function _physical_to_world(bundle::BoardBundle)
    board_center = _canvas_mm(board_config(bundle)) ./ 2

    # y-flip and translate to center
    # world_x = physical_x - cx
    # world_y = cy - physical_y  (flip y)
    A = @SMatrix [1.0 0.0; 0.0 -1.0]
    b = SVector(-board_center[1], board_center[2])

    return AffineMap(A, b)
end

# =============================================================================
# Rect
# =============================================================================

"""
    Rect(bundle::BoardBundle, reference_frame::Symbol) -> Rect2

Construct a rectangle for the board in the specified coordinate system.

# Arguments
- `bundle::BoardBundle`: Board bundle containing canvas size in configuration
- `reference_frame::Symbol`: Which area/coordinate system to use
  - `:pattern` - Pattern area in pattern coords (excludes borders)
  - `:board` - Full board in pattern coords (includes borders)
  - `:canvas` - Full canvas in pattern coords (includes margins)
  - `:world` - Canvas in world coords (centered, y-up, unitless mm)

# Returns
- `Rect2{Float64}`: Rectangle with size in mm (unitless)

# Examples
```julia
bundle, _ = BlobBoards.load("board.png")

# Pattern coords
rect = Rect(bundle, :pattern)  # Rect2(0.0, 0.0, pattern_width, pattern_height)
rect = Rect(bundle, :board)    # Rect2(-border, -border, board_width, board_height)

# World coords (for 3D visualization)
rect = Rect(bundle, :world)    # Rect2(-w/2, -h/2, w, h) - centered at origin
```
"""
function GeometryBasics.Rect(bundle::BoardBundle, reference_frame::Symbol)
    config = board_config(bundle)

    # :world is special: centered at origin, y-up, unitless mm
    if reference_frame == :world
        w, h = _canvas_mm(config)
        # Rect origin at bottom-left in y-up world coords
        return GeometryBasics.Rect2(-w / 2, -h / 2, w, h)
    end
    reference_frame in REFERENCE_FRAMES ||
        throw(ArgumentError("Unknown reference_frame :$reference_frame. Valid options: :pattern, :board, :canvas, :world"))

    # Get size in the specified reference frame (with mm units)
    width, height = _frame_size(config.layout_grid, config, reference_frame)

    # Get position of the reference frame's origin in pattern coordinates
    T = Translation(config, reference_frame, :pattern)
    offset_mm = (T.translation .* px) .* mm_per_px(stored_density(config))

    return GeometryBasics.Rect2(offset_mm[1], offset_mm[2], width, height)
end

# A frame's size in mm: what the layout draws on a point grid (whole grid units);
# the nominal size a v3 board always reported.
function _frame_size(::PixelGrid, config, reference_frame::Symbol)
    size = reference_frame == :canvas ? config.canvas_size : config.board_size
    reference_frame == :pattern || return (size.width, size.height)
    frame = config.ruler_width          # a v3 board's band
    return (size.width - 2 * frame, size.height - 2 * frame)
end
function _frame_size(g::PointGrid, config, reference_frame::Symbol)
    sizes = BoardLayout(config)
    s = reference_frame == :canvas ? sizes.canvas_size : reference_frame == :board ? sizes.board_size : sizes.pattern_size
    l, f = _stored_size(g, s), mm_per_px(stored_density(g))
    return (l.width * f, l.height * f)
end
