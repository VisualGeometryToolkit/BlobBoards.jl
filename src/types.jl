# =============================================================================
# Type Definitions - All struct definitions and related functionality
# =============================================================================

# =============================================================================
# Blob Types and Enums - Now imported from VisualGeometryCore
# =============================================================================

# All basic blob types (AbstractDetection, Disc, DiscDetection,
# FeaturePolarity, DEFAULT_PIXEL_TYPE) are now imported from VisualGeometryCore

# =============================================================================
# Detection Result Types
# =============================================================================

using VisualGeometryCore: EllipticFrame
using VisualGeometryFeatures: Detection

"""
    BlobDetection{T}

Type alias for a `Detection` with an `EllipticFrame` and metadata containing:
- `id::Int` — index into the original detection vector

Access fields via property forwarding: `det.center`, `det.id`.
"""
const BlobDetection{T} = Detection{EllipticFrame{T}, @NamedTuple{id::Int}}

# =============================================================================
# Ruler Types
# =============================================================================

"""
    RulerOrientation

Enum for specifying ruler orientation on calibration boards.
"""
@enum RulerOrientation begin
    hruler = 1
    vruler = 2
end
@doc "A horizontal ruler band (along the board's top or bottom edge); see [`RulerOrientation`](@ref)." hruler
@doc "A vertical ruler band (along the board's left or right edge); see [`RulerOrientation`](@ref)." vruler

"""
    RulerTick{T}

Tick specification for a printed ruler.
Fields
- `spacing::T`: center-to-center distance between ticks
- `width::T`: the tick's full drawn length across the frame band (physical)
"""
struct RulerTick{T} <: FieldVector{2, T}
    spacing::T
    width::T
end

# JSON3 serialization support for RulerTick
StructTypes.StructType(::Type{<:RulerTick}) = StructTypes.CustomStruct()

StructTypes.lower(rt::RulerTick) = (spacing = rt.spacing, width = rt.width)

function StructTypes.construct(::Type{RulerTick}, x)
    spacing = StructTypes.construct(ScalarOrQuantity, _getfield(x, "spacing"))
    width = StructTypes.construct(ScalarOrQuantity, _getfield(x, "width"))
    return RulerTick(spacing, width)
end

# Preserve RulerTick type under broadcasting operations
StaticArrays.similar_type(::Type{RulerTick{T}}, ::Type{T2}, s::StaticArrays.Size{(2,)}) where {T, T2} = RulerTick{T2}
StaticArrays.similar_type(::Type{RulerTick{T}}, ::Type{T2}) where {T, T2} = RulerTick{T2}

# Arithmetic operations for RulerTick (explicitly specialize to Numbers to avoid ambiguity)
Base.:*(rt::RulerTick, scalar::Number) = RulerTick(rt.spacing * scalar, rt.width * scalar)
Base.:*(scalar::Number, rt::RulerTick) = RulerTick(rt.spacing * scalar, rt.width * scalar)
Base.:/(rt::RulerTick, scalar::Number) = RulerTick(rt.spacing / scalar, rt.width / scalar)

"""
    RulerSpec{RT<:RulerTick}

Ruler specification: the `major_tick` and `minor_tick` spacings and widths.

A tick's `width` is its FULL drawn length across the ruler band, in physical
units, centred in the band. `BoardConfig`'s defaults (major = the padded band
`ruler_width − 2·frame_padding`, minor = half of it) fill the band with major
ticks. Requires `0 < minor ≤ major ≤ ruler_width − 2·frame_padding` (checked by
`BoardConfig` while the ruler is on).

Schema: board bundles before `blobboards/board/v3` stored a third field,
`width` (the "ruler width"), and HALF-extent tick widths; the reader drops that
field and doubles the tick widths (`StructTypes.construct` below), which
reproduces exactly the ruler those boards were drawn with. The stored
`hashes.config` stays their identity.
"""
struct RulerSpec{RT <: RulerTick}
    major_tick::RT
    minor_tick::RT
end

# JSON3 serialization support for RulerSpec
StructTypes.StructType(::Type{<:RulerSpec}) = StructTypes.CustomStruct()

StructTypes.lower(rs::RulerSpec) = (major_tick = rs.major_tick, minor_tick = rs.minor_tick)

function StructTypes.construct(::Type{RulerSpec}, x)
    major_tick = StructTypes.construct(RulerTick, _getfield(x, "major_tick"))
    minor_tick = StructTypes.construct(RulerTick, _getfield(x, "minor_tick"))
    # A pre-v3 ruler (board schema v1/v2) carries `width` and half-extent tick
    # widths: drop the one, double the others (see the docstring). Detected by
    # the field itself, so a bare preamble (a QR payload) migrates the same way.
    if _getfield(x, "width", nothing) !== nothing
        major_tick = RulerTick(major_tick.spacing, 2 * major_tick.width)
        minor_tick = RulerTick(minor_tick.spacing, 2 * minor_tick.width)
    end

    # Promote major_tick and minor_tick to have consistent types
    spacing_promoted = promote(major_tick.spacing, minor_tick.spacing)
    width_promoted = promote(major_tick.width, minor_tick.width)
    promoted_major = RulerTick(spacing_promoted[1], width_promoted[1])
    promoted_minor = RulerTick(spacing_promoted[2], width_promoted[2])

    return RulerSpec(promoted_major, promoted_minor)
end

# Broadcasting and arithmetic support for RulerSpec
Base.broadcastable(rs::RulerSpec) = Ref(rs)

Base.:*(rs::RulerSpec, scalar::Quantity) = RulerSpec(rs.major_tick * scalar, rs.minor_tick * scalar)
Base.:*(scalar::Quantity, rs::RulerSpec) = rs * scalar
Base.:/(rs::RulerSpec, scalar::Quantity) = RulerSpec(rs.major_tick / scalar, rs.minor_tick / scalar)

# =============================================================================
# Configuration Types
# =============================================================================

abstract type AbstractBundle end

"""
    PatternConfig

Deterministic configuration for pattern generation.

Fields (selected)
- `pattern_resolution::Size2`: logical size of the pattern region
- `min_scale`: minimum σ (in logical units)
- `max_scale`: maximum σ (in logical units) — the top of the scale ladder, exact
  (see `calc_blob_scales`). The constructor keyword's `nothing` (its default)
  stores a disc diameter of 20% of the short side; the field is always a σ.
  Users give sizes through `blob_sizes` (printed disc diameters) on
  `blob_pattern`/`blob_board`; this struct stores only σ in px (schema v2).
- `seed::UInt64`: RNG seed for reproducibility
- `pixel_type`: requested output colorant type (e.g. `Gray{N0f16}`)
- `polarity::Symbol`: blob polarity (`:dark` for dark blobs on light background,
  `:light` for light blobs on dark background)
"""
struct PatternConfig{U <: VgcQuantity, U2, PT}
    pattern_resolution::Size2{U}
    min_scale::U2
    seed::UInt64
    pixel_type::Type{PT}
    max_scale::U2
    scales_per_octave::Int
    max_trials::Int
    α::Float64
    cutoff::Float64
    # raster_aligned defaults to TRUE (2026-07-28): blob centers sit on pixel
    # centers, which makes direct 1× rendering symmetric per blob — the
    # `oversampling` field was removed in the same change (2×+Lanczos was
    # approximating an area integral of an already-continuous profile; with
    # aligned centers it bought nothing measurable). Old JSONs still parse:
    # `construct` ignores their stored `oversampling`, and board UIDs come
    # from the STORED config hash, never recomputed.
    raster_aligned::Bool
    polarity::Symbol

    function PatternConfig(;
            pattern_resolution::Size2,
            min_scale = MIN_SIGMA_PX * px,
            seed::Integer = rand(RandomDevice(), UInt64),
            pixel_type::Type{PT} = DEFAULT_PIXEL_TYPE,
            max_scale = nothing,
            scales_per_octave::Int = _pattern_dflt("scales_per_octave"),
            max_trials::Int = _pattern_dflt("max_trials"),
            α::Float64 = Float64(_pattern_dflt("alpha")),
            cutoff::Float64 = DEFAULT_CUTOFF,
            raster_aligned::Bool = true,
            polarity::Symbol = Symbol(_pattern_dflt("polarity"))
        ) where {PT}


        pattern_resolution =
        if eltype(pattern_resolution) <: Integer
            pattern_resolution .* px
        else
            pattern_resolution
        end

        # Ensure min_scale is a logical (px) quantity — plain numbers get
        # px attached; quantities must already be pixel-dimensioned (the
        # physical-length form belongs to blob_board, not blob_pattern).
        min_scale = if !(min_scale isa Quantity)
            min_scale * px
        else
            dimension(min_scale) == dimension(px) ||
                throw(ArgumentError("min_scale must be in pixels, got $(min_scale)"))
            ubase(min_scale)
        end

        # Convert seed to UInt64 if needed
        seed_uint64 = UInt64(seed)

        # Validate polarity
        polarity in (:dark, :light) || throw(ArgumentError("polarity must be :dark or :light, got :$polarity"))

        pattern_px = ustrip.(px, pattern_resolution)
        σ_min = Float64(ustrip(px, min_scale))
        σ_req = if max_scale === nothing
            # rounded as `_resolve_blob_sizes` rounds it: the same default board
            # must mint one identity whether built here or through blob_pattern
            round(DEFAULT_MAX_BLOB.fraction * minimum(pattern_px) / _diameter_factor(Float64(cutoff)); sigdigits = 12)
        else
            Float64(ustrip(px, max_scale isa Quantity ? max_scale : max_scale * px))   # a ConversionError unless pixels
        end
        max_scale = convert(typeof(min_scale),
            _check_max_scale(pattern_px, σ_min, σ_req, scales_per_octave, Float64(cutoff)) * px)

        U = eltype(pattern_resolution)
        U2 = typeof(min_scale)

        return new{U, U2, PT}(
            pattern_resolution, min_scale, seed_uint64, pixel_type,
            max_scale, scales_per_octave, max_trials,
            α, cutoff, raster_aligned, polarity
        )
    end
end

StructTypes.StructType(::Type{<:PatternConfig}) = StructTypes.CustomStruct()

# Custom serialization to handle the Type field. v1 spellings are
# language-neutral: `alpha` (not `α`) and `gray8`/`gray16` (not Julia type
# names) — these live in the preamble, so they only affect newly MINTED
# identities; stored hashes stay authoritative for existing boards.
StructTypes.lower(pc::PatternConfig) = (
    pattern_resolution = pc.pattern_resolution,
    min_scale = pc.min_scale,
    seed = string(pc.seed),
    pixel_type = _pixel_type_name(pc.pixel_type),
    max_scale = pc.max_scale,
    scales_per_octave = pc.scales_per_octave,
    max_trials = pc.max_trials,
    alpha = pc.α,
    cutoff = pc.cutoff,
    raster_aligned = pc.raster_aligned,
    polarity = string(pc.polarity),
)

# v1 pixel-type names; the string(T) fallback is v1-LOAD-BEARING for
# non-standard pixel types (e.g. Gray{Float64} training patterns), not a
# legacy path — v1 files round-trip those through the type's own spelling.
_pixel_type_name(::Type{Gray{N0f8}}) = "gray8"
_pixel_type_name(::Type{Gray{N0f16}}) = "gray16"
_pixel_type_name(T::Type) = string(T)

_pixel_type_from_name(s::AbstractString) =
    s == "gray8" ? Gray{N0f8} :
    s == "gray16" ? Gray{N0f16} : _parse_pixel_type(s)

function _parse_pixel_type(s::AbstractString)
    ex = try
        Meta.parse(s)
    catch
        throw(ArgumentError("Invalid pixel_type expression: $s"))
    end
    pt = try
        Core.eval(@__MODULE__, ex)
    catch
        throw(ArgumentError("Cannot evaluate pixel_type: $s"))
    end
    pt <: Gray || throw(ArgumentError("pixel_type must be a Gray{T} type, got $s"))
    return pt
end

function StructTypes.construct(::Type{PatternConfig}, x)
    pixel_type = _pixel_type_from_name(String(_getfield(x, "pixel_type")))

    pattern_resolution = StructTypes.construct(Size2, _getfield(x, "pattern_resolution"))
    min_scale = StructTypes.construct(ScalarOrQuantity, _getfield(x, "min_scale"))

    # Handle seed as either number or string (for full UInt64 precision)
    seed_raw = _getfield(x, "seed")
    seed = seed_raw isa AbstractString ? parse(UInt64, seed_raw) : UInt64(seed_raw)

    polarity_raw = _getfield(x, "polarity", nothing)
    polarity = polarity_raw !== nothing ? Symbol(polarity_raw) : :dark

    # v2 stores `max_scale`; a v1 file stores `max_diameter_fraction`, converted
    # here to the top rung it always produced (same ladder, so the blobs re-place
    # bit-identically; the stored hash stays the identity).
    max_scale_raw = _getfield(x, "max_scale", nothing)
    max_fraction_raw = _getfield(x, "max_diameter_fraction", nothing)
    (max_scale_raw === nothing) == (max_fraction_raw === nothing) && throw(
        ArgumentError("pattern_config must carry exactly one of max_scale (v2) or max_diameter_fraction (v1)")
    )
    cutoff = Float64(something(_getfield(x, "cutoff", _getfield(x, "sigma_cutoff", nothing)), DEFAULT_CUTOFF))
    scales_per_octave = Int(_getfield(x, "scales_per_octave"))
    max_scale = if max_scale_raw !== nothing
        StructTypes.construct(ScalarOrQuantity, max_scale_raw)
    else
        σ_min = Float64(ustrip(px, min_scale))
        k = _ladder_top_from_fraction(ustrip.(px, pattern_resolution), σ_min, Float64(max_fraction_raw), scales_per_octave, cutoff)
        σ_min * 2^(k / scales_per_octave) * px
    end

    alpha_raw = _getfield(x, "alpha", nothing)
    alpha_raw === nothing && throw(ArgumentError("pattern_config missing alpha"))

    return PatternConfig(
        pattern_resolution = pattern_resolution,
        min_scale = min_scale,
        seed = seed,
        pixel_type = pixel_type,
        max_scale = max_scale,
        scales_per_octave = scales_per_octave,
        max_trials = Int(_getfield(x, "max_trials")),
        α = Float64(alpha_raw),
        cutoff = cutoff,
        # `oversampling` (present in pre-2026-07-28 JSONs) is intentionally
        # ignored: the field was removed, old boards are never re-rendered,
        # and their UIDs come from the stored config hash.
        raster_aligned = Bool(_getfield(x, "raster_aligned")),
        polarity = polarity
    )
end

# Parameterized on the config type: `PatternConfig{U, U2, PT}` is a UnionAll, so
# a bare `pattern_config::PatternConfig` field would be abstract and every read
# of it a dynamic dispatch. One bundle carries one config, so `P` is settled at
# construction and inferred by the default constructor.
"""
    PatternBundle <: AbstractBundle

Container returned by `blob_pattern` with configuration and blob list.

Fields:
- `pattern_config::PatternConfig`
- `blobs::Vector{Disc}`
"""
struct PatternBundle{P <: PatternConfig, D <: Disc} <: AbstractBundle
    pattern_config::P
    # `D`, not `Disc`: that name is a UnionAll too (`Disc{Q}` over the quantity
    # type), so `Vector{Disc}` is a container with an ABSTRACT ELEMENT TYPE —
    # which `isconcretetype` reports as `true` for the Vector itself, so it hides
    # from a naive field audit. It also broke construction once `P` was added:
    # a parametric struct's generated constructor matches field types exactly,
    # and `Vector{Disc{Q}} <: Vector{Disc}` is false (invariance).
    blobs::Vector{D}
end

# Accessor functions
"""
    pattern_config(bundle) -> PatternConfig

The [`PatternConfig`](@ref) a pattern or board bundle was made with: its blob
sizes (σ, in the stored unit), seed, placement ladder, cutoff and polarity.
"""
pattern_config(pb::PatternBundle) = pb.pattern_config
"""
    cutoff(bundle) -> Float64

The bundle's blob truncation radius in σ (a disc's radius is `cutoff·σ`); see
[`DEFAULT_CUTOFF`](@ref).
"""
cutoff(pb::PatternBundle) = pb.pattern_config.cutoff

"""
    blobs(pb::PatternBundle) -> Vector{Disc}

Get blobs from a pattern bundle in logical units (pixels, COLMAP convention).

Blobs are stored and returned in COLMAP convention (pixel centers at 0.5, 0.5).
Use `EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})` to convert to other conventions if needed.

# Returns
Vector of Disc objects in logical units (pixels)

# Example
```julia
b = blobs(pattern_bundle)

# Convert to Julia convention if needed for array indexing
# (Disc is a primitive, so wrap the frame map's offset in a Translation —
# EuclideanMap has no primitive-action methods yet; VGC 0.7.1 candidate.)
b_julia = Translation(EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia}).t).(b)
```
"""
blobs(pb::PatternBundle) = pb.blobs

StructTypes.StructType(::Type{<:PatternBundle}) = StructTypes.CustomStruct()

# v1 blob encoding: columnar with ONE unit declaration — blobs are bulk
# derived data (never identity-bearing; the hash covers only the config), so
# ~650 identical per-scalar {value, unit} pairs per file were pure noise.
#
# `unit` is the declared unit of the stored numbers: "px" for a pattern and a
# v3 board, "pt" for a v4 (point-grid) board, whose stored px ARE points — the
# same numbers either way, declared as what they are.
_lower_blobs(blobs::Vector{<:Disc}; unit::String = "px") = (
    unit = unit,
    centers = [Float64[ustrip(px, b.center[1]), ustrip(px, b.center[2])] for b in blobs],
    sigmas = Float64[ustrip(px, b.σ) for b in blobs],
)

function _construct_blobs(data; unit::String = "px")
    data isa AbstractVector && throw(
        ArgumentError(
            "row-form blobs are the retired pre-v1 encoding; migrate the file " *
                "with BlobBoards@0.18's migrate_bundles! (the v0 reader and " *
                "migrator were removed in 0.19)"
        )
    )
    u = String(_getfield(data, "unit"))
    u == unit || throw(ArgumentError("columnar blobs must declare unit \"$unit\", got \"$u\""))
    return [
        Disc(Point2(Float64(c[1]) * px, Float64(c[2]) * px), Float64(s) * px)
            for (c, s) in zip(_getfield(data, "centers"), _getfield(data, "sigmas"))
    ]
end

function StructTypes.lower(pb::PatternBundle)
    return (pattern_config = pb.pattern_config, blobs = _lower_blobs(pb.blobs))
end

function StructTypes.construct(::Type{<:PatternBundle}, x)
    pc = StructTypes.construct(PatternConfig, _getfield(x, "pattern_config"))
    blobs_vec = _construct_blobs(_getfield(x, "blobs"))
    return PatternBundle(pc, blobs_vec)
end

# =============================================================================
# Layout grids — the integer grid a board is laid out on
# =============================================================================
#
# A board is laid out in whole units of its GRID, and its stored geometry (the
# pattern config, the blobs) is in its STORED unit, a whole number of grid units:
#
# - `PointGrid` (board schema v4, every new board): a fixed physical grid of
#   1/7200 inch (0.01 pt, 3.53 µm), independent of the print density; the stored
#   unit is the point (1/72 inch, a PHYSICAL unit), declared "pt". Why 1/7200 inch: it is the
#   finest grid that keeps a coordinate in points within 6 significant digits,
#   the precision the grid was sized for (until 2026-10 Ghostscript wrote the
#   master) — 0.01 pt, two decimals, on every
#   coordinate below 9999.99 pt (3.53 m, the largest side a board may have) — and
#   it is the least common multiple of 300, 600, 720, 1200, 1440 and 2400 dpi, the
#   coarsest grid on which every pixel edge at those densities is a grid line.
#   Blob centres are grid points, so the stored "pt" value (`141.73`) is the
#   number the PDF holds, character for character.
# - `PixelGrid` (schema ≤ v3): the printer's pixel grid at the board's print
#   density, the stored unit the printer's pixel, blob centres on pixel centres. Stored
#   boards keep it, so they re-render exactly as they were printed.
#
# In memory both stored units are `px`-dimensioned quantities (the placement
# machinery is pixel code; a v4 board runs it at 72 "px" per inch): the unit library
# normalizes every length to metres, which cannot hold 141.73 pt exactly, while a
# `px` value is the stored number itself. `stored_density` says what a `px` of a
# given board's geometry is.
abstract type LayoutGrid end

"The fixed 1/7200-inch layout grid of a v4 board; its stored unit is the point."
struct PointGrid <: LayoutGrid end

"The pixel grid of a board stored in schema ≤ v3, at the density it was printed at."
struct PixelGrid{D <: VgcQuantity} <: LayoutGrid
    density::D
end

# The point grid's two facts: its stored unit is the typographic point (PDF's
# unit, 72 per inch), and it divides the point into hundredths. Its density,
# its record and every points-per-unit ratio derive from these.
const POINT_DENSITY = 72.0dpi
const _POINT_GRID_PER_PT = 100

"Grid units per inch (as a density)."
grid_density(::PointGrid) = _POINT_GRID_PER_PT * POINT_DENSITY   # 7200 dpi
grid_density(g::PixelGrid) = g.density

"The density of a board's stored unit: what one stored `px` of its geometry is."
stored_density(::PointGrid) = POINT_DENSITY
stored_density(g::PixelGrid) = g.density

# Grid units per stored unit (an integer: stored values are whole grid units).
_grid_per_stored(::PointGrid) = _POINT_GRID_PER_PT
_grid_per_stored(::PixelGrid) = 1

# Typographic points per unit of a density (72/600 for a 600 dpi pixel).
_pt_per(density::Quantity) = ustrip(dpi, POINT_DENSITY) / ustrip(dpi, density)

# A grid count, in stored units: exact for a `PointGrid` (k / 100 is the double
# nearest k/100 pt, which prints as its two decimals), the count itself for a
# `PixelGrid`. The inverse is `_grid_units` (round, never multiply back:
# (k/100)·100 ≠ k for 13 % of k).
_stored(g::LayoutGrid, k::Integer) = k / _grid_per_stored(g)
_grid_units(g::LayoutGrid, x::Real) = round(Int, x * _grid_per_stored(g))

# A layout size (whole grid units, a `BoardLayout` field) in the stored unit.
_stored_size(g::LayoutGrid, s::Size2) =
    Size2(width = _stored(g, Int(ustrip(px, s.width))) * px, height = _stored(g, Int(ustrip(px, s.height))) * px)

# A physical length in whole grid units. A pixel grid keeps the floor every
# stored board was laid out with; the point grid forgives the floating-point
# product landing a hair under a grid line (8.5 inch is 61199.99999999999
# units, which a floor would lose).
# A pixel quantity floored to whole pixels (FlexUnits rounds numbers, not
# quantities: strip in px, floor, reattach).
_floor_px(q) = floor(ustrip(px, q)) * px

# Both take the length through `_length_to_px` (mm × px/mm, the arithmetic every
# stored board was laid out with; coordinates.jl).
_grid_count(g::PixelGrid, len) = floor(_length_to_px(len, g.density)) * px
function _grid_count(g::PointGrid, len)
    x = _length_to_px(len, grid_density(g))
    r = round(x)
    return (abs(x - r) < 1.0e-6 ? Int(r) : floor(Int, x)) * px
end

# The largest side a point-grid board may have: its coordinates must keep two
# decimals within 6 significant digits, the precision the grid was sized for.
const MAX_POINT_GRID_SIDE_PT = 9999.99

"""
    BoardLayout

A board's layout in whole units of its grid (`LayoutGrid`): canvas, board,
pattern, corner and band sizes, the pattern's inset from the board's edge, and
the margins.

A point-grid (v4) board's pattern IS the board: the frame's marks (corner cells,
ruler bands) are reserved rectangles on it, which placement keeps clear of
(`_frame_marks`, `_keepouts`). A stored v3 board's pattern is inset by its
band, as it was laid out.

The `margin` field is a 4-tuple (left, right, bottom, top) in whole grid
units: the white margin of the sheet on each side of the board.
"""
struct BoardLayout
    canvas_size::Size2{typeof(1px)}
    board_size::Size2{typeof(1px)}
    pattern_size::Size2{typeof(1px)}
    corner_size::Size2{typeof(1px)}
    ruler_width::typeof(1px)     # the ruler bands' width (0: no ruler)
    pattern_inset::typeof(1px)   # from the board's edge to the pattern's, on every side
    margin::NTuple{4, Int}  # (left, right, bottom, top), whole grid units

    function BoardLayout(
            physical_canvas_size::Size2{<:Quantity},
            physical_board_size::Size2{<:Quantity},
            physical_ruler_width::Quantity,
            physical_corner_size::Quantity,
            grid::LayoutGrid
        )
        canvas_size = _grid_count.(Ref(grid), physical_canvas_size)
        board_size = _grid_count.(Ref(grid), physical_board_size)
        ruler_width = _grid_count(grid, physical_ruler_width)
        corner = _grid_count(grid, physical_corner_size)
        corner_size = Size2(corner, corner)
        pattern_inset = grid isa PixelGrid ? ruler_width : zero(ruler_width)
        pattern_size = board_size .- 2 .* Size2(pattern_inset, pattern_inset)
        fractional_margin = (canvas_size - board_size) / 2

        @assert all(corner_size .< _floor_px.(board_size ./ 2)) "Frame too wide for page width"
        @assert all(pattern_size .>= 0px) "Frame too wide for page width"


        # Convert fractional margins to whole grid units
        # in the order (left, right, bottom, top)
        # Use floor/ceil to distribute rounding error and ensure proper centering
        margin = (
            floor(Int, ustrip(px, fractional_margin.width)),   # left
            ceil(Int, ustrip(px, fractional_margin.width)),    # right
            floor(Int, ustrip(px, fractional_margin.height)),  # bottom
            ceil(Int, ustrip(px, fractional_margin.height)),
        )   # top

        return new(
            canvas_size,
            board_size,
            pattern_size,
            corner_size,
            ruler_width,
            pattern_inset,
            margin
        )
    end
end

StructTypes.StructType(::Type{BoardLayout}) = StructTypes.Struct()

# The white margin the default board leaves on every side of the page (see
# `BoardConfig`): clears printers' unprintable borders (≤ 6.35 mm) and leaves
# room for the id label under the board.
const DEFAULT_BOARD_MARGIN = _board_dflt("margin_mm") * mm

# A board is centred on its canvas: its margin is a `Size2` — `width` the
# margin left and right of it, `height` above and below — and the board is the
# canvas less twice the margin: THE arithmetic, shared by `BoardConfig` and the
# board code's decoder, so a decoded margin rebuilds the board bit for bit.
_board_from_margin(canvas_size::Size2, margin::Size2) = Size2(
    width = (ustrip(mm, canvas_size.width) - 2 * ustrip(mm, margin.width)) * mm,
    height = (ustrip(mm, canvas_size.height) - 2 * ustrip(mm, margin.height)) * mm,
)   # in mm, as the board was always computed: a length held in metres differs in the last ulp
# the margin centring `board` on `canvas`, in mm (the inverse of `_board_from_margin`)
_margin_of(canvas_size::Size2, board_size::Size2) = Size2(
    width = (ustrip(mm, canvas_size.width) - ustrip(mm, board_size.width)) / 2 * mm,
    height = (ustrip(mm, canvas_size.height) - ustrip(mm, board_size.height)) / 2 * mm,
)

# The board a canvas gets when none is given: the canvas less
# `DEFAULT_BOARD_MARGIN` on every side (`BoardConfig`; `suggest_blob_sizes`
# sizes blobs for it from a paper name).
_default_board_size(canvas_size::Size2) =
    _board_from_margin(canvas_size, Size2(width = DEFAULT_BOARD_MARGIN, height = DEFAULT_BOARD_MARGIN))

# The ruler bands and the corner cells: constant, whatever the board's size —
# they hold the ruler ticks, the uid glyphs and the identity QR code, none of
# which grows with the board (the band was 7.5 % of the page's longer side until
# 2026-10: 22 mm on A4, 89 mm on A0).
const DEFAULT_RULER_WIDTH = _board_dflt("ruler_width_mm") * mm
# The corner cells' side: the QR code top-left, a uid character in each other corner.
const DEFAULT_CORNER_SIZE = _board_dflt("corner_size_mm") * mm
# The smallest disc any board may have (`_board_preamble`).
const MIN_PRINTABLE_DISC = _board_dflt("min_printable_disc_mm") * mm
# The most placement trials per blob size a board may ask for (`_board_preamble`).
const MAX_TRIALS_LIMIT = Int(_board_dflt("max_trials_limit"))

# The frame's fields, its keyword values given or `nothing`. On a point-grid
# (v4) board each mark is switched by its size: `corner_size` (the corner
# cells: QR code, uid) and `ruler_width` (the ruler bands); a zero length turns
# it off. The ticks are the ruler's: given while it is off, an `ArgumentError`
# (a value that would enter the identity while changing nothing on the sheet);
# absent, zero. A stored v3 board keeps
# what it recorded: its corner cells are its band's crossings.
function _frame_fields(
        grid::LayoutGrid, padding;
        ruler_width, corner_size, major_tick_spacing, minor_tick_spacing, major_tick_width, minor_tick_width
    )
    rw = something(ruler_width, DEFAULT_RULER_WIDTH)
    cs = something(corner_size, grid isa PixelGrid ? rw : DEFAULT_CORNER_SIZE)
    ruled = !iszero(rw)
    given = [k for (k, v) in pairs((; major_tick_spacing, minor_tick_spacing, major_tick_width, minor_tick_width)) if v !== nothing]
    grid isa PixelGrid || ruled || isempty(given) ||
        throw(ArgumentError("$(sort(given)) size the ruler's ticks, and ruler_width = 0 turns the ruler off"))
    # (zero while the ruler is off: a TOML default the sheet does not draw would
    # still enter the identity, and a later edit of it break every such board)
    ticked = ruled || grid isa PixelGrid
    major_s = something(major_tick_spacing, ticked ? _board_dflt("major_tick_spacing_mm") * mm : zero(rw))
    minor_s = something(minor_tick_spacing, ticked ? _board_dflt("minor_tick_spacing_mm") * mm : zero(rw))
    major_w = something(major_tick_width, ruled ? rw - 2 * padding : zero(rw))
    minor_w = something(minor_tick_width, ruled ? _board_dflt("minor_tick_fraction") * major_w : zero(rw))
    return rw, cs, major_s, minor_s, major_w, minor_w
end

"""
    BoardConfig

Physical board configuration: canvas, board, the frame's marks and the ruler,
in physical units (e.g. `mm` or `inch`), and the `layout_grid` the board is
laid out on (`PointGrid()`, the default, for every new board; a stored schema
≤ v3 board reads as `PixelGrid(density)`). The board is centred on the canvas:
its `margin` is a `Size2` (`width`: left and right of the board, `height`:
above and below it), `DEFAULT_BOARD_MARGIN` (10 mm) on every side by default,
so office printers' unprintable borders miss the board, and `board_size` is the
canvas less twice the margin. Give either `margin` or `board_size` (its margin
is then half the canvas's excess), not both; a zero margin is full bleed. A point-grid (v4) board records its margin and derives its board,
so the two cannot disagree; a stored v3 board keeps the `board_size` it recorded.
The marks are switched by their sizes: `corner_size` (the corner cells, QR code
and uid) and `ruler_width` (the ruler bands), 10 mm each by default, a zero
length turning a mark off. The print density of a new board is not part of its
configuration — the layout does not depend on it; it is a property of a print
or raster conversion, recorded with the render.

`frame_style` is a stored v3 board's (`:ruler`, or `:none`: a blank band) and
is not a keyword of a new one: its `:ruler` or `:none` says whether the ruler is
on.
"""
# A quantity (or a Size2 of them) in base units, FlexUnits' `ubase` (the
# dynamic quantity type, Float64); `nothing` kept.
_vgc(::Nothing) = nothing
_vgc(x) = ubase.(x)

struct BoardConfig{L <: VgcQuantity, G <: LayoutGrid, F <: VgcQuantity, P <: VgcQuantity, RS <: RulerSpec}
    canvas_size::Size2{L}
    board_size::Size2{L}
    margin::Size2{L}
    layout_grid::G
    ruler_width::F
    corner_size::F
    frame_padding::P
    ruler_spec::RS
    frame_style::Symbol

    function BoardConfig(;
            canvas_size::Size2{<:Quantity},
            board_size::Union{Nothing, Size2{<:Quantity}} = nothing,
            margin::Union{Nothing, Size2{<:Quantity}} = nothing,
            layout_grid::LayoutGrid = PointGrid(),
            # `nothing`: not given (`_frame_fields`)
            ruler_width = nothing,
            corner_size = nothing,
            frame_padding = _board_dflt("frame_padding_mm") * mm,
            major_tick_spacing = nothing,
            minor_tick_spacing = nothing,
            major_tick_width = nothing,
            minor_tick_width = nothing,
            frame_style::Union{Nothing, Symbol} = nothing     # a stored v3 board's only
        )
        # Stored as the dynamic quantity type (a quantity–unit product such as
        # `5mm / px` carries its unit until converted).
        canvas_size, board_size, margin = _vgc(canvas_size), _vgc(board_size), _vgc(margin)
        frame_padding = _vgc(frame_padding)
        if layout_grid isa PointGrid
            frame_style === nothing || throw(
                ArgumentError(
                    "frame_style is not a setting of a new board: its marks are switched by their sizes — " *
                        "corner_size and ruler_width, a zero length turning a mark off"
                )
            )
        else
            # (the checkerboard `:saddle` frame was removed in 2026-10: no board
            # in circulation used it, and nothing read its points)
            frame_style in (:ruler, :none) || throw(ArgumentError("a v3 board's frame_style is :ruler or :none, got $(repr(frame_style))"))
        end
        ruler_width, corner_size, major_tick_spacing, minor_tick_spacing, major_tick_width, minor_tick_width = _vgc.(_frame_fields(
            layout_grid, frame_padding;
            ruler_width, corner_size, major_tick_spacing, minor_tick_spacing, major_tick_width, minor_tick_width
        ))
        # a v3 board's band is ruled or blank as recorded; a new one's is the ruler, when on
        frame_style = something(frame_style, iszero(ruler_width) ? :none : :ruler)

        # The default board leaves `DEFAULT_BOARD_MARGIN` of white on every side
        # of the page: office printers cannot print the outer few mm of a sheet
        # (this kind of Ricoh: 4.23 mm; many Letter printers ¼ inch), and the id
        # label needs 9.25 mm under the board. Full bleed is a zero margin.
        board_size === nothing || margin === nothing ||
            throw(ArgumentError("board_size and margin both size the board (board = canvas − 2·margin): give one"))
        if layout_grid isa PixelGrid
            # a v3 board: its recorded board, exactly (its pixel layout depends on it)
            margin === nothing || throw(ArgumentError("a v3 board records its board_size, not a margin"))
            board_size = @something(board_size, _default_board_size(canvas_size))
            margin = _margin_of(canvas_size, board_size)
        else
            margin = if board_size !== nothing
                _margin_of(canvas_size, board_size)
            else
                @something(margin, Size2(width = DEFAULT_BOARD_MARGIN, height = DEFAULT_BOARD_MARGIN))
            end
            board_size = _board_from_margin(canvas_size, margin)     # the margin is the record
        end

        zero(margin.width) <= margin.width && zero(margin.height) <= margin.height &&
            board_size.width > zero(board_size.width) && board_size.height > zero(board_size.height) || throw(
            ArgumentError(
                "the margin must be ≥ 0 and leave a board on the canvas ($(ustrip.(mm, canvas_size)) mm); got " *
                    "margin $(ustrip.(mm, margin)) mm, leaving $(ustrip.(mm, board_size)) mm: give a smaller " *
                    "margin or board_size (a zero margin is full bleed)"
            )
        )

        _check_side(layout_grid, canvas_size)

        corner_size >= zero(corner_size) && ruler_width >= zero(ruler_width) ||
            throw(ArgumentError("corner_size and ruler_width must be ≥ 0, got $(corner_size) and $(ruler_width)"))
        if layout_grid isa PixelGrid
            # a v3 board is a 3 × 3 grid: its corner cells are where the bands cross
            corner_size == ruler_width ||
                throw(ArgumentError("a v3 board's corner cells are its band's crossings: corner_size must equal its band's width"))
        else
            2 * corner_size < min(board_size.width, board_size.height) || throw(
                ArgumentError(
                    "corner_size = $(corner_size): two corner cells do not fit the board's shorter side, " *
                        "$(min(board_size.width, board_size.height))"
                )
            )
            2 * ruler_width < min(board_size.width, board_size.height) || throw(
                ArgumentError(
                    "ruler_width = $(ruler_width): two ruler bands do not fit the board's shorter side, " *
                        "$(min(board_size.width, board_size.height))"
                )
            )
        end

        # frame_padding insets a mark's content (ticks, QR code, uid glyphs) from its edges
        if !iszero(ruler_width) && frame_style === :ruler && frame_padding >= ruler_width / 2
            throw(ArgumentError("frame_padding must be < ruler_width/2. Got frame_padding=$(frame_padding), ruler_width=$(ruler_width)"))
        end
        if layout_grid isa PointGrid && !iszero(corner_size) && frame_padding >= corner_size / 2
            throw(ArgumentError("frame_padding must be < corner_size/2. Got frame_padding=$(frame_padding), corner_size=$(corner_size)"))
        end

        # Tick widths are full lengths across the band (see RulerSpec): the
        # minor fits inside the major, the major inside the padded band.
        if frame_style === :ruler && !iszero(ruler_width)
            band = ruler_width - 2 * frame_padding
            zero(minor_tick_width) < minor_tick_width <= major_tick_width <= band || throw(
                ArgumentError(
                    "tick widths must satisfy 0 < minor_tick_width ≤ major_tick_width ≤ " *
                        "ruler_width − 2·frame_padding = $(band); got minor_tick_width = " *
                        "$(minor_tick_width), major_tick_width = $(major_tick_width)"
                )
            )
        end

        ruler_spec = RulerSpec(
            RulerTick(major_tick_spacing, major_tick_width),
            RulerTick(minor_tick_spacing, minor_tick_width)
        )

        L = eltype(canvas_size)
        G = typeof(layout_grid)
        F = typeof(ruler_width)
        P = typeof(frame_padding)
        RS = typeof(ruler_spec)

        return new{L, G, F, P, RS}(
            canvas_size,
            board_size,
            margin,
            layout_grid,
            ruler_width,
            convert(F, corner_size),
            frame_padding,
            ruler_spec,
            frame_style
        )
    end
end

# The integral layout a board is drawn with — the ONE derivation of its sizes
# and margins: the figure, the TrimBox and `Translation(:pattern, :canvas)` all
# read it. (Re-deriving the margins elsewhere put the pattern 1 px off on boards
# with an odd vertical slack, 2026-10.)
BoardLayout(bc::BoardConfig) = BoardLayout(bc.canvas_size, bc.board_size, bc.ruler_width, bc.corner_size, bc.layout_grid)

# The white margin kept between a corner cell and every blob — room around the
# uid characters, and more than the QR code's own quiet zone (inside its cell).
# Part of the frame's definition, like the grid: it decides where blobs may go,
# so changing it changes every board placed from its configuration.
const CORNER_CLEARANCE = 1.0mm

"""
    _frame_marks(bc::BoardConfig) -> Vector{NamedTuple}

The frame's marks — what is printed on a board besides its blobs — each once:
what it draws (`kind`), its rectangle and the clearance blobs keep from it, in
whole grid units with the board's top-left corner at the origin (y down):

- `:qr` (top-left) and `:uid` (`index` 1, 2, 3: top-right, bottom-left,
  bottom-right): the corner cells, `corner_size` square; none when it is 0;
- `:ruler`: the ruler bands between the cells, `ruler_width` wide (`index` 1,
  2, 3, 4: top, bottom, left, right); none when it is 0 — or `:blank`, a stored
  v3 `:none` board's band.

The sheet draws them (`_frame_content`) and placement keeps clear of them
(`_keepout_boxes`) from this one list, so the two cannot disagree.
"""
function _frame_marks(bc::BoardConfig)
    L = BoardLayout(bc)
    u(q) = Int(ustrip(px, q))
    bw, bh = u(L.board_size.width), u(L.board_size.height)
    c, f = u(L.corner_size.width), u(L.ruler_width)
    k = u(_grid_count(bc.layout_grid, CORNER_CLEARANCE))
    marks = @NamedTuple{kind::Symbol, index::Int, x::Int, y::Int, w::Int, h::Int, clearance::Int}[]
    if c > 0
        push!(marks, (; kind = :qr, index = 0, x = 0, y = 0, w = c, h = c, clearance = k))
        for (index, x, y) in ((1, bw - c, 0), (2, 0, bh - c), (3, bw - c, bh - c))
            push!(marks, (; kind = :uid, index, x, y, w = c, h = c, clearance = k))
        end
    end
    if f > 0
        kind = bc.frame_style === :ruler ? :ruler : :blank          # (`:blank`: a stored v3 `:none` board's band)
        for (index, x, y, w, h) in ((1, c, 0, bw - 2c, f), (2, c, bh - f, bw - 2c, f), (3, 0, c, f, bh - 2c), (4, bw - f, c, f, bh - 2c))
            push!(marks, (; kind, index, x, y, w, h, clearance = 0))
        end
    end
    return marks
end

# Each mark's reserved area — its rectangle grown by its clearance, clipped to
# the board — as (x0, y0, x1, y1) in whole grid units, board frame (y down): what
# placement keeps clear of and what a `:light` board paints white.
function _keepout_boxes(bc::BoardConfig)
    L = BoardLayout(bc)
    bw, bh = Int(ustrip(px, L.board_size.width)), Int(ustrip(px, L.board_size.height))
    return [
        (max(0, m.x - m.clearance), max(0, m.y - m.clearance), min(bw, m.x + m.w + m.clearance), min(bh, m.y + m.h + m.clearance))
            for m in _frame_marks(bc)
    ]
end

"""
    _keepouts(bc::BoardConfig) -> Vector{Rect2{Float64}}

The areas of the pattern no blob may touch, in the board's stored unit and the
`:pattern` frame: the marks' reserved areas (`_keepout_boxes`). A stored v3
board's pattern is inset by its band, so its marks lie outside it: none.
"""
_keepouts(bc::BoardConfig) = _keepouts(bc.layout_grid, bc)
_keepouts(::PixelGrid, _) = Rect2{Float64}[]
_keepouts(g::PointGrid, bc::BoardConfig) =
    [Rect2{Float64}(_stored(g, x0), _stored(g, y0), _stored(g, x1) - _stored(g, x0), _stored(g, y1) - _stored(g, y0)) for (x0, y0, x1, y1) in _keepout_boxes(bc)]

# A point-grid board's pattern size in points: the board's whole grid units,
# k/100 exactly (what `blob_board` stores and placement uses).
_board_pattern_points(bc::BoardConfig) = _stored_size(bc.layout_grid, BoardLayout(bc).pattern_size)

# A point-grid board's preamble: its pattern fills the board's pattern cell, and
# the grid (not `raster_aligned`) places the centres. `blob_board` and
# `parse_board_code` both build it here, so the two derived fields cannot drift.
# Its ladder is checked before anything is placed: a decoded board code or a
# size in the wrong unit must fail here, not exhaust memory in the placement.
function _board_preamble(bc::BoardConfig; pattern_kwargs...)
    pc = PatternConfig(; pattern_resolution = _board_pattern_points(bc), raster_aligned = false, pattern_kwargs...)
    σ0, σ1 = ustrip(px, pc.min_scale), ustrip(px, pc.max_scale)
    floor_mm = ustrip(mm, MIN_PRINTABLE_DISC)
    smallest_mm = ustrip(mm, 2 * pc.cutoff * σ0 * px / stored_density(bc))
    pc.cutoff > 0 && pc.max_trials >= 1 && pc.scales_per_octave >= 1 || throw(
        ArgumentError("a board needs cutoff > 0, max_trials ≥ 1 and scales_per_octave ≥ 1; got $(pc.cutoff), $(pc.max_trials), $(pc.scales_per_octave)")
    )
    pc.max_trials <= MAX_TRIALS_LIMIT || throw(
        ArgumentError("max_trials = $(pc.max_trials) is above the limit of $(MAX_TRIALS_LIMIT) placement trials per blob size (`max_trials_limit` in configs/authoring.toml)")
    )
    σ0 <= σ1 && smallest_mm >= floor_mm * (1 - 1.0e-9) || throw(
        ArgumentError(
            "a board's smallest disc must be at least $(_short_number(floor_mm)) mm across and no larger than its largest; " *
                "got $(round(smallest_mm; sigdigits = 3)) mm (σ $(round(σ0; sigdigits = 3)) to $(round(σ1; sigdigits = 3)) pt)"
        )
    )
    return BoardPreamble(bc, pc)
end

grid_density(bc::BoardConfig) = grid_density(bc.layout_grid)
stored_density(bc::BoardConfig) = stored_density(bc.layout_grid)

# A point-grid board's sides must keep two decimals within 6 significant digits.
_check_side(::PixelGrid, canvas_size) = nothing
function _check_side(::PointGrid, canvas_size)
    side = maximum(ustrip.(px, canvas_size .* stored_density(PointGrid())))
    side <= MAX_POINT_GRID_SIDE_PT || throw(
        ArgumentError(
            "the canvas side $(round(side; digits = 2)) pt ($(round(ustrip(mm, side * pt); digits = 1)) mm) is larger " *
                "than $(MAX_POINT_GRID_SIDE_PT) pt ($(round(ustrip(mm, MAX_POINT_GRID_SIDE_PT * pt); digits = 1)) mm): " *
                "beyond it the print master's coordinates lose their 0.01 pt precision"
        )
    )
    return nothing
end

StructTypes.StructType(::Type{<:BoardConfig}) = StructTypes.CustomStruct()

# The grid's record: a v3 board's density, under the key it was stored with
# (`print_density`: part of the stored v3 record, so it stays); a v4 board's
# declared grid. The key order is the v3 sidecar's, so a stored v3 board
# re-saves unchanged.
_grid_record(g::PixelGrid) = (; print_density = g.density)
_grid_record(g::PointGrid) = (; layout_grid = (value = 1 / _grid_per_stored(g), unit = _blob_unit(g)))

StructTypes.lower(bc::BoardConfig) = (;
    canvas_size = bc.canvas_size,
    _board_record(bc.layout_grid, bc)...,
    _grid_record(bc.layout_grid)...,
    _frame_record(bc.layout_grid, bc)...,
)
# A v4 board records its margin (its board is derived: `_board_from_margin`); a
# v3 board, the board size it was laid out with.
_board_record(::PointGrid, bc) = (; margin = bc.margin)
_board_record(::PixelGrid, bc) = (; board_size = bc.board_size)
# A v4 board records its marks' sizes (a zero turns one off) and the ruler's
# ticks; a v3 board, as it was written: its band (`frame_width`, whose crossings
# are its corner cells), its ticks and whether the band was ruled.
_frame_record(::PointGrid, bc) =
    (; ruler_width = bc.ruler_width, corner_size = bc.corner_size, frame_padding = bc.frame_padding, ruler_spec = bc.ruler_spec)
_frame_record(::PixelGrid, bc) =
    (; frame_width = bc.ruler_width, frame_padding = bc.frame_padding, ruler_spec = bc.ruler_spec, frame_style = string(bc.frame_style))

function _construct_grid(x)
    g = _getfield(x, "layout_grid", nothing)
    if g !== nothing
        want = _grid_record(PointGrid()).layout_grid
        (Float64(_getfield(g, "value")) == want.value && String(_getfield(g, "unit")) == want.unit) ||
            throw(ArgumentError("board_config.layout_grid must be {value: $(want.value), unit: \"$(want.unit)\"}, got $(g)"))
        return PointGrid()
    end
    return PixelGrid(StructTypes.construct(ScalarOrQuantity, _getfield(x, "print_density")))
end

# The ruler's ticks, as keywords: what a stored board is rebuilt with while its
# ruler is on (with it off they are zero, `_frame_fields`).
_tick_kwargs(rs) = (;
    major_tick_spacing = rs.major_tick.spacing, minor_tick_spacing = rs.minor_tick.spacing,
    major_tick_width = rs.major_tick.width, minor_tick_width = rs.minor_tick.width,
)

function StructTypes.construct(::Type{BoardConfig}, x)
    canvas_size = StructTypes.construct(Size2, _getfield(x, "canvas_size"))
    layout_grid = _construct_grid(x)
    # a v4 board's margin (its board derived from it); a v3 board's board, as recorded
    sized = layout_grid isa PixelGrid ? (; board_size = StructTypes.construct(Size2, _getfield(x, "board_size"))) :
        (; margin = StructTypes.construct(Size2, _getfield(x, "margin")))
    q(k) = StructTypes.construct(ScalarOrQuantity, _getfield(x, k))
    frame_padding = q("frame_padding")
    ruler_spec = StructTypes.construct(RulerSpec, _getfield(x, "ruler_spec"))
    marks = if layout_grid isa PixelGrid
        # a v3 board: its band and every tick as recorded, ruled or blank
        (; ruler_width = q("frame_width"), frame_style = Symbol(_getfield(x, "frame_style")), _tick_kwargs(ruler_spec)...)
    else
        rw = q("ruler_width")
        (; ruler_width = rw, corner_size = q("corner_size"), (iszero(rw) ? (;) : _tick_kwargs(ruler_spec))...)
    end
    return BoardConfig(; canvas_size, sized..., layout_grid, frame_padding, marks...)
end


# Parameterized for the same reason as `PatternBundle`: `BoardConfig{L,D,F,P,RS}`
# and `PatternConfig{U,U2,PT}` are both UnionAlls.
"""
    BoardPreamble

Metadata for a board, bundling the physical `BoardConfig` and the
embedded `PatternConfig` used to render the pattern region.
"""
struct BoardPreamble{B <: BoardConfig, P <: PatternConfig}
    board_config::B
    pattern_config::P
end

StructTypes.StructType(::Type{<:BoardPreamble}) = StructTypes.CustomStruct()

StructTypes.lower(bp::BoardPreamble) =
    (board_config = bp.board_config, pattern_config = _lower_pattern(bp.pattern_config, bp.board_config.layout_grid))

# The pattern config in the board's stored unit: px (v3) as `PatternConfig`
# writes it; on a point grid its σ range in "pt" — the same numbers, declared
# as what they are — and neither a pixel type (a point-grid board has none: its
# blobs are vector shadings, its TIFF/PNG 8-bit conversions of the PDF), a
# pattern size (it is the board, `_board_pattern_points`) nor `raster_aligned`
# (always false: its centres sit on the 0.01 pt grid): none is part of its
# identity.
_lower_pattern(pc::PatternConfig, ::PixelGrid) = pc
function _lower_pattern(pc::PatternConfig, g::PointGrid)
    pc.pixel_type == DEFAULT_PIXEL_TYPE ||
        throw(ArgumentError("a point-grid board has no pixel type; got a pattern config with $(pc.pixel_type)"))
    pt(q) = (value = Float64(ustrip(px, q)), unit = _blob_unit(g))
    return Base.structdiff(
        merge(StructTypes.lower(pc), (min_scale = pt(pc.min_scale), max_scale = pt(pc.max_scale))),
        NamedTuple{(:pixel_type, :pattern_resolution, :raster_aligned)}
    )
end

# The inverse: a point grid's "pt" lengths back to the stored px they are (the
# same numbers), in the form `PatternConfig`'s reader takes; the pattern size is
# the board's (a v4 file written before 2026-10-14 records it: that one is read).
function _pattern_from_points(x, bc::BoardConfig)
    function stored(q)
        u = String(_getfield(q, "unit"))
        u == _blob_unit(PointGrid()) || throw(ArgumentError("a point-grid board's pattern lengths are in \"$(_blob_unit(PointGrid()))\", got \"$u\""))
        return Dict{String, Any}("value" => _getfield(q, "value"), "unit" => "px")
    end
    res = _getfield(x, "pattern_resolution", nothing)
    d = Dict{String, Any}(string(k) => v for (k, v) in pairs(x))
    d["pattern_resolution"] = if res === nothing
        p = _board_pattern_points(bc)
        Dict{String, Any}(k => Dict{String, Any}("value" => ustrip(px, q), "unit" => "px") for (k, q) in (("width", p.width), ("height", p.height)))
    else
        Dict{String, Any}("width" => stored(_getfield(res, "width")), "height" => stored(_getfield(res, "height")))
    end
    for k in ("min_scale", "max_scale")
        v = _getfield(x, k, nothing)
        v === nothing || (d[k] = stored(v))      # (a v1 config has no max_scale)
    end
    # no pixel type (a v4 file written before 2026-10-13 records "gray8", the
    # only one a board then took into its QR code; its stored hash stays its identity)
    t = get(d, "pixel_type", _pixel_type_name(DEFAULT_PIXEL_TYPE))
    t == _pixel_type_name(DEFAULT_PIXEL_TYPE) ||
        throw(ArgumentError("a point-grid board has no pixel type; this one records $(repr(t))"))
    d["pixel_type"] = t
    # always false on a point grid (a v4 file written before 2026-10-14 records it)
    a = get(d, "raster_aligned", false)
    a === false || throw(ArgumentError("a point-grid board's centres sit on its grid: raster_aligned must be false, got $(repr(a))"))
    d["raster_aligned"] = false
    return d
end

function StructTypes.construct(::Type{<:BoardPreamble}, x)
    bc = StructTypes.construct(BoardConfig, _getfield(x, "board_config"))
    raw = _getfield(x, "pattern_config")
    pc = StructTypes.construct(PatternConfig, bc.layout_grid isa PointGrid ? _pattern_from_points(raw, bc) : raw)
    return BoardPreamble(bc, pc)
end

# Carries the preamble's parameters so `preamble` is a concrete field — the
# cascade from `BoardPreamble` gaining them.
"""
    BoardBundle <: AbstractBundle

Container returned by `blob_board`: the preamble and the placed blobs.

Fields:
- `preamble::BoardPreamble`
- `blobs::Vector{Disc}` - blob locations in the board's stored unit (COLMAP
  convention: the grid's corner at 0)

Blobs are stored in the `:pattern` reference frame, in the board's stored unit
(`stored_density`): points on a v4 (point-grid) board, printer pixels on a
stored v3 board. Use `blobs(bundle, reference_frame; units=mm)` to extract them
in other frames or units, or `units = px` with a `density` for pixels.
"""
struct BoardBundle{B <: BoardConfig, P <: PatternConfig, D <: Disc} <: AbstractBundle
    preamble::BoardPreamble{B, P}
    blobs::Vector{D}          # see the note on `PatternBundle.blobs`
end

# Basic accessor functions (raw storage access)
pattern_config(bb::BoardBundle) = bb.preamble.pattern_config
"""
    board_config(bundle::BoardBundle) -> BoardConfig

The [`BoardConfig`](@ref) of a board: canvas, margin and board size, the
frame's marks and the layout grid.
"""
board_config(bb::BoardBundle) = bb.preamble.board_config
cutoff(bb::BoardBundle) = bb.preamble.pattern_config.cutoff

StructTypes.StructType(::Type{<:BoardBundle}) = StructTypes.CustomStruct()

# The blobs' declared unit: the board's stored unit.
_blob_unit(::PixelGrid) = "px"
_blob_unit(::PointGrid) = "pt"

function StructTypes.lower(bb::BoardBundle)
    return (preamble = bb.preamble, blobs = _lower_blobs(bb.blobs; unit = _blob_unit(bb.preamble.board_config.layout_grid)))
end

function StructTypes.construct(::Type{<:BoardBundle}, x)
    preamble = StructTypes.construct(BoardPreamble, _getfield(x, "preamble"))
    grid = preamble.board_config.layout_grid
    blobs = _construct_blobs(_getfield(x, "blobs"); unit = _blob_unit(grid))

    # A file written before 2026-10 also carries `saddle_points` (always empty:
    # no board used the removed saddle frame); it is ignored.
    return BoardBundle(preamble, blobs)
end

# =============================================================================
# Constants
# =============================================================================

# Standard print sizes using Size2 for physical dimensions
"""
    PRINT_SIZES

Mapping of common paper sizes (ISO/ANSI names) to physical `Size2` dimensions.
Use with `blob_board(paper_size::String, ...)`.
"""
const PRINT_SIZES = ImmutableDict(
    # ISO A-series (mm)
    "A0" => Size2(height = 1189.0mm, width = 841.0mm),
    "A1" => Size2(height = 841.0mm, width = 594.0mm),
    "A2" => Size2(height = 594.0mm, width = 420.0mm),
    "A3" => Size2(height = 420.0mm, width = 297.0mm),
    "A4" => Size2(height = 297.0mm, width = 210.0mm),
    "A5" => Size2(height = 210.0mm, width = 148.0mm),

    # ANSI sizes (inches, exact)
    "ANSI A" => Size2(height = 11.0inch, width = 8.5inch),   # Letter
    "ANSI B" => Size2(height = 17.0inch, width = 11.0inch),  # Tabloid (portrait)
    "ANSI C" => Size2(height = 22.0inch, width = 17.0inch),
    "ANSI D" => Size2(height = 34.0inch, width = 22.0inch),
    "ANSI E" => Size2(height = 44.0inch, width = 34.0inch),
    "ANSI E1" => Size2(height = 42.0inch, width = 30.0inch),

    # Common-name aliases
    "Letter" => Size2(height = 11.0inch, width = 8.5inch),
    "US Letter" => Size2(height = 11.0inch, width = 8.5inch),
    "Legal" => Size2(height = 14.0inch, width = 8.5inch),
    "Tabloid" => Size2(height = 17.0inch, width = 11.0inch),
    "Ledger" => Size2(height = 11.0inch, width = 17.0inch),
)

# A paper name's page size (a key of `PRINT_SIZES`); anything else is an
# `ArgumentError` listing the names.
function _paper_size(paper::AbstractString)
    haskey(PRINT_SIZES, paper) && return PRINT_SIZES[paper]
    throw(ArgumentError("unknown paper size $(repr(paper)); one of $(join(sort!(collect(keys(PRINT_SIZES))), ", "))"))
end

# =============================================================================
# UID Generation
# =============================================================================

# --- blobboards canonical form v1 --------------------------------------------
# hash_config used to hash JSON3.write(config) verbatim, so board identity was
# hostage to JSON3's key order and float printing — a serializer upgrade could
# silently re-uid every board. v1 mints from a canonical encoding independent
# of any JSON library: keys sorted bytewise, no whitespace, integers bare,
# integral Float64s printed as integers, other floats via Base's shortest
# round-trip printing (Ryu). Existing boards never re-mint: the stored
# `hashes.config` is authoritative on load (see load_bundle).

function _plain(x)
    x isa Union{AbstractString, Bool, Nothing} && return x
    x isa Integer && return Int64(x)
    x isa AbstractFloat && return Float64(x)
    x isa Symbol && return string(x)
    x isa NamedTuple &&
        return Dict{String, Any}(string(k) => _plain(v) for (k, v) in pairs(x))
    x isa AbstractDict &&
        return Dict{String, Any}(string(k) => _plain(v) for (k, v) in x)
    # A custom StructTypes.lower takes precedence over container-ness, so the
    # canonical tree mirrors the serialized structure (Size2 is an
    # AbstractVector but serializes as {width, height}; quantities as
    # {value, unit}).
    # `StructTypes.lower` has no fallback method: a plain Vector{Any} (a
    # config_dict value, e.g. a tilt schedule) has none, so ask first.
    if hasmethod(StructTypes.lower, Tuple{typeof(x)})
        l = StructTypes.lower(x)
        l !== x && return _plain(l)
    end
    x isa Union{AbstractVector, Tuple} && return Any[_plain(v) for v in x]
    throw(ArgumentError("cannot canonicalize $(typeof(x)): no StructTypes.lower"))
end

# The canonical encoding itself (sorted keys, bare integral floats, Ryu for the
# rest, loud on non-finite numbers) is VisualGeometryIo's `canonical_json` since
# 0.7.6; the LOWERING of a config into that tree (`_plain`) stays here with the
# schema. Board identities are `sha256_hex` over exactly that string.
using VisualGeometryIo: canonical_json, sha256_hex

"""
    canonical_config_json(config) -> String

The blobboards canonical (v1) encoding of a config's lowered tree — the byte
string board identities are minted from. Stable across JSON3/serializer
versions by construction.
"""
canonical_config_json(config) = canonical_json(_plain(config))

"""
    hash_config(config)

Hash a configuration (PatternConfig or BoardPreamble) — SHA-256 over the
canonical v1 encoding ([`canonical_config_json`](@ref)); full hex string. A
point-grid board's identity also covers its schema and the blobs its
configuration places (placement runs to compute it), so a rebuild that places
different blobs has a different identity.

Minting only: identities of existing boards come from their STORED
`hashes.config` (authoritative on load), never from recomputation.
"""
hash_config(config) = sha256_hex(canonical_config_json(config))

# A board's identity names the board that is PRINTED: on a point-grid (v4) board
# it covers its schema, its configuration and its placed blobs, so a board
# rebuilt from its configuration — from its QR code — proves it is the same
# board: any change in what placement produces (its code, the frame's marks,
# Julia's random stream, the machine) changes the identity and fails the code's
# check, instead of silently printing another board under the old name. A
# stored v3 board's identity was minted from its configuration alone, and its
# stored hash stays authoritative (`load_bundle`).
hash_config(p::BoardPreamble) = _hash_board(p.board_config.layout_grid, p)
_hash_board(::PixelGrid, p::BoardPreamble) = sha256_hex(canonical_config_json(p))    # no placement: config only
_hash_board(::PointGrid, p::BoardPreamble) = _board_identity(p, _place_board_blobs(p))
# The identity of a board whose blobs are known (minted where they were placed,
# or read with them), never re-placing them. A v4 board's covers the blobs in the
# encoding its file stores them in (`_lower_blobs`).
_board_identity(p::BoardPreamble, blobs) = _board_identity(p.board_config.layout_grid, p, blobs)
_board_identity(g::PixelGrid, p::BoardPreamble, _) = _hash_board(g, p)
function _board_identity(g::PointGrid, p::BoardPreamble, blobs)
    record = Dict{String, Any}("schema" => BOARD_SCHEMA, "preamble" => _plain(p), "blobs" => _plain(_lower_blobs(blobs; unit = _blob_unit(g))))
    return sha256_hex(canonical_json(record))
end

"""
    uid_from_hash(hex_hash)

First 3 hex characters of a SHA-256 config hash — matches the characters
physically printed in the board corner squares.
"""
uid_from_hash(hex_hash::AbstractString) = hex_hash[1:3]
