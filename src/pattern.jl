# =============================================================================
# Pattern subsystem: placement, blob rendering, and pattern API (consolidated)
# =============================================================================

# =============================================================================
# Per-scale-level collision grid (2026-07-28), holding geometric shapes (2026-10)
# =============================================================================
#
# Placement keeps SHAPES apart in one index: each placed blob's footprint — the
# circle of its cutoff radius, `Circle(disc, cutoff)` — and the keep-out
# rectangles of the frame's marks (`_keepouts`). Two isbits types: a vector of
# their union is stored inline, and the collision loop splits on it (no dynamic
# dispatch, no allocation). The keep-outs are the FIRST level, one cell that
# every candidate checks in full, so a candidate landing on a mark (a ruler band
# is ~17% of an A4 board) is rejected before any blob is looked at. Measured
# 2026-10-14 against a circles-only grid with the rectangles checked after it
# (identical blobs): 0.83× the time on an A4 ruler board at 1 mm, 0.91× on an
# A3 one with 17 250 blobs, 1.08× on an A4 `:corners` board, whose keep-outs
# are only its corner cells.
#
# Blobs are placed in DESCENDING scale order and each level holds a single σ,
# so a level grid with cell size 2·cutoff·σℓ answers any later candidate
# (σ ≤ σℓ) EXACTLY from the 3×3 cell ring around its center: the collision
# radius cutoff·(σℓ + σ) never exceeds the cell size. At saturation the
# sampler runs millions of rejected trials; the O(1) ring lookup roughly halves
# their cost vs an R-tree descent.
#
# The circle test, |c₁ − c₂|² < (r₁ + r₂)² with r = cutoff·σ, replaced the σ
# form |c₁ − c₂|² < (cutoff·(σ₁ + σ₂))² in 2026-10. The two can round apart at
# the last bit, so the switch was measured, not assumed: on 88 stored boards
# (2.6·10⁹ pair checks, 40 exact ties) and 241 new ones (9.7·10⁸ checks) the two
# decided every pair alike, and every placement was bit-for-bit unchanged.

const _Shape = Union{Circle{Float64}, Rect2{Float64}}

struct _LevelGrid
    cell::Float64
    nx::Int
    ny::Int
    buckets::Matrix{Vector{Int}}   # indices into the shared `shapes` vector
end

function _LevelGrid(σ::Float64, cutoff::Float64, pattern_size)
    cell = 2.0 * cutoff * σ
    nx = max(1, ceil(Int, Float64(pattern_size.width) / cell))
    ny = max(1, ceil(Int, Float64(pattern_size.height) / cell))
    return _LevelGrid(cell, nx, ny, [Int[] for _ in 1:ny, _ in 1:nx])
end

@inline function _grid_cell(g::_LevelGrid, cx::Float64, cy::Float64)
    ix = clamp(floor(Int, cx / g.cell) + 1, 1, g.nx)
    iy = clamp(floor(Int, cy / g.cell) + 1, 1, g.ny)
    return (ix, iy)
end

@inline function _grid_insert!(g::_LevelGrid, cx::Float64, cy::Float64, j::Int)
    ix, iy = _grid_cell(g, cx, cy)
    push!(g.buckets[iy, ix], j)
    return nothing
end

# Two shapes overlap when their gap is under zero: strict `<`, so tangent
# shapes do not (a blob may touch a keep-out, and another blob, at its rim).
@inline function _overlaps(a::Circle{Float64}, b::Circle{Float64})
    d = a.center - b.center
    R = a.r + b.r
    return dot(d, d) < R * R
end
# the distance from the circle's centre to the rectangle (0 inside it), against its radius
@inline function _overlaps(c::Circle{Float64}, r::Rect2{Float64})
    lo = origin(r)
    hi = lo + widths(r)
    dx = c.center[1] - clamp(c.center[1], lo[1], hi[1])
    dy = c.center[2] - clamp(c.center[2], lo[2], hi[2])
    return dx * dx + dy * dy < c.r * c.r
end

# Exact collision check of a candidate against every placed shape: scan the
# 3×3 ring in EVERY level grid (levels are few — the keep-outs', then one per
# scale). Enumeration order does not affect the boolean, only which collision
# is seen first (and so how soon a doomed candidate is rejected).
function _grids_collide(grids::Vector{_LevelGrid}, shapes::Vector{_Shape}, q::Circle{Float64})
    cx, cy = q.center
    @inbounds for g in grids
        ix, iy = _grid_cell(g, cx, cy)
        for jx in max(1, ix - 1):min(g.nx, ix + 1)
            for jy in max(1, iy - 1):min(g.ny, iy + 1)
                for j in g.buckets[jy, jx]
                    s = shapes[j]
                    (s isa Circle{Float64} ? _overlaps(s, q) : _overlaps(q, s)) && return true
                end
            end
        end
    end
    return false
end

# A candidate blob is placed when its footprint overlaps no shape: the blob is
# kept (`blobs`, the output) and its footprint indexed (`shapes`). `r` is the
# level's footprint radius, `cutoff * σ` — exactly `Circle(Disc(c, σ), cutoff)`'s,
# computed once per level; the `Disc` is built only for a placed blob.
@inline function _try_insert_grid!(
        grids::Vector{_LevelGrid}, g::_LevelGrid,
        shapes::Vector{_Shape}, blobs::Vector{Disc{Float64}},
        cx::Float64, cy::Float64, σ::Float64, r::Float64
    )
    footprint = Circle(Point2(cx, cy), r)
    _grids_collide(grids, shapes, footprint) && return false
    push!(blobs, Disc(cx, cy, σ))
    push!(shapes, footprint)
    _grid_insert!(g, cx, cy, length(shapes))
    return true
end

# Module-level helper functions for _place_blobs (avoid closure overhead)

@inline function _calc_sampling_rect(pattern_size, σ::Float64, cutoff::Float64)
    margin = σ * cutoff
    lo = Point2(margin, margin)
    hi = pattern_size .- margin
    return Rect(lo, hi - lo)
end

# Device-aligned: sample uniformly over the pixel centers k + 0.5 that lie
# inside [lower, upper]. Fixed 2026-07-28 — the previous version computed
# integer bounds for points AT integers (`ceil(lower)`/`floor(upper)`) while
# returning k + 0.5 (half-pixel convention mismatch), and drew from
# `floor(rand·n)` which can never produce the top value (exclusive-upper
# fencepost). Depending on frac(margin), each edge silently lost up to one
# admissible row/column. Correct bounds: k ∈ [ceil(lower−½), floor(upper−½)],
# inclusive draw via rand·(n+1). This changes seeded patterns — a deliberate
# versioned generation change; old boards are never regenerated.
@inline function _sample_center(rng::MersenneTwister, bbox::Rect2, ::Val{true})
    lower, upper = extrema(bbox)
    lo_x = ceil(Int, lower[1] - 0.5)
    lo_y = ceil(Int, lower[2] - 0.5)
    hi_x = floor(Int, upper[1] - 0.5)
    hi_y = floor(Int, upper[2] - 0.5)
    ix = lo_x + floor(Int, rand(rng) * (hi_x - lo_x + 1))
    iy = lo_y + floor(Int, rand(rng) * (hi_y - lo_y + 1))
    return (Float64(ix) + 0.5, Float64(iy) + 0.5)
end

# Point grid (a v4 board, coordinates in pt): sample uniformly over the
# 0.01 pt grid points k/100 inside [lower, upper] — the numbers the print master
# holds exactly (see `PointGrid`). The integer k is drawn exactly as the pixel
# sampler draws its pixel, and k/100 is the double nearest k/100.
@inline function _sample_center(rng::MersenneTwister, bbox::Rect2, ::Val{:point_grid})
    n = _grid_per_stored(PointGrid())                 # grid units per point
    lower, upper = extrema(bbox)
    lo_x, lo_y = ceil(Int, lower[1] * n), ceil(Int, lower[2] * n)
    hi_x, hi_y = floor(Int, upper[1] * n), floor(Int, upper[2] * n)
    ix = lo_x + floor(Int, rand(rng) * (hi_x - lo_x + 1))
    iy = lo_y + floor(Int, rand(rng) * (hi_y - lo_y + 1))
    return (_stored(PointGrid(), ix), _stored(PointGrid(), iy))
end

# Continuous: sample anywhere in the rectangle
@inline function _sample_center(rng::MersenneTwister, bbox::Rect2, ::Val{false})
    o = bbox.origin
    w = bbox.widths
    return (o[1] + rand(rng) * w[1], o[2] + rand(rng) * w[2])
end

# Place isotropic blobs without overlap, larger scales first (better packing),
# clear of `keepouts`, with per-level grids for the collision checks. `max_trials` caps the
# insertion trials per blob. `sampler` is the centre sampler's `Val`:
# `Val(true)` pixel centres, `Val(false)` continuous, `Val(:point_grid)` the
# 0.01 pt grid of a v4 board.
function _place_blobs(
        rng::MersenneTwister,
        pattern_size::Size2,
        blob_counts,
        sampler::Val;
        max_trials::Int,
        cutoff::Float64,
        keepouts::Vector{Rect2{Float64}} = Rect2{Float64}[]
    )

    @assert pattern_size > zero(pattern_size) "Pattern size must be positive"
    pattern_size = pattern_size isa Size2{<:Quantity} ? ustrip.(px, pattern_size) : pattern_size

    # The placed blobs, and every shape they keep clear of in a vector the
    # per-level grids index (see the _LevelGrid header). The keep-outs are the
    # first level, a single cell every candidate checks in full; with none, the
    # grids, the trials and every placement are those of a board without.
    blobs = Disc{Float64}[]
    shapes = _Shape[]
    grids = _LevelGrid[]
    if !isempty(keepouts)
        k = _LevelGrid(max(Float64(pattern_size.width), Float64(pattern_size.height)), 1, 1, [Int[] for _ in 1:1, _ in 1:1])
        for r in keepouts
            push!(shapes, r)
            push!(k.buckets[1, 1], length(shapes))
        end
        push!(grids, k)
    end

    sorted_blob_counts = sort(blob_counts, by = x -> x.σ, rev = true)
    # Function barrier on the sampler Val: constructing a Val from a
    # RUNTIME Bool inside the trial loop is a dynamic dispatch PER TRIAL —
    # measured ~36 ns × millions of saturation trials ≈ 0.18 s on an A4-class
    # pattern. One dispatch per scale level instead; identical computations.
    for bc in sorted_blob_counts
        σ = Float64(strip_or_bare(px, bc.σ))  # Strip units once per scale level, ensure Float64
        g = _LevelGrid(σ, cutoff, pattern_size)
        push!(grids, g)
        sampling_rect = _calc_sampling_rect(pattern_size, σ, cutoff)
        successes = _place_scale!(
            rng, grids, g, shapes, blobs, sampling_rect, σ, cutoff,
            bc.count, max_trials, sampler
        )
        @debug "Placed $successes out of $(bc.count) for scale $(bc.σ)"
    end

    return blobs
end

# Trial loop for one scale level, specialized on the sampler Val — see the
# barrier comment in `_place_blobs`. Bit-for-bit the same trial sequence.
function _place_scale!(
        rng, grids::Vector{_LevelGrid}, g::_LevelGrid,
        shapes::Vector{_Shape}, blobs::Vector{Disc{Float64}}, sampling_rect,
        σ::Float64, cutoff::Float64, count::Integer, max_trials::Int,
        sampler::Val
    )
    successes = 0
    r = cutoff * σ                       # the footprint radius, as `Circle(disc, cutoff)` has it
    for _ in 1:count
        for trial in 1:max_trials
            cx, cy = _sample_center(rng, sampling_rect, sampler)
            if _try_insert_grid!(grids, g, shapes, blobs, cx, cy, σ, r)
                successes += 1
                break
            end
            if trial == max_trials
                @debug "Cannot place blob of scale $σ after $max_trials trials"
                break
            end
        end
    end
    return successes
end

# -------------------
# Blob pixel drawing
# -------------------

"""
    blob_profile(t, cutoff) -> Float64

THE blob profile — the one definition every rendering of a blob uses: the
printed intensity of a dark blob (0 = black, 1 = white) at normalised radius
`t = d / (cutoff·σ)`, the Gaussian truncated at its cutoff disc and normalised
over the CONTINUOUS disc, so the centre is exactly black and the rim exactly
white (full contrast at any resolution):

    v(t) = (1 − e^(−k t²)) / (1 − e^(−k)),   k = cutoff²/2;   v = 1 for t ≥ 1.

`render_pattern` samples it at pixel centres; a vector board draws it as each
blob's radial shading (the calculator function writes this formula, the sampled
function tabulates it). It is the profile the detector's scale-bias model
assumes (`_profile_log_response`: the same function, as darkness `1 − v`).
Until 2026-10 the raster normalised each blob over its SAMPLED pixels instead,
which stretched it by up to ~3.5 % near the rim depending on how the pixel grid
met the disc.
"""
function blob_profile(t::Real, cutoff::Real)
    t >= 1 && return 1.0
    k = cutoff^2 / 2
    return (1 - exp(-k * t^2)) / (1 - exp(-k))
end

# Sample `blob_profile` at the centres of the pixels inside the blob's cutoff
# disc (pixels outside keep the background).
function render_blob!(pattern, b::Disc; cutoff::Float64)
    # Pattern-render blobs are plain Float64 (no units); the pixel-origin
    # frame map is unitless by contract.
    cx, cy = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})(b.center)
    σ = b.σ
    r = σ * cutoff
    r² = r * r

    # Bounding box (clipped to pattern bounds)
    rows, cols = size(pattern)
    row_lo = max(1, ceil(Int, cy - r))
    row_hi = min(rows, floor(Int, cy + r))
    col_lo = max(1, ceil(Int, cx - r))
    col_hi = min(cols, floor(Int, cx + r))

    T = eltype(pattern)
    @inbounds for col in col_lo:col_hi
        dx² = (col - cx)^2
        for row in row_lo:row_hi
            dist² = dx² + (row - cy)^2
            dist² <= r² && (pattern[row, col] = T(blob_profile(sqrt(dist² / r²), cutoff)))
        end
    end
    return pattern
end

# -------------------
# Pattern API
# -------------------

"""
# Pattern Generation Algorithms

This module contains the core algorithms for generating blob patterns:

- Scale calculation and distribution
- Spatial placement with collision detection
- Pattern rendering and anti-aliasing

## Main Functions
- [`blob_pattern`](@ref): Generate patterns in pixel space
- [`calc_blob_scales`](@ref): The geometric scale ladder from min_scale to max_scale
- `_place_blobs`: Place blobs with spatial constraints
"""

# The blob scales are a geometric LADDER from `min_scale` to `max_scale`, BOTH
# exact: k steps, k the whole number of 1/scales_per_octave-octave steps nearest
# the span, spread evenly in log between the two ends (so a step is ≈ 1/3 octave
# at the default, never exactly unless max lies on the 2^(1/3) ladder — in which
# case the steps are exactly 2^(i/3), the original formula, bit for bit). Both
# ends are what PatternConfig stores (schema v2). A v1 file's
# `max_diameter_fraction` is converted to the rung it always produced: the
# largest 2^(i/3) σ whose 2·cutoff·σ footprint is ≤ that fraction of the short
# side — on the ladder, so v1 boards re-place bit-identically.

# The number of intervals k between min_scale and max_scale (both px, unitless).
function _ladder_intervals(min_scale::Float64, max_scale::Float64, scales_per_octave::Int)
    max_scale >= min_scale ||
        throw(ArgumentError("max_scale ($(max_scale)px) must be ≥ min_scale ($(min_scale)px)"))
    max_scale == min_scale && return 0
    return max(1, round(Int, scales_per_octave * log2(max_scale / min_scale)))
end

# The top rung index k a v1 `max_diameter_fraction` produced — the old
# `calc_blob_scales` arithmetic verbatim, so a v1 board's ladder is bit-identical.
function _ladder_top_from_fraction(pattern_size, min_scale::Float64, max_diameter_fraction::Float64,
                                   scales_per_octave::Int, cutoff::Float64)
    (0 < max_diameter_fraction < 1) || throw(ArgumentError("max_diameter_fraction ($max_diameter_fraction) must be between 0 and 1"))
    σ_cap = (minimum(pattern_size) * max_diameter_fraction) / (2 * cutoff)
    if σ_cap < min_scale
        min_required_size = ceil(Int, min_scale * 2 * cutoff / max_diameter_fraction)
        throw(
            ArgumentError(
                "Pattern size too small for requested parameters. " *
                    "Minimum pattern dimension is $(minimum(pattern_size)), " *
                    "but need at least $(min_required_size) px " *
                    "for min_scale=$(min_scale)px with max_diameter_fraction=$max_diameter_fraction. " *
                    "Try: (1) increasing pattern size, (2) reducing min_scale, or (3) increasing max_diameter_fraction."
            )
        )
    end
    num_scales = floor(Int, log2(σ_cap / min_scale) * scales_per_octave)
    num_scales <= 0 && throw(
        ArgumentError(
            "No valid blob scales can be generated: the range between min_scale=$(min_scale)px and " *
                "the σ cap $(σ_cap)px (max_diameter_fraction=$(max_diameter_fraction), cutoff=$(cutoff), " *
                "pattern_size=$(pattern_size)) is under one step at scales_per_octave=$(scales_per_octave). " *
                "Try: (1) decreasing min_scale, (2) increasing pattern size, (3) increasing max_diameter_fraction, or (4) decreasing scales_per_octave."
        )
    )
    return num_scales - 1
end

# Check the largest blob's 2·cutoff·σ footprint fits the pattern. Returns max_scale.
function _check_max_scale(pattern_size, min_scale::Float64, max_scale::Float64,
                          scales_per_octave::Int, cutoff::Float64)
    scales_per_octave > 0 || throw(ArgumentError("scales_per_octave ($scales_per_octave) must be positive"))
    _ladder_intervals(min_scale, max_scale, scales_per_octave)    # max ≥ min
    2 * cutoff * max_scale < minimum(pattern_size) || throw(
        ArgumentError(
            "max_scale $(max_scale)px does not fit: its 2·cutoff·σ footprint $(2 * cutoff * max_scale)px " *
                "is not smaller than the pattern's short side $(minimum(pattern_size))px"
        )
    )
    return max_scale
end

# =============================================================================
# blob_sizes — the user-facing size specification (printed disc DIAMETERS)
# =============================================================================

struct ShortSide end

"""
    short_side

Marker for a size relative to the pattern's short side: `0.2short_side` is 20% of
it. Valid as the MAX of `blob_sizes`, a disc diameter like any other size
(`blob_sizes = (2mm, 0.2short_side)` makes the largest disc's diameter 20% of
the short side).
"""
const short_side = ShortSide()

"""
    ShortSideFraction(fraction)

A size given as a fraction of the pattern's short side — what `fraction * short_side`
builds. Resolved to px once the pattern size is known; never stored.
"""
struct ShortSideFraction
    fraction::Float64
    function ShortSideFraction(fraction::Real)
        0 < fraction < 1 || throw(ArgumentError("a short_side fraction must be in (0, 1), got $fraction"))
        return new(Float64(fraction))
    end
end
Base.:*(fraction::Real, ::ShortSide) = ShortSideFraction(fraction)

# A blob's printed disc has diameter 2·cutoff·σ: `blob_sizes` are those
# diameters, the stored pair is σ.

# The smallest disc that prints well at a printer's density: 3 mm
# at 1200 dpi, inversely proportional to the density (6 mm at 600, 12 mm at
# 300). A printer makes grey from halftone cells, each a fixed number of printer
# dots, so a cell's size in mm — and the smallest disc whose profile spans
# enough of them — grows as the density drops. At 1200 dpi under a typical
# 150 lpi screen (0.17 mm cells) a 3 mm disc's radius spans ~9 cells. 6 mm at
# 600 dpi matches a campus Ricoh laser measured 2026-10-11 (A4 test board
# scanned at 600 dpi: discs from ~5–6 mm up print the profile within ~10
# levels of 255; at 2.5 mm the shoulder is ~20 levels light, below 1 mm ~40;
# examples/print_quality).
# The rule's two numbers are configs/authoring.toml's `[blob_board]`.
"""
    suggest_blob_sizes(density, extent) -> (smallest, largest)

The `blob_sizes` of a board printed on a printer of `density` (e.g. `600dpi`):
the smallest and largest printed disc DIAMETERS, as a tuple of lengths in mm.

- smallest: the disc that prints its profile well at `density` — 3 mm at
  1200 dpi, inversely proportional to the density (6 mm at 600 dpi, 12 mm at
  300). A printer makes grey from halftone cells, each a fixed number of
  printer dots, so the smallest disc whose profile spans enough of them grows
  as the density drops.
- largest: 20% of the board's shorter side, but never more than 64 mm (20% of
  an A0 board would be 164 mm).

`extent` is the board's size — a `Size2` or a `(width, height)` tuple of
lengths — or a paper name ([`BlobBoards.PRINT_SIZES`](@ref)), which stands for
the board `blob_board` makes on that paper by default: the paper less the
10 mm margin on every side (A4: 190 × 277 mm).

```julia
blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), dir = "./out")   # 6 mm … 38 mm
suggest_blob_sizes(1200dpi, (300mm, 200mm))                                      # (3.0 mm, 40.0 mm)
```

A board has no density of its own and `blob_sizes` no default, so this is how
a printer enters a board's design. Either end may also be given directly: a
length, or `0.2short_side` for the max (of the pattern's short side, no cap).
The rule's numbers are `configs/authoring.toml`'s `min_blob_diameter_mm` at
`min_blob_reference_dpi`, `max_blob_fraction` and `max_blob_diameter_mm`.
"""
function suggest_blob_sizes(density, extent)
    density isa Quantity && dimension(density) == dimension(dpi) || throw(
        ArgumentError("suggest_blob_sizes takes a printer's density first, e.g. suggest_blob_sizes(600dpi, \"A4\"); got $(repr(density))")
    )
    w, h = _board_extent(extent)
    smallest = _board_dflt("min_blob_diameter_mm") * (_board_dflt("min_blob_reference_dpi") / ustrip(dpi, density))
    largest = min(_pattern_dflt("max_blob_fraction") * min(w, h), _board_dflt("max_blob_diameter_mm"))
    return (smallest * mm, largest * mm)
end

# A board's (width, height) in mm: a `Size2` or a tuple of two lengths, or a
# paper name for the default board on that paper (less `DEFAULT_BOARD_MARGIN`).
_board_extent(paper::AbstractString) = _board_extent(_default_board_size(_paper_size(paper)))
_board_extent(s::Size2) = _board_extent((s.width, s.height))
function _board_extent(extent)
    extent isa Tuple && length(extent) == 2 && all(x -> x isa Quantity && dimension(x) == dimension(mm) && ustrip(mm, x) > 0, extent) || throw(
        ArgumentError(
            "suggest_blob_sizes takes the board's extent — a Size2, a (width, height) tuple of " *
                "positive lengths, or a paper name such as \"A4\"; got $(repr(extent))"
        )
    )
    return (Float64(ustrip(mm, extent[1])), Float64(ustrip(mm, extent[2])))
end
_diameter_factor(cutoff::Float64) = 2 * cutoff

# The default largest blob when `blob_sizes` gives no max: a disc whose DIAMETER is
# 20% of the short side (v1's `max_diameter_fraction = 0.2`).
const DEFAULT_MAX_BLOB = _pattern_dflt("max_blob_fraction") * short_side

# The smallest σ the placement accepts (`calc_blob_scales`), in px.
const MIN_SIGMA_PX = 2.0

# A number for a message: integral values without a trailing ".0", others to 2 decimals.
_short_number(x::Real) = isinteger(round(x; digits = 2)) ? string(Int(round(x; digits = 2))) : string(round(x; digits = 2))

# Resolve the user's `blob_sizes` (printed disc DIAMETERS) into PatternConfig's
# stored `min_scale`/`max_scale` (σ, the stored unit). `to_stored` turns one size quantity into the stored unit
# (identity for a pattern; × print density for a board). The default min is
# `default_min_sigma` (a pattern's: its diameter, 2·cutoff·σ, follows `cutoff`,
# so the default pattern is the same σ ladder at any cutoff); a board has no
# default (`nothing`: `blob_sizes` is required — its smallest disc is a choice
# about the printer, `suggest_blob_sizes(density, extent)`). `floor_text(d_px)` states the
# smallest allowed diameter in the CALLER's units. A board's max is never
# `nothing` either: a length (`suggest_blob_sizes`'s) or `f * short_side`. The
# stored pair is the ONLY representation: units and any short_side fraction
# are not recorded.
function _resolve_blob_sizes(
        to_stored, fname::Symbol, kw::NamedTuple, pattern_size_px;
        default_min_sigma = nothing, floor_text = d -> "$(_short_number(d)) px", min_sigma::Float64 = MIN_SIGMA_PX
    )
    for k in (:min_scale, :max_scale, :max_diameter_fraction, :blob_measure)
        haskey(kw, k) && throw(
            ArgumentError(
                "$fname: `$k` is not a keyword — give blob sizes as " *
                    "`blob_sizes = (min, max)`, the printed disc DIAMETERS " *
                    "(max may be `0.2short_side`" *
                    (default_min_sigma === nothing ? ", or a length), e.g. `blob_sizes = suggest_blob_sizes(600dpi, \"A4\")`" : " or `nothing` for the default)")
            )
        )
    end
    cutoff = Float64(get(kw, :cutoff, DEFAULT_CUTOFF))
    factor = _diameter_factor(cutoff)
    haskey(kw, :blob_sizes) || default_min_sigma !== nothing || throw(
        ArgumentError(
            "$fname: `blob_sizes = (min, max)` is required — the smallest and largest printed disc " *
                "diameters, e.g. blob_sizes = suggest_blob_sizes(600dpi, \"A4\") for a 600 dpi printer " *
                "and the default A4 board (6 mm … $(_short_number(100 * DEFAULT_MAX_BLOB.fraction))% of the board's " *
                "shorter side, at most $(_short_number(_board_dflt("max_blob_diameter_mm"))) mm)"
        )
    )
    sizes = get(() -> (factor * default_min_sigma, nothing), kw, :blob_sizes)
    (sizes isa Tuple && length(sizes) == 2) ||
        throw(ArgumentError("blob_sizes must be a (min, max) tuple, got $(repr(sizes))"))
    min_size, max_size = sizes
    short = minimum(pattern_size_px)

    min_size isa ShortSideFraction &&
        throw(ArgumentError("blob_sizes: the min must be a size, not a short_side fraction"))
    min_size === nothing && throw(ArgumentError("blob_sizes: the min cannot be nothing"))
    # Canonical σ: the same blob typed in mm or inch differs only in the last
    # ulp, which would mint a different identity for the same board. 12
    # significant digits is far below printable (~1e-11 px) and makes the stored
    # pair a function of the BLOB, not the spelling. Applied to new INPUT only —
    # a stored config (v1 or v2) is never re-rounded.
    min_scale = round(ustrip(px, to_stored(min_size)) / factor; sigdigits = 12)
    # (a relative hair of slack: a size typed as exactly the floor, in mm, can
    # land an ulp under it after the unit conversion)
    min_scale > 0 || throw(ArgumentError("$fname: the smallest blob diameter must be positive, got blob_sizes[1] = $(min_size)"))
    min_scale >= min_sigma * (1 - 1.0e-9) || throw(
        ArgumentError(
            "$fname: the smallest blob diameter must be ≥ $(floor_text(factor * min_sigma)) " *
                "(σ ≥ $(_short_number(min_sigma)) px at cutoff $(_short_number(cutoff))); got blob_sizes[1] = $(min_size)"
        )
    )

    max_scale = if max_size === nothing
        # a pattern's default; a board states its largest disc
        default_min_sigma === nothing && throw(
            ArgumentError(
                "$fname: the largest blob cannot be `nothing` — give blob_sizes[2] as a length " *
                    "(`suggest_blob_sizes(density, extent)[2]`: $(_short_number(100 * DEFAULT_MAX_BLOB.fraction))% of the " *
                    "board's shorter side, at most $(_short_number(_board_dflt("max_blob_diameter_mm"))) mm) or as `0.2short_side`"
            )
        )
        DEFAULT_MAX_BLOB.fraction * short / factor
    elseif max_size isa ShortSideFraction
        max_size.fraction * short / factor
    else
        ustrip(px, to_stored(max_size)) / factor
    end
    max_scale = round(max_scale; sigdigits = 12)              # canonical, as min_scale
    max_scale >= min_scale || throw(
        ArgumentError(
            "$fname: the smallest blob diameter ($(floor_text(factor * min_scale))) is larger than the largest " *
                "($(floor_text(factor * max_scale)))" *
                (haskey(kw, :blob_sizes) ? "" : "; the default smallest blob is too large for this pattern") *
                " — give blob_sizes = (min, max)"
        )
    )
    rest = Base.structdiff(kw, NamedTuple{(:blob_sizes,)})
    return merge(rest, (min_scale = min_scale * px, max_scale = max_scale * px))
end

# The pattern size as unitless px (an Integer Size2 is px).
_pattern_px(pattern_resolution::Size2{<:Integer}) = Float64.(pattern_resolution)
_pattern_px(pattern_resolution::Size2) = ustrip.(px, pattern_resolution)

# A pattern's sizes are already in px; a bare number is px.
function _pattern_to_px(q)
    q isa Quantity || return Float64(q) * px
    dimension(q) == dimension(px) || throw(ArgumentError("blob_sizes must be in pixels, got $(q)"))
    return q
end

"""
    calc_blob_scales(min_scale, max_scale; scales_per_octave) -> Vector

The blob-scale ladder from `min_scale` to `max_scale`, both exact: `k + 1` scales,
`k` the whole number of `1/scales_per_octave`-octave steps nearest the span, spaced
evenly in log. When `max_scale` lies on the `2^(1/scales_per_octave)` ladder the
scales are exactly `min_scale · 2^(i / scales_per_octave)`. Unitless px.
`min_sigma` is the smallest `min_scale` accepted (2 px, the rendering floor).
"""
function calc_blob_scales(
        min_scale::Real, max_scale::Real; scales_per_octave::Int = _pattern_dflt("scales_per_octave"), min_sigma::Real = MIN_SIGMA_PX
    )
    scales_per_octave > 0 || throw(ArgumentError("scales_per_octave ($scales_per_octave) must be positive"))
    min_scale >= min_sigma || throw(
        ArgumentError(
            "the smallest blob σ is $(min_scale) px; it must be at least $(min_sigma) px to render — " *
                "raise the min of blob_sizes (or the print density)"
        )
    )
    lo, hi = Float64(min_scale), Float64(max_scale)
    k = _ladder_intervals(lo, hi, scales_per_octave)
    # On the 2^(1/spo) ladder (every v1 board, and any max that lands on a rung):
    # the original formula, so those boards re-place bit-identically.
    hi == lo * 2^(k / scales_per_octave) && return [lo * 2^(i / scales_per_octave) for i in 0:k]
    octaves = log2(hi / lo) / k                               # octaves per interval
    scales = [lo * 2^(i * octaves) for i in 0:k]
    scales[end] = hi                                          # the requested max, exactly
    return scales
end

function calc_scale_counts(
        pattern_size::Size2, scales; α = Float64(_pattern_dflt("alpha")), cutoff::Float64,
        free_area = VgtkCore.area(pattern_size)
    )
    isempty(scales) && throw(ArgumentError("Empty candidate scale list; check blob scale parameters (see calc_blob_scales diagnostics)."))
    @assert all(scales .>= zero(eltype(scales))) @error "Candidate scales must be positive"
    scales_dimensionless = scales
    pattern_area = free_area          # the pattern less what keep-outs take
    weights = normalize(1.0 ./ scales_dimensionless .^ α, 1)
    blob_areas = pi .* (cutoff .* scales_dimensionless) .^ 2
    avg_area = dot(weights, blob_areas)
    total_blobs = ceil(Int, pattern_area / avg_area)
    counts = ceil.(Int, total_blobs .* weights)
    freqs = StructArray((σ = scales, count = counts))

    return freqs
end

"""
    blob_pattern(config::PatternConfig; dir=nothing)

Generate a deterministic blob pattern from a `PatternConfig`.
RNG is `MersenneTwister(config.seed)` for reproducible output.

Returns `(pattern, blobs)` where pattern is the rendered image and blobs is a vector of placed blobs.
If `dir` is provided, saves the pattern as PNG with JSON sidecar.
"""
function blob_pattern(config::PatternConfig; dir::Union{Nothing, AbstractString} = nothing)
    blobs = _place_pattern_blobs(config)
    # Direct 1× render (oversampling removed 2026-07-28): `blob_profile` is
    # continuous and pinned to the background at the truncation boundary, and
    # raster-aligned centers make the point-sampled Gaussian symmetric per blob
    # — the former 2×-render + Lanczos downsample approximated an area integral
    # of a function smooth far below Nyquist. `render_pattern` is the render
    # half of this function. Placement forbids cutoff-disc overlap, so
    # rendering the sorted list is bit-identical to rendering it in placement
    # order (the golden hashes pin it).
    pattern = render_pattern(config, blobs)

    bundle = PatternBundle(config, blobs)
    !isnothing(dir) && save(pattern, bundle; dir = dir, overwrite = true)

    return pattern, blobs
end

# The placement half of `blob_pattern`: the deterministic blob list (sorted,
# px-united), without rendering — what a vector board needs, where the raster
# would be a full-resolution matrix thrown away.
#
# `sampler` places the centres (default: the config's own, pixel centres or
# continuous); a v4 board supplies `Val(:point_grid)`. `min_sigma` is the
# smallest σ the ladder accepts, in the config's units: 2 px for a pattern; a
# v4 board, in points, checks its floor against its print density instead
# (`blob_board`), so it supplies 0.
function _place_pattern_blobs(
        config::PatternConfig; sampler::Val = Val(config.raster_aligned), min_sigma::Real = MIN_SIGMA_PX,
        keepouts::Vector{Rect2{Float64}} = Rect2{Float64}[]
    )
    rng = MersenneTwister(config.seed)

    pattern_resolution = ustrip.(px, config.pattern_resolution)
    min_scale = ustrip(px, config.min_scale)

    blob_scales = calc_blob_scales(
        min_scale, ustrip(px, config.max_scale);
        scales_per_octave = config.scales_per_octave, min_sigma
    )
    blob_counts = calc_scale_counts(
        pattern_resolution,
        blob_scales;
        α = config.α,
        cutoff = config.cutoff,
        # the area blobs may fill: the pattern less the keep-outs (inside it; they may overlap)
        free_area = VgtkCore.area(pattern_resolution) - _union_area(keepouts)
    )
    placed_blobs = _place_blobs(
        rng,
        pattern_resolution,
        blob_counts,
        sampler;
        max_trials = config.max_trials,
        cutoff = config.cutoff,
        keepouts
    )
    blobs_unitless = sort(placed_blobs, by = b -> Tuple(origin(b)))
    return blobs_unitless .* (1.0px)     # px units for storage
end

# The area of a union of rectangles (a few: a board's frame marks), exactly, on
# the grid their edges cut the plane into.
function _union_area(rects::Vector{Rect2{Float64}})
    isempty(rects) && return 0.0
    xs = sort!(unique!(vcat([[origin(r)[1], origin(r)[1] + widths(r)[1]] for r in rects]...)))
    ys = sort!(unique!(vcat([[origin(r)[2], origin(r)[2] + widths(r)[2]] for r in rects]...)))
    a = 0.0
    for i in 1:(length(xs) - 1), j in 1:(length(ys) - 1)
        cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
        any(r -> origin(r)[1] < cx < origin(r)[1] + widths(r)[1] && origin(r)[2] < cy < origin(r)[2] + widths(r)[2], rects) &&
            (a += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]))
    end
    return a
end

# A board's placement, on its grid (`LayoutGrid`): a v3 board's on its pixel
# centres, exactly as it was placed; a point-grid board's on 0.01 pt grid points,
# its σ floor already checked at its print density (`blob_board`), clear of its
# frame's marks (`_keepouts`).
_place_board_blobs(p::BoardPreamble) = _place_board_blobs(p.board_config.layout_grid, p)
_place_board_blobs(::PixelGrid, p::BoardPreamble) = _place_pattern_blobs(p.pattern_config)
_place_board_blobs(g::PointGrid, p::BoardPreamble) =
    _place_pattern_blobs(p.pattern_config; sampler = Val(:point_grid), min_sigma = 0.0, keepouts = _keepouts(g, p.board_config))

"""
    render_pattern(config::PatternConfig, blobs) -> Matrix

Rasterize a STORED blob list — the render half of [`blob_pattern`](@ref)
without the placement half: `BlobBoards.blob_profile` sampled exactly at
every pixel centre, in `config.pixel_type` (so `Gray{N0f16}` or float patterns
keep their precision). A board's own raster outputs (TIFF, PNG, the gallery's
images) are conversions of its vector PDF instead, painted from the same
recorded blobs (`render_board`); the two agree to within one 8-bit level.
"""
function render_pattern(config::PatternConfig, blobs)
    res = ustrip.(px, config.pattern_resolution)
    pattern = ones(config.pixel_type, Int(res.height), Int(res.width))
    # stored blobs are px-united by construction (`blobs_unitless .* (1.0px)`
    # above), so the px strip is the exact inverse
    discs = ustrip.(px, blobs)
    # order-independent: placement forbids cutoff-disc overlap, so each blob
    # writes a disjoint region
    render_blob!.(Ref(pattern), discs; cutoff = config.cutoff)
    clamp!(pattern, zero(eltype(pattern)), one(eltype(pattern)))
    if config.polarity == :light
        pattern .= one(eltype(pattern)) .- pattern
    end
    return pattern
end

# The pixel type is a TYPE argument, so it is positional and leads
# (`blob_pattern(Gray{N0f16}, size)`) — never a keyword. `PatternConfig` keeps
# the FIELD (a config struct is data), but the generation entry points forward
# their keywords into it, so without this check `pixel_type = …` would still be
# accepted there. `form` is the positional spelling the error points to.
function _reject_pixel_type_keyword(fname::Symbol, form::AbstractString, kwargs)
    haskey(kwargs, :pixel_type) || return nothing
    throw(ArgumentError("$fname: the pixel type is a positional argument, not a keyword — call `$form`"))
end

"""
    blob_pattern(pixel_type::Type, pattern_resolution::Size2; kwargs...)

Generate a blob pattern with specified pixel type and resolution.

# Arguments
- `pixel_type`: Output pixel type (e.g., `Gray{Float64}`, `Gray{N0f16}`)
- `pattern_resolution`: Pattern size as `Size2` with logical units

# Keyword Arguments
- `blob_sizes`: `(min, max)`, the smallest and largest disc DIAMETER
  (`2·cutoff·σ`) in **logical units** (`12px`; a bare number is px). `max` may be
  `0.2short_side` (a fraction of the short side) or `nothing` (default: 20% of
  the short side). The default min is `2·cutoff·2px` (8 px at the default
  cutoff), which is also the floor (σ ≥ 2 px). Sizes are stored as σ in px
  (`min_scale`, `max_scale`), both exact; the sizes between are spaced evenly in
  log at about `1/scales_per_octave` octave (see `calc_blob_scales`).
- `seed`: Random seed for reproducible patterns
- `scales_per_octave`: Number of scales per octave (default 3)
- `α`: Scale sampling bias (default 1.2)
- `cutoff`: Gaussian truncation radius in units of σ (default `DEFAULT_CUTOFF` = 2; use it)
- `max_trials`: Maximum placement trials per blob (default 5000)
- `raster_aligned`: Align blob centers to raster grid (default true)
- `polarity`: Blob polarity, `:dark` (default) or `:light`
- `dir`: Output directory for saving files (optional)

Returns `(pattern, blobs, config)`.
"""
function blob_pattern(pixel_type::Type{<:Union{Gray, Normed, AbstractFloat}}, pattern_resolution::Size2{<:Union{Integer, Quantity}}; kwargs...)
    _reject_pixel_type_keyword(:blob_pattern, "blob_pattern(pixel_type, pattern_resolution; kwargs...)", kwargs)
    kw = _resolve_blob_sizes(
        _pattern_to_px, :blob_pattern, (; kwargs...), _pattern_px(pattern_resolution); default_min_sigma = MIN_SIGMA_PX * px
    )
    pattern_kwargs, dropped = filter_kwargs(PatternConfig; kw...)
    config = PatternConfig(pattern_resolution = pattern_resolution, pixel_type = pixel_type; pattern_kwargs...)

    return blob_pattern(config; dropped...)..., config
end

"""
    blob_pattern(width, height; kwargs...)

Generate a blob pattern with specified dimensions in pixels.

# Arguments
- `width`, `height`: Pattern dimensions in pixels (Integer) or logical units (px quantities)

# Keyword Arguments
- `blob_sizes`: disc diameter range (see the `pixel_type` form)
- Other arguments same as `blob_pattern(pixel_type, pattern_resolution; kwargs...)`,
  except the pixel type: it is positional, so supply `blob_pattern(T, Size2(…))`
  (a `pixel_type` keyword is an `ArgumentError`)

Returns `(pattern, blobs, config)`.

# Example
```julia
# Generate 800x600 pattern: discs from 12 px across to 20% of the short side
pattern, blobs, config = blob_pattern(800, 600; blob_sizes=(12px, 0.2short_side), cutoff=2.0, seed=0x1234)
```
"""
function blob_pattern(width::Union{Integer, Quantity}, height::Union{Integer, Quantity}; kwargs...)
    _reject_pixel_type_keyword(:blob_pattern, "blob_pattern(pixel_type, Size2(width = width, height = height); kwargs...)", kwargs)
    pattern_resolution = Size2(width = width, height = height)

    return blob_pattern(pattern_resolution; kwargs...)
end

"""
    blob_pattern(pattern_resolution::Size2; kwargs...)

Generate a blob pattern with specified resolution.

# Arguments
- `pattern_resolution`: Pattern size as `Size2` with pixel or logical units

# Keyword Arguments
- `blob_sizes`: disc diameter range (see the `pixel_type` form)
- Other arguments same as `blob_pattern(pixel_type, pattern_resolution; kwargs...)`,
  except the pixel type: it is positional, `blob_pattern(T, pattern_resolution)`
  (a `pixel_type` keyword is an `ArgumentError`; the default is `DEFAULT_PIXEL_TYPE`)

Returns `(pattern, blobs, config)`.

# Example
```julia
using VisualGeometryToolkitCore: Size2, px
pattern, blobs, config = blob_pattern(Size2(width=800px, height=600px); blob_sizes=(12px, 120px), cutoff=2.0)
```
"""
function blob_pattern(pattern_resolution::Size2{<:Union{Integer, Quantity}}; kwargs...)
    _reject_pixel_type_keyword(:blob_pattern, "blob_pattern(pixel_type, pattern_resolution; kwargs...)", kwargs)
    kw = _resolve_blob_sizes(
        _pattern_to_px, :blob_pattern, (; kwargs...), _pattern_px(pattern_resolution); default_min_sigma = MIN_SIGMA_PX * px
    )
    pattern_kwargs, dropped = filter_kwargs(PatternConfig; kw...)
    config = PatternConfig(pattern_resolution = pattern_resolution; pattern_kwargs...)

    return blob_pattern(config; dropped...)..., config
end
