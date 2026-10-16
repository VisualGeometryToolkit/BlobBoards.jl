# =============================================================================
# Board outputs — what a board is written AS: Pdf(), Tiff(density), Png(density)
# =============================================================================
#
# A board is laid out in physical units and has no pixel density; a density is
# a property of one kind of output, a RASTER, and of nothing else. So the
# output is a value that carries exactly the options its format has, and the
# writer (BlobBoards.Authoring) dispatches on it:
#
#   Pdf(; shading_function, icc_profile)   the print master, PDF/X-4 — no density
#   Tiff(density; icc_profile)             the sheet converted at `density`
#   Png(density)                           the same, for digital use — no profile
#
# A combination a format would ignore cannot be written down: a PDF has no
# density, a PNG no ICC profile, a raster no shading function. (Until 2026-10
# these were five keywords — `format`, `pdf_standard`, `icc_profile`,
# `shading_function`, `print_density` — checked against each other at run time,
# and `print_density` was also a positional argument of `blob_board`, where it
# meant three things: a raster's density, the gallery image's, and the printer
# a board was sized for. The last is `suggest_blob_sizes(density, extent)` now.)

"""
    BoardOutput

What a board is written as: [`Pdf`](@ref), [`Tiff`](@ref) or [`Png`](@ref).
`blob_board(…; dir, output)` and `render_board(bundle; dir, output)` dispatch
on it.
"""
abstract type BoardOutput end

# The colour functions a PDF's blob shadings may carry, in the order an error lists them.
const SHADING_FUNCTIONS = (:calculator, :sampled)

function _check_shading_function(name)
    name in SHADING_FUNCTIONS && return name
    name === :raster && throw(
        ArgumentError(
            "shading_function = :raster was removed: a board's blobs are always vector, and its raster " *
                "outputs (Tiff, Png) are conversions of the vector PDF"
        )
    )
    name in (:vector, :vector_sampled) && throw(
        ArgumentError(
            "shading_function = $(repr(name)) is now $(repr(_LEGACY_SHADING_FUNCTIONS[String(name)])) " *
                "(the blobs are vector either way)"
        )
    )
    throw(ArgumentError("shading_function must be one of $(SHADING_FUNCTIONS), got $(repr(name))"))
end

# The profile as an output stores it — `nothing` for the built-in
# linear-reflectance profile (`:linear`), else the absolute path of an existing
# grey ICC profile — checked when the output is made, not when it is written.
_icc_path(::Nothing) = nothing
_icc_path(icc::Symbol) = icc === :linear ? nothing : _bad_icc(icc)
_icc_path(icc::AbstractString) = _is_grey_icc(icc) ? abspath(String(icc)) : _bad_icc(icc)
# An ICC profile's header: the signature "acsp" at byte 36, the data colour
# space at byte 16 ("GRAY": the writer declares the profile one-component).
function _is_grey_icc(path::AbstractString)
    isfile(path) && filesize(path) >= 128 || return false
    header = open(io -> read(io, 40), path)
    return header[37:40] == codeunits("acsp") && header[17:20] == codeunits("GRAY")
end
_icc_path(icc) = _bad_icc(icc)
_bad_icc(icc) = throw(ArgumentError("icc_profile must be :linear or the path of a grey ICC profile, got $(repr(icc))"))

# A raster's density: pixels per inch of the image, a whole number (the page is
# converted on Poppler's pixel grid, which takes one).
const _RasterDensity = typeof(1.0dpi)
function _output_density(density)
    density isa Quantity && dimension(density) == dimension(dpi) ||
        throw(ArgumentError("a raster output takes its density, e.g. Tiff(600dpi); got $(repr(density))"))
    # (the unit library stores a density in base units, px/m, so a dpi value
    # can read back a hair off: a whole number is one within rounding of it)
    d = ustrip(dpi, density)
    r = round(d)
    d > 0 && isapprox(d, r; rtol = 1.0e-12) ||
        throw(ArgumentError("a raster output needs a whole-number density in dpi, got $(_short_number(d)) dpi"))
    return r * dpi
end

"""
    Pdf(; shading_function = :calculator, icc_profile = :linear, bleed = 0mm)

The print master: a PDF/X-4 file (also PDF/A-2b) that BlobBoards writes itself
— vector frame, text in one embedded font, and every blob a radial shading the
printer evaluates at its own resolution. It has NO density: the same file
prints at any.

- `shading_function`: the shading's colour function — `:calculator` (the exact
  profile, as a PostScript calculator function) or `:sampled` (a 1024-entry
  table, for a RIP that mishandles calculator functions).
- `icc_profile`: the grey profile of the OutputIntent, declaring how the values
  print — `:linear` (linear reflectance, the default) or the path of a
  single-channel ICC file.
- `bleed`: the trim allowance a print shop asks for, a length ≥ 0 on every
  side. The canvas (`"A4"`, `1000mm × 1000mm`) is the FINISHED sheet and the
  PDF's TrimBox; the page (MediaBox, = BleedBox) is the canvas plus `bleed` on
  each side, a white strip the shop cuts off. `margin` is measured from the
  trim. A rendering option, like `id_label`: the board, its uid and its QR are
  the same with or without it. Rounded to the board's layout grid.
"""
struct Pdf <: BoardOutput
    shading_function::Symbol
    icc_profile::Union{Nothing, String}      # `nothing`: the built-in linear profile
    bleed::typeof(1.0mm)
    Pdf(; shading_function::Symbol = :calculator, icc_profile = :linear, bleed = 0mm) =
        new(_check_shading_function(shading_function), _icc_path(icc_profile), _check_bleed(bleed))
end

function _check_bleed(bleed)
    bleed isa Quantity && dimension(bleed) == dimension(mm) && ustrip(mm, bleed) >= 0 ||
        throw(ArgumentError("bleed must be a length ≥ 0 (e.g. Pdf(bleed = 2mm)), got $(repr(bleed))"))
    return Float64(ustrip(mm, bleed)) * mm
end

"""
    Tiff(density; icc_profile = :linear)

The sheet as an 8-bit grey TIFF. It is the print master converted at `density`
(pixels per inch, a whole number; the page rounded up to whole pixels),
LZW-compressed, the density in its resolution tags and `icc_profile` embedded.
It is a print output: the loader never reads it (a board kept as a TIFF is
rendered from its JSON for detection, like one kept as JSON alone).
The board's smallest blob must be at least 2 pixels
per σ at `density`.
"""
struct Tiff <: BoardOutput
    density::_RasterDensity
    icc_profile::Union{Nothing, String}
    Tiff(density; icc_profile = :linear) = new(_output_density(density), _icc_path(icc_profile))
end

"""
    Png(density)

The sheet as a single-channel, linear-gamma PNG for digital use: the print
master converted at `density` (pixels per inch, a whole number). No ICC
profile. The board's smallest blob must be at least 2 pixels per σ at `density`.
"""
struct Png <: BoardOutput
    density::_RasterDensity
    Png(density) = new(_output_density(density))
end

# The file extension, which is also the sidecar's recorded `format`.
_format(::Pdf) = "pdf"
_format(::Tiff) = "tiff"
_format(::Png) = "png"

# What each output records of itself; `nothing` where a format has no such thing.
# The ICC profile, as the sidecar records it: "linear" for the built-in one;
# a custom profile's bytes, embedded (base64, with its name and sha256), so a
# rebuild anywhere reproduces the sheet without the original file (until
# 2026-10-14 its absolute path was recorded, which a rebuild on another machine
# could not find).
_recorded_icc(o::Union{Pdf, Tiff}) = _icc_record(o.icc_profile)
_icc_record(::Nothing) = "linear"
function _icc_record(path::String)
    bytes = read(path)
    return (; name = basename(path), sha256 = bytes2hex(sha256(bytes)), data = base64encode(bytes))
end
_recorded_icc(::Png) = nothing
_recorded_shading(o::Pdf) = String(o.shading_function)
_recorded_shading(::BoardOutput) = nothing             # a raster has none (it is the default PDF, converted)
# The density an output is rasterized at; a PDF has none.
_density(::Pdf) = nothing
_density(o::Union{Tiff, Png}) = o.density
_recorded_density(o::BoardOutput) = (d = _density(o); d === nothing ? nothing : (value = ustrip(dpi, d), unit = "dpi"))

# The JSON `render` block — how the file was made, so a reconstruction
# reproduces the sheet: one set of keys whatever the format, and whether the id
# label was actually drawn. The inverse of `_recorded_output ∘ _render_options`.
_render_record(o::BoardOutput, id_label) = (;
    format = _format(o), icc_profile = _recorded_icc(o), id_label,
    shading_function = _recorded_shading(o), density = _recorded_density(o), bleed = _recorded_bleed(o),
)
# A PDF's bleed, in mm; a raster has none (it is the default PDF, converted).
_recorded_bleed(o::Pdf) = (value = ustrip(mm, o.bleed), unit = "mm")
_recorded_bleed(::BoardOutput) = nothing

# The output a render record describes (`opts`: `_render_options`; the Python
# bridge hands its flat keywords through the same reading). A raster without a
# density cannot be made; the caller names one.
function _recorded_output(opts)
    fmt = lowercase(opts.format)
    icc = something(opts.icc_profile, :linear)
    fmt == "pdf" && return Pdf(; opts.shading_function, icc_profile = icc, bleed = something(get(opts, :bleed, nothing), 0mm))
    fmt in ("tiff", "tif", "png") || throw(ArgumentError("render.format must be \"pdf\", \"tiff\" or \"png\", got $(repr(opts.format))"))
    opts.density === nothing && throw(
        ArgumentError("a $(fmt) is a raster and needs its density, and none is given or recorded: name one (Png(300dpi); from Python, density = 300)")
    )
    return fmt == "png" ? Png(opts.density) : Tiff(opts.density; icc_profile = icc)
end
