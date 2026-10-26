# IO subsystem: file naming, saving figures/images, JSON sidecars (consolidated)
using Poppler_jll: pdftocairo

# NOTE: `load_camera` (the ad-hoc "camera_matrix"/"distortion_coefficients"
# JSON reader) is GONE (2026-08-18). Cameras are stored in VisualGeometryToolkitIO's
# canonical `vgio/cameras` schema (v3: a store of named `BoundedCamera`s; the
# retired v1/v2 files were migrated once, 2026-10). Read them with
#
#     import VisualGeometryToolkitIO as VgtkIO
#     cam = VgtkIO.load(CameraModel, path)                 # single-camera store
#     cam = VgtkIO.load(CameraModel, path; name = "…")     # multi-camera store
#
# A second reader here would be a second authority over the same bytes;
# VgtkIO owns serialization.

"""
    make_filebase(bundle::AbstractBundle; config_hash = <the bundle's identity>) -> String

Base filename (without extension) computed from the bundle type and configuration hash.
Returns "blob_board_<hash>" for `BoardBundle` and "blob_pattern_<hash>" for `PatternBundle`,
where <hash> is the first 8 characters of the hex-encoded SHA-256 hash of the complete configuration,
formatted as XXX_XXXXX (3 chars underscore 5 chars) for readability.

For BoardBundle: the board's identity (`hash_config`: on a point-grid board,
over its schema, configuration and blobs). For PatternBundle: the SHA-256 of
its PatternConfig. A bundle loaded from disk
must be given its stored `config_hash` (`hashes.config`), its identity.
"""
function make_filebase(
        # `config_hash`: the bundle's identity. A FRESH bundle mints it from its
        # config; a bundle loaded from disk must be given its STORED
        # `hashes.config` — the recomputed hash of a v1 (or migrated v0) board
        # is not its identity.
        bundle::Union{BoardBundle, PatternBundle};
        config_hash::AbstractString = _bundle_identity(bundle)
    )
    full_hash = config_hash
    uid = full_hash[1:3] * "_" * full_hash[4:8]
    return (bundle isa BoardBundle ? "blob_board_" : "blob_pattern_") * uid
end


# Bundle JSON schema. The pre-v1 ("v0") reader — per-scalar {value, unit}
# blobs, `α`/Julia-type-name spellings, schema-less files — was REMOVED in
# 0.19 after every known corpus was migrated in place (281 bundles,
# 2026-08-20, identity-preserving). A stray v0 file migrates with the
# permanently registered BlobBoards@0.18 (`migrate_bundles!`); this version
# refuses it with that instruction rather than guessing.
#
# v2 (0.36): `pattern_config.max_scale` (px, the top rung of the scale ladder)
# replaces v1's `max_diameter_fraction`. Both versions are READ — a v1 config's
# fraction converts to the rung it always produced, so v1 boards re-place
# bit-identically — and v2 is what is written. Identity is the stored
# `hashes.config`, never a recompute (already true for every migrated v0 board).
#
# board v3 (0.37): the ruler is its ticks — `ruler_spec.width` is gone and the
# tick widths are FULL lengths across the frame band. A v1/v2 board's ruler is
# migrated in memory on read (drop `width`, double the half-extent tick widths;
# `JSON.lift(::Type{<:RulerSpec}, _)`), drawing the same ruler; files
# are never rewritten by loading, and the stored hash stays the identity.
# Patterns are unaffected and stay at v2.
#
# board v4 (2026-10): physical boards. A board is laid out on a fixed
# 1/7200-inch grid (`PointGrid`), independent of the print density; its pattern
# config and blobs are stored in points ("pt"), blob centres on 0.01 pt grid
# points — the numbers the print master holds. `print_density` left the config
# (and so the identity: one layout, one uid, whatever it is printed at) for the
# `render` block. A v3 board reads as `PixelGrid(density)` and keeps its
# geometry, schema and identity; a re-save writes it as v3.
const BOARD_SCHEMA = "blobboards/board/v4"
const PATTERN_SCHEMA = "blobboards/pattern/v2"
const BOARD_SCHEMA_V3 = "blobboards/board/v3"   # a pixel-grid board's, written as it was
const READABLE_SCHEMAS = (
    BOARD_SCHEMA, PATTERN_SCHEMA,
    BOARD_SCHEMA_V3, "blobboards/board/v2", "blobboards/board/v1", "blobboards/pattern/v1",
)

_bundle_schema(b::BoardBundle) = _board_schema(b.preamble.board_config.layout_grid)
_board_schema(::PointGrid) = BOARD_SCHEMA
_board_schema(::PixelGrid) = BOARD_SCHEMA_V3
_bundle_schema(::PatternBundle) = PATTERN_SCHEMA
# A bundle's identity, minted from what it holds: a board's from its own blobs
# (no placement), a pattern's from its configuration.
_bundle_identity(b::BoardBundle) = _board_identity(b.preamble, b.blobs)
_bundle_identity(b::PatternBundle) = hash_config(b.pattern_config)

# `render`: how the board's file was rendered (`_render_record`: its format and
# that format's options, whether the id label was drawn) — recorded beside `hashes` and NEVER part of
# `hashes.config`: the same board rendered with or without its id label keeps
# one identity, but a reconstruction (`rebuild_board`, a regenerated gallery
# render) reproduces the sheet as it was printed. Absent in every file written
# before 0.36.2; readers then fall back to the as-printed defaults
# (`_render_options`).
function save_json(
        json_path, bundle::AbstractBundle;
        overwrite::Bool = false, pretty_json::Bool = true, render = nothing,
        config_hash::AbstractString = _bundle_identity(bundle)   # the STORED one for a loaded bundle
    )
    if !overwrite && ispath(json_path)
        throw(ArgumentError("JSON already exist: $json_path"))
    end

    # Single serialization path: lower() is the source of truth. Fresh saves
    # mint identity canonically, so hashes.config IS the canonical hash; a
    # re-save of a loaded bundle gives its stored `config_hash`, which stays the
    # identity across format migrations (a migrated v1 board re-saved as the
    # current schema keeps its uid).
    lowered = _lower(bundle)
    output = (
        schema = _bundle_schema(bundle), lowered...,
        hashes = (config = String(config_hash),),
    )
    render === nothing || (output = merge(output, (; render)))

    open(json_path, "w") do io
        _write_json(io, output; pretty = pretty_json)
    end
    return @info "Saved metadata" metadata = json_path
end


"""
    save(pattern::AbstractMatrix, bundle::PatternBundle; dir, overwrite=false, pretty_json=true)

Save a grayscale pattern image directly and its JSON sidecar.
Saves the colorant array directly as a grayscale PNG (single channel, no alpha).
Returns `(img_path, json_path)`.
"""
function save(pattern::AbstractMatrix, bundle::PatternBundle; dir, overwrite = false, pretty_json = true)
    valid_dir = validate_dir(dir; create = true)
    base = make_filebase(bundle)
    fig_path = joinpath(valid_dir, base * ".png")

    if !overwrite && ispath(fig_path)
        throw(ArgumentError("Pattern already exists: $fig_path."))
    end

    # Save colorant array directly to preserve grayscale format
    # Use file_gamma=1.0 for linear (non-gamma-corrected) output for measurement
    PNGFiles.save(fig_path, pattern; file_gamma = 1.0)
    @info "Saved pattern" image = fig_path

    json_path = joinpath(valid_dir, base * ".json")
    save_json(json_path, bundle; overwrite = overwrite, pretty_json = pretty_json)
    return fig_path, json_path
end

# =============================================================================
# PDF rasterization — a PDF page as pixels
# =============================================================================

# The rasterizer's name in a cache key (a raster of the same PDF by another
# renderer differs: `pdftoppm`, used before 2026-10, altered 36 % of a 300 dpi
# board's pattern pixels).
const _RASTERIZER = "pdftocairo"

"""
    rasterize_pdf(pdf_path, render_dpi::Integer) -> Matrix{Gray{N0f8}}

Render a PDF's first page to an 8-bit grayscale image at `render_dpi` dots per inch, with Poppler's
`pdftocairo` (Poppler_jll). Cairo copies an embedded image 1:1 and evaluates a
smooth shading at every pixel, so a board's pattern comes back as written: a
raster pattern pixel for pixel, a vector one within one grey level at every σ.
The pixel grid is Poppler's: the page rounded UP to whole pixels and anchored
at its top-left corner. It goes through an uncompressed TIFF (`pdftocairo`
cannot write one to stdout): ≈1 s for a 12000² board, 32 s as PNG.
"""
function rasterize_pdf(pdf_path::AbstractString, render_dpi::Integer)
    return mktempdir() do dir
        stem = joinpath(dir, "page")
        run(`$(pdftocairo()) -tiff -tiffcompression none -gray -r $render_dpi -singlefile $pdf_path $stem`)
        _read_grey_tiff(stem * ".tif")
    end
end

# An 8-bit grey TIFF, as `pdftocairo -gray` writes it, read with libtiff (the
# reference reader; a FileIO backend need not be installed) row by row.
function _read_grey_tiff(path::AbstractString)
    tif = @ccall libtiff.TIFFOpen(path::Cstring, "r"::Cstring)::Ptr{Cvoid}
    tif == C_NULL && error("rasterize_pdf: libtiff could not open $path")
    return try
        field(tag, ::Type{T}) where {T} = (v = Ref{T}(0); @ccall(libtiff.TIFFGetField(tif::Ptr{Cvoid}, tag::UInt32; v::Ptr{T})::Cint) == 1 || error("rasterize_pdf: $path lacks TIFF tag $tag"); v[])
        width, height = Int(field(UInt32(256), UInt32)), Int(field(UInt32(257), UInt32))
        bits, samples, photometric = field(UInt32(258), UInt16), field(UInt32(277), UInt16), field(UInt32(262), UInt16)
        (bits == 8 && samples == 1 && photometric in (0, 1)) ||
            error("rasterize_pdf: expected an 8-bit grey page, got $bits bits × $samples samples (photometric $photometric)")
        row = Vector{UInt8}(undef, @ccall libtiff.TIFFScanlineSize(tif::Ptr{Cvoid})::Int)
        img = Matrix{Gray{N0f8}}(undef, height, width)
        for r in 1:height
            @ccall(libtiff.TIFFReadScanline(tif::Ptr{Cvoid}, row::Ptr{UInt8}, (r - 1)::UInt32, 0::UInt16)::Cint) == 1 ||
                error("rasterize_pdf: libtiff could not read row $r of $path")
            for c in 1:width
                v = reinterpret(N0f8, row[c])
                img[r, c] = Gray(photometric == 0 ? one(N0f8) - v : v)       # 0: min-is-white
            end
        end
        img
    finally
        @ccall libtiff.TIFFClose(tif::Ptr{Cvoid})::Cvoid
    end
end

"""
    _resolve_paths(path) -> (json_path, image_path)

Resolve JSON and image file paths from any supported input path.
"""
function _resolve_paths(path::AbstractString)
    isfile(path) || error("File not found: $path")
    base_name, ext = splitext(path)

    if lowercase(ext) == ".json"
        json_path = path
        pdf_path = base_name * ".pdf"
        png_path = base_name * ".png"
        # `nothing` = no sibling render; load_bundle regenerates from the
        # PARSED bundle (stored blobs + stored identity), which the raw path
        # cannot do — a pre-v1 JSON re-fed to the generator re-RUNS placement
        # under today's algorithm and produces a different board (957 vs 1326
        # blobs on the 2025-era gallery, 2026-08-20 run).
        # (a TIFF is a print output and is never read: a board kept as one is
        # rendered from its JSON, like a board kept as JSON alone)
        image_path = isfile(pdf_path) ? pdf_path :
            isfile(png_path) ? png_path : nothing
    elseif lowercase(ext) in [".png", ".pdf"]
        image_path = path
        json_path = base_name * ".json"
        isfile(json_path) || error("Could not find JSON sidecar for $path (expected $json_path)")
    else
        error("Unsupported file extension: $ext (expected .json, .png, or .pdf)")
    end
    return json_path, image_path
end

# Module-owned so the scratch key is ALWAYS BlobBoards' UUID: @get_scratch!
# expands with the CALLING module, so a qualified BlobBoards.@get_scratch! in
# a test or script silently keys a different directory (bit us 2026-08-20 —
# a "purged" cache kept serving a stale render).
const _RENDER_LOCK = ReentrantLock()

# Keyed by the board, its id label, the writer's render record AND the bundle
# schema: a render made otherwise (or by an older writer — a QR carrying an
# older schema's preamble) is a different file, never served for this JSON.
_render_cache_dir(config_hash::AbstractString, id_label::Bool) = joinpath(
    @get_scratch!("board_renders"),
    String(config_hash)[1:16] * "_" * bytes2hex(sha256(codeunits(string(_render_record(Pdf(), id_label), BOARD_SCHEMA))))[1:8]
)

"""
    _regenerate_render(json_path, bundle::BoardBundle, config_hash; id_label) -> String

Fallback for a board definition with no sibling render: render the STORED
bundle (its recorded blobs, frame, QR and uid — `render_board`, never a
placement re-run) as the PDF print master into a scratch cache keyed by the
stored config hash, and return its path. `id_label`: as the sheet was printed
(the JSON's render record). The source directory is NEVER written to — board
definitions can live in a content-addressed artifact tree, which must stay
immutable.

A PDF, not a raster: it has no density, so one cached file serves every
density a caller rasterizes at (`_board_raster`'s own content-addressed cache,
which runs in parallel), and writing it is milliseconds.

The cache is content-stable:
the render is a pure function of the stored bundle, and the key is its
authoritative identity, so a cached render never goes stale and is shared by
every consumer of the same board.
"""
function _regenerate_render(json_path::AbstractString, bundle::BoardBundle, config_hash::AbstractString; id_label::Bool)
    # One render at a time: callers load boards on several threads (the
    # gallery's batch `insert!`, `load_board_gallery`). The check repeats under
    # the lock, so a board two threads ask for is rendered once.
    return lock(_RENDER_LOCK) do
        dir = _render_cache_dir(config_hash, id_label)
        pdfs = isdir(dir) ? filter(endswith(".pdf"), readdir(dir; join = true)) : String[]
        isempty(pdfs) || return only(pdfs)
        mkpath(dir)
        render_board(bundle; dir, output = Pdf(), id_label, config_hash)
        only(filter(endswith(".pdf"), readdir(dir; join = true)))
    end
end

# A render record's `blob_encoding` (written before 2026-10-13) → the
# `shading_function` it names; any other value goes on as it is, and is refused
# by the writer it reaches.
const _LEGACY_SHADING_FUNCTIONS = Dict(
    "vector" => :calculator, "vector_sampled" => :sampled,
    # a TIFF/PNG written 2026-10-10/11 records its sheet's raster pattern engine,
    # removed since; its blobs re-render as the calculator function, as a record
    # without one does
    "raster" => :calculator,
)

"""
    _render_options(json_obj) -> (; format, icc_profile, id_label, shading_function, density, bleed)

The render options a bundle JSON records in its `render` block — how its print
file was made (`_recorded_output` turns them into the output). A file without
one (every board written before 0.36.2) gets format "pdf", the linear profile
and NO id label. That is how every board before 0.36.1 was printed; boards
written by 0.36.1 itself carried a label by default but recorded nothing, and
rebuild without it (set `id_label = true` on `rebuild_board` to reproduce one).

`shading_function` missing means `:calculator`: a board's blobs are always
vector since 2026-10 (the raster pattern engine was removed after an A/B showed
its sheets and the vector ones agree to an 8-bit level), so a board printed
before then rebuilds with vector blobs. A record written before 2026-10-13
names it `blob_encoding`, "vector" or "vector_sampled" (`_LEGACY_SHADING_FUNCTIONS`).

`density` is a RASTER's (a TIFF's or PNG's pixels per inch; `nothing` for a
PDF, which has none). A record written before 2026-10-14 names it
`print_density` and carries one whatever the format; it is read for a raster
and means nothing for a PDF. A recorded `pdf_standard` (written until
2026-10-14) is not read: the print master is always PDF/X-4. A custom ICC
profile is embedded in the record (`_icc_record`) and written back to a
content-addressed scratch file for the rebuild.

`bleed` is a PDF's trim allowance (`Pdf(; bleed)`), a length; `nothing` when
none is recorded (every board written before it existed: no bleed).
"""
function _render_options(json_obj::_ParsedObject)
    # isa-narrowed field by field: dynamic JSON access is a union JET cannot
    # narrow (the core ratchet). A wrong-TYPED entry falls back to the default; a
    # well-typed but unknown value (`"shading_function": "x"`) is refused, loudly,
    # by the output it reaches.
    format = "pdf"
    icc_profile = nothing                   # the linear-reflectance default
    id_label = false
    shading_function = :calculator          # the calculator function: the exact profile
    density = nothing
    bleed = nothing                         # a PDF's trim allowance; none recorded: no bleed
    r = get(json_obj, :render, nothing)
    if r isa _ParsedObject
        f = get(r, :format, nothing)
        f isa AbstractString && (format = String(f))
        c = get(r, :icc_profile, nothing)
        c isa AbstractString && c != "linear" && (icc_profile = String(c))     # (a path: recorded before 2026-10-14)
        c isa _ParsedObject && (icc_profile = _icc_from_record(c))
        l = get(r, :id_label, nothing)
        l isa Bool && (id_label = l)
        e = get(r, :shading_function, nothing)
        e isa AbstractString && (shading_function = Symbol(e))
        legacy = get(r, :blob_encoding, nothing)
        e === nothing && legacy isa AbstractString &&
            (shading_function = get(_LEGACY_SHADING_FUNCTIONS, legacy, Symbol(legacy)))
        d = get(r, :density, nothing)
        d isa _ParsedObject || (d = get(r, :print_density, nothing))
        d isa _ParsedObject && lowercase(format) != "pdf" && (density = _density_record(d))
        b = get(r, :bleed, nothing)
        b isa _ParsedObject && lowercase(format) == "pdf" && (bleed = _bleed_record(b))
    end
    return (; format, icc_profile, id_label, shading_function, density, bleed)
end

# An embedded ICC profile (`_icc_record`) as a file the output can name: its
# bytes, checked against the recorded sha256, under a content-addressed scratch
# directory and the profile's own file name.
function _icc_from_record(c::_ParsedObject)
    data, recorded, name = get(c, :data, nothing), get(c, :sha256, nothing), get(c, :name, nothing)
    (data isa AbstractString && recorded isa AbstractString && name isa AbstractString) ||
        throw(ArgumentError("render.icc_profile: an embedded profile records its name, sha256 and data (base64), got $(c)"))
    bytes = base64decode(data)
    h = bytes2hex(sha256(bytes))
    h == recorded || throw(ArgumentError("render.icc_profile: the embedded profile does not match its recorded sha256"))
    path = joinpath(@get_scratch!("icc_profiles"), h, basename(name))
    if !isfile(path)
        mkpath(dirname(path))
        tmp = string(path, ".", getpid(), ".tmp")
        write(tmp, bytes)
        mv(tmp, path; force = true)
    end
    return path
end

# A recorded bleed, `{value, unit}` a length; anything else is refused.
function _bleed_record(b)
    q = _lift_scalar(b)
    (q isa Quantity && dimension(q) == dimension(mm)) || throw(ArgumentError("render.bleed must be a length, got $(b)"))
    return q
end

# A recorded density, `{value, unit}` in dpi; anything else is refused.
function _density_record(d)
    q = _lift_scalar(d)
    (q isa Quantity && dimension(q) == dimension(dpi)) || throw(ArgumentError("render.density must be a density, got $(d)"))
    return q
end

_regenerate_render(json_path::AbstractString, bundle, config_hash; id_label = false) = error(
    "no render beside $json_path and regeneration is only supported for " *
        "BoardBundle (got $(typeof(bundle))) — place the render next to the JSON"
)

"""
    _render_dpi(density) -> Int

The rasterization DPI for a density (a raster output's, a board image's). MUST use the
target-unit `ustrip(dpi, …)` form: a bare `ustrip` of a density yields
its base-unit magnitude (px/m; px/mm under the old (mm, px) basis), which
under the old basis silently
rendered boards at ~47 dpi instead of 1200 — every blob a ~2 px smudge
and canvas-frame coordinates pointing off the image (2026-07-29 gallery
regression, masked for months by the gallery feature cache).
"""
_render_dpi(density) = round(Int, ustrip(dpi, density))

# The whole-number density at which a board's smallest σ is at least `σ_px`
# pixels: the one conversion behind the raster floor and the board's own
# raster density.
function _density_for_sigma(preamble::BoardPreamble, σ_px::Real)
    σ_inch = ustrip(px, preamble.pattern_config.min_scale) / ustrip(dpi, stored_density(preamble.board_config))
    return ceil(σ_px / σ_inch - 1.0e-9) * dpi
end

# A v3 board is laid out on its printer's pixels: its rasters exist at that
# density and no other. `what` names the caller's request in the message.
function _check_v3_density(g::PixelGrid, density, what::AbstractString)
    isapprox(ustrip(dpi, density), ustrip(dpi, g.density)) || throw(
        ArgumentError("a v3 board is laid out at the density it was printed at, $(g.density); got $what")
    )
    return nothing
end
_check_v3_density(::PointGrid, density, what) = nothing

"""
    _default_density(bundle::BoardBundle) -> density

The density a board is converted to pixels at when nobody names one — the
detector's gallery image. A v3 board is laid out on its printer's pixels: its
grid's density. A v4 board has none, and its image no halftone, so it is
rendered just fine enough to sample its smallest blob well: the whole-number
dpi at which the smallest disc is at least `min_blob_px` pixels across
(`configs/authoring.toml`, `[board_raster]`) — and never under the floor every
raster of a board must meet (`MIN_SIGMA_PX` per σ, which a cutoff above 4
would otherwise undercut).
"""
_default_density(bundle::BoardBundle) = _default_density(bundle.preamble.board_config.layout_grid, bundle)
_default_density(g::PixelGrid, _) = g.density
function _default_density(::PointGrid, bundle::BoardBundle)
    across = _raster_dflt("min_blob_px") / _diameter_factor(bundle.preamble.pattern_config.cutoff)   # as σ, in pixels
    return _density_for_sigma(bundle.preamble, max(across, MIN_SIGMA_PX))
end

"""
    _image_density(image_path, bundle::BoardBundle; density = nothing) -> density

The density of the pixels a board's image gives. A raster beside the JSON IS
at a density — a v3 board's grid's, else the one its `render` block records.
A PDF has none: it is converted at `density` when the caller names one,
else at the board's own (`_default_density`). A v3 board's layout is its
printer's pixels, so another density is refused.
"""
function _image_density(image_path::AbstractString, bundle::BoardBundle; density = nothing)
    g = bundle.preamble.board_config.layout_grid
    density === nothing || _check_v3_density(g, density, "density = $(density)")
    g isa PixelGrid && return g.density
    # (a PDF is converted on Poppler's pixel grid, at a whole number of dpi)
    lowercase(splitext(image_path)[2]) == ".pdf" && return density === nothing ? _default_density(bundle) : _output_density(density)
    json_path = splitext(image_path)[1] * ".json"
    rec = isfile(json_path) ? _render_options(JSON.parse(read(json_path, String))::_ParsedObject) : nothing
    # the record describes the board's LAST output: its density is this raster's
    # only if that output was this format (a PNG and a TIFF share one JSON)
    ext = lowercase(splitext(image_path)[2])
    fmt, rd = rec === nothing ? ("", nothing) : (lowercase(rec.format), rec.density)
    d = ext in (fmt == "png" ? (".png",) : fmt in ("tiff", "tif") ? (".tiff", ".tif") : ()) ? rd : nothing
    d === nothing && density !== nothing && return _output_density(density)     # the caller knows it
    d === nothing && error(
        "$(json_path) records no density (`render.density`) for the raster $(basename(image_path)) of this " *
            "board (schema v4) — none, or that of another output written after it — and its layout has none: " *
            "the density of its pixels is unknown (render it again, or load the board's PDF)"
    )
    density === nothing || isapprox(ustrip(dpi, density), ustrip(dpi, d)) || throw(
        ArgumentError("$(basename(image_path)) is a raster at $(d); got density = $(density)")
    )
    return d
end

# The DPI a bundle's PDF is rasterized at: the board's image density
# (`_image_density`); a pattern's PDF at 300.
_raster_dpi(image_path, bundle, density) =
    bundle isa BoardBundle ? _render_dpi(_image_density(image_path, bundle; density)) : 300

"""
    _load_image(image_path, bundle; density = nothing) -> image

Load a bundle's image; a PDF is rasterized at `density` when the caller
has it (`load_board(path).density`), else at the board's own
(`_image_density`).
"""
function _load_image(image_path::AbstractString, bundle; density = nothing)
    return if lowercase(splitext(image_path)[2]) == ".pdf"
        rasterize_pdf(image_path, _raster_dpi(image_path, bundle, density))
    else
        _load_raster(image_path, bundle)
    end
end

# A raster as its file holds it. A v4 board's PNG is written linear
# (`file_gamma = 1.0`: the grey values ARE the profile), and a PNG reader's
# default applies a display gamma to such a file — 127 comes back 186 — so it
# is read with none: the pixels it holds, exactly (measured 2026-10-14; the
# gallery's image of a board must not depend on which file sits beside its
# JSON). Every board's and pattern's PNG alike (v3 ones too, since 2026-10-14:
# `_GALLERY_VERSION` 6 — they were read display-gamma converted before).
_load_raster(image_path, _) = PNGFiles.load(String(image_path); gamma = 1.0)

"""
    _board_raster(image_path, bundle, config_hash; density = nothing) -> Matrix{Gray{N0f8}}

A board's raster for feature extraction, through a lossless scratch cache. A
PDF is rasterized once (`rasterize_pdf`, ~1.3 s for a 150 mm board at 1200
dpi) and the result kept as PNG (0.5 s to decode, 1/6 of a raw PGM's disk), so
a warm build never rasterizes. Any other image is decoded as before.

The key is the board's identity (`config_hash`), the render DPI, the rasterizer
(`_RASTERIZER`) and the
SHA-256 of the PDF's own bytes. The bytes make it content-addressed: a print
run's PDF and a regenerated render of the same board (which differ
radiometrically at the blob rims) never share an entry, so the cached raster
is always the one this PDF produces, whoever populated the cache first, and
an entry can never go stale. The source directory is never written to; the
PNG is written to a temporary name and renamed, so a concurrent build sees
either no entry or a whole one.
"""
function _board_raster(image_path::AbstractString, bundle, config_hash::AbstractString; density = nothing)
    lowercase(splitext(image_path)[2]) == ".pdf" || return _load_image(image_path, bundle; density)
    render_dpi = _raster_dpi(image_path, bundle, density)
    key = string(
        String(config_hash)[1:16], "_", render_dpi, "dpi_", _RASTERIZER, "_",
        bytes2hex(open(sha256, image_path))[1:16]
    )
    path = joinpath(@get_scratch!("board_rasters"), key * ".png")
    isfile(path) && return PNGFiles.load(String(path); gamma = 1.0)
    raster = rasterize_pdf(image_path, render_dpi)
    tmp = string(path, ".", getpid(), ".", objectid(raster), ".tmp.png")
    PNGFiles.save(tmp, raster; file_gamma = 1.0)
    mv(tmp, path; force = true)
    return raster
end

"""
    load_bundle(path) -> (bundle, image_path, config_hash)

Parse the JSON sidecar to get the typed bundle (BoardBundle or PatternBundle),
resolve the image path, and return the stored configuration hash. A board
with no render beside its JSON is rendered into a cache as its PDF print
master (`_regenerate_render`), which has no density.

The `config_hash` is the hex-encoded SHA-256 hash stored at write time in the
`hashes.config` field of the JSON.  This is the authoritative board identity —
use `uid_from_hash(config_hash)` to derive the 3-character UID rather than
recomputing from the deserialized config (which may differ after format migrations).

See also: [`load`](@ref) for loading bundle + image together.
"""
function load_bundle(path::AbstractString)
    if lowercase(splitext(path)[2]) in [".tracks", ".h5", ".hdf5"]
        error("load_bundle does not support track files; use load() instead")
    end

    json_path, image_path = _resolve_paths(path)
    bundle, config_hash = _parse_bundle(json_path)

    if image_path === nothing
        id_label = _render_options(JSON.parse(read(json_path, String))::_ParsedObject).id_label
        image_path = _regenerate_render(json_path, bundle, config_hash; id_label)
    end

    return bundle, image_path, config_hash
end

# A bundle sidecar parsed — the typed bundle and its stored configuration hash —
# without resolving or regenerating its render (the design resolver reads
# geometry only).
function _parse_bundle(json_path::AbstractString)
    json_str = read(json_path, String)
    # Top-level typeassert pins the JSON schema — JSON.parse infers as a
    # wide union otherwise and every haskey/getindex below union-splits
    # (the 6 JET reports that broke the ratchet, 2026-08-06).
    json_obj = JSON.parse(json_str)::_ParsedObject

    schema = get(json_obj, :schema, nothing)
    schema in READABLE_SCHEMAS || error(
        schema === nothing ?
            "$json_path has no schema key — a pre-v1 bundle. The v0 reader " *
            "was removed in 0.19; migrate the corpus once with the " *
            "permanently registered BlobBoards@0.18 " *
            "(BlobBoards.migrate_bundles!(dir))." :
            "$json_path declares unknown bundle schema $(repr(String(schema))); " *
            "this BlobBoards reads $(join(READABLE_SCHEMAS, ", "))"
    )
    bundle = if haskey(json_obj, :preamble)
        JSON.lift(BoardBundle, json_obj)
    elseif haskey(json_obj, :pattern_config)
        JSON.lift(PatternBundle, json_obj)
    else
        error("Unknown bundle format in $json_path (missing both :preamble and :pattern_config)")
    end

    config_hash = _stored_config_hash(json_obj)
    if config_hash === nothing
        @warn "No stored config hash in $json_path, recomputing"
        config_hash = _bundle_identity(bundle)
    end
    return bundle, config_hash
end

# A bundle JSON's stored identity, `hashes.config` — authoritative across
# format versions (a v1 board's recomputed hash is not its identity); `nothing`
# when the file stores none. isa-narrowing instead of bare getindex: a
# malformed `hashes` value is "none stored", not a MethodError.
function _stored_config_hash(json_obj::_ParsedObject)
    hashes = get(json_obj, :hashes, nothing)
    h = hashes isa _ParsedObject ? get(hashes, :config, nothing) : nothing
    return h isa AbstractString ? String(h) : nothing
end

"""
    load(path::AbstractString) -> (bundle, image)

Load a blob board or pattern from disk, including both the JSON metadata and image.

Accepts any of the paired files (image or JSON) and automatically finds the sidecar:
- `.png` → finds `.json` sidecar
- `.pdf` → finds `.json` sidecar (a board's PDF is converted at the board's own raster density, a pattern's at 300 dpi)
- `.json` → finds image (`.png` or `.pdf`)

Returns a tuple `(bundle, image)` where:
- `bundle`: Either `BoardBundle` or `PatternBundle` (automatically detected from JSON)
- `image`: Loaded image array

See also: [`load_bundle`](@ref) for metadata-only loading (no rasterization).
"""
function load(path::AbstractString)
    # NOTE: a `.tracks`/`.h5`/`.hdf5` branch used to dispatch to
    # `_load_tracks`, which lived in src/training.jl. Commit 197beeb deleted
    # training.jl but left this call site, so that branch has thrown
    # UndefVarError ever since. Track I/O is no longer part of this package;
    # the branch is removed rather than resurrected (found by JET, 2026-07-26).
    bundle, image_path = load_bundle(path)
    image = _load_image(image_path, bundle)
    return bundle, image
end
