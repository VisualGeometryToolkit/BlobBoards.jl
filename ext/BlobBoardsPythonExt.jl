module BlobBoardsPythonExt

using BlobBoards
using BlobBoards: BoardBundle, PatternBundle
using PythonCall: PythonCall, Py, pybuiltins, pyconvert, pydict, pyhasattr,
    pyimport, pyisinstance, pylist
using JSON3: JSON3
import VisualGeometryIo
import VisualGeometryCore
using VisualGeometryCore: CameraModel   # dispatch token for VisualGeometryIo.load
using VisualGeometryCore: ustrip, m, mm, cm, μm
using VisualGeometryCore: px, dpi, pt, inch, Size2, parse_unit, strip_or_bare
using FlexUnits: Quantity, dimension
using VisualGeometryCore: EuclideanMap, PixelFrame  # pixel-origin frame maps (VGC 0.7)

# Origin shift over the blob list: the labeled frame map acts on feature
# frames directly (VisualGeometryCore 0.21.1 gave `EuclideanMap{2}` the
# primitive actions only `Translation` had, so the `.t` rewrap is gone).
_origin_shift(from::Symbol, to::Symbol) = EuclideanMap(PixelFrame{from}, PixelFrame{to})
using VisualGeometryFeatures: is_dark
using StaticArrays: SMatrix
using Colors: Gray
using FixedPointNumbers: N0f8

_filter_kwargs(; kwargs...) = Dict{Symbol, Any}(k => v for (k, v) in pairs(kwargs) if v !== nothing)

_py_none(x) = x === nothing || (x isa Py && PythonCall.pyis(x, pybuiltins.None))

# One `blob_sizes` entry from Python: None → nothing, `f * blobboards.short_side`
# → `f * short_side`, a bare number → `default_unit` (mm for boards, px for
# patterns; NOT rounded — 2.5 px is a valid σ), a Pint quantity → its own unit.
function _py_blob_size(x, default_unit)
    _py_none(x) && return nothing
    x isa Real && return Float64(x) * default_unit
    if x isa Py
        pyhasattr(x, "_blobboards_short_side") && return pyconvert(Float64, x.fraction) * BlobBoards.short_side
        (pyisinstance(x, pybuiltins.float) || pyisinstance(x, pybuiltins.int)) &&
            return pyconvert(Float64, x) * default_unit
    end
    return pint_to_julia_length(x; default_unit, round_px = false)
end

# Python `blob_sizes=(min, max)` → the Julia `(min, max)` tuple.
function _py_blob_sizes(sizes, default_unit)
    _py_none(sizes) && return nothing
    items = sizes isa Union{Tuple, AbstractVector} ? collect(sizes) :
        [sizes[i] for i in 0:(pyconvert(Int, pybuiltins.len(sizes)) - 1)]
    length(items) == 2 || error("blob_sizes must be a (min, max) pair, got $(length(items)) entries")
    return (_py_blob_size(items[1], default_unit), _py_blob_size(items[2], default_unit))
end

_py_symbol(x) = _py_none(x) ? nothing : Symbol(x isa Py ? pyconvert(String, x) : string(x))

# The config fields a Python caller reads back, as stored, in `unit`: "px" for a
# pattern (and a v3 board), "pt" for a v4 board (its stored unit).
# Pixel sizes are whole pixels (ints); a v4 board's are points (floats).
_py_size(x, unit) = unit == "px" ? Int(x) : Float64(x)
_py_pattern_config(pc; unit::String = "px") = (
    width = _py_size(ustrip(px, pc.pattern_resolution.width), unit),
    height = _py_size(ustrip(px, pc.pattern_resolution.height), unit),
    unit,
    min_scale = Float64(ustrip(px, pc.min_scale)),
    max_scale = Float64(ustrip(px, pc.max_scale)),
    seed = Int128(pc.seed),
    scales_per_octave = pc.scales_per_octave,
    max_trials = pc.max_trials,
    alpha = pc.α,
    cutoff = pc.cutoff,
    raster_aligned = pc.raster_aligned,
    polarity = string(pc.polarity),
)

# =============================================================================
# Pattern + Board generators
# =============================================================================


# The JSON entry points' unit names, read by VisualGeometryCore's `parse_unit`
# (the stack's one guarded `uparse`): one registered unit symbol of pixels or a
# length ("px", "pd", "mm", "inch", "in", …).
function _json_units(units::AbstractString)
    u = try
        parse_unit(units)
    catch e
        e isa ArgumentError || rethrow()
        nothing
    end
    u !== nothing && (dimension(u) == dimension(px) || dimension(u) == dimension(mm)) && return u
    return error("Unknown units: $units. Use 'px', 'mm', or 'inch'.")
end

function BlobBoards.blob_pattern_python(
        width, height;
        cutoff = nothing,
        dir = nothing,
        blob_sizes = nothing,
        seed = nothing,
        scales_per_octave = nothing,
        max_trials = nothing,
        α = nothing,
        raster_aligned = nothing,
        polarity = nothing
    )
    width_jl = pint_to_julia_length(width; default_unit = px)
    height_jl = pint_to_julia_length(height; default_unit = px)
    pattern_size = Size2(; width = width_jl, height = height_jl)

    julia_kwargs = _filter_kwargs(
        dir = dir === nothing ? nothing : string(dir),
        blob_sizes = _py_blob_sizes(blob_sizes, px),
        seed = seed === nothing ? nothing : UInt64(seed),
        scales_per_octave = scales_per_octave === nothing ? nothing : Int(scales_per_octave),
        max_trials = max_trials === nothing ? nothing : Int(max_trials),
        α = α === nothing ? nothing : Float64(α),
        cutoff = _py_none(cutoff) ? nothing : Float64(cutoff),
        raster_aligned = raster_aligned === nothing ? nothing : Bool(raster_aligned),
        polarity = _py_symbol(polarity),
    )

    pattern, blob_list, config = BlobBoards.blob_pattern(pattern_size; julia_kwargs...)

    np = pyimport("numpy")

    pattern_float = Float64.(pattern)
    pattern_np = np.array(pattern_float)

    blobs_array = zeros(Float64, length(blob_list), 3)
    for (i, blob) in enumerate(blob_list)
        blobs_array[i, 1] = ustrip(px, blob.center[1])
        blobs_array[i, 2] = ustrip(px, blob.center[2])
        blobs_array[i, 3] = ustrip(px, blob.σ)
    end
    blobs_np = np.array(blobs_array)

    config_nt = _py_pattern_config(config)

    return (pattern = pattern_np, blobs = blobs_np, config = config_nt, config_hash = BlobBoards.hash_config(config))
end

# A board's lengths from Python: a pint quantity, or a bare number in mm;
# `nothing` (not supplied) stays `nothing`.
_py_mm(x) = x === nothing ? nothing : pint_to_julia_length(x; default_unit = mm)
# A (width, height) pair — a Julia tuple, or a Python tuple or list — as a Size2
# of lengths (`board_size`, `margin`, a canvas); `nothing` stays `nothing`.
_py_pair(x) = x isa Tuple || (x isa Py && (pyisinstance(x, pybuiltins.tuple) || pyisinstance(x, pybuiltins.list)))
_py_size2(::Nothing) = nothing
function _py_size2(x)
    _py_pair(x) || throw(ArgumentError("expected a (width, height) pair of lengths, got $(typeof(x))"))
    w, h = x isa Tuple ? (x[1], x[2]) : (x[0], x[1])
    return Size2(; width = _py_mm(w), height = _py_mm(h))
end

function BlobBoards.blob_board_python(
        canvas_size;
        cutoff = nothing,
        dir = nothing,
        format = nothing,
        density = nothing,
        icc_profile = nothing,
        id_label = nothing,
        shading_function = nothing,
        bleed = nothing,
        board_size = nothing,
        margin = nothing,
        ruler_width = nothing,
        corner_size = nothing,
        frame_padding = nothing,
        major_tick_spacing = nothing,
        minor_tick_spacing = nothing,
        major_tick_width = nothing,
        minor_tick_width = nothing,
        blob_sizes = nothing,
        seed = nothing,
        scales_per_octave = nothing,
        max_trials = nothing,
        α = nothing,
        polarity = nothing,
        image_origin = "colmap"
    )
    # a paper name, a (width, height) pair, or one length (a square page)
    canvas_size_jl = if canvas_size isa AbstractString || (canvas_size isa Py && pyisinstance(canvas_size, pybuiltins.str))
        string(canvas_size)
    elseif _py_pair(canvas_size)
        _py_size2(canvas_size)
    else
        side = _py_mm(canvas_size)
        Size2(; width = side, height = side)
    end

    density_jl = _py_none(density) ? nothing : pint_to_julia_density(density)
    julia_kwargs = _filter_kwargs(
        dir = dir === nothing ? nothing : string(dir),
        output = _py_output(format, density_jl, icc_profile, shading_function, bleed),
        id_label = _py_none(id_label) ? nothing : (id_label isa Py ? pyconvert(Bool, id_label) : Bool(id_label)),
        board_size = _py_size2(board_size), margin = _py_size2(margin),
        ruler_width = _py_mm(ruler_width), corner_size = _py_mm(corner_size), frame_padding = _py_mm(frame_padding),
        major_tick_spacing = _py_mm(major_tick_spacing), minor_tick_spacing = _py_mm(minor_tick_spacing),
        major_tick_width = _py_mm(major_tick_width), minor_tick_width = _py_mm(minor_tick_width),
        blob_sizes = _py_blob_sizes(blob_sizes, mm),
        seed = seed === nothing ? nothing : UInt64(seed),
        scales_per_octave = scales_per_octave === nothing ? nothing : Int(scales_per_octave),
        max_trials = max_trials === nothing ? nothing : Int(max_trials),
        α = α === nothing ? nothing : Float64(α),
        cutoff = _py_none(cutoff) ? nothing : Float64(cutoff),
        polarity = _py_symbol(polarity),
    )

    bundle = BlobBoards.blob_board(canvas_size_jl; julia_kwargs...)

    return _py_board_result(
        bundle, image_origin, BlobBoards._bundle_identity(bundle),
        something(density_jl, BlobBoards._default_density(bundle))
    )
end

# Python names an output by strings and numbers; Julia by a value
# (`BlobBoards._recorded_output`, the same reading a JSON's render record
# gets). `nothing` when Python names none of the four (Julia's default then
# applies: a PDF). What a format has no use for the Julia types refuse by
# construction; Python's flat keywords can still SAY it, so it is refused here:
# a raster has no shading function and no bleed, a PNG no profile. (A PDF's
# file has no density: Python's `density` is then only the density of the pixel
# coordinates it gets back.)
function _py_output(format, density, icc_profile, shading_function, bleed = nothing)
    str(x) = x isa Py ? pyconvert(String, x) : string(x)
    fmt = _py_none(format) ? "pdf" : lowercase(str(format))
    icc = _py_none(icc_profile) ? nothing : str(icc_profile)
    shading = _py_symbol(shading_function)
    bleed_jl = _py_none(bleed) ? nothing : _py_mm(bleed)
    _py_none(format) && icc === nothing && shading === nothing && bleed_jl === nothing && return nothing
    if fmt != "pdf"
        bleed_jl === nothing || throw(
            ArgumentError("bleed applies to PDF output only; format = $(repr(fmt)) is a raster, which has no page boxes")
        )
        shading === nothing || throw(
            ArgumentError("shading_function applies to PDF output only; format = $(repr(fmt)) is a raster converted from the PDF")
        )
        fmt == "png" && icc !== nothing && throw(
            ArgumentError("icc_profile applies to PDF and TIFF output; PNG is written as linear-gamma greyscale without a profile")
        )
    end
    return BlobBoards._recorded_output(
        (; format = fmt, icc_profile = icc == "linear" ? nothing : icc, shading_function = something(shading, :calculator), density, bleed = bleed_jl)
    )
end

# What every board entry point hands Python: the blobs in the chosen pixel
# convention at `density` (given, else the board's own raster density),
# the configs, and the board's identity — minted for a fresh board, the STORED
# one for a loaded or re-rendered board.
function _py_board_result(bundle::BoardBundle, image_origin, config_hash::AbstractString, density)
    np = pyimport("numpy")
    image_origin_sym = image_origin isa Py ? Symbol(pyconvert(String, image_origin)) : Symbol(image_origin)
    blob_list = BlobBoards.blobs(bundle, :pattern; units = px, density)
    # px by the `units = px` call above, stripped in px; the per-field ustrip
    # in the loop below is then an identity.
    blob_list = _origin_shift(:colmap, image_origin_sym).(ustrip.(px, blob_list))

    blobs_array = zeros(Float64, length(blob_list), 3)
    for (i, blob) in enumerate(blob_list)
        blobs_array[i, 1] = ustrip(blob.center[1])
        blobs_array[i, 2] = ustrip(blob.center[2])
        blobs_array[i, 3] = ustrip(blob.σ)
    end
    blobs_np = np.array(blobs_array)

    bc = bundle.preamble.board_config
    rs = bc.ruler_spec
    board_config_nt = (
        canvas_width_mm = Float64(ustrip(mm, bc.canvas_size.width)),
        canvas_height_mm = Float64(ustrip(mm, bc.canvas_size.height)),
        board_width_mm = Float64(ustrip(mm, bc.board_size.width)),
        board_height_mm = Float64(ustrip(mm, bc.board_size.height)),
        margin_width_mm = Float64(ustrip(mm, bc.margin.width)),
        margin_height_mm = Float64(ustrip(mm, bc.margin.height)),
        ruler_width_mm = Float64(ustrip(mm, bc.ruler_width)),
        corner_size_mm = Float64(ustrip(mm, bc.corner_size)),
        frame_padding_mm = Float64(ustrip(mm, bc.frame_padding)),
        major_tick_spacing_mm = Float64(ustrip(mm, rs.major_tick.spacing)),
        minor_tick_spacing_mm = Float64(ustrip(mm, rs.minor_tick.spacing)),
        major_tick_width_mm = Float64(ustrip(mm, rs.major_tick.width)),
        minor_tick_width_mm = Float64(ustrip(mm, rs.minor_tick.width)),
    )

    return (
        blobs = blobs_np,
        density_dpi = Float64(ustrip(dpi, density)),   # what `blobs` are at; the layout has none
        board_config = board_config_nt,
        pattern_config = _py_pattern_config(bundle.preamble.pattern_config; unit = BlobBoards._blob_unit(bc.layout_grid)),
        config_hash = String(config_hash),   # the board's identity (uid = first 3 hex)
    )
end

# A stored board and the density its pixels are at: the caller's, else the
# image's own (`_image_density`: a raster's recorded one, a v3 board's grid's,
# else the board's raster density).
function _py_load_board(json_path, density, fname)
    d = _py_none(density) ? nothing : pint_to_julia_density(density)
    bundle, image_path, config_hash = BlobBoards.load_bundle(string(json_path))
    bundle isa BoardBundle || throw(ArgumentError("$fname: $(json_path) is not a board bundle"))
    return bundle, config_hash, BlobBoards._image_density(image_path, bundle; density = d)
end

function BlobBoards.load_board_python(json_path; image_origin = "colmap", density = nothing)
    bundle, config_hash, d = _py_load_board(json_path, density, "load_board")
    return _py_board_result(bundle, image_origin, config_hash, d)
end

# The file is written at the OUTPUT's density (a raster's own; a PDF has none),
# whatever sits beside the JSON; the blobs handed back are at that density too,
# or — for a PDF — at `density` when given, else the board's own raster density.
function BlobBoards.render_board_python(
        json_path; dir, format = nothing, density = nothing, icc_profile = nothing,
        id_label = false, shading_function = nothing, bleed = nothing, image_origin = "colmap",
    )
    bundle, _, config_hash = BlobBoards.load_bundle(string(json_path))
    bundle isa BoardBundle || throw(ArgumentError("render_board: $(json_path) is not a board bundle"))
    given = _py_none(density) ? nothing : pint_to_julia_density(density)
    grid = bundle.preamble.board_config.layout_grid
    given === nothing || BlobBoards._check_v3_density(grid, given, "density = $(given)")     # before any file is written
    output = something(_py_output(format, given, icc_profile, shading_function, bleed), Pdf())
    BlobBoards.render_board(
        bundle; dir = string(dir), output, id_label = id_label isa Py ? pyconvert(Bool, id_label) : Bool(id_label), config_hash
    )
    pixels = something(BlobBoards._density(output), given, BlobBoards._default_density(bundle))
    return _py_board_result(bundle, image_origin, config_hash, pixels)
end

# =============================================================================
# Unit conversion helpers — Pint ↔ FlexUnits (Pint converts its own quantities)
# =============================================================================

# `round_px`: pixel COUNTS (sizes, frame widths) round to whole pixels; a blob σ
# does not (2.5 px is a valid σ), so `_py_blob_size` supplies `round_px = false`.
function pint_to_julia_length(pint_quantity; default_unit = px, require_units::Bool = false, round_px::Bool = true)
    if pint_quantity isa Real
        if require_units
            error("min_scale must have units (Pint quantity or string like '1.0mm'), got unitless value: $pint_quantity")
        end
        return default_unit == px ? Int(round(pint_quantity)) * default_unit :
            Float64(pint_quantity) * default_unit
    end

    if pyisinstance(pint_quantity, pybuiltins.float) || pyisinstance(pint_quantity, pybuiltins.int)
        if require_units
            error("min_scale must have units (Pint quantity or string like '1.0mm'), got unitless value: $pint_quantity")
        end
        return default_unit == px ? Int(round(pint_quantity)) * default_unit :
            Float64(pint_quantity) * default_unit
    end

    return try
        py_qty = Py(pint_quantity)
        if !pyhasattr(py_qty, "magnitude") || !pyhasattr(py_qty, "units")
            error("Not a valid Pint quantity — missing magnitude/units attributes. Got type: $(typeof(pint_quantity))")
        end
        # Pint converts its own quantities: a length into millimetres, a
        # printing unit (pixel/dot) into pixels; a bare Pint number is pixels.
        dimensionless = pyconvert(Bool, py_qty.dimensionless)
        if pyconvert(Bool, py_qty.check("[length]"))
            pyconvert(Float64, py_qty.to("millimeter").magnitude) * mm
        elseif dimensionless || pyconvert(Bool, py_qty.check("[printing_unit]"))
            magnitude = pyconvert(Float64, dimensionless ? py_qty.magnitude : py_qty.to("pixel").magnitude)
            round_px ? Int(round(magnitude)) * px : magnitude * px
        else
            error("Unsupported unit: $(pyconvert(String, string(py_qty.units)))")
        end
    catch
        try
            val = pyconvert(Float64, pint_quantity)
            return default_unit == px ? Int(round(val)) * default_unit :
                val * default_unit
        catch
            error("Could not convert $(typeof(pint_quantity)) to Julia units. Expected number or Pint quantity.")
        end
    end
end

pint_to_julia_density(q::Quantity) = q
pint_to_julia_density(x::Real) = Float64(x) * dpi
function pint_to_julia_density(pint_quantity)
    if pyisinstance(pint_quantity, pybuiltins.float) || pyisinstance(pint_quantity, pybuiltins.int)
        return pyconvert(Float64, pint_quantity) * dpi
    end

    # a Pint density, converted by Pint itself: per inch of its dots/pixels
    # (Pint's `DPI`/`ppi`; `dot` is its alias of `pixel`) or of a plain count
    # (`n / ureg.inch`, or per mm). Pint's `dpi` is the DRY PINT, a volume: refused.
    py_qty = Py(pint_quantity)
    if pyhasattr(py_qty, "to")
        for target in ("pixel / inch", "1 / inch")
            v = try
                pyconvert(Float64, py_qty.to(target).magnitude)
            catch
                nothing
            end
            v === nothing || return v * dpi
        end
        error("Unsupported density unit: $(pyconvert(String, pybuiltins.str(py_qty.units))) — give dots per inch, e.g. 300 or 300 * ureg.DPI")
    end
    return try
        pyconvert(Float64, pint_quantity) * dpi
    catch
        error("Could not convert $(typeof(pint_quantity)) to Julia density units.")
    end
end

# =============================================================================
# JSON-loaded blob extraction
# =============================================================================

function BlobBoards.blobs_python(filepath; units = "px", reference_frame = "pattern", image_origin = "colmap", density = nothing)
    json_str = read(string(filepath), String)

    if occursin("\"board_config\"", json_str)
        bundle = JSON3.read(json_str, BoardBundle)
        reference_frame_sym = Symbol(reference_frame)

        units_jl = _json_units(units)
        # a board's pixels exist only at a density the caller names
        density = _py_none(density) ? nothing : pint_to_julia_density(density)

        blob_list = BlobBoards.blobs(bundle, reference_frame_sym; units = units_jl, density)
    else
        bundle = JSON3.read(json_str, PatternBundle)
        if units != "px" && units != "pd"
            error("PatternBundle only supports pixel units. Use 'px'.")
        end
        blob_list = BlobBoards.blobs(bundle)
    end

    if (units == "px" || units == "pd")
        blob_list = _origin_shift(:colmap, Symbol(image_origin)).(ustrip.(px, blob_list))
    else
        # a length, stripped in the unit asked for (:world blobs are bare mm already)
        blob_list = [strip_or_bare(units_jl, b) for b in blob_list]
    end

    # The blobs are plain numbers in the requested unit by now (stripped above),
    # so these strips are identities.
    blob_array = zeros(Float64, length(blob_list), 3)
    for (i, blob) in enumerate(blob_list)
        blob_array[i, 1] = ustrip(blob.center[1])
        blob_array[i, 2] = ustrip(blob.center[2])
        blob_array[i, 3] = ustrip(blob.σ)
    end

    np = pyimport("numpy")
    return np.array(blob_array)
end

# =============================================================================
# Detection
# =============================================================================

# Only live knobs. `method`, `first_octave`, `octave_resolution` were accepted
# and never read (detection is the ASIFT/IMAS-25 pipeline only), and the Python
# wrapper additionally forwarded `overlap_threshold`, which this function has
# never accepted — so `blobboards.detect_blobs()` was an outright MethodError.
# Removed 2026-08 rather than left as decoration.
function BlobBoards.detect_blobs_python(
        img;
        peak_threshold = 0.01,
        edge_threshold = 5.0,
        polarity = "dark",
        image_origin = "opencv"
    )
    img_jl = pyconvert(Array, img)

    polarity_str = pyisinstance(polarity, pybuiltins.str) ? string(polarity) : string(polarity)
    polarity_sym = Symbol(polarity_str)

    detections = BlobBoards.detect_blobs(
        img_jl;
        config = BlobBoards.DetectConfig(
            peak_threshold = Float64(peak_threshold),
            edge_threshold = Float64(edge_threshold),
            polarity = polarity_sym,
        ),
    )

    image_origin_sym = pyisinstance(image_origin, pybuiltins.str) ? Symbol(string(image_origin)) : Symbol(image_origin)
    T = EuclideanMap(PixelFrame{:colmap}, PixelFrame{image_origin_sym})

    result = Vector{Any}(undef, length(detections))
    # Defensive identity-or-strip (audited): detections arrive bare (adapted
    # frames) or px-quantities (raw detect output); both are px regime.
    for (i, det) in enumerate(detections)
        shifted = T(det.center)
        x = ustrip(shifted[1])
        y = ustrip(shifted[2])
        σ = ustrip(det.σ)
        resp = det.response
        pol = is_dark(det) ? "dark" : "light"
        result[i] = (x, y, σ, resp, pol)
    end

    return result
end

# =============================================================================
# find_boards — THE detection entry point (Python)
# =============================================================================

# Device string → KernelAbstractions backend through the package's one
# authority (VisualGeometryCore.default_device); "cuda" requires BlobBoardsCUDAExt to have
# resolved a functional GPU. `caller` names the Python entry point in the errors.
function _python_backend(device, caller::AbstractString)
    string(device) == "cpu" && return BlobBoards.KernelAbstractions.CPU()
    string(device) == "cuda" || error("$(caller): device must be \"cuda\" or \"cpu\", got $(device)")
    d = BlobBoards.default_device()
    d isa BlobBoards.KernelAbstractions.CPU &&
        error("device=\"cuda\" but no functional CUDA backend — `import CUDA, cuDNN` (Julia side) before calling $(caller)")
    return d
end

# The network, gallery features and bundles are the expensive, image-independent
# half of a `find_boards` call (an ONNX Runtime session, 50 cached gallery
# reads): built once per process and reused, keyed by everything that
# determines them — including the files' modification times, so an edited
# board or model is rebuilt rather than served stale.
const _BOARDS_CACHE = Dict{Any, Any}()

# The network, the board gallery's descriptors (staged on `backend`) and the
# board bundles, each keyed by board uid — what every image entry point needs.
function _python_boards_cached(board_paths, model_path, gallery_path, backend)
    bpaths = [string(p) for p in board_paths]
    key = (
        bpaths, map(mtime, bpaths), string(model_path), mtime(string(model_path)),
        string(gallery_path), typeof(backend),
    )
    return get!(() -> _python_boards(bpaths, model_path, gallery_path, backend), _BOARDS_CACHE, key)
end

function _python_boards(board_paths, model_path, gallery_path, backend)
    # The network runs where the kernels do: device = "cpu" means ONNX Runtime's
    # CPU provider too. `DescriptorNetwork`'s default (:cuda, no fallback) throws
    # in an environment without CUDA + cuDNN — the default authoring env — so
    # without this, device = "cpu" could never run.
    provider = backend isa BlobBoards.KernelAbstractions.CPU ? :cpu : :cuda
    model = BlobBoards.DescriptorNetwork(string(model_path); execution_provider = provider)
    gallery = BlobBoards.BoardGallery(string(gallery_path), model)
    bpaths = [string(p) for p in board_paths]
    board_feature_dict = Dict(
        uid => feats for (uid, feats) in
            [insert!(gallery, bp, model; device = backend) for bp in bpaths]
    )
    board_bundles = Dict(
        let (b, _, ch) = BlobBoards.load_bundle(bp)
                BlobBoards.uid_from_hash(ch) => b
        end for bp in bpaths
    )
    return model, board_feature_dict, board_bundles
end

# A board → image homography in BlobBoards' COLMAP pixels, re-expressed in the
# `origin` convention: the pixel-frame map composed on the image side.
function _homography_in(origin::Symbol, H::SMatrix{3, 3, Float64})
    t = EuclideanMap(PixelFrame{:colmap}, PixelFrame{origin}).t
    return SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, t[1], t[2], 1) * H
end

#     find_boards_python(board_paths, model_path, img; kwargs...) -> list
#
# Python wrapper for the full multi-board detection pipeline. Mirrors the Julia
# entry point: builds the descriptor model + board gallery, runs `find_boards`,
# and returns one dict per verified board.
#
# Arguments (the gallery and the network lead, the image follows — Julia's
# `find_boards(board_feature_dict, descriptor_network, img)` order):
# - `board_paths`: list of board-bundle JSON paths
# - `model_path`: ONNX descriptor model path
# - `img`: grayscale image as a 2-D numpy array (uint8 or float in [0,1])
# - `gallery_path`: JLD2 feature-cache path (created if missing)
# - `camera_json`: optional camera-calibration JSON → calibrated P3P path;
#   omit for the homography path
# - `device`: "cuda" (default; needs a functional CUDA backend on the Julia
#   side) or "cpu" — one KernelAbstractions backend for gallery AND query
# - `min_inliers`, `outlier_halfwidth`: geometry gates
# - `verify_similarity`, `min_appearance_support`: appearance-verification
#   knobs (the mandatory per-inlier descriptor gate; Julia-side defaults
#   0.6 / 0.5)
#
# - `image_origin`: the pixel convention of the returned homography, `"opencv"`
#   (default, as `detect_blobs_python`: top-left pixel centre at (0, 0)),
#   `"colmap"` ((0.5, 0.5), BlobBoards' own) or `"julia"` ((1, 1))
#
# Returns a list of dicts with keys `uid`, `model`, `quality`, `n_inliers`,
# `n_csponds`, `weights` (numpy array). `model` is the 3×3 homography H mapping a
# point of the board's world plane — centred, y up, in MILLIMETRES — to a pixel
# in `image_origin`'s convention; on the pose path (`camera_json`) it is a dict
# with `R` (3×3) and `t` (3-vector, millimetres), the board → camera pose, which
# carries no pixel convention.
function BlobBoards.find_boards_python(
        # Typed so a call in the retired `(img, board_paths, model_path)` order
        # is a MethodError: juliacall hands a Python list over as a `PyList`
        # (an `AbstractVector`), a str as a `String` and a 2-D numpy array as
        # a `PyArray` (an `AbstractMatrix`); a raw `Py` is accepted for each.
        board_paths::Union{AbstractVector, Py},
        model_path::Union{AbstractString, Py},
        img::Union{AbstractMatrix, Py};
        gallery_path,
        camera_json = nothing,
        device = "cuda",
        min_inliers = nothing,
        outlier_halfwidth = nothing,
        verify_similarity = nothing,
        min_appearance_support = nothing,
        image_origin = "opencv",
    )
    origin = Symbol(string(image_origin))
    origin in (:opencv, :colmap, :julia) ||
        throw(ArgumentError("find_boards_python: image_origin must be \"opencv\", \"colmap\" or \"julia\", got $(repr(image_origin))"))
    backend = _python_backend(device, "find_boards_python")
    img_jl = _py_gray(img)
    model, board_feature_dict, board_bundles = _python_boards_cached(board_paths, model_path, gallery_path, backend)
    provider = string(model.session.execution_provider)
    dev_name = backend isa BlobBoards.KernelAbstractions.CPU ? "cpu" : "cuda"

    cam = camera_json === nothing ? nothing :
        VisualGeometryIo.load(CameraModel, string(camera_json))
    geo_kw = _filter_kwargs(
        min_inliers = min_inliers === nothing ? nothing : Int(min_inliers),
        outlier_halfwidth = outlier_halfwidth === nothing ? nothing : Float64(outlier_halfwidth)
    )
    geometry = cam === nothing ? BlobBoards.HomographyGeometry(; geo_kw...) :
        BlobBoards.PoseGeometry(cam; geo_kw...)
    cfg_kw = _filter_kwargs(
        verify_similarity = verify_similarity === nothing ? nothing : Float64(verify_similarity),
        min_appearance_support = min_appearance_support === nothing ? nothing : Float64(min_appearance_support)
    )
    config = BlobBoards.FindBoardsConfig(; geometry, cfg_kw...)

    accepted = BlobBoards.find_boards(
        board_feature_dict, model, img_jl;
        config, device = backend, board_bundles
    )

    np = pyimport("numpy")
    out = Vector{Any}(undef, length(accepted))
    for (i, (uid, m, q, sc)) in enumerate(accepted)
        w = sc.weights
        model_py = if cam === nothing
            np.array(collect(_homography_in(origin, SMatrix{3, 3, Float64}(m))))
        else
            pydict(
                Dict(
                    "R" => np.array(collect(Float64.(SMatrix{3, 3}(m.R)))),
                    "t" => np.array(collect(Float64, m.t))
                )
            )  # Extrinsics3.t is bare (canonical mm)
        end
        out[i] = pydict(
            Dict(
                "uid" => uid,
                "model" => model_py,
                "quality" => Float64(q),
                "n_inliers" => count(VisualGeometryCore.is_map_inlier, w),
                "n_csponds" => length(sc),
                "weights" => np.array(Float64.(w)),
                "device" => dev_name,
                "execution_provider" => provider,
            )
        )
    end
    return pylist(out)
end

# A grey image from Python: 8-bit values are 0–255 (as PIL/OpenCV load them)
# and become [0, 1] exactly as N0f8 does; floats are taken as [0, 1].
function _py_gray(img)
    A = pyconvert(Array, img)
    eltype(A) == UInt8 && return Gray{Float32}.(reinterpret(N0f8, A))
    return Gray{Float32}.(A)
end

# A model as Python sees it: the homography in `origin` pixels, or a pose dict.
_py_model(np, m, origin, posed::Bool) = posed ?
    pydict(Dict("R" => np.array(collect(Float64.(SMatrix{3, 3}(m.R)))), "t" => np.array(collect(Float64, m.t)))) :
    np.array(collect(_homography_in(origin, SMatrix{3, 3, Float64}(m))))

function BlobBoards.track_boards_python(
        board_paths::Union{AbstractVector, Py}, model_path::Union{AbstractString, Py}, frames;
        gallery_path, camera_json = nothing, device = "cuda", image_origin = "opencv",
    )
    origin = Symbol(string(image_origin))
    origin in (:opencv, :colmap, :julia) ||
        throw(ArgumentError("track_boards_python: image_origin must be \"opencv\", \"colmap\" or \"julia\", got $(repr(image_origin))"))
    backend = _python_backend(device, "track_boards_python")
    model, board_feature_dict, board_bundles = _python_boards_cached(board_paths, model_path, gallery_path, backend)
    provider = string(model.session.execution_provider)
    cam = camera_json === nothing ? nothing : VisualGeometryIo.load(CameraModel, string(camera_json))
    geometry = cam === nothing ? BlobBoards.HomographyGeometry() : BlobBoards.PoseGeometry(cam)
    config = BlobBoards.FindBoardsConfig(; geometry)
    # A video path streams; anything else is a sequence of grey frames.
    input = frames isa AbstractString ? String(frames) :
        (frames isa Py && pyisinstance(frames, pybuiltins.str)) ? pyconvert(String, frames) :
        [_py_gray(f) for f in frames]
    tracks = BlobBoards.track_boards(board_feature_dict, model, input; board_bundles, config, device = backend)
    np = pyimport("numpy")
    out = Dict{String, Any}()
    for (uid, dets) in tracks
        out[uid] = pylist([
            pydict(Dict(
                "frame" => d.frame, "model" => _py_model(np, d.model, origin, cam !== nothing),
                "quality" => d.quality, "n_inliers" => d.n_inliers, "source" => string(d.source),
                "device" => backend isa BlobBoards.KernelAbstractions.CPU ? "cpu" : "cuda",
                "execution_provider" => provider,
            )) for d in dets
        ])
    end
    return pydict(out)
end

end # module BlobBoardsPythonExt
