# =============================================================================
# Board subsystem: layout, QR/rulers, and board assembly (consolidated)
# =============================================================================

# The QR code's error-correction level: Medium (15% of the codewords
# recoverable). A fact of the printed sheet — it fixes the module counts the
# board code is sized for (41 × 41) — not a keyword.
const QR_ERROR_CORRECTION = QREncode.Medium()

# QR Code Generation
# A point-grid (v4) board's QR carries its DEFINITION, `board_code` (format
# BB4, ~120-150 alphanumeric characters: 41 × 41 modules, 0.24 mm in a 10 mm
# corner, read by any phone), from which `parse_board_code`
# rebuilds the board under its identity. A stored v3 board — laid out on its
# printer's pixels, its stored identity not always recomputable from its
# config — carries its identity instead: the 64-hex sha256 the uid abbreviates,
# in upper case (QR's alphanumeric mode). (Until 2026-10 the QR held the
# preamble as JSON, ~880 characters, 113 × 113 modules: 0.18 mm even in A4's
# 22 mm corner, too fine to read off a print.)
# What the QR holds: the board code, when the board's configuration reproduces
# the identity it is printed under; else that identity.
_qr_text(code, config_hash) = something(code, uppercase(config_hash))
# The board code of a board whose blobs are `blobs`, or `nothing` when there is
# none to print: a v3 board (laid out on its printer's pixels), or a board whose
# configuration and blobs do not give the identity it is printed under.
# `verified`: `config_hash` was just minted from these blobs (no re-hash).
_code_or_nothing(preamble, blobs, config_hash; verified::Bool = false) =
    preamble.board_config.layout_grid isa PointGrid && (verified || BlobBoards._board_identity(preamble, blobs) == config_hash) ?
    BlobBoards._board_code(preamble, config_hash) : nothing

function make_qr_code(text::AbstractString)
    qr_matrix = qrcode(text, eclevel = QR_ERROR_CORRECTION)   # `true` = a dark module
    # the 4-module quiet zone is LIGHT (`false`). Until 2026-10 it was padded with
    # `true` and `true` was drawn white: every printed QR came out colour-inverted
    # with a light quiet zone, which no decoder reads.
    h, w = size(qr_matrix)
    padded = fill(false, h + 8, w + 8)
    padded[5:(end - 4), 5:(end - 4)] = qr_matrix
    return padded
end

# Validate the writer keywords BEFORE placement. `nothing` means "not given":
# `output` and `id_label` describe a file, so giving one without `dir` (nothing
# is written) is a mistake. Returns the output (`Pdf()` when omitted).
function _check_writer(dir, output, id_label)
    if dir === nothing
        given = [k for (k, v) in (:output => output, :id_label => id_label) if v !== nothing]
        isempty(given) || throw(
            ArgumentError("$(given) describe the output file, but dir = nothing writes none — give a dir")
        )
    end
    return something(output, Pdf())
end

# The five keywords the typed outputs replaced (2026-10): named so the error
# says where each went.
const _RETIRED_OUTPUT_KWARGS = (:format, :pdf_standard, :icc_profile, :shading_function, :print_density)
function _refuse_retired_output_kwargs(fname, kw)
    retired = intersect(keys(kw), _RETIRED_OUTPUT_KWARGS)
    isempty(retired) || throw(
        ArgumentError(
            "$fname: $(sort(collect(retired))) are no longer keywords — the output is one value, " *
                "`output = Pdf(; shading_function, icc_profile)`, `Tiff(density; icc_profile)` or `Png(density)` " *
                "(a PDF has no density, and is always PDF/X-4)"
        )
    )
    return nothing
end

#     blob_board(preamble::BoardPreamble; dir=nothing, output=Pdf(), id_label=true)
#
# Generate a calibration board from a complete preamble: the core every other
# `blob_board` method reduces to. The layout is the preamble's own
# (`BoardLayout(board_config)`, the one derivation).
function blob_board(
        preamble::BoardPreamble; dir::Union{Nothing, AbstractString} = nothing,
        output::Union{Nothing, BoardOutput} = nothing, id_label::Union{Nothing, Bool} = nothing, kwargs...
    )
    _refuse_unknown_kwargs("blob_board(preamble)", (; kwargs...))
    return _generate_board(preamble; dir, output, id_label)
end

# The forms that take the output group only (a preamble, a JSON): anything else
# is refused by name, a retired output keyword saying where it went.
function _refuse_unknown_kwargs(fname, kw)
    _refuse_retired_output_kwargs(fname, kw)
    isempty(kw) || throw(
        ArgumentError("$fname: unknown keyword(s) $(sort(collect(keys(kw)))); it takes dir, output and id_label")
    )
    return nothing
end

# Placement + render under the board's identity, minted from what was placed
# (`_board_identity`), whose uid is printed on the corner squares. `id_label` as
# given (`nothing`: the default, on).
function _generate_board(preamble::BoardPreamble; dir, output, id_label)
    out = _check_writer(dir, output, id_label)
    board_sizes = BoardLayout(preamble.board_config)
    _check_output(out, preamble, board_sizes)                     # before placement
    blobs_stored = BlobBoards._place_board_blobs(preamble)        # the board's stored unit (pt; px for v3)
    config_hash = BlobBoards._board_identity(preamble, blobs_stored)
    return _assemble_board(preamble, board_sizes, blobs_stored, config_hash; dir, out, id_label = something(id_label, BlobBoards._board_dflt("id_label")::Bool), verified = true)
end

# The board's identity in one printed line, so every printed sheet can be
# traced, frameless boards included (a frame carries only the 3-character uid
# and the QR): the file stem, the board's DEFINITION — its board code, what
# the QR holds, from which `parse_board_code` rebuilds the board — and the
# BlobBoards version that rendered it. No density: a point-grid board has none
# (one sheet, any printer). Where the code would not reproduce the identity
# the board is printed under, the line carries that identity instead, as the
# QR does (`_qr_text`).
function _id_label_text(bundle::BoardBundle, config_hash::AbstractString, code)
    preamble = bundle.preamble
    stem = make_filebase(bundle; config_hash)                 # the file's own stem
    return "$stem · $(_label_identity(preamble.board_config.layout_grid, preamble, config_hash, code)) · BlobBoards $(pkgversion(BlobBoards))"
end
_label_identity(::PointGrid, preamble, config_hash, code) = something(code, "sha256 $config_hash")
# (the label of a bundle, its code derived from the bundle's own blobs)
_id_label_text(bundle::BoardBundle, config_hash::AbstractString) =
    _id_label_text(bundle, config_hash, _code_or_nothing(bundle.preamble, bundle.blobs, config_hash))
# A stored v3 board has no code, and IS laid out on its printer's pixels: its
# hash, its size and that density, the line such boards were printed with.
function _label_identity(g::PixelGrid, preamble, config_hash, _)
    bc = preamble.board_config
    w, h = (round(ustrip(mm, s); digits = 1) for s in (bc.board_size.width, bc.board_size.height))
    mm_str(x) = isinteger(x) ? string(Int(x)) : string(x)
    return "sha256 $config_hash · $(mm_str(w)) × $(mm_str(h)) mm @ $(mm_str(round(ustrip(dpi, g.density); digits = 1))) dpi"
end

# A Cairo image surface cannot exceed 32767 px a side, and Cairo refuses to
# draw an image over 2^15 - 23: the bound on a raster CONVERSION of a board
# (`Tiff`, `Png`: pdftocairo converts through one; `_check_output`). A PDF has
# none — its blobs are shadings and its frame vector. A 1 m side at 1200 dpi
# is 47244 px.
const _CAIRO_MAX_IMAGE_PX = 2^15 - 23

# The RENDER half of board authoring: everything after placement. Shared by
# `blob_board` (fresh generation) and `render_board` (faithful reconstruction
# from a stored bundle), so the two cannot drift apart.
#
# `out` is the checked output (`_check_output`); the writer dispatches on it.
# The sheet is laid out in whole units of the board's layout grid
# (`board_sizes`), so every frame element lands on the grid; the writer paints
# the blobs into the pattern's cell as vector shadings (`_pattern_content`).
function _assemble_board(
        preamble::BoardPreamble, board_sizes, blobs_stored, config_hash::AbstractString;
        dir, out::BoardOutput, id_label::Bool, verified::Bool = false
    )
    bundle = BoardBundle(preamble, blobs_stored)
    isnothing(dir) && return bundle
    # the board code (QR and label), once: it re-derives nothing from placement
    code = _code_or_nothing(preamble, blobs_stored, config_hash; verified)
    # what the render block records: whether the label was actually drawn
    sheet, labelled = _frame_content(
        preamble, board_sizes, config_hash, _qr_text(code, config_hash), id_label ? _id_label_text(bundle, config_hash, code) : nothing
    )
    _save_board(out, sheet, bundle; dir, id_label = labelled, config_hash)
    return bundle
end

#     blob_board(width::Quantity, height::Quantity; blob_sizes, kwargs...)
#
# Generate a calibration board with specified physical dimensions.
#
# # Arguments
# - `width`, `height`: Physical canvas dimensions (e.g., `210mm`, `8.5inch`)
#
# # Keyword Arguments
# - `blob_sizes` (required): `(min, max)` printed disc diameters in **physical
#   units** (e.g., `(1mm, 12mm)`), or `suggest_blob_sizes(600dpi, "A4")`
#   (the smallest disc a 600 dpi printer prints well; 20% of the board's
#   shorter side, at most 64 mm). The max may also be
#   `0.2short_side` (that fraction of the pattern's short side, no cap);
#   `nothing` is an ArgumentError. Stored as σ in points; min and max are exact.
# - `seed`: Random seed for reproducible patterns
# - `dir`: Output directory for saving files
# - `output`: what is written — `Pdf()` (default: the PDF/X-4 print master),
#   `Tiff(density)` or `Png(density)`
# - `margin`: the white margin around the centred board (default
#   `DEFAULT_BOARD_MARGIN` = 10 mm on every side): a `Size2` — `width` left and
#   right of the board, `height` above and below; zero is full bleed. Or `board_size`: the board itself (its margin is
#   half the canvas's excess). The two are mutually exclusive.
# - `corner_size`: the corner cells (QR code, uid), 10 mm by default; `0mm` turns them off
# - `ruler_width`: the ruler bands between them, 10 mm by default; `0mm` turns the ruler off
# - `frame_padding`: Padding between frame content and edges (default: 0mm). Applied to
#   ruler ticks, QR code, and UID characters.
# - Pattern parameters: `scales_per_octave`, `α`, `cutoff`, etc.
#
# # Example
# ```julia
# # Basic board
# bundle = blob_board(210mm, 297mm; blob_sizes=(4.0mm, 0.2short_side), seed=0x1234, dir="./out")
#
# # Board with corner cells but no ruler, as a 600 dpi TIFF
# bundle = blob_board(210mm, 297mm;
#     blob_sizes=(4.0mm, 0.2short_side), seed=0x1234,
#     ruler_width=0mm, dir="./out", output=Tiff(600dpi))
# ```
function blob_board(width::Quantity, height::Quantity; kwargs...)
    return blob_board(Size2(width = width, height = height); kwargs...)
end

# BoardConfig constructor keywords that are NOT fields (they build `ruler_spec`).
const _RULER_KWARGS = (:major_tick_spacing, :minor_tick_spacing, :major_tick_width, :minor_tick_width)
# What `blob_board` takes for each config: the fields not derived from its
# positional arguments or other keywords, and the keywords that build them.
const _BOARD_KWARGS = (setdiff(fieldnames(BoardConfig), (:canvas_size, :layout_grid, :ruler_spec, :frame_style))..., _RULER_KWARGS...)
const _PATTERN_KWARGS = (
    :blob_sizes, setdiff(fieldnames(PatternConfig), (:pattern_resolution, :min_scale, :max_scale, :raster_aligned, :pixel_type))...,
)

# The keywords `blob_board` refuses by name, each with its reason and the way
# out. Most are config FIELDS: `filter_kwargs` matches field names, so it would
# capture them and either record an inert value (`raster_aligned`) or splat one
# over the value derived from the positional argument (`canvas_size`). The
# retired output keywords are `_refuse_retired_output_kwargs`, shared with the
# preamble form; an unknown name is refused after the split.
const _GRID_ONLY = "every new board is laid out on the 1/7200-inch point grid"
const _MARKS_BY_SIZE = "the frame's marks are switched by their sizes: corner_size (the corner cells: QR code, uid) " *
    "and ruler_width (the ruler bands), a zero length turning one off"
const _FROM_CANVAS = "it is derived from the positional argument (the canvas size)"
const _REFUSED_BOARD_KWARGS = Dict{Symbol, String}(
    # the `blob_board(T, …)` forms were removed 2026-10-13
    :pixel_type => "a board has no pixel type: its blobs are vector shadings and its TIFF/PNG 8-bit " *
        "conversions of the PDF (a pattern takes one: blob_pattern(T, …))",
    :raster_aligned => "a board's blob centres are always on its 1/7200-inch layout grid",
    :layout_grid => _GRID_ONLY, :grid => _GRID_ONLY,
    :ruler_spec => "set $(join(_RULER_KWARGS, ", "))",
    # the marks have been switched by their sizes since 2026-10-14
    :frame_style => _MARKS_BY_SIZE, :frame_width => _MARKS_BY_SIZE,
    :canvas_size => _FROM_CANVAS, :pattern_resolution => _FROM_CANVAS,
)
function _refuse_board_kwargs(kw)
    for k in sort!(collect(keys(kw)))
        reason = get(_REFUSED_BOARD_KWARGS, k, nothing)
        reason === nothing || throw(ArgumentError("blob_board: `$k` is not a keyword — $reason"))
    end
    return nothing
end

function blob_board(canvas_size::Size2{<:Quantity}; kwargs...)
    kw = (; kwargs...)
    _refuse_retired_output_kwargs(:blob_board, kw)
    _refuse_board_kwargs(kw)

    # Split for BoardConfig. `filter_kwargs` matches FIELD names, and the ruler
    # geometry is authored through constructor KEYWORDS that build the
    # `ruler_spec` field — so pull those out by name first (they used to fall
    # through to the unknown-keyword error below; docs/guide/boards.md and
    # examples/generate/board.jl supply them).
    ruler_kwargs = NamedTuple{Tuple(k for k in keys(kw) if k in _RULER_KWARGS)}(kw)
    # (a band or tick keyword its frame style draws nothing with is refused by
    # `BoardConfig`, wherever a config is built)
    board_kwargs, dropped1 = filter_kwargs(BoardConfig; Base.structdiff(kw, ruler_kwargs)...)
    board_config = BoardConfig(; canvas_size = canvas_size, layout_grid = PointGrid(), board_kwargs..., ruler_kwargs...)
    grid = board_config.layout_grid

    # the pattern in the board's stored unit, points: k/100, exact
    pattern_pt = BlobBoards._board_pattern_points(board_config)

    # Blob sizes: `blob_sizes`, printed disc DIAMETERS in PHYSICAL lengths,
    # stored as σ in points (`min_scale`/`max_scale`). Required: a board has no
    # density, so its smallest disc is the caller's choice about the printer
    # (`suggest_blob_sizes(density, extent)`), and a raster output checks it against its own
    # pixels (`_check_output`).
    to_pt = q -> begin
        q isa Quantity && dimension(q) == dimension(mm) ||
            throw(ArgumentError("blob_board: blob_sizes must be physical lengths (e.g. 0.2mm), got $(q)"))
        BlobBoards._length_to_px(q, stored_density(grid)) * px   # mm × points/mm
    end
    sized = BlobBoards._resolve_blob_sizes(
        to_pt, :blob_board, (; dropped1...), ustrip.(px, pattern_pt);
        floor_text = d_pt -> "$(BlobBoards._short_number(ustrip(mm, d_pt * pt))) mm", min_sigma = 0.0
    )
    pattern_kwargs, dropped2 = filter_kwargs(PatternConfig; sized...)

    preamble = BlobBoards._board_preamble(board_config; pattern_kwargs...)

    # Anything that matched neither config and is not a board-writing option is
    # a MISTAKE, not a leftover. This used to `filter` the whitelist and discard
    # the rest, so `frame_stlye = :ruler` produced a default-framed board and
    # reported success — and the artifact is something you PRINT. blob_pattern
    # already fails loud here (its leftovers hit a concretely-keyworded method);
    # this brings blob_board into line, and matches the TOML loader's rule that
    # a knob which silently does nothing is the failure worth preventing.
    _writer_opts = (:dir, :output, :id_label)
    unknown = setdiff(keys(dropped2), _writer_opts)
    isempty(unknown) || throw(
        ArgumentError(
            "blob_board got unknown keyword(s) $(sort(collect(unknown))). Valid keywords: " *
                "board $(_BOARD_KWARGS), pattern $(_PATTERN_KWARGS), output $(_writer_opts)."
        )
    )
    preamble_kwargs = filter(kv -> kv[1] in _writer_opts, pairs(dropped2))

    return blob_board(preamble; preamble_kwargs...)
end


#     blob_board(paper_size::String; blob_sizes, kwargs...)
#
# Generate a calibration board using a standard paper size.
#
# # Arguments
# - `paper_size`: Standard paper size string (e.g., "A4", "Letter", "A3")
#
# # Keyword Arguments
# - `blob_sizes` (required): printed disc diameter range in **physical units** (see above).
# - `output`: `Pdf(; shading_function, icc_profile)` (default — the PDF/X-4
#   print master, no density), `Tiff(density; icc_profile)` or `Png(density)`
# - `id_label`: print the board's identity (file stem, board code, BlobBoards
#   version) in 6 pt under the board, in the sheet margin outside the
#   TrimBox (default `true`; skipped with a warning when the margin has no room)
# - `margin`: the white margin around the centred board (default
#   `DEFAULT_BOARD_MARGIN` = 10 mm on every side): a `Size2` — `width` left and
#   right of the board, `height` above and below; zero is full bleed. Or `board_size`: the board itself (its margin is
#   half the canvas's excess). The two are mutually exclusive.
# - `corner_size`, `ruler_width`: the corner cells and the ruler bands, 10 mm each; `0mm` turns one off
# - `frame_padding`: Padding between frame content and edges (default: 0mm). Applied to
#   ruler ticks, QR code, and UID characters.
# - Other arguments same as other blob_board methods
#
# # Example
# ```julia
# # Standard board with ruler frame
# bundle = blob_board("A4"; blob_sizes=(4.0mm, 0.2short_side), seed=0x1234, dir="./out")
#
# # Square board centered in A4 with margins
# bundle = blob_board("A4";
#     blob_sizes=(4.0mm, 0.2short_side),
#     board_size=Size2(width=180mm, height=180mm),
#     dir="./out")
# ```
#
# # Available Paper Sizes
# The keys of `BlobBoards.PRINT_SIZES`: A0–A5, ANSI A–E1, Letter, US Letter, Legal, Tabloid, Ledger.
function blob_board(paper_size::String; kwargs...)
    # (a board's JSON is rebuilt by `rebuild_board`; say so rather than "unknown paper size")
    startswith(lstrip(paper_size), '{') &&
        throw(ArgumentError("blob_board takes a paper name; a board's JSON is rebuilt with BlobBoards.rebuild_board(json)"))
    return blob_board(BlobBoards._paper_size(paper_size); kwargs...)
end


#     rebuild_board(json_or_path; dir=nothing, output=nothing, id_label=nothing)
#
# Reconstruct a stored board from its JSON (the text, or the file's path) —
# documented on the stub (src/authoring_stubs.jl). `output` and
# `id_label` default to the JSON's `render` block (how the board was printed); a
# file without one renders as a PDF with no id label.
#
# # Example
# ```julia
# bundle1 = blob_board("A4"; blob_sizes=(4.0mm, 0.2short_side), seed=0x1234, dir="./out")
# bundle2 = BlobBoards.rebuild_board("./out/blob_board_1234.json"; dir="./out2")
# ```
function rebuild_board(
        json_or_path::AbstractString; dir::Union{Nothing, AbstractString} = nothing,
        # reconstruction: as printed — each option defaults to the JSON's `render`
        # block (BlobBoards._render_options); an explicit keyword overrides it
        output::Union{Nothing, BoardOutput} = nothing, id_label::Union{Nothing, Bool} = nothing, kwargs...
    )
    _refuse_unknown_kwargs("rebuild_board", (; kwargs...))
    json_config = if startswith(lstrip(json_or_path), '{')
        String(json_or_path)
    elseif isfile(json_or_path)
        read(json_or_path, String)
    else
        throw(ArgumentError("rebuild_board takes a board's JSON, as text or a file's path; got $(repr(first(json_or_path, 80)))"))
    end
    # the caller's keywords are checked as given, before parsing and placement
    # (no output keyword without a dir); the recorded options then fill in
    _check_writer(dir, output, id_label)
    # Try to read as BoardBundle first (the common case), extract preamble
    bundle = JSON3.read(json_config, BoardBundle)
    preamble = bundle.preamble
    obj = JSON3.read(json_config)::JSON3.Object
    rec = BlobBoards._render_options(obj)
    # a v3 board's rasters are at its printer's density; its writers recorded none
    g = preamble.board_config.layout_grid
    g isa BlobBoards.PixelGrid && rec.density === nothing && (rec = merge(rec, (; density = g.density)))
    # the recorded output, when a file is written and the caller names none
    out = output !== nothing ? output : dir === nothing ? Pdf() : BlobBoards._recorded_output(rec)
    board_sizes = BoardLayout(preamble.board_config)
    _check_output(out, preamble, board_sizes)

    # A rebuild prints what the JSON says was printed: its stored blobs (an older
    # board's were placed by older code, which today's need not reproduce). A
    # point-grid board also proves its configuration still places them — its
    # identity covers them — and is refused if not; a file with no blobs is
    # placed. The identity is the STORED hash (a v1 board's recomputed hash is
    # not, nor is a migrated older-schema board's); the QR carries the board
    # code, or that identity when the configuration does not reproduce it.
    blobs_stored = bundle.blobs
    if isempty(blobs_stored)
        blobs_stored = BlobBoards._place_board_blobs(preamble)
    elseif get(obj, :schema, nothing) == BlobBoards.BOARD_SCHEMA         # a file of today's schema
        BlobBoards._place_board_blobs(preamble) == blobs_stored || throw(
            ArgumentError(
                "rebuild_board: this board's configuration no longer places the blobs its JSON records " *
                    "(placement changed since it was made, schema $(BlobBoards.BOARD_SCHEMA)): its stored blobs are " *
                    "drawn by render_board(bundle; dir, output), which does not re-place"
            )
        )
    end
    stored_hash = BlobBoards._stored_config_hash(obj)
    stored_hash === nothing && (stored_hash = BlobBoards._board_identity(preamble, blobs_stored))
    return _assemble_board(
        preamble, board_sizes, blobs_stored, stored_hash; dir, out, id_label = something(id_label, rec.id_label)
    )
end

#     render_board(bundle::BoardBundle; dir, output = Pdf(), id_label = false,
#                  config_hash = <the bundle's identity>) -> BoardBundle
#
# Render a stored board WITHOUT re-running placement — documented on the stub
# (src/authoring_stubs.jl). The vector blobs come from the bundle's recorded
# blobs, the QR from the board code (or the identity, `_qr_text`), and the
# corner uid glyphs from `config_hash`, the authoritative identity carried in
# the bundle JSON.
function render_board(
        bundle::BoardBundle; dir::AbstractString, output::BoardOutput = Pdf(),
        # off by default: a reconstruction reproduces the board AS PRINTED, and the
        # boards in circulation (and the gallery renders made from them) have none
        id_label::Bool = false, config_hash::AbstractString = BlobBoards._bundle_identity(bundle)
    )
    preamble = bundle.preamble
    board_sizes = BoardLayout(preamble.board_config)
    _check_output(output, preamble, board_sizes)
    return _assemble_board(preamble, board_sizes, bundle.blobs, String(config_hash); dir, out = output, id_label)
end

# What an output asks of a board, checked before anything is placed or drawn.
# A PDF asks nothing: its blobs are shadings and its frame vector, at any size.
# A raster is the sheet in pixels at its density, so:
# - a v3 board, laid out on its printer's pixels, converts at that density only;
# - the smallest blob must be at least `MIN_SIGMA_PX` pixels per σ (a disc of
#   2·cutoff·2 pixels) — a board with 1 mm discs is not silently rendered at 150 dpi;
# - the sheet must fit a Cairo image surface (`_CAIRO_MAX_IMAGE_PX` per side:
#   pdftocairo converts through one).
_check_output(::Pdf, preamble::BoardPreamble, board_sizes) = nothing
function _check_output(o::Union{Tiff, Png}, preamble::BoardPreamble, board_sizes)
    bc, pc = preamble.board_config, preamble.pattern_config
    grid = bc.layout_grid
    d = ustrip(dpi, o.density)
    BlobBoards._check_v3_density(grid, o.density, "$(nameof(typeof(o)))($(o.density))")
    floor_dpi = ustrip(dpi, BlobBoards._density_for_sigma(preamble, BlobBoards.MIN_SIGMA_PX))
    if d < floor_dpi
        across = BlobBoards._diameter_factor(pc.cutoff) * pc.min_scale / stored_density(bc)   # the smallest disc's diameter, a length
        throw(
            ArgumentError(
                "this board's smallest disc ($(BlobBoards._short_number(ustrip(mm, across))) mm) is " *
                    "$(BlobBoards._short_number(ustrip(px, across * o.density))) pixels across at " *
                    "$(BlobBoards._short_number(d)) dpi, below the floor of " *
                    "σ ≥ $(BlobBoards._short_number(BlobBoards.MIN_SIGMA_PX)) pixels; render it at ≥ $(round(Int, floor_dpi)) dpi"
            )
        )
    end
    c = board_sizes.canvas_size
    # the size the conversion will have (`_raster_size`), not a rounding of its own
    m = maximum(_raster_size(ustrip.(px, (c.width, c.height)), grid, d))
    m > _CAIRO_MAX_IMAGE_PX && throw(
        ArgumentError(
            "$(nameof(typeof(o))) rasterizes the whole sheet through Cairo, which " *
                "cannot exceed $(_CAIRO_MAX_IMAGE_PX) px per side; this canvas is " *
                "$(m) px at $(BlobBoards._short_number(d)) dpi. Use Pdf() (vector, any size) or a lower density"
        )
    )
    return nothing
end
