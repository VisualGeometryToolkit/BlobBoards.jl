# =============================================================================
# Authoring + inspection entry points (documented here, implemented later)
# =============================================================================
#
# Board AUTHORING (the sheet's layout + its PDF/TIFF/PNG writers) is the
# submodule BlobBoards.Authoring (src/authoring/), which adds the methods. The
# package never loads Makie for it.
#
# INSPECTION recipes (`board!` overlay, `inspect_pipeline`) live in
# BlobBoardsMakieExt, triggered by any `using <Makie backend>`.

"""
    blob_board(paper_size::String; blob_sizes, kwargs...) -> BoardBundle
    blob_board(width, height; blob_sizes, kwargs...) -> BoardBundle
    blob_board(canvas_size::Size2; blob_sizes, kwargs...) -> BoardBundle
    blob_board(preamble::BoardPreamble; dir, output, id_label) -> BoardBundle

Generate a printable calibration board (blob pattern, ruler frame, QR code and uid
corners) at an exact physical size, and optionally write it and its JSON.

Sizes are physical (`mm`, `inch`). The board is laid out on a fixed physical grid
of 1/7200 inch (0.01 pt): blob centres, margins, frame and page are whole grid
units, so the print master holds exactly the stored geometry (stored in points).
A board has NO pixel density: the same sheet prints on any printer. A density
belongs to a raster output only — `Tiff(600dpi)`, `Png(300dpi)` — and the
printer a board is sized for enters through `blob_sizes`, with
[`suggest_blob_sizes`](@ref)`(density, extent)`.

# Arguments
- `paper_size`: a key of [`BlobBoards.PRINT_SIZES`](@ref) — `"A0"`…`"A5"`,
  `"ANSI A"`…`"ANSI E1"`, `"Letter"`, `"US Letter"`, `"Legal"`, `"Tabloid"`,
  `"Ledger"` (portrait, except `"Ledger"`). Anything else is an `ArgumentError`.
- `width`, `height` or `canvas_size = Size2(width = …, height = …)`: the page
  (canvas) size, physical lengths.
- `preamble`: the low-level form the others reduce to — a complete
  [`BlobBoards.BoardPreamble`](@ref), laid out as its board config says.

# Keywords

The paper-size, width/height and `Size2` forms take every keyword below; the JSON and preamble forms take the Output group only.
A keyword outside this list is an `ArgumentError`, never silently ignored.

**Output** (checked before placement)
- `dir = nothing`: output directory, created if missing. Writes
  `blob_board_<uid>.<pdf|tiff|png>` and `blob_board_<uid>.json` (the sidecar
  that reproduces the board), overwriting. `nothing` writes no file and only
  returns the bundle; giving `output` or `id_label` without a `dir` is then an
  `ArgumentError`. The JSON also records, outside the board's identity, how the
  file was rendered (its `render` block).
- `output = Pdf()`: what the board is written as — a [`Pdf`](@ref), a
  [`Tiff`](@ref) or a [`Png`](@ref), each carrying only its own options:
  - `Pdf(; shading_function = :calculator, icc_profile = :linear, bleed = 0mm)` — the print
    master: PDF/X-4 (also PDF/A-2b), written directly by BlobBoards with no
    external tools; vector frame, text in one embedded font, no image, and
    one radial shading per blob, so the printer evaluates the Gaussian at its
    own resolution and the file stays small at any size. No density.
    `shading_function` is the shading's colour function, `:calculator` (a
    PostScript calculator function) or `:sampled` (a sampled table, for a RIP
    that mishandles calculator functions); `icc_profile` the grey profile of
    the OutputIntent, `:linear` (linear reflectance) or the path of a
    single-channel ICC file; `bleed` a print shop's trim allowance on every
    side: the canvas stays the finished sheet and the TrimBox, the page
    (MediaBox = BleedBox) grows by `bleed` each way — a white strip cut off.
    A rendering option: the board and its uid do not change.
  - `Tiff(density; icc_profile = :linear)` — the whole sheet converted at
    `density` (pixels per inch, a whole number; the page rounded up to whole
    pixels: a pixel coordinate is the page's points × dpi/72), 8-bit grey, LZW,
    the density in the resolution tags and the profile embedded.
  - `Png(density)` — the same conversion as single-channel grey with linear
    gamma, for digital use; no profile.
  A raster's smallest blob must be at least 2 pixels per σ at its density
  (reported with the density that would do), a stored v3 board converts at its
  own density only, and a sheet over 32745 px a side cannot be converted (a
  PDF has no such bound).
- `id_label = true`: print the board's identity (file stem, board code,
  BlobBoards version) in one 6 pt line under the board, in the sheet
  margin. Skipped when the margin has no room (with a warning
  for a frameless board, which then carries no printed identity). A rendering
  option, not part of the identity.

`format`, `pdf_standard`, `icc_profile`, `shading_function` and `print_density`
were keywords until 2026-10; each is an `ArgumentError` naming `output`.

**Pattern** (recorded in the board's `PatternConfig`; see [`blob_pattern`](@ref))
- `blob_sizes` (REQUIRED, `(min, max)`): the smallest and largest printed disc
  DIAMETER (`2·cutoff·σ`), physical lengths. A board has no density, so there
  is no default: the smallest disc is a choice about the printer.
  [`suggest_blob_sizes`](@ref)`(density, extent)` gives both ends: the
  smallest disc that prints well — 3 mm at 1200 dpi, 6 mm at 600, 12 mm at 300
  (a printer's halftone cells grow as its density drops, and a disc must span
  enough of them to print smoothly) — and the largest, 20% of the board's
  shorter side, at most 64 mm (`suggest_blob_sizes(600dpi, "A4")` is
  6 mm … 38 mm). Either end may be given directly: a length, or for the max a
  fraction of the pattern's short side (`0.2short_side`: exactly that
  fraction, no cap). Neither end may be `nothing`, and the min may not be a
  fraction. Both ends are exact, and the largest
  must be smaller than the pattern's short side. Stored as σ in points
  (`min_scale`, `max_scale`; those names are not keywords).
- `seed`: the placement RNG seed (any integer; default a random `UInt64`,
  recorded in the bundle). Same seed and keywords → the same board.
- `polarity = :dark`: `:dark` blobs on white or `:light` blobs on black.
- `scales_per_octave = 3`: rungs per octave of the blob-scale ladder.
- `α = 1.2`: scale-sampling bias (how many blobs each scale gets).
- `cutoff = 2.0` (`DEFAULT_CUTOFF`): the disc radius in σ; keep the default.
- `max_trials = 5000`: placement attempts per blob before a scale gives up.

A board's blob centres always sit on its 0.01 pt grid (`raster_aligned` is
recorded as `false` in its `PatternConfig` — they are not pixel centres at any
density — and is not a keyword). A canvas side may not exceed 9999.99 pt
(3.53 m): beyond it the print master keeps no 0.01 pt.

**Board, frame and rulers** (recorded in the board's `BoardConfig`)
- `margin = Size2(width = 10mm, height = 10mm)` (`DEFAULT_BOARD_MARGIN`): the
  white margin around the board, which is always centred on the page — `width`
  left and right of it, `height` above and below. The board is the canvas less
  twice the margin. The canvas is the finished sheet, the PDF's TrimBox (a
  print shop's trim allowance is `Pdf(bleed)`, outside it). The default clears office
  printers' unprintable border (3–6.35 mm) and leaves the id label its 9.25 mm
  under the board. A zero margin is full bleed. The board records its margin.
- `board_size`: the board instead, a `Size2` of lengths ≤ the canvas; its margin
  is half the canvas's excess on each axis. `margin` and `board_size` are
  mutually exclusive (an `ArgumentError`).
- The blobs fill the whole board; the frame's marks are reserved rectangles on
  it that placement keeps clear of (`BlobBoards._frame_marks`). Each mark is
  switched by its size — a zero length (in any unit) turns it off:
- `corner_size = 10mm` (`DEFAULT_CORNER_SIZE`): the four corner cells — the QR
  code top left, a uid character in each other corner — which blobs keep 1 mm
  clear of. The QR carries the board's definition
  ([`BlobBoards.board_code`](@ref): ~120–150 characters, 0.24 mm modules in a
  10 mm corner), so the print alone rebuilds the board
  ([`BlobBoards.parse_board_code`](@ref)); it is omitted with a warning when it
  does not fit the corner.
- `ruler_width = 10mm` (`DEFAULT_RULER_WIDTH`): the ruler bands along each side
  between the corner cells, independent of `corner_size`.
- `frame_padding = 0mm`: white inset between a mark's content (ticks, QR code,
  uid glyphs) and its edges; must be `< corner_size / 2` and `< ruler_width / 2`.
- `major_tick_spacing = 10mm`, `minor_tick_spacing = 5mm`: ruler tick intervals
  (these and the tick widths are `ArgumentError`s with the ruler off).
  The ticks are centred along each side; check the printed scale against them.
- `major_tick_width = ruler_width − 2·frame_padding`, `minor_tick_width =
  major_tick_width / 2`: each tick's full drawn length across the ruler band
  (physical units, centred in the band). The defaults fill the padded band with
  major ticks and draw minor ticks half as long. Requires `0 < minor_tick_width
  ≤ major_tick_width ≤ ruler_width − 2·frame_padding`, else an `ArgumentError`.
- `frame_style` and `frame_width` (until 2026-10-14) are `ArgumentError`s naming
  `corner_size` and `ruler_width`.

Never keywords: `canvas_size`, `pattern_resolution` (derived
from the positional argument), `min_scale`, `max_scale` (use `blob_sizes`),
`pixel_type` (a board has none: its blobs are vector shadings, its TIFF/PNG
8-bit conversions), `raster_aligned` and `blob_measure` (sizes are always
diameters) are each an `ArgumentError`. A board's JSON is not a paper size:
it is rebuilt with [`BlobBoards.rebuild_board`](@ref).

# Returns
A [`BlobBoards.BoardBundle`](@ref):
- `bundle.preamble`: the [`BoardConfig`](@ref) (`board_config(bundle)`) and
  [`PatternConfig`](@ref) (`pattern_config(bundle)`) — everything needed to
  re-generate the board; its hash is the uid printed in the corners.
- `bundle.blobs`: the placed blobs (`Disc`s in points, pattern frame; the
  board's stored unit, a `px`-dimensioned value); use
  [`blobs`](@ref)`(bundle, frame; units)` for `:board`, `:canvas` or `:world`
  coordinates, mm, or pixels at a `density`.

# Examples
```julia
using BlobBoards

# An A4 board (190 mm wide inside its 10 mm margins) for a 600 dpi printer, written as PDF/X-4 + JSON
bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 0xBEEF, dir = "./out")

# A custom page, a 180 mm square board on it, 2–12 mm printed discs
bundle = blob_board(200mm, 250mm;
    board_size = Size2(width = 180mm, height = 180mm),
    blob_sizes = (2mm, 12mm),
    ruler_width = 0mm, frame_padding = 1mm, dir = "./out")

# TIFF (8-bit grey, the sheet converted at 1200 dpi) for a photo lab
bundle = blob_board("A3"; blob_sizes = (3mm, 0.2short_side), output = Tiff(1200dpi), dir = "./out")

# Re-generate a stored board, as it was printed
bundle = BlobBoards.rebuild_board("./out/blob_board_abc_12345.json"; dir = "./out2")
```

# See also
[`blob_pattern`](@ref) (the pattern alone, in pixels), [`PatternConfig`](@ref),
[`BoardConfig`](@ref), [`BlobBoards.PRINT_SIZES`](@ref). `BlobBoards.render_board`
re-renders a stored bundle without re-running placement.
"""
function blob_board end

"""
    rebuild_board(json_or_path::AbstractString; dir = nothing, output = nothing, id_label = nothing) -> BoardBundle

Rebuild a STORED board as it was printed, from its JSON sidecar — the JSON text
or the path of the file. The blobs drawn are the ones the JSON records (an older
board's were placed by older code, which today's need not reproduce); a
point-grid (v4) board also proves its configuration still places them — its
identity covers them — and is an `ArgumentError` if not. `output` and
`id_label` default to the JSON's `render` block (how the board was printed; a
file without one renders as a PDF with no id label); its identity is the stored
one. Without `dir` nothing is written. To draw a bundle you already hold in
another output, see [`BlobBoards.render_board`](@ref). Implemented in
`BlobBoards.Authoring`.
"""
function rebuild_board end

"""
    render_board(bundle::BoardBundle; dir, output = Pdf(), id_label = false,
                 config_hash = <the bundle's identity>) -> BoardBundle

Render a STORED board as `output` — [`Pdf`](@ref)`()` (the default),
[`Tiff`](@ref)`(density)` or [`Png`](@ref)`(density)` — into `dir`: draw the bundle's recorded blobs
(vector shadings) and its frame, QR and uid glyphs, without re-running
placement. `config_hash` is the authoritative identity from the bundle's JSON
(`hashes.config`); supply it so the printed uid matches the physical board even
for pre-v1 bundles whose recomputed hash differs. A raster follows
[`blob_board`](@ref)'s rules (2 pixels per σ; a stored v3 board at its own
density only). Implemented in `BlobBoards.Authoring`.

`id_label` defaults to `false` (a reconstruction reproduces the board as
printed, and the boards in circulation carry no label; supply the JSON's
recorded value, as [`BlobBoards.rebuild_board`](@ref) does). The QR code carries the board
code ([`BlobBoards.board_code`](@ref)) when the configuration reproduces
`config_hash`, else (a stored v3 board, or one whose stored identity is not
its configuration's hash) the identity itself, in upper-case hex. The uid
glyphs and the file name come from `config_hash`.
"""
function render_board end

"""
    save_canvas_texture(path, bundle::BoardBundle; density, relative_to = dirname(path),
                        id_label = false, config_hash = <the bundle's identity>) -> VisualGeometryToolkitCore.CanvasTexture

Write the board's printed page — the sheet [`render_board`](@ref) draws, as a
[`Png`](@ref)`(density)` (8-bit grey, linear gamma) — to `path`, and return
the VisualGeometryToolkitCore `CanvasTexture` that names it: the path relative
to `relative_to` (the directory of the file that will name it), the SHA-256 of
the bytes written, and the rectangle of the design chart the image covers. The
design chart is the board's world frame (canvas centre, y up; the plane `z = 0`
of `TargetFrame{uid}`, where [`board_points`](@ref) lie). The image is the
page on the raster's pixel grid — anchored at the canvas's top-left corner,
`1/density` per pixel, rounded UP to whole pixels — so the covered rectangle is
that many pixels at `density`, from the canvas's top-left corner: the canvas
itself when the page is a whole number of pixels, otherwise up to one blank
pixel more. Give it to [`board_target`](@ref)`(uid, boards; texture)`. A raster
follows [`render_board`](@ref)'s rules. Implemented in `BlobBoards.Authoring`.
"""
function save_canvas_texture end

"""
    write_tiff(path, grey::AbstractMatrix{UInt8}, density, icc::Vector{UInt8}) -> path

Write an 8-bit grey image (a board's sheet, converted) as an LZW-compressed TIFF (horizontal
predictor) with `density` in its resolution tags (pixels per inch) and `icc`
embedded. `grey` is (rows, columns), first row on top. Implemented in
`BlobBoards.Authoring` (src/authoring/tiff.jl, libtiff).
"""
function write_tiff end

"""
    board!(ax, proj, rect, blobs; cutoff)

Project a board's world rect + blobs into an image axis. Requires a Makie
backend (BlobBoardsMakieExt).
"""
function board! end

"Non-mutating form of [`board!`](@ref). Requires a Makie backend."
function board end
