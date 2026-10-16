# Authoring a Board

This guide is about making a printable calibration board with
[`blob_board`](@ref): setting up the environment, worked examples, output
types, printing and reproducing a board later. Every method form and every
keyword (defaults, units, valid values) is listed once, in the
[`blob_board`](@ref) reference. This guide shows them in use but does not repeat
that list.

## Setup

Board authoring is part of BlobBoards: `using BlobBoards` is all it needs. The
package writes the PDF print master itself (no Cairo, no Ghostscript), and its
TIFF and PNG conversions with Poppler and libtiff, which it brings along.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'           # once, in a checkout
julia --project=. examples/generate/board.jl                   # A4, for a 300 dpi printer, PDF
```

(Until 2026-10-14 authoring was an extension loaded by `using CairoMakie`.)

## Your first board

```julia
using BlobBoards      # re-exports the units: mm, inch, px, dpi, …

bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 0xBEEF, dir = "./output")
```

`blob_sizes` is required: the smallest and largest printed disc diameters.
`suggest_blob_sizes(600dpi, "A4")` gives both for a 600 dpi printer and the
default A4 board: 6 mm, the smallest disc that prints well, and 38 mm, 20% of
the board's shorter side (see Blobs, below). This
writes two files named after the board's uid:

- `blob_board_<uid>.pdf`: the print master, PDF/X-4.
- `blob_board_<uid>.json`: the sidecar. It records the configuration and every
  blob, which is enough to reproduce or re-render the board and to detect it.

The returned `BoardBundle` holds the same data, so with `dir` left out nothing
is written and you get the bundle in memory. `seed` makes the placement
reproducible; leave it out and a random seed is drawn and recorded.

## Physical layout: a board has no density

A board is laid out in physical units, on a fixed grid of 1/7200 inch (0.01 pt,
3.5 µm): every blob centre, margin and frame edge, and the page itself, is a
whole number of grid units. It has no pixel density. The print master is a
vector PDF that prints on any printer, and nothing in the file or printed on
the sheet mentions a density.

A density appears in two places only, and neither is part of the board:

- **A raster output has one.** `Tiff(600dpi)` and `Png(300dpi)` are the sheet
  converted to pixels at that density (see Output, below). It is recorded
  in the JSON's `render` block, not in the board's configuration.
- **The printer enters through `blob_sizes`.** The smallest disc that prints
  well depends on the printer's density, and
  [`suggest_blob_sizes`](@ref)`(density, extent)` gives it (see Blobs, below). Only the resulting length is stored.

The grid is the print master's. Every coordinate is written to the PDF as an
exact two-decimal number of points, up to 9999.99 pt (3.53 m, the largest side
a board may have; the grid was sized for a writer that kept 6 significant
digits). Every grid value therefore reaches the PDF unchanged: the sidecar's blob centre `141.73` (pt) is the
number the PDF holds. 7200 is also the least common multiple of 300, 600, 720,
1200, 1440 and 2400 dpi, so pixel edges at those densities fall on grid lines.
The sidecar stores the pattern and its blobs in points (`"unit": "pt"`) and
declares the grid (`"layout_grid": {"value": 0.01, "unit": "pt"}`).

Boards stored in schema v3 are laid out on the printer's pixels at their own
density. They load, re-render and rebuild exactly as printed, under their
stored identity, and convert to a raster at that density only.

## Page and board size

The page (canvas) can be a paper name, a width and height, or a `Size2`. The
board can be smaller than the page; it is then centred, with white margins:

```julia
sizes = suggest_blob_sizes(600dpi, (180mm, 180mm))    # 6 mm … 36 mm: suits every board below

bundle = blob_board("Letter"; blob_sizes = sizes, dir = "./output")          # a paper name (BlobBoards.PRINT_SIZES)
bundle = blob_board(200mm, 280mm; blob_sizes = sizes, dir = "./output")      # width × height
bundle = blob_board(8inch, 10inch; blob_sizes = sizes, dir = "./output")     # inches work too
bundle = blob_board(Size2(width = 200mm, height = 280mm); blob_sizes = sizes, dir = "./output")

# a 180 mm square board centred on an A4 page
bundle = blob_board("A4"; blob_sizes = sizes,
    board_size = Size2(width = 180mm, height = 180mm), dir = "./output")
```

The layout is two nested rectangles: the canvas (the page) and the board,
centred on it — the canvas less twice the `margin` (a `Size2`: `width` left and
right, `height` above and below; or give `board_size` instead, not both), which the blobs fill around the frame's marks (see
Frame and ruler, below). By default the board leaves a **10 mm white margin** on
every side of the page: office printers cannot print the outer few millimetres
of a sheet (3–6.35 mm; the campus Ricoh 4.23 mm), and anything there — the
corner glyphs, the QR code, the ruler's ends — would be cut off. The margin
also holds the id label under the board. For a print shop that trims, set
a zero margin (full bleed). The board records its margin, and its QR code
carries it.

## Blobs

`blob_sizes = (min, max)` are the smallest and largest printed disc
**diameters**, as **physical lengths**. It is required and both ends are
stated: leaving it out, or giving `nothing` for either end, is an
`ArgumentError`. [`suggest_blob_sizes`](@ref)`(density, extent)` gives both
ends, the smallest from the printer and the largest from the board:

```julia
# a 600 dpi printer and the default A4 board (190 × 277 mm inside its 10 mm margins): 6 mm … 38 mm
bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 0x1234, dir = "./output")

# the board's extent as a Size2 or a (width, height) tuple
sizes = suggest_blob_sizes(1200dpi, (300mm, 200mm))      # (3.0 mm, 40.0 mm)

# printed discs from 2 mm to 12 mm across
bundle = blob_board("A4"; blob_sizes = (2mm, 12mm), dir = "./output")

# the max as a fraction of the pattern's short side
bundle = blob_board("A4"; blob_sizes = (4.0mm, 0.2short_side), dir = "./output")

# light blobs on a black background
bundle = blob_board("A4"; blob_sizes = (4.0mm, 0.2short_side), polarity = :light, dir = "./output")
```

**The smallest disc** is a choice about the printer, which is why it has no
default. `suggest_blob_sizes(density, extent)[1]` is the smallest diameter
that prints its profile well at a printer's density: 3 mm at 1200 dpi,
inversely proportional to the density.

| Printer | smallest disc |
|---------|---------------|
| 300 dpi | 12 mm |
| 600 dpi | 6 mm |
| 1200 dpi | 3 mm |

A printer makes grey from halftone cells, small blocks of printer dots, so the
cells grow as the density drops, and a disc must span enough of them for its
profile to print smoothly. At 600 dpi this matches a measured office laser
printer: discs from about 5–6 mm up reproduce the profile within ~10 grey
levels of 255, while smaller ones print their mid-tones too light
(`examples/print_quality`). A smaller min is accepted; the suggestion is
not a limit. The rule's two numbers are `min_blob_diameter_mm`
and `min_blob_reference_dpi` in `configs/authoring.toml`.

**The largest disc** is a length or a fraction of the pattern's short side.
`suggest_blob_sizes(density, extent)[2]` is 20% of the board's shorter
side, but never more than 64 mm: on a large sheet a fixed fraction is a very
big disc (20% of an A0 board is 164 mm). A paper name as the extent stands for
the board `blob_board` makes on it by default, the paper less 10 mm on every
side.

| Board's shorter side | largest disc |
|----------------------|--------------|
| 128 mm (A5 board) | 25.6 mm |
| 190 mm (A4 board) | 38 mm |
| 320 mm and up | 64 mm |

`0.2short_side` is instead exactly that fraction of the pattern's short side
(the board's: a board's pattern is the whole board), with no cap, and any length (`12mm`) is taken as given; the largest disc must
be smaller than the pattern's short side. The cap exists only inside
`suggest_blob_sizes`. Its two numbers are `max_blob_fraction` and
`max_blob_diameter_mm` in `configs/authoring.toml`.

The placement keywords (`scales_per_octave`, `α`, `cutoff`, `max_trials`) are
described in the reference and the [Patterns guide](patterns.md).

## Frame and ruler

```
┌──────┬───────────────────┬─────────┐
│  QR  │    North ruler    │  uid 1  │
├──────┘                   └─────────┤
│ West                          East │
│ruler    blobs fill the     ruler   │
│         whole board                │
├──────┐                   ┌─────────┤
│uid 2 │    South ruler    │  uid 3  │
└──────┴───────────────────┴─────────┘
```

The blobs are placed on the whole board. Everything else printed on it — the
frame's **marks** — is a reserved rectangle that placement keeps clear of. There
are two, and each is switched by its size, a zero length (in any unit) turning
it off:

- `corner_size` (10 mm by default): the four **corner cells** — the QR code
  top-left, a uid character in each other corner. Blobs keep 1 mm clear of them
  (more than the QR code's own quiet zone, which is inside its cell);
- `ruler_width` (10 mm by default): the four **ruler bands** along the sides,
  between the cells. Blobs may touch a band.

So `ruler_width = 0mm` keeps only the corner cells, `corner_size = 0mm` only the
ruler, and both at `0mm` give a board of blobs alone. The two are independent —
a narrow ruler with large corner cells is `corner_size = 14mm, ruler_width =
6mm` — and the blob counts are taken from the board's area less the reserved
one. The tick keywords belong to the ruler: with it off they are refused. On a
`:light` board the marks are painted white over the black background.

The QR code carries the board's definition as a short text, the **board code**
(`BlobBoards.board_code`):

```
BB4/210MM/297MM/10MM/10MM/10MM/10MM/0MM/10MM/5MM/10MM/5MM/A4/4.25196850394/26.929/3/5000/1.2/2/D/0D435510
```

— the format, page size and margin, ruler width, corner size, padding, ruler
ticks, seed (hex), the smallest and largest σ (pt), the ladder and placement
settings, polarity, and the first 8 hex of the identity as a check. A length is
a number and its unit, the unit symbol of VisualGeometryCore's FlexUnits
registry upper-cased so the code stays in QR's alphanumeric set: `210MM`, or
`8.5INCH` on Letter, whichever of mm and inch is shorter and reads back to the
identical value. The reader lower-cases the symbol and resolves it through the
registry, so any registered length unit reads (`21CM`); a number without a unit
is refused, and so is a code in the older mm/`IN` spelling. Any phone reads it
(a 41 × 41 code, its modules about 0.24 mm in the 10 mm corner);
`BlobBoards.parse_board_code` turns it back into the board's configuration,
refusing a code whose recomputed identity disagrees with the check, so a print
alone rebuilds its board. A board's identity covers its schema, its
configuration and the blobs that configuration places, so the check also proves
the rebuild placed the same blobs: a code decoded by software that places
differently is refused, never rebuilt into another board. `BB4` is the board
schema's version; a change to placement or the frame's marks is a new schema. If the QR code does not fit its corner, it is omitted
with a warning. The ruler's ticks sit at exact multiples of their spacing (to
the 0.01 pt grid; within 3 µm in the print master) and are drawn 0.1 mm wide.

```julia
# both marks (the default), a wider ruler, ticks every 10 mm / 5 mm
bundle = blob_board("A4"; blob_sizes = (4.0mm, 0.2short_side), ruler_width = 15mm,
    major_tick_spacing = 10mm, minor_tick_spacing = 5mm, dir = "./output")

# a narrow ruler and large corner cells
bundle = blob_board("A4"; blob_sizes = (4.0mm, 0.2short_side), corner_size = 14mm, ruler_width = 6mm, dir = "./output")

# no ruler: blobs fill the board around the four corner cells (QR, uid)
bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), ruler_width = 0mm, dir = "./output")

# no marks at all: blobs fill the whole board
bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), corner_size = 0mm, ruler_width = 0mm, dir = "./output")
```

`frame_padding` adds white space between a mark's content (ticks, QR code,
uid glyphs) and its edges. A stored v3 board keeps the layout it was printed
with: a band around an inset pattern, the corner cells where the bands cross
(its `:none` is a blank band).

### Ruler widths

`major_tick_width` and `minor_tick_width` are each tick's **full drawn length
across the band**, centred in it. By default the major ticks fill the padded band
(`major_tick_width = ruler_width - 2frame_padding`) and the minor ticks are half
as long. Widths must satisfy `0 < minor_tick_width ≤ major_tick_width ≤
ruler_width - 2frame_padding`; anything else is an `ArgumentError`.

```julia
# 15 mm band; major ticks 8 mm long, minor ticks 4 mm, centred in the band
bundle = blob_board("A4"; blob_sizes = (4.0mm, 0.2short_side), ruler_width = 15mm,
    major_tick_width = 8mm, minor_tick_width = 4mm, dir = "./output")
```

## Output

The `output` keyword says what the board is written as. It is a value that
carries exactly the options its format has:

| `output` | File | Options | Density |
|----------|------|---------|---------|
| [`Pdf`](@ref)`()` (default) | the print master, PDF/X-4 | `shading_function = :calculator`, `icc_profile = :linear` | none |
| [`Tiff`](@ref)`(density)` | the sheet converted to an 8-bit grey TIFF | `icc_profile = :linear` | the image's, e.g. `Tiff(600dpi)` |
| [`Png`](@ref)`(density)` | the sheet converted to a grey PNG, for digital use | — | the image's, e.g. `Png(300dpi)` |

```julia
sizes = suggest_blob_sizes(600dpi, "A4")
bundle = blob_board("A4"; blob_sizes = sizes, dir = "./output")                             # Pdf()
bundle = blob_board("A4"; blob_sizes = sizes, dir = "./output", output = Pdf(shading_function = :sampled))
bundle = blob_board("A4"; blob_sizes = sizes, dir = "./output", output = Tiff(600dpi))
bundle = blob_board("A4"; blob_sizes = sizes, dir = "./output", output = Png(300dpi))
```

The output is checked before placement, and `output` without a `dir` is an
`ArgumentError`. A raster's density is pixels per inch of the image, a whole
number. Three rules apply to a raster and none to a PDF:

- the board's smallest blob must be at least 2 pixels per σ at the density
  (the error names the density that would do);
- a stored v3 board converts at its own density only;
- a sheet over 32745 px a side cannot be converted.

### PDF (default)

```julia
bundle = blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), dir = "./output")   # PDF/X-4
```

- always **PDF/X-4** (what most print shops ask for). A PDF/X-4 master is
  also **PDF/A-2b**, so [veraPDF](https://verapdf.org) (which has no PDF/X
  profile) can validate it: `verapdf --flavour 2b board.pdf`. For the PDF/X
  side itself, ask the print shop to preflight the file (most do, with callas
  pdfToolbox or Enfocus PitStop).
- an OutputIntent declaring the print condition (linear-reflectance grey by
  default; see ICC profiles below) and page boxes: the TrimBox is the
  finished sheet (the canvas: `"A4"` is 210 × 297 mm), and with
  `Pdf(bleed = 2mm)` the page (MediaBox = BleedBox) is 2 mm larger on every
  side, a white strip the shop cuts off (see Print-shop bleed below)
- the blobs as **vector shadings** (below)
- vector rulers, and the text (the uid characters in the corners, the id
  label) set in one embedded font, so the page is searchable and the label's
  hash can be copied; the file holds no image

BlobBoards writes the file directly: one page with a Flate-compressed content
stream, and one shared radial shading painted once per blob.

#### Blob encoding

A board's blobs are always vector graphics — its one source of truth: one
radial shading per blob, whose colour function is the blob profile (the
Gaussian truncated at the cutoff disc and normalised to 0 at the centre and 1
at the rim). `Pdf(; shading_function)` picks how that function is written:

| `shading_function` | The profile is | Use |
|---|---|---|
| `:calculator` (default) | a PostScript calculator function (the exact formula) | everything; the printer evaluates the Gaussian at its own resolution |
| `:sampled` | a 1024-entry sampled table | a RIP that mishandles calculator functions |

A vector board is small at any size — the file grows with the number of blobs,
not the area (a 1 m² board: ~0.25 MB, against ~83 MB as a 1200 dpi raster) —
and has no size limit. Every raster of a board (TIFF, PNG, the
detection gallery's images) is a conversion of this PDF; `blob_pattern`, which
makes pattern images directly, samples the same profile at pixel centres, and
the two agree to within one 8-bit level.

How it looks on screen depends on the viewer. A renderer that evaluates the
shading per pixel (Cairo — `pdftocairo`, which `rasterize_pdf` uses) reproduces
the profile within one grey level at every size; Ghostscript, MuPDF and
poppler's own renderer draw shadings in ~1-device-pixel bands, which steps
blobs only a few pixels wide (σ = 2 px: tens of levels) and is invisible for
large ones (σ ≥ 47 px: ≤ 5 levels). What the printer does is the RIP's: check
a test print.

TIFF and PNG are the PDF converted at their own density, so `Tiff` and `Png`
have no `shading_function`. The function written is recorded in the JSON's
`render` block, and a reconstruction reproduces it; a JSON recording none
rebuilds as `:calculator`.

### TIFF

For print shops and photo labs that take raster files (a print output only: the
loader never reads a TIFF, and renders a board kept as one from its JSON):

```julia
bundle = blob_board("A4"; blob_sizes = (3mm, 0.2short_side), output = Tiff(1200dpi), dir = "./output")
```

- the whole sheet converted from the vector PDF at the TIFF's density, 8-bit
  grey: the page rounded up to whole pixels, so a pixel coordinate is the
  page's points × dpi/72 (blob centres are not pixel centres)
- the density in the resolution tags, so the print software sizes it exactly
- the ICC profile embedded (`Tiff(density; icc_profile)`, linear reflectance by default)
- LZW compression with the horizontal predictor

### PNG

For digital use (rendering, training):

```julia
bundle = blob_board("A4"; blob_sizes = (12mm, 0.2short_side), output = Png(300dpi), dir = "./output")
```

The PNG is single-channel grey with linear gamma (`file_gamma = 1.0`) and
carries no ICC profile.

### ICC profiles

A PDF/X file's OutputIntent says how its grey values are to be printed. The
default, `:linear`, declares them **linear reflectance**: a value of 0.5 reflects
half the light of bare paper, so a Gaussian in the file is a Gaussian in the light
the camera sees. A colour-managed RIP (print shops) honours it; an office printer
ignores it and applies its own tone curve.

```julia
sizes = suggest_blob_sizes(600dpi, "A4")
bundle = blob_board("A4"; blob_sizes = sizes, output = Pdf(icc_profile = :linear), dir = "./output")   # the default
bundle = blob_board("A4"; blob_sizes = sizes, output = Pdf(icc_profile = "/path/to/press_gray.icc"), dir = "./output")
```

`Tiff(density; icc_profile)` embeds the same profile; a PNG has none.

## Identity Label

Every new board prints its identity in one 6 pt line under the board, in the
sheet margin (inside the trimmed sheet, clear of the printer's unprintable border):

```
blob_board_0d4_35510 · BB4/210MM/297MM/10MM/10MM/10MM/10MM/0MM/10MM/5MM/10MM/5MM/A4/4.25196850394/26.929/3/5000/1.2/2/D/0D435510 · BlobBoards 1.1.2
```

The middle part is the board code, the same string the QR code holds:
`parse_board_code` rebuilds the board from it. It is text in the PDF, so it can
be searched for and copied. The label mentions no density. (A stored v3
board's label keeps its hash, size and density.)

Every printed sheet — frameless boards included — can so be traced to its JSON.
The label is a rendering option, not part of the board's identity: `id_label = false`
turns it off without changing the UID. A board that fills its sheet has no room
for it, and the label is skipped (with a warning for a frameless board, which
would otherwise carry no printed identity). The JSON's `render` block records
whether it was drawn: `BlobBoards.rebuild_board` and the gallery's regeneration rebuild
the sheet as recorded, and a board with no `render` block (written before
0.36.2) rebuilds without one. `render_board` defaults to `false`.

## Printing

A board is a measuring instrument, so it must be printed at **exactly 100%**.

- In the print dialog choose **"Actual size" / 100%**. Turn off "Fit", "Shrink
  oversized pages" and "Scale to printable area". An A4 page is larger than any
  printer's printable area, so a dialog left on "Fit" shrinks the whole board
  by about 3%. PDF/X-4 output asks Acrobat/Reader for actual size
  (`/PrintScaling /None`); other viewers may ignore it.
- If your printer cannot print to the page edge and keeps scaling, make the
  board smaller than the page: a wider `margin` leaves more white, so the
  page prints at 100% without anything being clipped.
- Print **grey / black ink only**, at the highest quality, with image
  enhancement off.
- **Check every print.** Measure the rulers: the major ticks are
  `major_tick_spacing` apart, so at the default spacing ten intervals must
  span exactly 100.0 mm. On a frameless board, measure the pattern edge to
  edge. A "fit" print comes out about 3% short.
- Print shops print PDF/X at 100% by default and cut to the TrimBox, the
  finished sheet.

### Print-shop bleed

A print shop that trims its sheets asks for a trim allowance ("2 mm bleed on
every side"). Give it to the PDF, not to the board:

```julia
blob_board("A4"; blob_sizes = suggest_blob_sizes(1200dpi, "A4"), output = Pdf(bleed = 2mm), dir = "./out")
```

The page is then 214 × 301 mm (MediaBox = BleedBox), the TrimBox the A4 sheet
2 mm in from every edge, and the board where it is on a plain A4 sheet: 10 mm
`margin` from the trim. The bleed strip is white. Bleed is a rendering option,
like `id_label`: the board, its uid, QR code and JSON definition are those of
the same board without bleed, and the JSON's `render` block records it, so
`rebuild_board` prints the same sheet. Do NOT enlarge the canvas and the margin
yourself instead: the TrimBox would land on the larger sheet and the board
would get a different uid.

## Reproducing a board

The JSON sidecar is the board's definition. There are three ways to use it.

```julia
# 1. Load the bundle (and its image): the input to detection and coordinates
bundle, image = BlobBoards.load("./output/blob_board_<uid>.json")

# 2. Re-render the stored board (its recorded blobs, frame, QR and uid);
#    no placement is re-run, so it matches the printed board exactly.
#    The output is the first argument; `dir` is required.
BlobBoards.render_board(bundle; dir = "./rerender", output = Png(300dpi))
BlobBoards.render_board(bundle; dir = "./rerender")                        # Pdf() by default

# 3. Re-generate from the configuration (placement is re-run from the seed)
bundle = BlobBoards.rebuild_board("./output/blob_board_<uid>.json"; dir = "./output2")
```

Form 3 writes the board as its JSON's `render` block records (`output` and
`id_label`); a keyword overrides it, e.g. `output = Tiff(600dpi)`.

## Reading a board back

```julia
bundle, image = BlobBoards.load("./output/blob_board_<uid>.json")

blobs(bundle, :pattern; units = mm)                  # mm, pattern frame
blobs(bundle, :board; units = mm)                    # mm, board frame (includes the frame band)
blobs(bundle, :canvas; density = 300dpi)       # px of a 300 dpi conversion (COLMAP convention)
blobs(bundle, :world)                                # centred, y-up, unitless mm: the z = 0 calibration plane

board_config(bundle).ruler_width     # the BoardConfig
pattern_config(bundle).seed          # the PatternConfig
```

A board has no density, so `units = px` (the default) needs the density of
the pixels you ask for, `density`; without it `blobs` is an
`ArgumentError`. See
[Coordinate Systems](coordinates.md) for the reference frames and transforms.

## Full keyword reference

All method forms and keywords, with defaults and valid values, are in the
[`blob_board`](@ref) docstring. Paper names are the keys of
[`BlobBoards.PRINT_SIZES`](@ref).
