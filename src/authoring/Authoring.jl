# =============================================================================
# BlobBoards.Authoring — board AUTHORING: layout, frame, PDF/X-4 writer, rasters
# =============================================================================
#
# Everything needed to *make* physical boards: board assembly (`blob_board`,
# `rebuild_board`, `render_board`), the ruler/QR/uid/label geometry, the PDF/X-4
# writer (Flate via Zlib_jll, the sheet font through FreeTypeAbstraction) and
# its TIFF (libtiff) and PNG conversions. Part of the package since 2026-10-14
# (an extension triggered by `using CairoMakie` before): nothing draws with
# Makie or Cairo, and the sheet font, which Makie used to supply, ships in
# assets/fonts (VENDORED.toml).

module Authoring

using FreeTypeAbstraction: FreeTypeAbstraction, FTFont
using FreeType: FT_Get_Postscript_Name
using Zlib_jll: libz
using Libtiff_jll: libtiff
using SHA: sha256
using Dates: Dates

using ..BlobBoards
using ..BlobBoards: BoardPreamble, BoardConfig, BoardBundle, BoardOutput,
    Size2, make_filebase, save_json,
    PatternConfig, BoardLayout,
    # the layout grids (types.jl)
    LayoutGrid, PointGrid, PixelGrid, stored_density,
    # the ruler vocabulary
    hruler, vruler, uid_from_hash,
    # the outputs a board is written as (board_output.jl)
    Pdf, Tiff, Png
import ..BlobBoards: blob_board, render_board, rebuild_board, write_tiff
using VisualGeometryCore: filter_kwargs, px, validate_dir
using FlexUnits: Quantity, dimension
using GeometryBasics: GeometryBasics
using CoordinateTransformations: Translation
using JSON3: JSON3
using FileIO: FileIO
using VisualGeometryCore: ustrip, mm, pt
using VisualGeometryCore: dpi

# The QR encoder of QRCoders.jl 1.4.5, vendored (qrencode/, VENDORED.toml):
# only the board QR corner needs it.
include("qrencode/QREncode.jl")
using .QREncode: QREncode, qrcode

include("pdf.jl")
include("glyphs.jl")
include("sheet.jl")
include("vector_blobs.jl")
include("board.jl")
include("io_save.jl")
include("tiff.jl")

end # module Authoring
