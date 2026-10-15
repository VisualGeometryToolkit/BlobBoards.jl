# QREncode — the QR encoder of QRCoders.jl 1.4.5, vendored.
#
# Upstream: https://github.com/JuliaImages/QRCoders.jl, release 1.4.5 (source
# tree b3e5fcc7a7ade2d43f0ffd178c299b7a264c268a), MIT licence, Copyright (c)
# 2019 Jérémie Gillet and contributors — see LICENSE beside this file and the
# entry in the repository's VENDORED.toml.
#
# Only the encoder: message → BitMatrix. QRCoders' image, GIF, SVG and
# UnicodePlots exports and its styled codes are not vendored, which is what
# drops its ImageCore/FileIO/StatsBase/UnicodePlots dependencies (whose old
# compat bounds kept the General release from resolving with this stack).
# Local changes: the files are flattened into one directory; this module file
# replaces src/QRCoders.jl; qrcode.jl holds the two `qrcode` methods of
# src/export.jl, verbatim; types.jl drops `Base.show(io, ::QRCode)`. The
# encoding code is unchanged, so the matrices are bit-identical to QRCoders'
# (BlobBoards' fork of it, 1.4.6, changed only the export layer).

module QREncode

export qrcode, QRCode, Mode, Numeric, Alphanumeric, Byte, Kanji, UTF8,
    ErrCorrLevel, Low, Medium, Quartile, High, EncodeError

include("types.jl")           # Mode, ErrCorrLevel, EncodeError, QRCode
include("tables.jl")          # the specification's tables (+ kanji.jl)
include("polynomial.jl")      # Galois-field polynomial arithmetic (Reed–Solomon)
using .Polynomial
include("matrix.jl")          # finder/alignment patterns, data placement, masks, penalty
include("encode.jl")          # mode/version selection, message → codewords
include("qrcode.jl")          # qrcode(message; eclevel, …) → BitMatrix

end # module QREncode
