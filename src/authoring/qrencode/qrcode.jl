# Generate QR code matrices — the two `qrcode` methods of QRCoders 1.4.5's
# src/export.jl, verbatim; the image/GIF/SVG export functions are not vendored.

"""
    qrcode( message::AbstractString
          ; eclevel::ErrCorrLevel = Medium()
          , version::Int = 0
          , mode::Mode = Numeric()
          , mask::Int = -1
          , width::Int=0)

Create a `BitArray{2}` with the encoded `message`, with `true` (`1`) for the black
areas and `false` (`0`) as the white ones.

The error correction level `eclevel` can be picked from four values: `Low()`
(7% of missing data can be restored), `Medium()` (15%), `Quartile()` (25%) or
`High()` (30%). Higher levels make denser QR codes.

The version of the QR code can be picked from 1 to 40. If the assigned version is 
too small to contain the message, the first available version is used.

The encoding mode `mode` can be picked from five values: `Numeric()`, `Alphanumeric()`,
`Byte()`, `Kanji()` or `UTF8()`. If the assigned `mode` is `nothing` or failed to contain the message,
the mode is automatically picked.

The mask pattern `mask` can be picked from 0 to 7. If the assigned `mask` is `nothing`,
the mask pattern will picked by the penalty rules.
"""
function qrcode( message::AbstractString
               ; eclevel::ErrCorrLevel = Medium()
               , version::Int = 0
               , mode::Mode = Numeric()
               , mask::Int = -1
               , compact::Bool = false
               , width::Int = 0)
    isempty(message) && @warn(
        "Most QR code scanners don't support empty message!")
    # Determining mode and version of the QR code
    bestmode = getmode(message)
    mode = bestmode ⊆ mode ? mode : bestmode
    
    version > 40 && throw(EncodeError("Version $version should be no larger than 40"))
    minversion = getversion(message, mode, eclevel)
    if version < minversion # the specified version is too small
        version = minversion
    end

    # encode message
    data = encodemessage(message, mode, eclevel, version)

    # Generate qr code matrix
    matrix = emptymatrix(version)
    masks = makemasks(matrix) # 8 masks
    matrix = placedata!(matrix, data) # fill in data bits
    addversion!(matrix, version) # fill in version bits

    # Apply mask and add format information
    maskedmats = [addformat!(xor.(matrix, mat), i-1, eclevel) 
                  for (i, mat) in enumerate(masks)]
    
    # Pick the best mask
    if !(0 ≤ mask ≤ 7) # invalid mask
        mask = argmin(penalty.(maskedmats)) - 1
    end
    matrix = maskedmats[mask + 1]

    # white border
    (compact || width == 0) && return matrix # keyword compact will be removed in the future
    return addborder(matrix, width)
end

"""
    qrcode(code::QRCode)

Create a QR code matrix by the `QRCode` object.

Note: It would raise an error if failed to use the specified `mode`` or `version`.
"""
function qrcode(code::QRCode)
    # raise error if failed to use the specified mode or version
    mode, eclevel, version, mask = code.mode, code.eclevel, code.version, code.mask
    message, width = code.message, code.border
    getmode(message) ⊆ mode || throw(EncodeError("Mode $mode can not encode the message"))
    getversion(message, mode, eclevel) ≤ version ≤ 40 || throw(EncodeError("The version $version is too small"))

    # encode message
    data = encodemessage(message, mode, eclevel, version)

    # Generate qr code matrix
    matrix = emptymatrix(version)
    maskmat = makemask(matrix, mask)
    matrix = placedata!(matrix, data) # fill in data bits
    addversion!(matrix, version) # fill in version bits
    matrix = addformat!(xor.(matrix, maskmat), mask, eclevel)

    # white border
    addborder(matrix, width)
end
