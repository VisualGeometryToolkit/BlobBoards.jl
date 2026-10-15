# The board TIFF, written by libtiff — the reference TIFF library the print RIPs
# read with (Libtiff_jll, which Poppler_jll loads anyway). TiffImages.jl was not used: it decodes LZW/Deflate but writes only
# uncompressed strips (139 MB for a 20 cm board at 1200 dpi).


# libtiff tag numbers and values (tiff.h)
const TIFFTAG_IMAGEWIDTH = UInt32(256)
const TIFFTAG_IMAGELENGTH = UInt32(257)
const TIFFTAG_BITSPERSAMPLE = UInt32(258)
const TIFFTAG_COMPRESSION = UInt32(259)
const TIFFTAG_PHOTOMETRIC = UInt32(262)
const TIFFTAG_SAMPLESPERPIXEL = UInt32(277)
const TIFFTAG_ROWSPERSTRIP = UInt32(278)
const TIFFTAG_XRESOLUTION = UInt32(282)
const TIFFTAG_YRESOLUTION = UInt32(283)
const TIFFTAG_PLANARCONFIG = UInt32(284)
const TIFFTAG_RESOLUTIONUNIT = UInt32(296)
const TIFFTAG_SOFTWARE = UInt32(305)
const TIFFTAG_PREDICTOR = UInt32(317)
const TIFFTAG_ICCPROFILE = UInt32(34675)
const COMPRESSION_LZW = Cint(5)
const PHOTOMETRIC_MINISBLACK = Cint(1)
const PLANARCONFIG_CONTIG = Cint(1)
const RESUNIT_INCH = Cint(2)
const PREDICTOR_HORIZONTAL = Cint(2)

# TIFFSetField is variadic: integer tags take an int (default promotion of
# uint16/uint32), the resolution tags a double, the ICC profile (count, bytes).
_set(tif, tag, v::Integer) = @ccall libtiff.TIFFSetField(tif::Ptr{Cvoid}, tag::UInt32; v::Cint)::Cint
_set(tif, tag, v::Float64) = @ccall libtiff.TIFFSetField(tif::Ptr{Cvoid}, tag::UInt32; v::Cdouble)::Cint
_set(tif, tag, v::String) = @ccall libtiff.TIFFSetField(tif::Ptr{Cvoid}, tag::UInt32; v::Cstring)::Cint
_set(tif, tag, v::Vector{UInt8}) =
    @ccall libtiff.TIFFSetField(tif::Ptr{Cvoid}, tag::UInt32; length(v)::UInt32, v::Ptr{UInt8})::Cint

function write_tiff(path::AbstractString, grey::AbstractMatrix{UInt8}, density::Real, icc::Vector{UInt8})
    height, width = size(grey)
    tif = @ccall libtiff.TIFFOpen(path::Cstring, "w"::Cstring)::Ptr{Cvoid}
    tif == C_NULL && error("write_tiff: libtiff could not open $path for writing")
    try
        for (tag, v) in (
                (TIFFTAG_IMAGEWIDTH, width), (TIFFTAG_IMAGELENGTH, height),
                (TIFFTAG_BITSPERSAMPLE, 8), (TIFFTAG_SAMPLESPERPIXEL, 1),
                (TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK), (TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG),
                # LZW, the compression every prepress RIP reads; the horizontal
                # predictor (row differences) is what makes smooth Gaussians compress
                (TIFFTAG_COMPRESSION, COMPRESSION_LZW), (TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL),
                (TIFFTAG_ROWSPERSTRIP, 64),
                (TIFFTAG_XRESOLUTION, Float64(density)), (TIFFTAG_YRESOLUTION, Float64(density)),
                (TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH),
                (TIFFTAG_SOFTWARE, "BlobBoards.jl $(pkgversion(BlobBoards))"),
                (TIFFTAG_ICCPROFILE, icc),
            )
            _set(tif, tag, v) == 1 || error("write_tiff: libtiff rejected tag $tag = $(repr(v))")
        end
        # libtiff takes rows (scanlines); a Julia matrix is column-major, so a
        # transposed copy makes each image row one contiguous column.
        rows = permutedims(grey)
        GC.@preserve rows for r in 1:height
            line = pointer(rows, (r - 1) * width + 1)
            @ccall(libtiff.TIFFWriteScanline(tif::Ptr{Cvoid}, line::Ptr{UInt8}, (r - 1)::UInt32, 0::UInt16)::Cint) == 1 ||
                error("write_tiff: libtiff failed writing row $r of $path")
        end
        # TIFFClose returns nothing, yet it writes the last strip and the
        # directory: flush first so a full disk or quota is an error, not a TIFF
        # with no directory (the caller deletes the partial file).
        @ccall(libtiff.TIFFFlush(tif::Ptr{Cvoid})::Cint) == 1 ||
            error("write_tiff: libtiff failed to flush $path (disk full?)")
    finally
        @ccall libtiff.TIFFClose(tif::Ptr{Cvoid})::Cvoid
    end
    return path
end

