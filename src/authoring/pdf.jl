# =============================================================================
# The PDF file — objects, streams, cross-reference table, PDF/X-4 + PDF/A-2b
# =============================================================================
#
# The print master is written here, byte for byte: no Cairo, no Ghostscript, no
# qpdf. A board's sheet is a handful of objects — one page, its content stream,
# one shared radial shading and its colour function, the OutputIntents' ICC
# profile, the XMP packet, the font its text is set in — so every number in the file is the one the layout
# computed, and the object syntax is strict as written (an end-of-line before
# every `endobj` and `endstream`, ISO 19005 6.1.7.1 / 6.1.9).

# --- numbers ------------------------------------------------------------------

# A real as a PDF number: ≤ 10 significant digits, fixed notation (PDF has no
# exponent form), no trailing zeros. A value on a board's grid comes out as the
# decimal it is (124.68 → "124.68").
function _pdf_num(x::Real)
    v = round(Float64(x); sigdigits = 10)
    iszero(v) && return "0"
    isfinite(v) || throw(ArgumentError("not a PDF number: $x"))
    decimals = clamp(9 - floor(Int, log10(abs(v))), 0, 15)
    s = Base.Ryu.writefixed(v, decimals)
    occursin('.', s) && (s = rstrip(rstrip(s, '0'), '.'))
    return s == "-0" ? "0" : String(s)
end

# k hundredths of a point as a decimal string, exactly: 14173 → "141.73".
function _hundredths(k::Integer)
    a = abs(k)
    return string(k < 0 ? "-" : "", div(a, 100), ".", lpad(rem(a, 100), 2, '0'))
end

# A text string: ASCII as a literal, `(…)`, its delimiters escaped; anything
# else (a custom ICC profile's file name, a title) as UTF-16BE with its byte
# order mark, in hex, `<FEFF…>` — the PDF's own form for Unicode text.
function _pdf_string(s::AbstractString)
    isascii(s) && return "(" * replace(s, "\\" => "\\\\", "(" => "\\(", ")" => "\\)") * ")"
    units = transcode(UInt16, String(s))
    return "<FEFF" * join(uppercase(string(u; base = 16, pad = 4)) for u in units) * ">"
end

# --- Flate --------------------------------------------------------------------

# zlib's one-call deflate, at its best level: the streams are lossless either
# way, so the level is only the file's size.
function _deflate(data::Vector{UInt8})
    bound = ccall((:compressBound, libz), Culong, (Culong,), length(data))
    out = Vector{UInt8}(undef, bound)
    len = Ref{Culong}(bound)
    rc = ccall((:compress2, libz), Cint, (Ptr{UInt8}, Ref{Culong}, Ptr{UInt8}, Culong, Cint), out, len, data, length(data), 9)
    rc == 0 || error("zlib compress2 failed ($rc)")
    return resize!(out, len[])
end

# Its inverse, for a reader that must see a stream's text (the suite's checks).
function _inflate(data::AbstractVector{UInt8})
    src = Vector{UInt8}(data)
    cap = max(4 * length(src), 1024)
    while true
        out = Vector{UInt8}(undef, cap)
        len = Ref{Culong}(cap)
        rc = ccall((:uncompress, libz), Cint, (Ptr{UInt8}, Ref{Culong}, Ptr{UInt8}, Culong), out, len, src, length(src))
        rc == 0 && return resize!(out, len[])
        rc == -5 || error("zlib uncompress failed ($rc)")      # Z_BUF_ERROR: the output is larger
        cap *= 4
    end
    return
end

# --- objects ------------------------------------------------------------------

# One indirect object: its dictionary's entries (without `<<` `>>`) and, for a
# stream, its data as stored. A stream's `/Length` (and `/Filter`) are written
# by `_stream`, never by the caller.
struct PdfObject
    dict::String
    data::Union{Nothing, Vector{UInt8}}
end
PdfObject(dict::AbstractString) = PdfObject(String(dict), nothing)

# A stream object: Flate-compressed unless `compress = false` (the XMP packet,
# which PDF/A wants readable without a filter).
function _stream(dict::AbstractString, data::Vector{UInt8}; compress::Bool = true)
    stored = compress ? _deflate(data) : data
    filter = compress ? " /Filter /FlateDecode" : ""
    return PdfObject(strip(String(dict) * filter * " /Length $(length(stored))"), stored)
end
_stream(dict::AbstractString, data::AbstractString; kwargs...) = _stream(dict, Vector{UInt8}(codeunits(data)); kwargs...)

# The file: header, the objects numbered in order from 1, a classic
# cross-reference table (no object or xref streams: readable by any RIP) and
# the trailer. The file identifier is the body's own hash, so the same sheet
# written at the same instant is the same file.
function _pdf_file(objects::Vector{PdfObject}; root::Int, info::Int, version::AbstractString)
    io = IOBuffer()
    # the comment's four bytes > 127 mark the file as binary (ISO 19005 6.1.2)
    write(io, "%PDF-$version\n%", UInt8[0xe2, 0xe3, 0xcf, 0xd3], "\n")
    offsets = Int[]
    for (n, o) in enumerate(objects)
        push!(offsets, position(io))
        write(io, "$n 0 obj\n<< ", o.dict, " >>\n")
        o.data === nothing || write(io, "stream\n", o.data, "\nendstream\n")
        write(io, "endobj\n")
    end
    id = bytes2hex(sha256(take!(copy(io)))[1:16])
    xref = position(io)
    write(io, "xref\n0 $(length(objects) + 1)\n0000000000 65535 f \n")
    for off in offsets
        write(io, lpad(off, 10, '0'), " 00000 n \n")
    end
    write(io, "trailer\n<< /Size $(length(objects) + 1) /Root $root 0 R /Info $info 0 R /ID [ <$id> <$id> ] >>\n")
    write(io, "startxref\n$xref\n%%EOF\n")
    return take!(io)
end

# --- the print condition and the metadata ---------------------------------------

# The OutputIntent's profile: `:linear` (default) declares the grey values linear
# reflectance — a Gaussian in the file is a Gaussian in reflected light, which is
# what the detector assumes. It is an ICC v2 monochrome OUTPUT ('prtr') profile,
# the form PDF/X accepts (generated by dev/make_linear_gray_icc.py; it
# replaced a v4 monitor profile, which preflight rejects as an OutputIntent). A
# path supplies any other single-channel profile. `icc` is a spec's checked
# profile (`_icc_path`): `nothing` for the built-in one. Returns (path, condition).
function _icc_profile(icc::Union{Nothing, String})
    icc === nothing || return icc, splitext(basename(icc))[1]
    pkg_dir = pkgdir(@__MODULE__)
    pkg_dir === nothing && error("BlobBoards: cannot locate the package directory for the built-in ICC profile")
    return joinpath(pkg_dir, "assets", "Gray-linear-v2-prtr.icc"), "Linear reflectance grayscale (gamma 1.0)"
end

# One OutputIntent of subtype `S` (`GTS_PDFX`, `GTS_PDFA1`) on the profile object `icc`.
_output_intent(S, condition, icc::Int) =
    "/Type /OutputIntent /S /$S /OutputCondition $(_pdf_string(condition)) /OutputConditionIdentifier (Custom) " *
    "/RegistryName (http://www.color.org) /Info $(_pdf_string(condition)) /DestOutputProfile $icc 0 R"

_xml_text(s::AbstractString) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

# One PDF/A extension-schema entry describing one Text property.
_pdfa_schema(name, uri, prefix, property, description) =
    "<rdf:li rdf:parseType=\"Resource\"><pdfaSchema:schema>$(name)</pdfaSchema:schema>" *
    "<pdfaSchema:namespaceURI>$(uri)</pdfaSchema:namespaceURI><pdfaSchema:prefix>$(prefix)</pdfaSchema:prefix>" *
    "<pdfaSchema:property><rdf:Seq><rdf:li rdf:parseType=\"Resource\">" *
    "<pdfaProperty:name>$(property)</pdfaProperty:name><pdfaProperty:valueType>Text</pdfaProperty:valueType>" *
    "<pdfaProperty:category>internal</pdfaProperty:category><pdfaProperty:description>$(description)</pdfaProperty:description>" *
    "</rdf:li></rdf:Seq></pdfaSchema:property></rdf:li>"

# The XMP packet. PDF/X-4 is identified here (`pdfxid:GTS_PDFXVersion`), with
# the title, the dates, the document id and the trapping state. A PDF/X-4
# master is ALSO PDF/A-2b (dual conformance, ISO 19005-2 level b: the
# appearance), so veraPDF — the PDF Association's open validator, which has no
# PDF/X profile — can check the file end to end: PDF/A adds its own
# identification and schema descriptions for the two properties PDF/X puts in
# the XMP that PDF/A does not predefine (`pdfxid:GTS_PDFXVersion`, `pdf:Trapped`).
function _xmp(; title, creator, producer, date, document_id)
    return """
    <?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">\
    <pdf:Producer>$(_xml_text(producer))</pdf:Producer><pdf:Trapped>False</pdf:Trapped></rdf:Description>
    <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">\
    <xmp:CreateDate>$date</xmp:CreateDate><xmp:ModifyDate>$date</xmp:ModifyDate><xmp:MetadataDate>$date</xmp:MetadataDate>\
    <xmp:CreatorTool>$(_xml_text(creator))</xmp:CreatorTool></rdf:Description>
    <rdf:Description rdf:about="" xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/">\
    <xmpMM:DocumentID>uuid:$document_id</xmpMM:DocumentID><xmpMM:RenditionClass>default</xmpMM:RenditionClass>\
    <xmpMM:VersionID>1</xmpMM:VersionID></rdf:Description>
    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:format>application/pdf</dc:format>\
    <dc:title><rdf:Alt><rdf:li xml:lang="x-default">$(_xml_text(title))</rdf:li></rdf:Alt></dc:title></rdf:Description>
    <rdf:Description rdf:about="" xmlns:pdfxid="http://www.npes.org/pdfx/ns/id/">\
    <pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion></rdf:Description>
    <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">\
    <pdfaid:part>2</pdfaid:part><pdfaid:conformance>B</pdfaid:conformance></rdf:Description>
    <rdf:Description rdf:about="" xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/" \
    xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#" xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#">
    <pdfaExtension:schemas><rdf:Bag>
    $(_pdfa_schema("PDF/X ID Schema", "http://www.npes.org/pdfx/ns/id/", "pdfxid", "GTS_PDFXVersion", "the PDF/X version"))
    $(_pdfa_schema("Adobe PDF Schema", "http://ns.adobe.com/pdf/1.3/", "pdf", "Trapped", "whether the document is trapped"))
    </rdf:Bag></pdfaExtension:schemas></rdf:Description>
    </rdf:RDF>
    </x:xmpmeta>
    <?xpacket end="w"?>"""
end

# The instant of writing, as the Info dictionary and the XMP each spell it
# (local time with its offset): ("D:20261013214124+02'00'", "2026-10-13T21:41:24+02:00").
# With `SOURCE_DATE_EPOCH` set (the reproducible-builds convention: seconds
# since 1970, UTC) that instant instead, in UTC — two runs then write
# byte-identical files (the document ID is a hash of the content already).
function _pdf_dates()
    e = strip(get(ENV, "SOURCE_DATE_EPOCH", ""))
    isempty(e) && return _pdf_dates(time())
    t = tryparse(Int, e)
    t === nothing && throw(ArgumentError("SOURCE_DATE_EPOCH must be a whole number of seconds since 1970, got $(repr(e))"))
    d = Dates.unix2datetime(t)
    return ("D:" * Dates.format(d, "yyyymmddHHMMSS") * "Z", Dates.format(d, "yyyy-mm-ddTHH:MM:SS") * "Z")
end
function _pdf_dates(t::Real)
    # the zone's offset from the local time itself: `%z` is the zone's NAME on Windows
    s = floor(Int, t)
    lt = Libc.TmStruct(s)
    here = Dates.DateTime(lt.year + 1900, lt.month + 1, lt.mday, lt.hour, lt.min, lt.sec)
    offset = round(Int, Dates.value(here - Dates.unix2datetime(s)) / 60_000)   # minutes east of UTC
    hours, minutes = lpad(abs(offset) ÷ 60, 2, '0'), lpad(abs(offset) % 60, 2, '0')
    sign = offset < 0 ? "-" : "+"
    return (
        "D:" * Dates.format(here, "yyyymmddHHMMSS") * sign * hours * "'" * minutes * "'",
        Dates.format(here, "yyyy-mm-ddTHH:MM:SS") * sign * hours * ":" * minutes,
    )
end

# --- the embedded font ------------------------------------------------------------

# The sheet font, embedded whole, as the five
# objects a composite font takes, numbered from `first`: the Type 0 font the
# page names (`/F`), its CIDFont, the descriptor, the font program and the
# ToUnicode map. The program is the OpenType/CFF file as it is (not subset:
# subsetting CFF is the one hard part of font embedding, and the whole file
# is ~100 kB deflated); with Identity-H a text code is a glyph index. Widths
# and the Unicode map list exactly `chars`, what the sheet sets.
function _font_objects(font::FTFont, chars::Vector{Char}, first::Int)
    type0, cid, descriptor, program, tounicode = first:(first + 4)
    face = unsafe_load(font.ft_ptr)
    name = unsafe_string(FT_Get_Postscript_Name(font))
    k = 1000 / Float64(face.units_per_EM)                       # font units → thousandths of an em
    u(x) = round(Int, k * x)
    cap = round(Int, 1000 * Float64(FreeTypeAbstraction.get_extent(font, 'H').scale[2]))
    glyphs = sort!([(Int(FreeTypeAbstraction.glyph_index(font, ch)), ch) for ch in chars])
    widths = join(("$g [ $(_glyph_width(font, ch)) ]" for (g, ch) in glyphs), ' ')
    code(g) = string(g; base = 16, pad = 4)
    utf16(ch) = join(string(w; base = 16, pad = 4) for w in transcode(UInt16, string(ch)))
    cmap = IOBuffer()
    print(
        cmap, "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n",
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n"
    )
    for part in Iterators.partition(glyphs, 100)                # a block holds at most 100 entries
        print(cmap, length(part), " beginbfchar\n")
        for (g, ch) in part
            print(cmap, "<", code(g), "> <", utf16(ch), ">\n")
        end
        print(cmap, "endbfchar\n")
    end
    print(cmap, "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n")
    return PdfObject[
        PdfObject(
            "/Type /Font /Subtype /Type0 /BaseFont /$name /Encoding /Identity-H " *
                "/DescendantFonts [ $cid 0 R ] /ToUnicode $tounicode 0 R"
        ),
        PdfObject(
            "/Type /Font /Subtype /CIDFontType0 /BaseFont /$name " *
                "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> " *
                "/FontDescriptor $descriptor 0 R /DW 1000 /W [ $widths ]"
        ),
        PdfObject(
            "/Type /FontDescriptor /FontName /$name /Flags 4 " *
                "/FontBBox [ $(u(face.bbox.xMin)) $(u(face.bbox.yMin)) $(u(face.bbox.xMax)) $(u(face.bbox.yMax)) ] " *
                "/ItalicAngle 0 /Ascent $(u(face.ascender)) /Descent $(u(face.descender)) /CapHeight $cap /StemV 80 " *
                "/FontFile3 $program 0 R"
        ),
        _stream("/Subtype /OpenType", read(_sheet_font_file())),
        _stream("", take!(cmap)),
    ]
end

# --- the print master -----------------------------------------------------------

# The bytes of a one-page PDF/X-4 + PDF/A-2b print master.
#
# `page` and `trimbox` are in points (strings, as the file is to hold them);
# `content` is the page's content stream; `shading` the blobs' shared radial
# shading as (dictionary entries of its function, the function's data), named
# `/B` in the page's resources; `chars` the characters the content sets in the
# sheet font (`/F`; none: no font is embedded).
# X-4 (PDF 1.6) asks the print dialog for ACTUAL
# SIZE (/PrintScaling /None): an A4 page on A4 paper is larger than any
# printer's printable area, and dialogs that default to "fit" would shrink the
# board ~3%. Acrobat/Reader honour it; other viewers may not, so the docs still
# say to print at 100% and measure.
function _print_master(;
        page::NTuple{2, String}, trimbox::NTuple{4, String}, content::AbstractString,
        shading::Tuple{String, Vector{UInt8}}, chars::Vector{Char},
        icc_profile::Union{Nothing, String}, title::AbstractString
    )
    icc_path, condition = _icc_profile(icc_profile)
    creator = "BlobBoards $(pkgversion(BlobBoards))"
    info_date, xmp_date = _pdf_dates()
    h = bytes2hex(sha256(content * title)[1:16])
    document_id = join((h[1:8], h[9:12], h[13:16], h[17:20], h[21:32]), "-")
    # object numbers, in the order the objects are written
    catalog, pages, metadata, intent_x, intent_a, icc, page_obj, contents, shading_obj, fn, info = 1:11
    fn_dict, fn_data = shading
    font_objects = isempty(chars) ? PdfObject[] : _font_objects(_sheet_font(), chars, info + 1)
    font_resource = isempty(chars) ? "" : " /Font << /F $(info + 1) 0 R >>"
    objects = PdfObject[
        PdfObject(
            "/Type /Catalog /Pages $pages 0 R /Metadata $metadata 0 R " *
                "/OutputIntents [ $intent_x 0 R $intent_a 0 R ] /ViewerPreferences << /PrintScaling /None >>"
        ),
        PdfObject("/Type /Pages /Kids [ $page_obj 0 R ] /Count 1"),
        _stream(
            "/Type /Metadata /Subtype /XML",
            _xmp(; title, creator, producer = creator, date = xmp_date, document_id); compress = false
        ),
        # PDF/A requires every intent's profile to be one stream: both name `icc`
        PdfObject(_output_intent("GTS_PDFX", condition, icc)),
        PdfObject(_output_intent("GTS_PDFA1", condition, icc)),
        _stream("/N 1", read(icc_path)),
        PdfObject(
            "/Type /Page /Parent $pages 0 R /MediaBox [ 0 0 $(page[1]) $(page[2]) ] /TrimBox [ $(join(trimbox, ' ')) ] " *
                "/Resources << /Shading << /B $shading_obj 0 R >>$font_resource >> /Contents $contents 0 R"
        ),
        _stream("", content),
        PdfObject(
            "/ShadingType 3 /ColorSpace /DeviceGray /Coords [ 0 0 0 0 0 1 ] /Extend [ false false ] /Function $fn 0 R"
        ),
        _stream(fn_dict, fn_data),
        PdfObject(
            "/Title $(_pdf_string(title)) /Creator $(_pdf_string(creator)) /Producer $(_pdf_string(creator)) " *
                "/CreationDate ($info_date) /ModDate ($info_date) /Trapped /False"
        ),
        font_objects...,
    ]
    return _pdf_file(objects; root = catalog, info, version = "1.6")
end
