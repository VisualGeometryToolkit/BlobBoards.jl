# The one print file / sidecar a board directory holds
onlypdf(d) = only(filter(endswith(".pdf"), readdir(d; join = true)))
onlyjson(d) = only(filter(endswith(".json"), readdir(d; join = true)))

# 8-bit levels of an image
levels(a) = Int16[reinterpret(x.val) for x in a]

# A board's raster at `dpi_val` (a conversion of its PDF, or its TIFF) against
# the exact sampler: `blob_profile` at every pixel centre, the blobs at their
# exact (sub-pixel) positions in the page's pixels at that density. Returns the
# max |difference| in 8-bit levels over the pixels wholly inside the pattern and
# clear of the frame's marks (reserved areas on a v4 board's pattern).
function vs_sampler(img, b, dpi_val)
    pc, bc = b.preamble.pattern_config, b.preamble.board_config
    ref = ones(size(img))
    BlobBoards.render_blob!.(Ref(ref), ustrip.(px, blobs(b, :canvas; density = dpi_val * dpi)); cutoff = pc.cutoff)
    pc.polarity == :light && (ref .= 1 .- ref)
    s = dpi_val / ustrip(dpi, BlobBoards.stored_density(bc))       # px per stored unit
    x0, y0 = BlobBoards.CoordinateTransformations.Translation(bc, :pattern, :canvas).translation .* s
    res = ustrip.(px, pc.pattern_resolution) .* s
    rows, cols = (ceil(Int, y0) + 1):floor(Int, y0 + res.height), (ceil(Int, x0) + 1):floor(Int, x0 + res.width)
    clear = trues(size(img))
    for r in BlobBoards._keepouts(bc)                    # in pattern units; a pixel touching one is skipped
        lo, w = BlobBoards.origin(r) .* s, BlobBoards.widths(r) .* s
        clear[max(1, floor(Int, y0 + lo[2])):min(end, ceil(Int, y0 + lo[2] + w[2]) + 1),
              max(1, floor(Int, x0 + lo[1])):min(end, ceil(Int, x0 + lo[1] + w[1]) + 1)] .= false
    end
    d = abs.(levels(img[rows, cols]) .- round.(Int16, 255 .* ref[rows, cols]))
    return maximum(d[clear[rows, cols]])
end

# k hundredths of a point, as the exact two-decimal string: 14173 → "141.73"
hundredths(k::Integer) = string(div(k, 100), ".", lpad(rem(k, 100), 2, '0'))
# a number in the file, as the grid value it denotes ("129" is 129.00);
# `nothing` when it is not a 0.01 pt grid value
on_grid(x::AbstractString) = (v = parse(Float64, x) * 100; abs(v - round(v)) < 1e-9 ? hundredths(round(Int, v)) : nothing)

# The print master's page content, inflated: the one Flate stream its page names.
function page_content(pdf)
    AE = BlobBoards.Authoring
    bytes = read(pdf)
    s = String(copy(bytes))
    n = match(r"/Contents (\d+) 0 R", s)[1]
    m = match(Regex("\\n$n 0 obj\\n<< /Filter /FlateDecode /Length (\\d+) >>\\nstream\\n"), s)
    start = m.offset + ncodeunits(m.match)
    return String(AE._inflate(bytes[start:(start + parse(Int, m[1]) - 1)]))
end
# Every blob the content paints: the shading's matrix `[r 0 0 r x y]`, as written.
blob_matrices(pdf) = [split(m.captures[1]) for m in eachmatch(r"q ([^\n]*) cm /B sh Q", page_content(pdf))]

# Every blob of a point-grid board in its print master: the shading's centre is
# the stored centre in canvas points, exactly (as the string it is), and its
# radius cutoff·σ to 1e-5. Returns (n blobs, n shadings, n exact, max radius error).
function print_master_blobs(b, pdf)
    mats = blob_matrices(pdf)
    bc, pc = b.preamble.board_config, b.preamble.pattern_config
    g = bc.layout_grid
    H = Int(ustrip(px, BlobBoards.BoardLayout(bc).canvas_size.height))
    x0, y0 = BlobBoards._grid_units.(Ref(g), BlobBoards.CoordinateTransformations.Translation(bc, :pattern, :canvas).translation)
    expected = Dict(
        (hundredths(x0 + BlobBoards._grid_units(g, c[1])), hundredths(H - (y0 + BlobBoards._grid_units(g, c[2])))) =>
            pc.cutoff * ustrip(px, d.σ) for (d, c) in ((d, ustrip.(px, d.center)) for d in b.blobs)
    )
    found = [get(expected, (on_grid(m[5]), on_grid(m[6])), nothing) for m in mats]
    radius = maximum((abs(parse(Float64, m[1]) / r - 1) for (m, r) in zip(mats, found) if r !== nothing); init = 0.0)
    return (length(b.blobs), length(mats), count(!isnothing, found), radius)
end

@testset "Board Generation" begin
    @testset "Frame styles" begin
        @testset "Ruler frame (default)" begin
            bundle = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234)
            # Ruler geometry is authored through BoardConfig KEYWORDS (not fields);
            # blob_board must route them — this path used to throw "unknown keyword".
            custom = blob_board(
                "A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234,
                major_tick_spacing = 20.0mm, minor_tick_spacing = 4.0mm
            )
            @test hash_config(custom.preamble) != hash_config(bundle.preamble)
            # a board has no pixel type (vector blobs, 8-bit conversions): neither
            # a keyword nor the retired leading positional form
            @test_throws ArgumentError blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, pixel_type = Float32)
            @test !applicable(blob_board, BlobBoards.Gray{BlobBoards.N0f16}, "A5", 100dpi)
            @test !applicable(blob_board, BlobBoards.Gray{BlobBoards.N0f8}, 100mm, 100mm, 100dpi)
            # the low-level form takes no layout of its own: the preamble's is the one
            @test !applicable(blob_board, bundle.preamble, BlobBoards.BoardLayout(bundle.preamble.board_config), 100dpi)
            @test blob_board(bundle.preamble).blobs == bundle.blobs
            @test length(bundle.blobs) > 0
            @test bundle.preamble.board_config.ruler_width == 10mm && bundle.preamble.board_config.corner_size == 10mm   # both marks, by default
        end

        @testset "Saddle frame: removed" begin
            # the checkerboard frame was removed (2026-10; no board used it): refused
            @test_throws ArgumentError blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, frame_style = :saddle)   # (no frame_style at all now)
            # a sidecar written before still carries an (empty) `saddle_points`: it loads
            mktempdir() do dir
                blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, dir)
                obj = JSON.parse(read(onlyjson(dir), String); dicttype = Dict{String, Any})
                @test !haskey(obj, "saddle_points")              # no longer written
                obj["saddle_points"] = []
                back = JSON.parse(JSON.json(obj), BlobBoards.BoardBundle)
                @test back.blobs == JSON.parse(read(onlyjson(dir), String), BlobBoards.BoardBundle).blobs
            end
        end

        @testset "No frame: blobs fill the whole board" begin
            bundle = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, corner_size = 0mm, ruler_width = 0mm)
            @test length(bundle.blobs) > 0
            @test iszero(bundle.preamble.board_config.corner_size) && iszero(bundle.preamble.board_config.ruler_width)
            @test isempty(BlobBoards._frame_marks(bundle.preamble.board_config))     # no cells, no bands
            @test isempty(BlobBoards._keepouts(bundle.preamble.board_config))
        end

        @testset "Corners frame: no band, the pattern fills the board around the corner cells" begin
            ustrip = BlobBoards.VisualGeometryToolkitCore.ustrip
            footprints(b) = [
                BlobBoards.Circle(BlobBoards.Disc(ustrip(px, d.center[1]), ustrip(px, d.center[2]), ustrip(px, d.σ)), b.preamble.pattern_config.cutoff)
                    for d in b.blobs
            ]
            for polarity in (:dark, :light)
                b = blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0xC0, ruler_width = 0mm, polarity)
                bc, pc = b.preamble.board_config, b.preamble.pattern_config
                # the pattern IS the board: its frames coincide
                @test pc.pattern_resolution == BlobBoards._stored_size(bc.layout_grid, BlobBoards.BoardLayout(bc).board_size)
                @test BlobBoards.Translation(bc, :pattern, :board).translation == [0.0, 0.0]
                # four keep-outs, each a corner cell grown by the clearance (10 mm + 1 mm), and no blob touches one
                ko = BlobBoards._keepouts(bc)
                @test length(ko) == 4 && all(r -> isapprox(r.widths, [11 * 72 / 25.4, 11 * 72 / 25.4]; atol = 0.05), ko)   # to the 0.01 pt grid
                @test !any(c -> any(r -> BlobBoards._overlaps(c, r), ko), footprints(b))
                # blobs reach the board's edges (no band): within 3 mm of every side
                fp = footprints(b); W, H = ustrip.(px, (pc.pattern_resolution.width, pc.pattern_resolution.height))
                mm3 = 3 * 72 / 25.4
                @test minimum(c -> c.center[1] - c.r, fp) < mm3 && minimum(c -> c.center[2] - c.r, fp) < mm3
                @test minimum(c -> W - c.center[1] - c.r, fp) < mm3 && minimum(c -> H - c.center[2] - c.r, fp) < mm3
                # rebuilt from its preamble and from its board code (ruler width 0) alike
                @test blob_board(b.preamble).blobs == b.blobs
                code = BlobBoards.board_code(b.preamble)
                @test split(code, "/")[6] == "0MM"
                @test hash_config(BlobBoards.parse_board_code(code)) == hash_config(b.preamble)
                # on paper: the cells' clearance strip is white (a `:light` board's black background painted over)
                mktempdir() do dir
                    b2 = blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0xC0, ruler_width = 0mm, polarity, dir, output = Png(150dpi), id_label = false)
                    img = BlobBoards.FileIO.load(only(filter(endswith(".png"), readdir(dir; join = true))); gamma = 1.0)
                    s = 150 / 25.4                                   # px per mm
                    r0, c0 = round(Int, 10 * s), round(Int, 10 * s)  # the board's top-left corner (10 mm margin)
                    strip = img[(r0 + round(Int, 10.3 * s)):(r0 + round(Int, 10.7 * s)), (c0 + 2):(c0 + round(Int, 10 * s))]   # under the QR cell (clear of the board's anti-aliased edge)
                    @test all(==(1), Float64.(strip))
                end
            end
            # the keep-outs take their area out of the blob counts: fewer blobs than the same board with no keep-outs
            b = blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0xC0, ruler_width = 0mm)
            free = BlobBoards._place_pattern_blobs(b.preamble.pattern_config; sampler = Val(:point_grid), min_sigma = 0.0)
            @test length(b.blobs) < length(free)
            # the style needs corner cells, and the point grid
            @test_throws ArgumentError blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 1, frame_width = 0mm)          # retired: ruler_width
            @test_throws ArgumentError blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 1, frame_style = :band)
        end

        @testset "Frame marks: reserved areas on the board, corners and bands sized apart" begin
            ustrip = BlobBoards.VisualGeometryToolkitCore.ustrip
            kw = (; blob_sizes = (6mm, 0.2short_side), seed = 0x3A)
            for extra in ((;), (; corner_size = 0mm, ruler_width = 0mm), (; ruler_width = 0mm), (; corner_size = 0mm),
                          (; corner_size = 14mm, ruler_width = 6mm))
                b = blob_board("A5"; kw..., extra...)
                bc, pc = b.preamble.board_config, b.preamble.pattern_config
                # every style: the pattern is the board, and no blob touches a mark
                @test pc.pattern_resolution == BlobBoards._stored_size(bc.layout_grid, BlobBoards.BoardLayout(bc).board_size)
                ko = BlobBoards._keepouts(bc)
                @test !any(b.blobs) do d
                    c = BlobBoards.Circle(BlobBoards.Disc(ustrip(px, d.center[1]), ustrip(px, d.center[2]), ustrip(px, d.σ)), pc.cutoff)
                    any(r -> BlobBoards._overlaps(c, r), ko)
                end
                # the marks: four cells (unless :none) and four ruled bands (:ruler only); never a blank band
                kinds = [m.kind for m in BlobBoards._frame_marks(bc)]
                @test count(in((:qr, :uid)), kinds) == (iszero(bc.corner_size) ? 0 : 4)
                @test count(==(:ruler), kinds) == (iszero(bc.ruler_width) ? 0 : 4)
                @test !in(:blank, kinds)
                # stored and coded, and rebuilt from either (the bundle's code needs no placement)
                @test hash_config(BlobBoards.parse_board_code(BlobBoards.board_code(b))) == BlobBoards._bundle_identity(b)
                back = JSON.parse(JSON.json(b), BlobBoards.BoardBundle)
                @test hash_config(back.preamble) == hash_config(b.preamble) && back.preamble.board_config.corner_size == bc.corner_size
            end
            # corner cells and band are sized apart: a 6 mm band runs between 14 mm cells
            b = blob_board("A5"; kw..., corner_size = 14mm, ruler_width = 6mm)
            marks = BlobBoards._frame_marks(b.preamble.board_config)
            m = Dict(:qr => only(filter(x -> x.kind === :qr, marks)), :top => only(filter(x -> x.kind === :ruler && x.index == 1, marks)))
            grid_mm(u) = u / 100 * 25.4 / 72
            @test grid_mm(m[:qr].w) ≈ 14 atol = 0.01
            @test grid_mm(m[:top].h) ≈ 6 atol = 0.01
            @test m[:top].x == m[:qr].w                          # the band starts where the cell ends
            # a v4 board records its corner cells; its pattern size is the board's, not recorded
            obj = JSON.parse(JSON.json(b); dicttype = Dict{String, Any})
            @test haskey(obj["preamble"]["board_config"], "corner_size")
            @test !haskey(obj["preamble"]["pattern_config"], "pattern_resolution")
            # a keyword the style draws nothing with is refused, not stored in the identity
            @test_throws ArgumentError blob_board("A5"; kw..., ruler_width = 0mm, major_tick_spacing = 20mm)   # ticks of an off ruler
            @test_throws ArgumentError blob_board("A5"; kw..., ruler_width = 0inch, minor_tick_width = 2mm)    # zero in any unit
            # the retired keywords say where they went
            for retired in ((; frame_style = :corners), (; frame_style = :ruler), (; frame_width = 8mm))
                err = try; blob_board("A5"; kw..., retired...); nothing; catch e; e; end
                @test err isa ArgumentError && occursin("corner_size", err.msg) && occursin("ruler_width", err.msg)
            end
            # any zero length turns a mark off, whatever its unit
            @test isempty(BlobBoards._frame_marks(blob_board("A5"; kw..., corner_size = 0inch, ruler_width = 0.0mm).preamble.board_config))
            @test_throws ArgumentError blob_board("A5"; kw..., corner_size = 70mm)                         # two cells exceed a side
            @test_throws ArgumentError blob_board("A5"; kw..., corner_size = 4mm, frame_padding = 2mm)      # content inset ≥ half a cell
            # the type refuses them too, however a config is built (not only blob_board's keywords)
            A5 = BlobBoards.PRINT_SIZES["A5"]
            @test_throws ArgumentError BlobBoards.BoardConfig(; canvas_size = A5, ruler_width = 0mm, major_tick_width = 2mm)
            @test_throws ArgumentError BlobBoards.BoardConfig(; canvas_size = A5, frame_style = :ruler)          # a v3 board's only
            # ... while an off ruler's ticks are fixed: they cannot vary the identity
            @test BlobBoards.BoardConfig(; canvas_size = A5, ruler_width = 0mm).ruler_spec ==
                BlobBoards.BoardConfig(; canvas_size = A5, ruler_width = 0mm, frame_padding = 1mm).ruler_spec
        end
    end

    @testset "Frame padding" begin
        @testset "Padding leaves the pattern alone" begin
            b1 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, frame_padding = 0.0mm)
            b2 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, frame_padding = 3.0mm)
            # Same blobs (same seed, same pattern area)
            @test b1.blobs == b2.blobs
        end

        @testset "Ruler with padding" begin
            b1 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, frame_padding = 0.0mm)
            b2 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, frame_padding = 2.0mm)

            # Same blobs
            @test length(b1.blobs) == length(b2.blobs)
            # Config captures padding
            @test b1.preamble.board_config.frame_padding != b2.preamble.board_config.frame_padding
        end
    end

    @testset "Tick widths are drawn as full lengths across the band" begin
        # The longest/shortest dark runs across the north band of the PDF at 300
        # dpi (a 0.1 mm tick is 1.2 px wide there, so some pixel of it is darker
        # than half): major / minor ticks.
        function tick_lengths(; kw...)
            mktempdir() do dir
                # full bleed: the band starts at the page's top edge
                blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), seed = 0x1234, ruler_width = 20mm, board_size = BlobBoards.PRINT_SIZES["A5"], dir, kw...)
                img = BlobBoards.rasterize_pdf(onlypdf(dir), 300)
                c = round(Int, 20 / 25.4 * 300)                          # the band, in px
                north = img[1:c, (c + 1):(size(img, 2) - c)]
                counts = vec(sum(x -> Float64(x) < 0.5, north; dims = 1))
                (minimum(filter(>(0), counts)), maximum(counts))
            end
        end
        # a tick of length w is drawn w long: w·300/25.4 px
        drawn(w) = ustrip(mm, w) / 25.4 * 300
        minor, major = tick_lengths()                          # defaults: the band, half the band
        @test abs(major - drawn(20mm)) <= 2 && abs(minor - drawn(10mm)) <= 2
        minor, major = tick_lengths(major_tick_width = 12mm, minor_tick_width = 6mm)
        @test abs(major - drawn(12mm)) <= 2 && abs(minor - drawn(6mm)) <= 2
        # widths that do not fit are rejected, not clipped: 0 < minor ≤ major ≤ band
        kw = (; seed = 0x1234, ruler_width = 20mm)
        @test_throws ArgumentError blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), kw..., major_tick_width = 21mm)
        @test_throws ArgumentError blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), kw..., frame_padding = 2mm, major_tick_width = 17mm)
        @test_throws ArgumentError blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), kw..., minor_tick_width = 0mm)
        @test_throws ArgumentError blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), kw..., major_tick_width = 8mm, minor_tick_width = 9mm)
        # defaults: the padded band and half of it
        cfg(; k...) = BoardConfig(; canvas_size = BlobBoards.PRINT_SIZES["A5"], ruler_width = 20mm, k...)
        rs = cfg().ruler_spec
        @test rs.major_tick.width == 20mm && rs.minor_tick.width == 10mm
        rs = cfg(frame_padding = 2mm).ruler_spec
        @test rs.major_tick.width == 16mm && rs.minor_tick.width == 8mm
    end

    @testset "A pre-v3 board's ruler migrates on read" begin
        # board v1/v2 stored `ruler_spec.width` and HALF-extent tick widths; the
        # reader drops the one and doubles the others, keeping the stored hash.
        mktempdir() do dir
            bundle = blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), seed = 0x1234, ruler_width = 20mm, dir, output = Png(400dpi))
            json = only(filter(endswith(".json"), readdir(dir; join = true)))
            d = JSON.parse(read(json, String); dicttype = Dict{String, Any})
            rs = d["preamble"]["board_config"]["ruler_spec"]
            half(t) = Dict("spacing" => t["spacing"], "width" => Dict("value" => t["width"]["value"] / 2, "unit" => t["width"]["unit"]))
            d["preamble"]["board_config"]["ruler_spec"] = Dict(
                "width" => Dict("value" => 10, "unit" => "mm"),
                "major_tick" => half(rs["major_tick"]), "minor_tick" => half(rs["minor_tick"]),
            )
            d["schema"] = "blobboards/board/v1"
            d["hashes"] = Dict("config" => "0123456789abcdef" ^ 4)   # a stored identity, never recomputed
            legacy = joinpath(mktempdir(), "blob_board_legacy.json")
            open(io -> JSON.json(io, d), legacy, "w")
            cp(only(filter(endswith(".png"), readdir(dir; join = true))), replace(legacy, ".json" => ".png"))
            b, _, h = BlobBoards.load_bundle(legacy)
            @test b.preamble.board_config.ruler_spec == bundle.preamble.board_config.ruler_spec
            @test h == "0123456789abcdef" ^ 4
            @test hash_config(b.preamble) == hash_config(bundle.preamble)
        end
    end

    @testset "The output is one value, checked before placement" begin
        root = mktempdir()
        kw = (; blob_sizes = (9.0mm, 0.2short_side), seed = 0x1234)
        # an output is valid by construction: what a format has no use for cannot be written
        @test_throws MethodError Pdf(300dpi)                     # a PDF has no density
        @test_throws MethodError Png(300dpi; icc_profile = :linear)
        @test_throws MethodError Tiff(300dpi; shading_function = :sampled)
        @test_throws ArgumentError Png(300)                      # a density, not a number
        @test_throws ArgumentError Tiff(300.5dpi)                # whole dpi: Poppler's pixel grid
        # whole dpi within rounding: the unit library holds 60dpi as 59.999999999999993 dpi
        @test all(d -> BlobBoards._render_dpi(Png(d * dpi).density) == d, (60, 72, 96, 100, 150, 200, 300, 600, 1200))
        @test_throws ArgumentError Tiff(300mm)
        @test_throws ArgumentError Pdf(icc_profile = "/no/such/profile.icc")
        @test_throws ArgumentError Pdf(shading_function = :svg)
        @test Pdf() == Pdf(shading_function = :calculator, icc_profile = :linear)
        # the five keywords the outputs replaced are refused, naming `output`
        for okw in ((format = "png",), (pdf_standard = :x4,), (icc_profile = :linear,), (shading_function = :sampled,), (print_density = 300dpi,))
            err = try; blob_board("A5"; kw..., dir = root, okw...); nothing; catch e; e; end
            @test err isa ArgumentError && occursin("output", err.msg)
        end
        # the retired positional density is no method at all
        @test !applicable(blob_board, "A5", 300dpi)
        @test !applicable(blob_board, 148mm, 210mm, 300dpi)
        @test !applicable(blob_board, BlobBoards.PRINT_SIZES["A5"], 300dpi)
        @test !applicable(blob_board, blob_board("A5"; kw...).preamble, 300dpi)
        # the output leads `render_board`; the helper the positional density became is exported
        # the output is a keyword everywhere (`render_board(output, bundle)` until 2026-10-14)
        @test hasmethod(BlobBoards.render_board, Tuple{BlobBoards.BoardBundle}, (:dir, :output, :id_label, :config_hash))
        @test !hasmethod(BlobBoards.render_board, Tuple{Pdf, BlobBoards.BoardBundle})
        @test !isdefined(BlobBoards, :default_min_blob_diameter) && isdefined(BlobBoards, :suggest_blob_sizes)
        @test !isdefined(BlobBoards, :smallest_blob_diameter) && !isdefined(BlobBoards, :largest_blob_diameter)
        # every pixel density is `density`: the retired keyword names are gone
        @test hasmethod(blobs, Tuple{BlobBoards.BoardBundle, Symbol}, (:density,)) && !hasmethod(blobs, Tuple{BlobBoards.BoardBundle, Symbol}, (:print_density,))
        @test !hasmethod(BlobBoards.AffineMap, Tuple{BlobBoards.BoardBundle, Symbol, Symbol}, (:print_density,))
        @test hasmethod(load_board, Tuple{String}, (:density,)) && !hasmethod(load_board, Tuple{String}, (:raster_density,))
        # output keywords without a dir: nothing would be written
        for okw in ((output = Png(400dpi),), (output = Pdf(),), (id_label = true,))
            @test_throws ArgumentError blob_board("A5"; kw..., okw...)
        end
        @test isempty(readdir(root))                          # every rejection came before writing
        bundle = blob_board("A5"; kw...)                       # in memory: no output keyword, no file
        json = JSON.json(bundle)
        @test_throws ArgumentError BlobBoards.rebuild_board(json; output = Png(400dpi))      # no dir
        @test_throws ArgumentError BlobBoards.rebuild_board(json; dir = root, format = "png")
        @test_throws ArgumentError BlobBoards.rebuild_board(json; dir = root, frame_style = :none)   # a rebuild takes the output only
        @test_throws MethodError BlobBoards.render_board(Pdf(), bundle; dir = root)   # the retired order fails loudly
        @test_throws UndefKeywordError BlobBoards.render_board(bundle)               # a file needs its dir
        dir = mktempdir()
        blob_board("A5"; kw..., dir, output = Tiff(400dpi))
        @test any(endswith(".tiff"), readdir(dir))
    end

    @testset "blob_sizes are printed disc diameters, and required" begin
        # a board has no density, so its smallest disc has no default: refused, naming the helper
        err = try; blob_board("A5"; seed = 0x1234); nothing; catch e; e; end
        @test err isa ArgumentError && occursin("blob_sizes", err.msg) && occursin("suggest_blob_sizes", err.msg)
        # suggest_blob_sizes(density, extent): the disc that prints well at a printer's density
        # (3 mm × 1200 dpi / density) … 20% of the board's shorter side, at most 64 mm
        @test suggest_blob_sizes(1200dpi, (300mm, 200mm)) == (3.0mm, 40.0mm)
        @test suggest_blob_sizes(600dpi, BlobBoards.Size2(width = 190mm, height = 277mm)) == (6.0mm, 38.0mm)
        @test suggest_blob_sizes(300dpi, (841mm, 1189mm)) == (12.0mm, 64.0mm)                 # capped
        @test suggest_blob_sizes(600dpi, (7.5inch, 10inch))[2] ≈ 38.1mm
        # a paper name: the board `blob_board` makes on it, the paper less 10 mm on every side
        @test suggest_blob_sizes(600dpi, "A4") == (6.0mm, 38.0mm)
        @test suggest_blob_sizes(600dpi, "A0") == (6.0mm, 64.0mm)
        @test suggest_blob_sizes(600dpi, "A4") == suggest_blob_sizes(600dpi, blob_board("A4"; blob_sizes = (6mm, 0.2short_side), seed = 1).preamble.board_config.board_size)
        @test_throws ArgumentError suggest_blob_sizes(300, "A4")
        @test_throws ArgumentError suggest_blob_sizes(3mm, "A4")
        @test_throws ArgumentError suggest_blob_sizes("A4", 600dpi)
        @test_throws ArgumentError suggest_blob_sizes(600dpi, "A9")
        @test_throws ArgumentError suggest_blob_sizes(600dpi, (190, 277))
        @test_throws ArgumentError suggest_blob_sizes(600dpi, (190mm, 0mm))
        @test_throws ArgumentError suggest_blob_sizes(600dpi, (190mm,))
        @test_throws ArgumentError suggest_blob_sizes(600dpi, 190mm)
        d = blob_board("A5"; blob_sizes = (suggest_blob_sizes(600dpi, "A5")[1], 0.2short_side), seed = 0x1234)
        e = blob_board("A5"; seed = 0x1234, blob_sizes = (6.0mm, 0.2short_side))
        @test hash_config(d.preamble) == hash_config(e.preamble)
        # a smallest disc larger than the largest the pattern takes is refused
        @test_throws ArgumentError blob_board("A5"; blob_sizes = suggest_blob_sizes(100dpi, "A5"), seed = 1)
        # σ is stored in points (the board's stored unit): 6 mm / (2·cutoff)
        @test d.preamble.pattern_config.min_scale ≈ (6 / 4 * 72 / 25.4) * px
        c3 = blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0x1234, cutoff = 3.0).preamble.pattern_config
        @test 2 * c3.cutoff * c3.min_scale ≈ (6 * 72 / 25.4) * px
        # the largest disc is stated too: `nothing` is refused, naming the two ways to give it
        err = try; blob_board("A5"; blob_sizes = (6mm, nothing), seed = 1); nothing; catch e; e; end
        @test err isa ArgumentError && occursin("suggest_blob_sizes", err.msg) && occursin("short_side", err.msg)
        maxd(b) = (pc = b.preamble.pattern_config; 2 * pc.cutoff * ustrip(px, pc.max_scale) * 25.4 / 72)   # mm
        @test maxd(blob_board("A4"; blob_sizes = suggest_blob_sizes(600dpi, "A4"), seed = 3)) ≈ 38
        @test maxd(blob_board("A0"; blob_sizes = (8mm, suggest_blob_sizes(600dpi, "A0")[2]), seed = 3)) ≈ 64
        @test maxd(blob_board("A0"; blob_sizes = (8mm, 100mm), seed = 3)) ≈ 100     # any length is taken as given
        # `f * short_side` is exactly that fraction of the pattern's short side — a v4 board's
        # pattern is the board (its frame marks are reserved areas on it) — with no cap
        @test maxd(d) ≈ 0.2 * (148 - 2 * 10) atol = 0.01                           # A5: 25.6 mm
        @test maxd(blob_board("A0"; blob_sizes = (8mm, 0.2short_side), seed = 3)) ≈ 0.2 * (841 - 20) atol = 0.01
        # `blob_measure` was removed: rejected by name
        @test_throws ArgumentError blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0x1234, blob_measure = :diameter)
        # a size must be a positive physical length
        @test_throws ArgumentError blob_board("A4"; seed = 0x1234, blob_sizes = (0mm, 0.2short_side))
        @test_throws ArgumentError blob_board("A4"; seed = 0x1234, blob_sizes = (4, 0.2short_side))
    end

    @testset "raster_aligned is not a keyword: centres are 0.01 pt grid points" begin
        # a board's blob centres are always on its 1/7200-inch grid; the keyword is
        # rejected by name, before placement, as is `grid`
        for v in (true, false)
            @test_throws ArgumentError blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), seed = 0x1234, raster_aligned = v)
            @test_throws ArgumentError blob_board(148mm, 210mm; blob_sizes = (9.0mm, 0.2short_side), seed = 0x1234, raster_aligned = v)
        end
        @test_throws ArgumentError blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), seed = 0x1234, grid = BlobBoards.PointGrid())
        b = blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), seed = 0x1234)
        @test b.preamble.board_config.layout_grid === BlobBoards.PointGrid()
        @test all(d -> all(c -> on_grid(string(c)) !== nothing, ustrip.(px, d.center)), b.blobs)
    end

    @testset "the layout is physical: a board has no density" begin
        kw = (; blob_sizes = (2mm, 0.2short_side), seed = 0x55, board_size = BlobBoards.Size2(width = 120mm, height = 150mm))
        a = blob_board("A5"; kw...)
        # a PDF asks nothing of the board; a raster needs 2 pixels per σ at ITS density:
        # 2 mm discs (σ = 0.5 mm) need ≥ 101.6 dpi, and the error names the density that does
        mktempdir() do root
            blob_board("A5"; kw..., dir = joinpath(root, "pdf"))
            blob_board("A5"; kw..., dir = joinpath(root, "ok"), output = Png(102dpi))
            err = try; blob_board("A5"; kw..., dir = joinpath(root, "low"), output = Png(100dpi)); nothing; catch e; e; end
            @test err isa ArgumentError && occursin("≥ 102 dpi", err.msg)
            @test !isdir(joinpath(root, "low"))                # refused before placement
            @test_throws ArgumentError BlobBoards.render_board(a; output = Tiff(100dpi), dir = joinpath(root, "low"))
            # one board, whatever it is written as
            ids = [JSON.parse(read(onlyjson(joinpath(root, k)), String)).hashes.config for k in ("pdf", "ok")]
            @test all(==(hash_config(a.preamble)), ids)
        end
        # a side beyond 9999.99 pt (3.53 m) cannot keep 0.01 pt in the print master
        @test_throws ArgumentError blob_board(3600mm, 1000mm; blob_sizes = (12.0mm, 0.2short_side), seed = 1)
        # 8.5 inch is exactly 61200 grid units (the floating-point product is 61199.99…)
        @test BlobBoards.BoardLayout(blob_board("Letter"; blob_sizes = (9.0mm, 0.2short_side), seed = 1).preamble.board_config).canvas_size.width == 61200px
    end

    @testset "Frame width" begin
        b1 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, ruler_width = 10.0mm)
        b2 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, ruler_width = 20.0mm)

        # Different frame widths mean different pattern areas, potentially different blob counts
        @test b1.preamble.board_config.ruler_width != b2.preamble.board_config.ruler_width
    end

    @testset "Paper sizes" begin
        for paper in ["A4", "A5", "Letter"]
            bundle = blob_board(paper; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234)
            @test length(bundle.blobs) > 0
        end
    end

    @testset "Board reproducibility" begin
        b1 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0xBEEF)
        b2 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0xBEEF)

        @test length(b1.blobs) == length(b2.blobs)

        # Check blob positions match
        for (bl1, bl2) in zip(b1.blobs, b2.blobs)
            @test bl1.center == bl2.center
            @test bl1.σ == bl2.σ
        end
    end

    @testset "Board round-trip idempotency" begin
        for marks in [(;), (; corner_size = 0mm, ruler_width = 0mm)]
            dir1 = mktempdir()
            dir2 = mktempdir()

            # Generate original
            b1 = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234, marks..., dir = dir1)

            # Load JSON and regenerate
            json_file = only(filter(f -> endswith(f, ".json"), readdir(dir1; join = true)))
            json_str = read(json_file, String)
            b2 = BlobBoards.rebuild_board(json_str; dir = dir2)

            # Compare JSONs (excluding file hash which varies)
            json1 = JSON.parse(read(only(filter(f -> endswith(f, ".json"), readdir(dir1; join = true))), String))
            json2 = JSON.parse(read(only(filter(f -> endswith(f, ".json"), readdir(dir2; join = true))), String))

            @test JSON.json(json1.preamble) == JSON.json(json2.preamble)
            @test JSON.json(json1.blobs) == JSON.json(json2.blobs)
            @test json1.hashes.config == json2.hashes.config
        end
    end

    @testset "Blob extraction" begin
        bundle = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x1234)

        @testset "Pixel units: at a density the caller names" begin
            # the layout has no density: pixels exist only at one
            @test_throws ArgumentError blobs(bundle, :pattern)
            b_px = blobs(bundle, :pattern; density = 300dpi)
            @test length(b_px) == length(bundle.blobs)
            @test eltype(b_px[1].center) <: BlobBoards.Quantity
            # points × 300/72
            @test ustrip.(px, b_px[1].center) ≈ ustrip.(px, bundle.blobs[1].center) .* (300 / 72)
            AffineMap = BlobBoards.CoordinateTransformations.AffineMap
            @test_throws ArgumentError AffineMap(bundle, :pixel, :world)
            H = AffineMap(bundle, :pixel, :world; density = 300dpi)
            w = blobs(bundle, :world)
            @test H(collect(ustrip.(px, blobs(bundle, :canvas; density = 300dpi)[1].center))) ≈ collect(w[1].center)
            # a stored v3 board's grid IS its printer's pixels: its own density by default
            v3, _, _ = BlobBoards.load_bundle(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"))
            @test blobs(v3, :pattern) == blobs(v3, :pattern; density = 300dpi)
            @test AffineMap(v3, :pixel, :world) == AffineMap(v3, :pixel, :world; density = 300dpi)
            @test_throws ArgumentError AffineMap(v3, :logical, :world)          # renamed :pixel
        end

        @testset "Physical units" begin
            b_mm = blobs(bundle, :pattern; units = mm)
            @test length(b_mm) == length(bundle.blobs)
        end

        @testset "Reference frames" begin
            b_pattern = blobs(bundle, :pattern; units = mm)
            b_board = blobs(bundle, :board; units = mm)
            b_canvas = blobs(bundle, :canvas; units = mm)

            # a v4 board's pattern is the board; the canvas is offset by the margin
            @test b_pattern[1].center == b_board[1].center
            @test b_pattern[1].center != b_canvas[1].center
        end
    end
end

@testset "render regeneration from a bare board definition" begin
    # A board JSON with NO sibling render regenerates its print master through
    # BlobBoards.Authoring into a scratch cache (keyed by config hash) — the
    # mechanism that lets the published example-data artifact ship board
    # DEFINITIONS instead of 280 MB of print-resolution renders. The cached
    # file is the PDF, which has no density: one file serves every density a
    # caller rasterizes at. The source directory is never written (definitions
    # can live in a content-addressed artifact tree).
    mktempdir() do dir
        bundle = blob_board("A5"; blob_sizes = (4.0mm, 0.2short_side), seed = 0x0000b10b, dir)
        base = BlobBoards.make_filebase(bundle)
        json = joinpath(dir, base * ".json")
        # keep only the definition, note the original render's pixels
        ref = BlobBoards.rasterize_pdf(joinpath(dir, base * ".pdf"), 100)
        rm(joinpath(dir, base * ".pdf"))
        # purge any cached render from a previous test run — this asserts
        # REGENERATION, not a cache hit
        h = String(JSON.parse(read(json, String)).hashes.config)
        cache = BlobBoards._render_cache_dir(h, true)            # the sheet was written with its id label
        isdir(cache) && rm(cache; recursive = true)

        back, img_path, ch = BlobBoards.load_bundle(json)
        @test occursin("board_renders", img_path) && endswith(img_path, ".pdf")   # scratch, not the source dir
        @test readdir(dir) == [base * ".json"]      # source dir untouched
        # FULL v1 board round trip — the write→read contract, field by field:
        # geometry, frame layout, and identity all survive serialization.
        @test back.blobs == bundle.blobs
        @test back.preamble == bundle.preamble
        @test ch == h                               # stored identity returned
        # the regenerated render is the original's pixels at the same density, id label included
        @test BlobBoards.rasterize_pdf(img_path, 100) == ref
        # second resolve is a cache hit on the same path
        @test BlobBoards.load_bundle(json)[2] == img_path
        # the detector's image: at the density asked for …
        @test ustrip(dpi, load_board(json; density = 100dpi).density) == 100
        # … else at the board's own — its smallest disc (4 mm) 16 px across
        lb = load_board(json)
        own = BlobBoards._default_density(bundle)
        @test lb.density == own && ustrip(dpi, own) == ceil(16 / (4 / 25.4))   # 102 dpi
        @test lb.image_path == img_path             # one cached file, whatever the density
        small = minimum(d -> 2 * BlobBoards.cutoff(lb.bundle) * ustrip(px, d.σ), blobs(lb.bundle, :canvas; density = own))
        @test 16 <= small < 16.2
    end
end

# The page boxes of a PDF, in points: Dict("MediaBox" => [[x0, y0, x1, y1], …], …).
pdf_boxes(s::AbstractString) = Dict(
    k => [parse.(Float64, split(m.captures[1])) for m in eachmatch(Regex("/$k\\s*\\[([^\\]]*)\\]"), s)]
        for k in ("MediaBox", "BleedBox", "TrimBox")
)

@testset "PDF output: the PDF/X-4 print contract" begin
    # The print master carries the contract a print shop's preflight checks:
    # identification, an OutputIntent with its ICC profile and page boxes:
    # the TrimBox is the finished sheet (the canvas), so a shop cutting to it
    # keeps the margin and the id label. It holds no image, its text is set in one
    # embedded font, and its objects are plain: no object streams, no
    # cross-reference stream.
    mktempdir() do root
        dir = joinpath(root, "x4")
        blob_board(
            "A5"; board_size = BlobBoards.Size2(width = 80mm, height = 80mm), corner_size = 0mm, ruler_width = 0mm,
            blob_sizes = (1mm, 6mm), seed = 0x77, dir
        )
        s = String(read(onlypdf(dir)))
        @test startswith(s, "%PDF-1.6")
        @test occursin("/OutputIntents", s) && occursin("/GTS_PDFX", s)
        @test occursin("<pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>", s)
        @test !occursin("/Subtype /Image", s) && !occursin("/ObjStm", s) && !occursin("/XRef", s)
        @test occursin("/FontFile3", s) && occursin("/ToUnicode", s)        # this board carries its id label
        @test occursin("/PrintScaling /None", s)
        # the trailer names the catalog, the Info dictionary and a file identifier
        @test occursin(r"trailer\n<< /Size \d+ /Root 1 0 R /Info \d+ 0 R /ID \[ <[0-9a-f]{32}> <[0-9a-f]{32}> \] >>", s)
        # the cross-reference table points at every object
        xref = parse(Int, match(r"startxref\n(\d+)\n%%EOF\n$", s)[1])
        bytes = codeunits(s)
        @test String(bytes[(xref + 1):(xref + 4)]) == "xref"
        entries = [parse(Int, m[1]) for m in eachmatch(r"\n(\d{10}) 00000 n ", s)]
        @test all(startswith(String(bytes[(off + 1):(off + 12)]), "$n 0 obj\n") for (n, off) in enumerate(entries))
        # one each of MediaBox, BleedBox and TrimBox; with no bleed all three are
        # the A5 page (the 80 mm board centred on it is NOT the trim)
        boxes = pdf_boxes(s)
        @test all(k -> length(boxes[k]) == 1, ("MediaBox", "BleedBox", "TrimBox"))
        a5 = [0, 0, 148 / 25.4 * 72, 210 / 25.4 * 72]
        @test all(k -> isapprox(only(boxes[k]), a5; atol = 0.02), ("MediaBox", "BleedBox", "TrimBox"))
        # the one standard: a board recorded under another (X-3, X-1a, a plain PDF:
        # they went with Ghostscript, 2026-10) is rebuilt as PDF/X-4, which is
        # not recorded (`pdf_standard`, written until 2026-10-14, is not read)
        obj = JSON.parse(read(onlyjson(dir), String); dicttype = Dict{String, Any})
        obj["render"]["pdf_standard"] = "x3"
        BlobBoards.rebuild_board(JSON.json(obj); dir = joinpath(root, "re"))
        @test !haskey(JSON.parse(read(onlyjson(joinpath(root, "re")), String)).render, :pdf_standard)
    end
end

@testset "PDF bleed: the canvas is the trim, the page the canvas plus the bleed" begin
    # A print shop asks for a trim allowance: `Pdf(bleed)` makes the page the
    # canvas plus `bleed` on every side (MediaBox = BleedBox) and keeps the TrimBox
    # at the finished sheet. It is a rendering option: the board, its uid and its
    # blobs are those of the same board without bleed, and the record carries it
    # so a rebuild prints the same sheet.
    mktempdir() do root
        kw = (; blob_sizes = (3mm, 12mm), seed = 0x5eed, corner_size = 15mm, ruler_width = 0mm)
        plain = blob_board("A5"; kw..., dir = joinpath(root, "plain"))
        bled = blob_board("A5"; kw..., output = Pdf(bleed = 2mm), dir = joinpath(root, "bleed"))
        @test bled.blobs == plain.blobs
        @test basename(onlypdf(joinpath(root, "bleed"))) == basename(onlypdf(joinpath(root, "plain")))   # one uid
        pt(x) = x / 25.4 * 72
        boxes = pdf_boxes(String(read(onlypdf(joinpath(root, "bleed")))))
        @test isapprox(only(boxes["MediaBox"]), [0, 0, pt(152), pt(214)]; atol = 0.02)
        @test only(boxes["BleedBox"]) == only(boxes["MediaBox"])
        @test isapprox(only(boxes["TrimBox"]), [pt(2), pt(2), pt(150), pt(212)]; atol = 0.02)
        rec = JSON.parse(read(onlyjson(joinpath(root, "bleed")), String)).render
        @test rec.bleed.value == 2 && rec.bleed.unit == "mm"
        # the rebuild prints what was printed: the same boxes
        BlobBoards.rebuild_board(onlyjson(joinpath(root, "bleed")); dir = joinpath(root, "re"))
        @test pdf_boxes(String(read(onlypdf(joinpath(root, "re"))))) == boxes
        # a raster is the sheet without bleed (it has no page boxes)
        @test BlobBoards._recorded_bleed(Png(300dpi)) === nothing
        @test_throws ArgumentError Pdf(bleed = -1mm)
        @test_throws ArgumentError Pdf(bleed = 2)
        @test_throws ArgumentError Pdf(bleed = 300dpi)
        # the page, canvas plus bleed, keeps the point grid's precision, as the canvas does
        @test_throws ArgumentError blob_board(
            3520mm, 100mm; blob_sizes = (3mm, 12mm), corner_size = 0mm, ruler_width = 0mm,
            output = Pdf(bleed = 10mm), dir = joinpath(root, "big")
        )
    end
end

@testset "PDF numbers: fixed notation, ten significant digits" begin
    AE = BlobBoards.Authoring
    @test AE._pdf_num(124.68) == "124.68" && AE._pdf_num(129.0) == "129" && AE._pdf_num(0) == "0"
    @test AE._pdf_num(1.0e-5) == "0.00001" && AE._pdf_num(-3.125) == "-3.125"       # PDF has no exponent form
    @test AE._pdf_num(1 / 3) == "0.3333333333" && AE._pdf_num(12345.678912345) == "12345.67891"
    @test AE._pdf_num(-1.0e-12) == "-0.000000000001" && AE._pdf_num(4.0e-17) == "0"
    @test_throws ArgumentError AE._pdf_num(Inf)
    @test AE._inflate(AE._deflate(Vector{UInt8}("q 1 0 0 1 0 0 cm Q\n"^500))) == Vector{UInt8}("q 1 0 0 1 0 0 cm Q\n"^500)
end

@testset "sheet text: set in the embedded font, searchable" begin
    # The uid characters and the id label are text in one embedded font
    # (Identity-H: a code is a glyph index), with a width and a Unicode value
    # for every character the sheet sets — what makes the page searchable and
    # the hash copyable. A board with no text embeds no font.
    AE = BlobBoards.Authoring
    font = AE._sheet_font()
    hex(text) = join(AE._glyph_code(font, ch) for ch in text)
    mktempdir() do root
        b = blob_board("A5"; blob_sizes = (12.0mm, 0.2short_side), seed = 0x91, dir = joinpath(root, "framed"))
        pdf = onlypdf(joinpath(root, "framed"))
        s = String(read(pdf))
        content = page_content(pdf)
        h = hash_config(b.preamble)
        label = AE._id_label_text(b, h)
        @test occursin("0 Tr", content) && occursin("<$(hex(label))> Tj", content)       # the label: filled
        for ch in uid_from_hash(h)
            @test occursin(Regex("2 Tr [-0-9. ]+ Td <$(hex(string(ch)))> Tj"), content)   # a uid character: filled and stroked
        end
        @test occursin("/Subtype /Type0", s) && occursin("/Encoding /Identity-H", s) && occursin("/Subtype /CIDFontType0", s)
        @test occursin("/FontFile3", s) && occursin("/Subtype /OpenType", s)
        # every character set has its width in the font object, as the font program gives it
        w = match(r"/W \[ ((?:\d+ \[ \d+ \] )+)\]", s)[1]
        stated = Dict(parse(Int, m[1]) => parse(Int, m[2]) for m in eachmatch(r"(\d+) \[ (\d+) \]", w))
        chars = unique(label * uid_from_hash(h))
        @test all(stated[AE._glyph_index(font, ch)] == AE._glyph_width(font, ch) for ch in chars)
        @test length(stated) == length(chars)
        # the line is centred by those widths
        @test AE._line_width(font, "ab") ≈ (AE._glyph_width(font, 'a') + AE._glyph_width(font, 'b')) / 1000

        blob_board("A5"; blob_sizes = (12.0mm, 0.2short_side), seed = 0x91, dir = joinpath(root, "bare"), corner_size = 0mm, ruler_width = 0mm, id_label = false)
        @test !occursin("/Font", String(read(onlypdf(joinpath(root, "bare")))))
    end
    @test_throws ArgumentError AE._glyph_code(font, '\U1F600')       # no glyph: refused, not a missing-glyph box
end

@testset "TIFF output: the vector sheet converted, LZW, density and ICC tags" begin
    mktempdir() do dir
        bundle = blob_board(
            "A5"; board_size = BlobBoards.Size2(width = 80mm, height = 80mm), corner_size = 0mm, ruler_width = 0mm,
            blob_sizes = (1mm, 6mm), seed = 0x77, dir, output = Tiff(300dpi)
        )
        img = TiffImages.load(only(filter(endswith(".tiff"), readdir(dir; join = true))))
        ifd = first(img.ifds)
        @test ifd[TiffImages.COMPRESSION].data == 5                     # LZW
        @test ifd[TiffImages.PREDICTOR].data == 2                       # horizontal
        @test ifd[TiffImages.XRESOLUTION].data == 300 && ifd[TiffImages.RESOLUTIONUNIT].data == 2
        @test !isempty(ifd[TiffImages.ICCPROFILE].data)
        # the whole page on Poppler's grid at the density: rounded up to whole pixels
        c = BlobBoards.BoardLayout(bundle.preamble.board_config).canvas_size
        @test size(img) == ceil.(Int, ustrip.(px, (c.height, c.width)) .* (300 / 7200) .- 1e-6)
        # the pattern: the profile sampled exactly at the blobs' sub-pixel centres, to one 8-bit level
        @test vs_sampler(img, bundle, 300) <= 1
    end
end

@testset "id_label: the board's identity printed under it" begin
    # 80 mm board on A5: 65 mm sheet margin under it. Rasterized at 100 dpi the
    # board's lower edge is at 145 mm (row 571); the label starts 1.5 mm below.
    under_board(pdf) = let img = BlobBoards.rasterize_pdf(pdf, 100)
        Float64.(img[575:600, 150:430]) ./ Float64(maximum(img))
    end
    mk(dir; kw...) = blob_board(
        "A5"; board_size = BlobBoards.Size2(width = 80mm, height = 80mm), corner_size = 0mm, ruler_width = 0mm,
        blob_sizes = (1mm, 6mm), seed = 0x77, dir, kw...
    )
    mktempdir() do root
        on = mk(joinpath(root, "on"))
        off = mk(joinpath(root, "off"); id_label = false)
        pdf(d) = only(filter(endswith(".pdf"), readdir(joinpath(root, d); join = true)))
        @test minimum(under_board(pdf("on"))) < 0.5            # dark glyphs under the board
        @test minimum(under_board(pdf("off"))) > 0.99          # white paper
        @test hash_config(on.preamble) == hash_config(off.preamble)   # a rendering option, not identity
        # the label's content: stem, the board code (what the QR holds), version —
        # and no density: the sheet is the same at any
        AE = BlobBoards.Authoring
        h = hash_config(on.preamble)
        txt = AE._id_label_text(on, h)
        @test txt == "blob_board_$(h[1:3])_$(h[4:8]) · $(BlobBoards.board_code(on.preamble)) · BlobBoards $(pkgversion(BlobBoards))"
        @test !occursin("dpi", txt)
        # the code in the line rebuilds the board
        @test hash_config(BlobBoards.parse_board_code(split(txt, " · ")[2])) == h
        # under another identity than its configuration's, the line carries that identity
        other = "0"^64
        @test occursin("sha256 $other", AE._id_label_text(on, other))
        # a stored v3 board (laid out on its printer's pixels, no code): hash, size and its density
        v3, _, h3 = BlobBoards.load_bundle(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"))
        t3 = AE._id_label_text(v3, h3)
        @test occursin("sha256 $h3", t3) && occursin(r"mm @ 300 dpi$", split(t3, " · BlobBoards")[1])
        # a frameless board that fills its sheet has no room and nothing else to identify it
        @test_logs (:warn, r"id_label skipped") match_mode = :any blob_board(
            "A5"; corner_size = 0mm, ruler_width = 0mm, board_size = BlobBoards.PRINT_SIZES["A5"], blob_sizes = (4mm, 0.2short_side), seed = 0x78, dir = joinpath(root, "full")
        )
    end
end

@testset "render block: how the sheet was printed, outside the identity" begin
    under_board(pdf) = let img = BlobBoards.rasterize_pdf(pdf, 100)
        Float64.(img[575:600, 150:430]) ./ Float64(maximum(img))
    end
    labelled(pdf) = minimum(under_board(pdf)) < 0.5
    mk(dir; kw...) = blob_board(
        "A5"; board_size = BlobBoards.Size2(width = 80mm, height = 80mm), corner_size = 0mm, ruler_width = 0mm,
        blob_sizes = (1mm, 6mm), seed = 0x77, dir, kw...
    )
    mktempdir() do root
        mk(joinpath(root, "on")); mk(joinpath(root, "off"); id_label = false)
        jon = JSON.parse(read(onlyjson(joinpath(root, "on")), String))
        joff = JSON.parse(read(onlyjson(joinpath(root, "off")), String))
        @test jon.render.format == "pdf" && !haskey(jon.render, :pdf_standard) && jon.render.id_label === true
        @test jon.render.icc_profile == "linear"
        @test joff.render.id_label === false
        @test jon.hashes.config == joff.hashes.config                 # one identity either way
        @test JSON.json(jon.preamble) == JSON.json(joff.preamble)

        # rebuild_board reproduces the sheet as recorded; a keyword overrides it
        str_on, str_off = read(onlyjson(joinpath(root, "on")), String), read(onlyjson(joinpath(root, "off")), String)
        BlobBoards.rebuild_board(str_on; dir = joinpath(root, "re_on"))
        BlobBoards.rebuild_board(str_off; dir = joinpath(root, "re_off"))
        BlobBoards.rebuild_board(str_on; dir = joinpath(root, "re_override"), id_label = false)
        @test labelled(onlypdf(joinpath(root, "re_on")))
        @test !labelled(onlypdf(joinpath(root, "re_off")))
        @test !labelled(onlypdf(joinpath(root, "re_override")))

        # a PDF has no density, and records none; a raster records its own
        @test jon.render.density === nothing && !haskey(jon.render, :print_density)
        @test BlobBoards._render_options(jon) ==
            (format = "pdf", icc_profile = nothing, id_label = true, shading_function = :calculator, density = nothing, bleed = 0mm)
        @test BlobBoards._recorded_output(BlobBoards._render_options(jon)) == Pdf()
        mk(joinpath(root, "png"); output = Png(300dpi))
        jpng = JSON.parse(read(onlyjson(joinpath(root, "png")), String))
        @test jpng.render.format == "png" && jpng.render.density.value == 300 && jpng.render.density.unit == "dpi"
        @test !haskey(jpng.render, :pdf_standard) && jpng.render.icc_profile === nothing
        @test BlobBoards._recorded_output(BlobBoards._render_options(jpng)) == Png(300dpi)
        @test jpng.hashes.config == jon.hashes.config                 # the output is not the board
        # rebuilt as recorded: a PNG at its density; an output named instead overrides
        BlobBoards.rebuild_board(read(onlyjson(joinpath(root, "png")), String); dir = joinpath(root, "re_png"))
        @test only(filter(endswith(".png"), readdir(joinpath(root, "re_png")))) == basename(onlyjson(joinpath(root, "png")))[1:(end - 5)] * ".png"
        BlobBoards.rebuild_board(read(onlyjson(joinpath(root, "png")), String); dir = joinpath(root, "re_png_pdf"), output = Pdf())
        @test isfile(onlypdf(joinpath(root, "re_png_pdf")))

        # a record written before 2026-10-14 names a raster's density `print_density`,
        # and carries one for a PDF too (the density the board was "made for"): read
        # for the raster, nothing to the PDF
        old = JSON.parse(read(onlyjson(joinpath(root, "png")), String); dicttype = Dict{String, Any})
        old["render"]["print_density"] = pop!(old["render"], "density")
        @test BlobBoards._render_options(JSON.parse(JSON.json(old))).density == 300dpi
        oldpdf = JSON.parse(str_on; dicttype = Dict{String, Any})
        oldpdf["render"]["print_density"] = Dict("value" => 300, "unit" => "dpi")
        @test BlobBoards._render_options(JSON.parse(JSON.json(oldpdf))).density === nothing
        # a raster record with no density cannot be reproduced: the caller names the output
        delete!(old["render"], "print_density")
        @test_throws ArgumentError BlobBoards.rebuild_board(JSON.json(old); dir = joinpath(root, "re_nodensity"))
        BlobBoards.rebuild_board(JSON.json(old); dir = joinpath(root, "re_nodensity"), output = Png(300dpi))

        # a file without a render block: the as-printed defaults (a PDF, no label)
        legacy = JSON.parse(str_on; dicttype = Dict{String, Any}); delete!(legacy, "render")
        @test BlobBoards._render_options(JSON.parse(JSON.json(legacy))) ==
            (format = "pdf", icc_profile = nothing, id_label = false, shading_function = :calculator, density = nothing, bleed = nothing)
        BlobBoards.rebuild_board(JSON.json(legacy); dir = joinpath(root, "re_legacy"))
        @test !labelled(onlypdf(joinpath(root, "re_legacy")))

        # the gallery's regeneration path (a JSON with no render beside it) honours the label too
        bare = joinpath(root, "bare"); mkpath(bare)
        cp(onlyjson(joinpath(root, "on")), joinpath(bare, basename(onlyjson(joinpath(root, "on")))))
        cache = BlobBoards._render_cache_dir(String(jon.hashes.config), true)
        isdir(cache) && rm(cache; recursive = true)
        _, img_path, _ = BlobBoards.load_bundle(onlyjson(bare))
        @test labelled(img_path)
        rm(cache; recursive = true)
    end
end

@testset "reconstruction keeps the STORED identity; the QR carries the board code" begin
    # A v1 board's in-memory preamble lowers to the current schema, so its
    # recomputed hash is not its identity. Build a v1 file from a fresh board:
    # same config, v1 fields, a stored hash that differs from any recomputation.
    AE = BlobBoards.Authoring
    # a sidecar's preamble, read back, equals the bundle's — same schema, same values
    sidecar_matches(bundle, json_path) =
        JSON.parse(JSON.json(bundle.preamble); dicttype = Dict{String, Any}) ==
        JSON.parse(read(json_path, String); dicttype = Dict{String, Any})["preamble"]
    # what a board's QR carries (`make_qr_code`)
    qr_text(bundle, h) = AE._qr_text(AE._code_or_nothing(bundle.preamble, bundle.blobs, h), h)
    mktempdir() do root
        fresh = blob_board(
            "A5"; board_size = BlobBoards.Size2(width = 80mm, height = 80mm),
            blob_sizes = (4mm, 0.2short_side), seed = 0x99, dir = joinpath(root, "fresh")
        )
        fresh_json = only(filter(endswith(".json"), readdir(joinpath(root, "fresh"); join = true)))
        @test sidecar_matches(fresh, fresh_json)
        # a fresh board's QR is its board code, which decodes to the sidecar's identity
        h_fresh = String(JSON.parse(read(fresh_json, String)).hashes.config)
        @test qr_text(fresh, h_fresh) == BlobBoards.board_code(fresh.preamble)
        @test hash_config(BlobBoards.parse_board_code(qr_text(fresh, h_fresh))) == h_fresh

        obj = JSON.parse(read(fresh_json, String); dicttype = Dict{String, Any})
        pc = obj["preamble"]["pattern_config"]
        delete!(pc, "max_scale"); pc["max_diameter_fraction"] = 0.2
        rs = obj["preamble"]["board_config"]["ruler_spec"]
        half(t) = Dict("spacing" => t["spacing"], "width" => Dict("value" => t["width"]["value"] / 2, "unit" => t["width"]["unit"]))
        obj["preamble"]["board_config"]["ruler_spec"] = Dict(
            "width" => half(rs["major_tick"])["width"], "major_tick" => half(rs["major_tick"]), "minor_tick" => half(rs["minor_tick"])
        )
        obj["schema"] = "blobboards/board/v1"
        stored = "abcdef" * "0"^58
        obj["hashes"] = Dict("config" => stored)
        delete!(obj, "render")
        v1 = JSON.json(obj)

        # (a synthetic file: a point-grid layout under the v1 spellings, so it
        # records no density and the rebuild names one)
        rebuilt = BlobBoards.rebuild_board(v1; dir = joinpath(root, "rebuilt"))
        files = readdir(joinpath(root, "rebuilt"))
        @test "blob_board_abc_def00.json" in files && "blob_board_abc_def00.pdf" in files
        rebuilt_json = joinpath(root, "rebuilt", "blob_board_abc_def00.json")
        side = JSON.parse(read(rebuilt_json, String))
        @test side.hashes.config == stored                       # the stored identity
        @test side.schema == BlobBoards.BOARD_SCHEMA             # written in the current schema
        @test sidecar_matches(rebuilt, rebuilt_json)
        # its config does not reproduce the stored identity, so its QR is the identity, not a code
        @test qr_text(rebuilt, stored) == uppercase(stored)
        @test !haskey(side.preamble.pattern_config, :max_diameter_fraction)
        @test !haskey(side.preamble.board_config.ruler_spec, :width)

        # render_board of the loaded v1 board: same rules
        b, _, h = BlobBoards.load_bundle(begin
            p = joinpath(root, "v1", "blob_board_abc_def00.json"); mkpath(dirname(p)); write(p, v1)
            cp(joinpath(root, "rebuilt", "blob_board_abc_def00.pdf"), replace(p, ".json" => ".pdf")); p
        end)
        rr = BlobBoards.render_board(b; output = Png(100dpi), dir = joinpath(root, "rerender"), config_hash = h)
        rj = only(filter(endswith(".json"), readdir(joinpath(root, "rerender"); join = true)))
        @test basename(rj) == "blob_board_abc_def00.json" && sidecar_matches(rr, rj)
    end
end

@testset "a raster records its own options; the label, whether it was drawn" begin
    mktempdir() do root
        mk(; kw...) = blob_board(
            "A5"; board_size = BlobBoards.Size2(width = 80mm, height = 80mm), corner_size = 0mm, ruler_width = 0mm,
            blob_sizes = (4mm, 0.2short_side), seed = 0x5, dir = root, kw...
        )
        # a TIFF embeds its profile and records it, with its density; it has no PDF standard
        mk(; output = Tiff(300dpi), dir = joinpath(root, "t"))
        r = JSON.parse(read(only(filter(endswith(".json"), readdir(joinpath(root, "t"); join = true))), String)).render
        @test r.format == "tiff" && !haskey(r, :pdf_standard) && r.icc_profile == "linear" && r.density.value == 300
        # the render block records whether the label was DRAWN: a frameless board
        # filling its sheet has no room for it
        @test_logs (:warn, r"id_label skipped") match_mode = :any blob_board(
            "A5"; corner_size = 0mm, ruler_width = 0mm, board_size = BlobBoards.PRINT_SIZES["A5"], blob_sizes = (4mm, 0.2short_side), seed = 0x6, dir = joinpath(root, "full")
        )
        rf = JSON.parse(read(only(filter(endswith(".json"), readdir(joinpath(root, "full"); join = true))), String)).render
        @test rf.id_label === false
    end
end

@testset "raster conversions are bounded by Cairo's image size; a PDF is not" begin
    # TIFF/PNG are the PDF converted through a Cairo image surface (≤ 32745 px a
    # side): refused before placement. A PDF's blobs are shadings — no bound.
    @test_throws ArgumentError blob_board(1000mm, 1000mm; blob_sizes = (1mm, 0.2short_side), output = Tiff(1200dpi), dir = mktempdir())
end

@testset "shading function: always vector, the function as asked, recorded as written" begin
    # A board's blobs are radial shadings — a calculator function by default, a
    # sampled one on request; the raster engine was removed (2026-10). The writer
    # records what it WROTE, a reconstruction reproduces it, and a JSON that
    # records nothing rebuilds as :calculator. Every raster of a board is the PDF
    # converted, and it agrees with the profile sampled exactly (`render_pattern`)
    # to one 8-bit level.
    recorded(d) = JSON.parse(read(onlyjson(d), String)).render.shading_function
    rendered(d, res) = Int16.(reinterpret(UInt8, BlobBoards.rasterize_pdf(onlypdf(d), res)))
    # the converted pattern against the exact sampler, max |difference| in 8-bit levels
    sheet_vs_sampler(d) = vs_sampler(
        BlobBoards.rasterize_pdf(onlypdf(d), 300), JSON.parse(read(onlyjson(d), String), BlobBoards.BoardBundle), 300
    )
    mk(dir; kw...) = blob_board(
        "A5"; blob_sizes = (1.6mm, 0.2short_side), seed = 0x42, dir, id_label = false, kw...
    )
    mktempdir() do root
        d(name) = joinpath(root, name)
        mk(d("calculator"))                                  # the default
        mk(d("sampled"); output = Pdf(shading_function = :sampled))
        @test recorded(d("calculator")) == "calculator" && recorded(d("sampled")) == "sampled"
        # what each file carries (the dictionaries are plain text)
        sv, ss = (String(read(onlypdf(d(n)))) for n in ("calculator", "sampled"))
        @test occursin("/ShadingType 3", sv) && occursin("/FunctionType 4", sv)
        @test occursin("/ShadingType 3", ss) && occursin("/FunctionType 0", ss)
        # the removed raster engine is refused, saying why
        @test_throws ArgumentError Pdf(shading_function = :raster)

        @test sheet_vs_sampler(d("calculator")) <= 1
        @test sheet_vs_sampler(d("sampled")) <= 1
        for (name, kw) in (
                "light" => (; polarity = :light),
                "frameless_margin" => (; corner_size = 0mm, ruler_width = 0mm, board_size = BlobBoards.Size2(width = 120mm, height = 160mm)),
                "corners_only" => (; ruler_width = 0mm),
            )
            mk(d(name); kw...)
            @test sheet_vs_sampler(d(name)) <= 1
        end

        # either function, in the print master
        for enc in (:calculator, :sampled)
            dir = d("x4_$(enc)")
            blob_board("A5"; blob_sizes = (4mm, 0.2short_side), seed = 0x43, dir, output = Pdf(shading_function = enc))
            f = String(read(onlypdf(dir)))
            @test occursin(enc === :calculator ? "/FunctionType 4" : "/FunctionType 0", f)
            @test recorded(dir) == string(enc)
            # strict object syntax: an EOL before every `endobj` and `endstream`
            # (ISO 19005 6.1.7.1, 6.1.9)
            @test !occursin(r"[^\r\n]endobj", f) && !occursin(r"[^\r\n]endstream", f)
            # an X-4 master is also PDF/A-2b
            @test occursin("/GTS_PDFA1", f) && occursin("<pdfaid:part>2</pdfaid:part>", f)
        end
        # veraPDF (the PDF Association's validator; no PDF/X profile) finds the
        # X-4 master compliant as PDF/A-2b, when it is installed (a flatpak here):
        # both functions, a framed board with its label, a light one
        if Sys.which("flatpak") !== nothing && success(pipeline(`flatpak info org.verapdf.veraPDF`; stdout = devnull, stderr = devnull))
            for name in ("x4_calculator", "x4_sampled", "calculator", "light")
                report = read(
                    pipeline(`flatpak run --filesystem=$(d(name)):ro --command=verapdf org.verapdf.veraPDF
                        --flavour 2b --format xml $(onlypdf(d(name)))`; stderr = devnull), String
                )
                @test occursin("isCompliant=\"true\"", report)
            end
        else
            @info "veraPDF not installed: the PDF/A-2b validation is skipped"
        end

        # reconstruction: as recorded; an output named instead overrides and is what is recorded
        rv = rendered(d("calculator"), 300)
        json_v = read(onlyjson(d("calculator")), String)
        BlobBoards.rebuild_board(json_v; dir = d("re_vector"))
        @test recorded(d("re_vector")) == "calculator" && rendered(d("re_vector"), 300) == rv
        BlobBoards.rebuild_board(json_v; dir = d("re_override"), output = Pdf(shading_function = :sampled))
        @test recorded(d("re_override")) == "sampled"
        # a JSON recording no encoding rebuilds as :calculator
        legacy = JSON.parse(json_v; dicttype = Dict{String, Any}); delete!(legacy["render"], "shading_function")
        @test BlobBoards._render_options(JSON.parse(JSON.json(legacy))).shading_function === :calculator
        BlobBoards.rebuild_board(JSON.json(legacy); dir = d("re_legacy"))
        @test recorded(d("re_legacy")) == "calculator" && rendered(d("re_legacy"), 300) == rv
        # a record written before 2026-10-13 names it `blob_encoding`
        for (old, new) in ("vector" => :calculator, "vector_sampled" => :sampled)
            legacy["render"]["blob_encoding"] = old
            @test BlobBoards._render_options(JSON.parse(JSON.json(legacy))).shading_function === new
        end
        # and the old values are refused, naming the new ones
        err = try; Pdf(shading_function = :vector_sampled); nothing; catch e; e; end
        @test err isa ArgumentError && occursin(":sampled", err.msg)

        # a raster has no shading function (it is the default PDF, converted): it records none,
        # and a record without one reads as the calculator function
        blob_board("A5"; blob_sizes = (4mm, 0.2short_side), seed = 0x44, dir = d("tiff"), output = Tiff(100dpi))
        @test recorded(d("tiff")) === nothing
        @test BlobBoards._render_options(JSON.parse(read(onlyjson(d("tiff")), String))).shading_function === :calculator
    end
end

@testset "outputs: the paths a default board does not take" begin
    AE = BlobBoards.Authoring
    mktempdir() do root
        d(name) = joinpath(root, name)
        kw = (; blob_sizes = (4mm, 0.2short_side), seed = 0x99)
        # a profile of the caller's: embedded in the PDF's OutputIntents and in the TIFF, and recorded
        icc = joinpath(root, "my-grey.icc")
        cp(joinpath(pkgdir(BlobBoards), "assets", "Gray-linear-v2-prtr.icc"), icc)
        blob_board("A5"; kw..., dir = d("icc_pdf"), output = Pdf(icc_profile = icc))
        s = String(read(onlypdf(d("icc_pdf"))))
        @test occursin("/OutputCondition (my-grey)", s) && count("/DestOutputProfile", s) == 2
        # recorded with its bytes, not its path: the rebuild needs the original file no more
        rec = JSON.parse(read(onlyjson(d("icc_pdf")), String)).render.icc_profile
        @test rec.name == "my-grey.icc" && rec.sha256 == bytes2hex(BlobBoards.sha256(read(icc)))
        blob_board("A5"; kw..., dir = d("icc_tiff"), output = Tiff(100dpi; icc_profile = icc))
        rm(icc)
        BlobBoards.rebuild_board(read(onlyjson(d("icc_pdf")), String); dir = d("icc_re"))            # rebuilt with the recorded profile
        @test occursin("/OutputCondition (my-grey)", String(read(onlypdf(d("icc_re")))))
        @test !isempty(first(TiffImages.load(only(filter(endswith(".tiff"), readdir(d("icc_tiff"); join = true)))).ifds)[TiffImages.ICCPROFILE].data)
        # a light board with the sampled function: the table is inverted, and the sheet agrees with the sampler
        b = blob_board("A5"; kw..., blob_sizes = (1.6mm, 0.2short_side), polarity = :light, id_label = false,
            dir = d("light_sampled"), output = Pdf(shading_function = :sampled))
        @test vs_sampler(BlobBoards.rasterize_pdf(onlypdf(d("light_sampled")), 300), b, 300) <= 1
        # the width/height form writes; a board without its ruler keeps its uid characters and its label
        blob_board(148mm, 210mm; kw..., ruler_width = 0mm, dir = d("wh"))
        c = page_content(onlypdf(d("wh")))
        @test count("2 Tr", c) == 3 && count("0 Tr", c) == 1 && !occursin(" l\nS", c)   # three uid characters, the label, no ticks
        @test JSON.parse(read(onlyjson(d("wh")), String)).render.id_label === true
        # a v4 PNG is read back linear, as written: the pixels its PDF converts to (a PNG reader's
        # default applies a display gamma to a linear file)
        bp = blob_board("A5"; kw..., dir = d("png"), output = Png(100dpi))
        blob_board("A5"; kw..., dir = d("pdf"))
        png = only(filter(endswith(".png"), readdir(d("png"); join = true)))
        img = BlobBoards._load_image(png, bp)
        ref = BlobBoards.rasterize_pdf(onlypdf(d("pdf")), 100)
        @test img == ref[axes(img)...]
        @test load_board(onlyjson(d("png"))).image_path == png
        # a v3 board as a TIFF, at its own density
        v3, _, h3 = BlobBoards.load_bundle(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"))
        BlobBoards.render_board(v3; output = Tiff(300dpi), dir = d("v3_tiff"), config_hash = h3)
        @test size(TiffImages.load(only(filter(endswith(".tiff"), readdir(d("v3_tiff"); join = true))))) == (909, 1192)
        # the preamble and JSON forms take the output group only, and say where a retired keyword went
        for f in (kws -> blob_board(bp.preamble; kws...), kws -> BlobBoards.rebuild_board(read(onlyjson(d("pdf")), String); kws...))
            err = try; f((; dir = d("x"), format = "png")); nothing; catch e; e; end
            @test err isa ArgumentError && occursin("output", err.msg)
            @test_throws ArgumentError f((; dir = d("x"), frame_style = :none))
        end
        @test !isdir(d("x"))
        # the font's Unicode map splits past 100 characters (a CMap block holds at most 100 entries)
        chars = sort!(collect(filter(c -> !iszero(AE.FreeTypeAbstraction.glyph_index(AE._sheet_font(), c)), Char.(0x21:0x17f))))[1:150]
        objs = AE._font_objects(AE._sheet_font(), chars, 12)
        cmap = String(AE._inflate(objs[5].data))
        @test occursin("100 beginbfchar", cmap) && occursin("50 beginbfchar", cmap) && count("endbfchar", cmap) == 2
    end
end

@testset "the page is exactly the canvas, on the 0.01 pt grid" begin
    # The page is exactly the canvas's grid units · 0.01 pt (A5: 41952 × 59527
    # units = 419.52 × 595.27 pt, 1.5 µm inside ISO), written as those two
    # decimals. A conversion is that page rounded up to whole pixels.
    for (paper, mb) in (("A5", ("419.52", "595.27")), ("Letter", ("612.00", "792.00")))
        mktempdir() do dir
            b = blob_board(paper; blob_sizes = (4mm, 0.2short_side), seed = 0x81, dir, id_label = false)
            c = BlobBoards.BoardLayout(b.preamble.board_config).canvas_size
            w, h = Int.(ustrip.(px, (c.width, c.height)))
            @test (hundredths(w), hundredths(h)) == mb
            m = match(r"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", String(read(onlypdf(dir))))
            @test (on_grid(m[1]), on_grid(m[2])) == mb
            @test size(BlobBoards.rasterize_pdf(onlypdf(dir), 300)) == ceil.(Int, (h, w) .* (300 / 7200) .- 1e-6)
        end
    end
end

@testset "canvas frame: the layout's own margins, drawn where the blobs say" begin
    # A5 around a 120 × 161 mm board: `Translation(:pattern, :canvas)` (what
    # `blobs(bundle, :canvas)` and the gallery use) is the layout's margins in
    # whole grid units, and the sheet's blobs are where it puts them, to one
    # 8-bit level at three densities.
    mktempdir() do dir
        b = blob_board(
            "A5"; blob_sizes = (1mm, 0.2short_side), seed = 0x51, dir, id_label = false,
            board_size = BlobBoards.Size2(width = 120mm, height = 161mm)
        )
        bc = b.preamble.board_config
        sizes = BlobBoards.BoardLayout(bc)
        left, _, _, top = sizes.margin
        T = BlobBoards.CoordinateTransformations.Translation(bc, :pattern, :canvas).translation
        @test Tuple(T) == (left / 100, top / 100)          # the pattern is the board: at the margins
        for d in (300, 600, 1200)
            @test vs_sampler(BlobBoards.rasterize_pdf(onlypdf(dir), d), b, d) <= 1
        end
        # the world frame: centred on the page the print master holds
        x0, y0 = T
        mb = parse.(Float64, split(match(r"/MediaBox\s*\[([^\]]*)\]", String(read(onlypdf(dir)))).captures[1]))
        mmpt = 25.4 / 72
        for (d, wd) in zip(b.blobs, blobs(b, :world))
            c = ustrip.(px, d.center)
            @test collect(wd.center) ≈ [(x0 + c[1]) * mmpt - mb[3] * mmpt / 2, mb[4] * mmpt / 2 - (y0 + c[2]) * mmpt] atol = 1e-9
        end
    end
end

@testset "QR code: vector rectangles on the grid, dark = `true`" begin
    # The QR is drawn as rectangles, `rat` whole grid units per module, centred
    # in the corner cell — with the encoder's polarity (`true` dark, a light
    # quiet zone): until 2026-10 every printed QR was colour-inverted and
    # unreadable. With it, a vector board holds no image. Read back at 1200 dpi,
    # every module centre is the module's colour.
    AE = BlobBoards.Authoring
    for padding in (0mm, 2mm)
        mktempdir() do dir
            b = blob_board("A5"; blob_sizes = (12.0mm, 0.2short_side), seed = 0x61, dir, corner_size = 25mm, frame_padding = padding, id_label = false)
            pdf = onlypdf(dir)
            @test !occursin(r"/Subtype\s*/Image", String(read(pdf)))
            bc = b.preamble.board_config
            sizes = BlobBoards.BoardLayout(bc)
            left, _, _, top = sizes.margin
            ch, cw = Int.(ustrip.(px, (sizes.corner_size.height, sizes.corner_size.width)))
            qr = AE.make_qr_code(BlobBoards.board_code(b))
            pad = floor(Int, ustrip(px, bc.frame_padding * BlobBoards.grid_density(bc)))
            rat = minimum(floor.(Int, (ch - 2pad, cw - 2pad) ./ size(qr)))
            @test rat >= 1
            h, w = size(qr)
            y0, x0 = floor(Int, (ch - h * rat) / 2), floor(Int, (cw - w * rat) / 2)
            img = Float64.(BlobBoards.rasterize_pdf(pdf, 1200))
            px_at(u) = floor(Int, u * 1200 / 7200) + 1                     # grid units → the pixel holding them
            dark = [img[px_at(top + y0 + (i - 0.5) * rat), px_at(left + x0 + (j - 0.5) * rat)] < 0.5 for i in 1:h, j in 1:w]
            @test dark == qr
        end
    end
end

@testset "vector blobs: the print master carries every blob EXACTLY" begin
    # Each blob is the shared shading under its own matrix. Every centre is
    # written EXACTLY — a 0.01 pt grid point, two decimals — as the very string
    # the stored value denotes in canvas points, and every radius to 1e-9
    # relative (ten significant digits). Matched by position.
    for (paper, density, kw) in (
            ("A5", 600dpi, (; blob_sizes = (1.6mm, 0.2short_side), seed = 0x71)),
            ("A0", 300dpi, (; blob_sizes = (8mm, 0.2short_side), seed = 0x03)),           # 3370 pt high
            (BlobBoards.Size2(width = 201.3mm, height = 257.7mm), 1200dpi, (; blob_sizes = (1mm, 12mm), seed = 0x02,
                polarity = :light, board_size = BlobBoards.Size2(width = 150.7mm, height = 171.1mm), ruler_width = 11.3mm)),
        )
        mktempdir() do dir
            b = blob_board(paper; dir, id_label = false, kw...)
            n, nmats, exact, radius = print_master_blobs(b, onlypdf(dir))
            @test n == nmats == exact
            @test radius <= 1e-9
        end
    end
end

@testset "default layout: a 10 mm margin, a 10 mm frame" begin
    # The default board clears every printer's unprintable border (≤ 6.35 mm)
    # with room for the id label; the frame is constant, whatever the board.
    a4 = blob_board("A4"; blob_sizes = (12.0mm, 0.2short_side), seed = 1).preamble.board_config
    @test a4.board_size == BlobBoards.Size2(width = 190mm, height = 277mm)
    @test a4.ruler_width == 10mm && a4.corner_size == 10mm
    @test blob_board("A0"; seed = 1, blob_sizes = (8mm, 0.2short_side)).preamble.board_config.ruler_width == 10mm
    # an explicit board_size is used as given: full bleed is board_size = canvas
    @test blob_board("A5"; blob_sizes = (12.0mm, 0.2short_side), seed = 1, board_size = BlobBoards.PRINT_SIZES["A5"]).preamble.board_config.board_size ==
        BlobBoards.PRINT_SIZES["A5"]
    # a canvas too small for the margin is refused, naming the way out
    err = try; blob_board(15mm, 40mm; blob_sizes = (12.0mm, 0.2short_side), seed = 1); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("board_size", err.msg)
    # the default A4 board's ink stays ≥ 6 mm inside the page, the id label drawn
    mktempdir() do dir
        blob_board("A4"; blob_sizes = (12.0mm, 0.2short_side), seed = 2, dir)
        dark = Float64.(BlobBoards.rasterize_pdf(onlypdf(dir), 300)) .< 0.5
        rows, cols = findall(vec(any(dark; dims = 2))), findall(vec(any(dark; dims = 1)))
        inset = (rows[1] - 1, size(dark, 1) - rows[end], cols[1] - 1, size(dark, 2) - cols[end]) .* (25.4 / 300)
        @test minimum(inset) > 6
        @test JSON.parse(read(onlyjson(dir), String)).render.id_label === true
    end
end

@testset "a v4 identity covers the schema, the configuration and the placed blobs" begin
    b = blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0x1D)
    h = hash_config(b.preamble)
    # minted from what was placed: the bundle's own blobs give the same identity, without placing
    @test BlobBoards._bundle_identity(b) == h
    # the same configuration with other blobs is another board: another identity
    @test BlobBoards._board_identity(b.preamble, b.blobs[2:end]) != h
    moved = [BlobBoards.Disc(d.center .+ (0.01px, 0px), d.σ) for d in b.blobs]
    @test BlobBoards._board_identity(b.preamble, moved) != h
    # ... and a code whose blobs would come out otherwise fails its check: a code for this board,
    # its check swapped for that of the same configuration with one blob fewer, is refused
    code = BlobBoards.board_code(b.preamble)
    forged = join([split(code, "/")[1:(end - 1)]; uppercase(BlobBoards._board_identity(b.preamble, b.blobs[2:end])[1:8])], "/")
    @test_throws ArgumentError BlobBoards.parse_board_code(forged)
    # the tag is the schema's version: another schema's code is refused by name
    err = try; BlobBoards.parse_board_code(replace(code, r"^BB4" => "BB5")); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("schema v5", err.msg)
    @test BlobBoards.BOARD_CODE_TAG == "BB" * last(split(BlobBoards.BOARD_SCHEMA, "/v"))
    # a stored v3 board's identity is still its configuration's alone (its stored hash)
    v3, _, h3 = BlobBoards.load_bundle(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"))
    @test hash_config(v3.preamble) == h3 == BlobBoards._bundle_identity(v3)
end

@testset "board code: the definition the QR carries, rebuilt exactly" begin
    # every value of the identity, positionally, in QR's alphanumeric set; the
    # decoder recomputes the identity and refuses a code whose check disagrees
    boards = [
        blob_board("A4"; blob_sizes = (6.0mm, 0.2short_side), seed = 0xA4),
        blob_board("Letter"; blob_sizes = (6.0mm, 0.2short_side), seed = 1, board_size = BlobBoards.Size2(width = 190.5mm, height = 252.73mm),
            ruler_width = 0.5inch, corner_size = 0.5inch, frame_padding = 0.3mm),                        # 8.5 inch = 215.89999999999998 mm
        blob_board("A5"; blob_sizes = (12.0mm, 0.2short_side)),                                                # a random 64-bit seed
        blob_board("A5"; seed = 7, polarity = :light, corner_size = 0mm, ruler_width = 0mm,
            cutoff = 2.5, scales_per_octave = 4, blob_sizes = (2mm, 9mm)),
    ]
    alnum = Set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ \$%*+-./:")
    for b in boards
        code = BlobBoards.board_code(b.preamble)
        @test all(in(alnum), code) && startswith(code, "BB4/")
        p = BlobBoards.parse_board_code(code)
        @test hash_config(p) == hash_config(b.preamble)
        @test JSON.json(p) == JSON.json(b.preamble)
        @test BlobBoards._place_board_blobs(p) == b.blobs                       # the same board
    end
    code = BlobBoards.board_code(boards[1].preamble)
    f = split(code, '/')
    edited = join([f[1:12]; "A5"; f[14:end]], "/")                              # another seed
    @test_throws ArgumentError BlobBoards.parse_board_code(edited)
    @test_throws ArgumentError BlobBoards.parse_board_code(join(f[1:(end - 1)], "/"))   # a field short
    @test_throws ArgumentError BlobBoards.parse_board_code("BB9" * code[4:end])
    @test_throws ArgumentError BlobBoards.parse_board_code(join([f[1:5]; "X"; f[7:end]], "/"))   # a ruler width that is no number
    # a field of the wrong kind, each
    field(i, v) = join([f[1:(i - 1)]; v; f[(i + 1):end]], "/")
    for (i, v) in ((2, "A4"), (16, "3.5"), (13, "XYZ"), (20, "Q"))   # number, integer, hex, polarity
        @test_throws ArgumentError BlobBoards.parse_board_code(field(i, v))
    end
    # a length is a number and an upper-cased registry unit symbol of length; it
    # reads back to the identical double, in mm and in inches, the A4 and Letter
    # sheets and their 10 mm margins (the derived board, 190 × 277 mm, included)
    for q in (210mm, 297mm, 10mm, 0mm, 5mm, 0.3mm, 190mm, 277mm, 190.5mm, 252.73mm,
              8.5inch, 11inch, 0.5inch, 0.25inch)
        c = BlobBoards._code_length(q)
        @test all(in(alnum), c)
        r = BlobBoards._parse_length(c)
        @test r == q && ustrip(r) === ustrip(q)                                   # the stored base value, bit for bit
    end
    @test BlobBoards._code_length(210mm) == "210MM"
    @test BlobBoards._code_length(8.5inch) == "8.5INCH"                         # 215.89999999999998MM is longer
    @test BlobBoards._parse_length("21CM") == 210mm                              # any registered length unit reads
    for bad in ("210", "10PX", "10DPI", "10FOO", "MM", "1.2.3MM", "10mm")        # no unit, not a length, unregistered, no number, lower case
        @test BlobBoards._parse_length(bad) === nothing
    end
    @test_throws ArgumentError BlobBoards.parse_board_code(field(2, "210"))
    # a code in the mm/IN spelling (until 2026-10-15) rebuilds no board
    @test_throws ArgumentError BlobBoards.parse_board_code("BB4/210/297/10/10/10/10/0/10/5/10/5/A4/4.25196850394/26.929/3/5000/1.2/2/D/0D435510")
    @test_throws ArgumentError BlobBoards.parse_board_code(replace(BlobBoards.board_code(boards[2].preamble), "MM" => ""))
    # a pre-2026-10-13 code (it carried a bit depth) rebuilds no board
    @test_throws ArgumentError BlobBoards.parse_board_code(join([f[1:(end - 1)]; "8"; f[end]], "/"))
    # a v3 board has none (its QR is its identity)
    v3, _, _ = BlobBoards.load_bundle(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"))
    @test_throws ArgumentError BlobBoards.board_code(v3.preamble)
    # nor a preamble whose pattern is not its board's cell on the grid: its code
    # would rebuild another board
    p = boards[1].preamble
    pc = p.pattern_config
    aligned = PatternConfig(; (k => getfield(pc, k) for k in fieldnames(PatternConfig))..., raster_aligned = true)
    @test_throws ArgumentError BlobBoards.board_code(BlobBoards.BoardPreamble(p.board_config, aligned))
end

@testset "a v4 board's identity and code are pinned" begin
    # any change to the v4 identity (the layout_grid record, pt lengths, key
    # order, a default) changes these; the round trips above would not notice.
    # Re-pinned 2026-10-14: the pattern became the board (the frame's marks are
    # reserved areas on it), `corner_size` joined the identity and the code, the
    # pattern size left the identity (it is the board's), the identity came to
    # cover the schema and the placed blobs, and the marks came to be switched by
    # their sizes (`ruler_width`, `corner_size`; no `frame_style`), and
    # `raster_aligned` left the record (always false on the point grid).
    # Re-pinned 2026-10-15: the board records its margin (its board is derived),
    # and the code carries the margin; blobs unchanged (165).
    # Re-spelled 2026-10-15: a length is a number and its registry unit symbol,
    # upper-cased (`210MM`); identity and blobs unchanged.
    b = blob_board("A4"; blob_sizes = (6.0mm, 0.2short_side), seed = 0xA4)
    @test hash_config(b.preamble) == "0d43551099e12e6181f74786bc81f39e0564e40267c08ce610dc717ef125ff67"
    @test BlobBoards.board_code(b.preamble) ==
        "BB4/210MM/297MM/10MM/10MM/10MM/10MM/0MM/10MM/5MM/10MM/5MM/A4/4.25196850394/26.929/3/5000/1.2/2/D/0D435510"
    @test length(b.blobs) == 165
end

@testset "blob_board and render_board refuse what they would ignore" begin
    @test_throws ArgumentError blob_board("A5"; blob_sizes = (12mm, 0.2short_side), frame_stlye = :none)        # unknown, with the valid list
    @test_throws ArgumentError blob_board("A5"; blob_sizes = (12mm, 0.2short_side), ruler_spec = nothing)       # built from the tick keywords
    v3, _, h = BlobBoards.load_bundle(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"))
    mktempdir() do dir
        # a v3 board's layout IS its printer's pixels (300 dpi here): a raster at no other density, by any route
        @test_throws ArgumentError BlobBoards.render_board(v3; output = Png(600dpi), dir, config_hash = h)
        @test_throws ArgumentError blob_board(v3.preamble; dir, output = Tiff(600dpi))
        @test_throws ArgumentError load_board(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"); density = 600dpi)
        BlobBoards.render_board(v3; output = Png(300dpi), dir, config_hash = h)
        @test size(BlobBoards._load_image(only(filter(endswith(".png"), readdir(dir; join = true))), v3)) == (909, 1192)   # its canvas, pixel for pixel
        # a raster needs 2 pixels per σ at its density, whatever made the board:
        # 2 mm discs (σ = 0.5 mm) are 8 px at 400 dpi, under 2 px per σ at 100
        small = blob_board("A5"; seed = 4, blob_sizes = (2mm, 0.2short_side))
        @test_throws ArgumentError BlobBoards.render_board(small; output = Png(100dpi), dir)
        @test_throws ArgumentError blob_board(small.preamble; dir, output = Png(100dpi))
        blob_board(small.preamble; dir = joinpath(dir, "pdf"))                     # a PDF asks nothing
    end
end

@testset "boards written before 2026-10-13 still load" begin
    # a v4 sidecar then recorded `pixel_type: "gray8"` and `render.blob_encoding`;
    # it loads under its STORED identity (its recomputed one changed when the
    # pixel type left the v4 identity), and the legacy record maps across
    mktempdir() do dir
        blob_board("A5"; blob_sizes = (9.0mm, 0.2short_side), seed = 0x7, dir)
        json = onlyjson(dir)
        obj = JSON.parse(read(json, String); dicttype = Dict{String, Any})
        obj["preamble"]["pattern_config"]["pixel_type"] = "gray8"
        delete!(obj["render"], "shading_function")
        obj["render"]["blob_encoding"] = "raster"                 # a TIFF/PNG of 2026-10-10/11
        write(json, JSON.json(obj))
        b, _, h = BlobBoards.load_bundle(json)
        @test h == obj["hashes"]["config"]
        @test BlobBoards._render_options(JSON.parse(read(json, String))).shading_function === :calculator
        obj["preamble"]["pattern_config"]["pixel_type"] = "gray16"     # never a board's
        write(json, JSON.json(obj))
        @test_throws ArgumentError BlobBoards.load_bundle(json)
    end
end

@testset "load_board: a v4 board's pixels, at its own raster density or the caller's (CPU)" begin
    mktempdir() do dir
        blob_board("A5"; blob_sizes = (12mm, 0.2short_side), seed = 5, dir)
        json = onlyjson(dir)
        # a PDF beside the JSON has no density: converted at the caller's …
        lb = load_board(json; density = 300dpi)
        @test ustrip(dpi, lb.density) ≈ 300
        img = BlobBoards._load_image(lb.image_path, lb.bundle; density = lb.density)
        @test size(img) == (2481, 1748)            # A5 at 300 dpi: 595.27 × 419.52 pt on the grid, Poppler's ceil
        # the largest blob's centre, in canvas pixels at that density, is dark
        j = argmax([d.σ for d in BlobBoards.blobs(lb.bundle, :world)])
        x, y = lb.canvas_frames[j].center                                           # :colmap, centre of pixel (1,1) at 0.5
        @test Float64(img[ceil(Int, y), ceil(Int, x)]) < 0.1
        # … else at the board's own: its smallest disc (12 mm) 16 px across, 34 dpi —
        # no render record needed, a PDF records no density
        obj = JSON.parse(read(json, String); dicttype = Dict{String, Any})
        delete!(obj, "render")
        write(json, JSON.json(obj))
        own = load_board(json)
        @test ustrip(dpi, own.density) == 34
        small = BlobBoards._board_raster(own.image_path, own.bundle, own.config_hash; density = own.density)
        @test size(small) == (282, 199)
        x, y = own.canvas_frames[j].center
        @test Float64(small[ceil(Int, y), ceil(Int, x)]) < 0.1
        # a raster beside the JSON is at the density its sidecar records; without the record it is unknown
        png = joinpath(dir, "png")
        blob_board("A5"; blob_sizes = (12mm, 0.2short_side), seed = 5, dir = png, output = Png(150dpi))
        @test ustrip(dpi, load_board(onlyjson(png)).density) == 150
        @test_throws ArgumentError load_board(onlyjson(png); density = 300dpi)
        obj = JSON.parse(read(onlyjson(png), String); dicttype = Dict{String, Any})
        delete!(obj, "render")
        write(onlyjson(png), JSON.json(obj))
        @test_throws ErrorException load_board(onlyjson(png))
    end
end

@testset "stored v3 boards render exactly as they did" begin
    # Two boards written before the point grid (schema v3: laid out on the
    # printer's pixels): rebuilt from the config and rendered from the stored
    # blobs, every blob shading and the page are the numbers their print
    # masters held (captured at fae07d5, when Ghostscript wrote them at 6
    # significant digits, the last one truncated: equal to one unit in the sixth,
    # the centres exactly).
    fixtures = joinpath(@__DIR__, "..", "fixtures", "v3_render")
    numbers(line) = parse.(Float64, split(line))
    # [the page; every blob's matrix, ordered by its centre (the exact part)]
    by_centre(mats) = sort(mats; by = m -> (m[5], m[6]))
    masters(dir) = let s = String(read(onlypdf(dir)))
        [numbers(match(r"/MediaBox\s*\[([^\]]*)\]", s).captures[1]), by_centre(map(m -> parse.(Float64, m), blob_matrices(onlypdf(dir))))...]
    end
    stored(text) = let lines = split(strip(text), '\n')
        [numbers(replace(lines[1], "MediaBox" => "")), by_centre(numbers.(lines[2:end]))...]
    end
    same(a, b) = length(a) == length(b) && all(u[5:end] == v[5:end] && isapprox(u[1], v[1]; rtol = 1.0e-5) for (u, v) in zip(a[2:end], b[2:end])) && a[1] == b[1]
    for name in ("framed600", "odd300")
        json = joinpath(fixtures, name * ".json")
        want = stored(read(joinpath(fixtures, name * ".matrices.txt"), String))
        b, _, h = BlobBoards.load_bundle(json)
        @test b.preamble.board_config.layout_grid isa BlobBoards.PixelGrid
        mktempdir() do dir
            BlobBoards.rebuild_board(read(json, String); dir)
            @test same(masters(dir), want)
            @test JSON.parse(read(onlyjson(dir), String)).schema == "blobboards/board/v3"
        end
        mktempdir() do dir
            BlobBoards.render_board(b; output = Pdf(), dir, config_hash = h)
            @test same(masters(dir), want)
        end
    end
end

@testset "rebuild_board prints the stored blobs; a v4 board proves its configuration places them" begin
    mktempdir() do dir
        b = blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0x4B, dir)
        json = read(onlyjson(dir), String)
        @test BlobBoards.rebuild_board(json).blobs == b.blobs                         # the stored blobs, verified
        @test BlobBoards.rebuild_board(onlyjson(dir)).blobs == b.blobs               # ... or from the file's path
        # blob_board takes a paper name only: a JSON there says where it goes
        err = try; blob_board(json); nothing; catch e; e; end
        @test err isa ArgumentError && occursin("rebuild_board", err.msg)
        @test_throws ArgumentError BlobBoards.rebuild_board("no/such/file.json")
        # a JSON whose blobs its configuration does not place is refused, not reprinted otherwise
        obj = JSON.parse(json; dicttype = Dict{String, Any})
        obj["blobs"]["centers"][1][1] += 1.0
        @test_throws ArgumentError BlobBoards.rebuild_board(JSON.json(obj))
    end
    # a stored v3 board is drawn from its JSON's blobs, never re-placed
    v3json = read(joinpath(@__DIR__, "..", "fixtures", "v3_render", "odd300.json"), String)
    @test BlobBoards.rebuild_board(v3json).blobs == JSON.parse(v3json, BlobBoards.BoardBundle).blobs
end

@testset "SOURCE_DATE_EPOCH: byte-identical print masters" begin
    withenv("SOURCE_DATE_EPOCH" => "1760000000") do                 # 2025-10-09T08:53:20Z
        a, b = mktempdir(), mktempdir()
        blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0x11, dir = a)
        blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0x11, dir = b)
        @test read(onlypdf(a)) == read(onlypdf(b))
        @test occursin("/CreationDate (D:20251009085320Z)", String(read(onlypdf(a))))
    end
    withenv("SOURCE_DATE_EPOCH" => "yesterday") do
        @test_throws ArgumentError blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0x11, dir = mktempdir())
    end
end

@testset "review regressions (2026-10-15)" begin
    # a board code's check is exactly 8 hex digits: an emptied or shortened one is refused
    b = blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0xC0, ruler_width = 0mm)
    code = BlobBoards.board_code(b)
    stem = code[1:findlast('/', code)]
    @test_throws ArgumentError BlobBoards.parse_board_code(stem)
    @test_throws ArgumentError BlobBoards.parse_board_code(stem * code[end - 7])

    # with the ruler off its ticks are zero, not the TOML's spacings: the identity
    # cannot move when those defaults are edited
    rs = b.preamble.board_config.ruler_spec
    @test iszero(rs.major_tick.spacing) && iszero(rs.minor_tick.spacing)

    # a size in the wrong unit is refused before placement; so is a ruler wider than the board
    @test_throws ArgumentError blob_board("A4"; blob_sizes = (0.006mm, 38mm))
    @test_throws ArgumentError blob_board("A5"; blob_sizes = (6mm, 0.2short_side), ruler_width = 70mm)
    # an absurd max_trials is refused before placement, from a keyword or an edited board code
    @test_throws ArgumentError blob_board("A5"; blob_sizes = (6mm, 0.2short_side), max_trials = 100_001)
    edited = replace(code, "/5000/" => "/1000000000/")
    @test_throws ArgumentError BlobBoards.parse_board_code(edited)

    # a small canvas with an explicit board size does not need the default margin
    @test BlobBoards.BoardConfig(;
        canvas_size = BlobBoards.Size2(width = 20mm, height = 20mm), board_size = BlobBoards.Size2(width = 20mm, height = 20mm),
        corner_size = 0mm, ruler_width = 0mm,
    ) isa BlobBoards.BoardConfig

    # an ICC profile must be a grey ICC profile, not any existing file
    @test_throws ArgumentError Pdf(icc_profile = joinpath(pkgdir(BlobBoards), "Project.toml"))

    # a stored v3 board whose render record names a raster but no density rebuilds at its print density
    mktempdir() do dir
        BlobBoards.rebuild_board(joinpath(pkgdir(BlobBoards), "test", "fixtures", "v3_raster_render_no_density.json"); dir)
        @test any(endswith(".png"), readdir(dir))
    end

    # a PDF is converted at a whole number of dpi: another density is refused, not rounded
    mktempdir() do dir
        bb = blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 0xC0, dir)
        json = only(filter(endswith(".json"), readdir(dir; join = true)))
        @test_throws ArgumentError BlobBoards.load_board(json; density = 101.6dpi)
    end

    # the PDF dates carry a numeric UTC offset whatever the platform's %z
    d, x = BlobBoards.Authoring._pdf_dates(0)
    @test occursin(r"^D:\d{14}[+-]\d\d'\d\d'$", d)
    @test occursin(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$", x)
end

@testset "margin: the board is centred, the canvas less twice the margin" begin
    bc(b) = b.preamble.board_config
    kw = (; blob_sizes = (6mm, 0.2short_side), seed = 0x3)
    # the default: 10 mm every side; the board derived from it
    a4 = bc(blob_board("A4"; kw...))
    @test a4.margin == BlobBoards.Size2(width = 10mm, height = 10mm)
    @test a4.board_size == BlobBoards.Size2(width = 190mm, height = 277mm)
    # a Size2 (width: left and right, height: above and below), or a board size (its margin: half the excess)
    @test bc(blob_board("A4"; kw..., margin = BlobBoards.Size2(width = 5mm, height = 20mm))).board_size == BlobBoards.Size2(width = 200mm, height = 257mm)
    sq = bc(blob_board("A4"; kw..., board_size = BlobBoards.Size2(width = 180mm, height = 180mm)))
    @test sq.margin == BlobBoards.Size2(width = 15mm, height = 58.5mm) && sq.board_size == BlobBoards.Size2(width = 180mm, height = 180mm)
    @test bc(blob_board("A5"; kw..., margin = BlobBoards.Size2(width = 0mm, height = 0mm))).board_size == BlobBoards.PRINT_SIZES["A5"]
    # the two keywords size the same thing: giving both is refused, as is a margin that leaves no board
    m(w, h) = BlobBoards.Size2(width = w, height = h)
    @test_throws ArgumentError blob_board("A4"; kw..., margin = m(5mm, 5mm), board_size = m(180mm, 180mm))
    @test_throws ArgumentError blob_board("A5"; kw..., margin = m(80mm, 80mm))
    @test_throws ArgumentError blob_board("A5"; kw..., margin = m(-1mm, 10mm))
    @test_throws Exception blob_board("A5"; kw..., margin = 10mm)                 # a Size2, not a bare length
    # a v4 sidecar records the margin, not the board; it reads back to the same config
    mktempdir() do dir
        b = blob_board("Letter"; kw..., margin = m(0.25inch, 0.25inch), dir)
        txt = read(onlyjson(dir), String)
        @test occursin("\"margin\"", txt) && !occursin("board_size", txt)
        @test BlobBoards.load_board(onlyjson(dir)).bundle.preamble.board_config == bc(b)
    end
    # the code carries the margin and spells a length in inches when that is shorter and exact
    letter = blob_board("Letter"; kw...)
    code = BlobBoards.board_code(letter)
    @test startswith(code, "BB4/8.5INCH/11INCH/10MM/10MM/") && length(code) < 135
    @test hash_config(BlobBoards.parse_board_code(code)) == BlobBoards._bundle_identity(letter)
end

@testset "PDF text: non-ASCII as UTF-16; the font found at run time" begin
    @test BlobBoards.Authoring._pdf_string("a(b)") == "(a\\(b\\))"
    @test BlobBoards.Authoring._pdf_string("glänzend") == "<FEFF0067006C00E4006E007A0065006E0064>"
    # a custom profile's file name becomes the OutputCondition: any name is written
    mktempdir() do dir
        icc = joinpath(dir, "Fogra39_glänzend.icc")
        cp(joinpath(pkgdir(BlobBoards), "assets", "Gray-linear-v2-prtr.icc"), icc)
        out = joinpath(dir, "out")
        blob_board("A5"; blob_sizes = (6mm, 0.2short_side), seed = 1, dir = out, output = Pdf(icc_profile = icc))
        @test occursin("/OutputCondition <FEFF", String(read(onlypdf(out))))
    end
    # the sheet font's path is the package's, resolved at run time (not baked in at precompile)
    @test isfile(BlobBoards.Authoring._sheet_font_file())
    @test startswith(BlobBoards.Authoring._sheet_font_file(), pkgdir(BlobBoards))
end
