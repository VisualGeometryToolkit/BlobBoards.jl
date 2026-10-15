# =============================================================================
# Print quality: how faithfully a printer reproduces the blob profile, by size
# =============================================================================
#
# A board's PDF holds every blob exactly (a radial shading whose colour
# function is `blob_profile`); what reaches paper is the printer's rendering of
# it — a halftone of its RIP's making. This measures that, per blob diameter,
# from a scan of a printed test board.
#
# The blobs are KNOWN, so nothing is detected beyond the board itself:
# `find_boards` gives H, the board's `:world` plane (mm) → scan pixels; every
# ground-truth blob is mapped through it, and every scan pixel near a blob is
# mapped back through inv(H) to its normalised radius t on the board, so each
# grey value is paired with the exact profile value `blob_profile(t)`. The
# printer's and scanner's tone curves are removed by a monotone curve fitted on
# the largest blobs (which both reproduce faithfully); what remains at a smaller
# size is the loss of detail.
#
# Per blob diameter it reports
#   profile — RMS of (reflectance − blob_profile) inside the disc, 8-bit levels
#   centre  — the blob's ink centroid against its true centre, µm
#   scale   — the apparent size: the radius scale that best fits the profile
#             (> 1 blurred wider, < 1 eroded smaller)
#
#   julia --project=examples examples/print_quality/print_quality.jl generate DIR
#       the A4 test board: discs 0.4–12 mm (16 sizes, ~13 500 blobs), made for 1200 dpi
#   julia --project=examples examples/print_quality/print_quality.jl selfcheck BOARD.json [DPI]
#       the measurement's floor: a perfect render of the PDF at DPI (default 600)
#   julia --project=examples examples/print_quality/print_quality.jl evaluate BOARD.json SCAN.tif [DPI]
#       a scan of the printed board (DPI default 1200; grey, 8-bit or more, every
#       automatic correction and descreen off, lossless — not JPEG, not a
#       256-colour palette); writes SCAN.print_quality.csv beside it
#
# Printing: at 100 %, grey, the printer's best resolution, and from a PDF path
# that keeps the shadings vector (a CUPS queue converting with an old
# Ghostscript's ps2write rasterises them coarsely: `-o pdftops-renderer=pdftops`).
#
# Measured floor (2026-10-11, selfcheck at 600 dpi): 0.33 levels RMS at every
# size, centres 0.1–0.2 µm, scale 1.00.

using BlobBoards, CairoMakie, CUDA, cuDNN
using BlobBoards: mm, dpi, px, ustrip
using FileIO: FileIO
using StaticArrays: SMatrix, SVector
using Statistics: mean, median
using VisualGeometryCore: is_map_inlier

const TEST_BOARD = (paper = "A4", density = 1200dpi, blob_sizes = (0.4mm, 12mm), seed = 0x7E57)
const FIND_DPI = 150          # find_boards runs on the scan block-averaged to about this density
const TONE_FROM_MM = 6.0      # the tone curve is fitted on discs at least this large

# -----------------------------------------------------------------------------
# The board in the scan
# -----------------------------------------------------------------------------

# a k-times block-averaged copy (find_boards does not need 1200 dpi)
blockmean(A, k) = [
    sum(Float64, @view A[((i - 1) * k + 1):(i * k), ((j - 1) * k + 1):(j * k)]) / k^2
        for i in 1:(size(A, 1) ÷ k), j in 1:(size(A, 2) ÷ k)
]

# H for the full-resolution scan: find_boards on a k-times block-averaged copy
# (COLMAP pixel coordinates scale about the image corner, x_full = k·x_small)
function locate_board(img, scan_dpi, json_path)
    k = max(1, round(Int, scan_dpi / FIND_DPI))
    net = BlobBoards.DescriptorNetwork(BlobBoards.descriptor_network_path())
    gallery = BlobBoards.BoardGallery(joinpath(mktempdir(), "gallery.jld2"), net)
    features = Dict(insert!(gallery, [json_path], net))
    board = BlobBoards.load_board(json_path)
    small = BlobBoards.Gray{Float32}.(blockmean(img, k))
    found = find_boards(features, net, small; board_bundles = Dict(board.uid => board.bundle))
    isempty(found) && error("find_boards found no board in the scan")
    uid, model, _, scored = first(found)
    uid == board.uid || error("found board $uid, expected $(board.uid)")
    H = SMatrix{3, 3, Float64}(k, 0, 0, 0, k, 0, 0, 0, 1) * SMatrix{3, 3, Float64}(model)
    return board.bundle, H, count(is_map_inlier, scored.weights)
end

# a point through a homography (static: no allocation in the per-pixel loop)
apply(H::SMatrix{3, 3}, x, y) = (v = H * SVector(x, y, 1.0); (v[1] / v[3], v[2] / v[3]))

# -----------------------------------------------------------------------------
# Samples: each pixel near each blob, with its exact profile value
# -----------------------------------------------------------------------------

# For every blob, the other blobs whose cutoff disc reaches into its sampled
# disc (`tmax` × its radius) — found once through a grid of `cell`-mm cells.
function neighbour_lists(ws, c, tmax)
    cell = maximum(c * w.σ for w in ws)
    grid = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (j, w) in enumerate(ws)
        push!(get!(() -> Int[], grid, (floor(Int, w.center[1] / cell), floor(Int, w.center[2] / cell))), j)
    end
    return map(enumerate(ws)) do (j, w)
        reach = tmax * c * w.σ + cell                      # this disc's reach plus the largest radius
        n = ceil(Int, reach / cell)
        i0, j0 = floor(Int, w.center[1] / cell), floor(Int, w.center[2] / cell)
        [
            k for di in -n:n, dj in -n:n for k in get(grid, (i0 + di, j0 + dj), Int[])
                if k != j && hypot(w.center[1] - ws[k].center[1], w.center[2] - ws[k].center[2]) < tmax * c * w.σ + c * ws[k].σ
        ]
    end
end

in_any(ws, c, ks, X, Y) = any(k -> hypot(X - ws[k].center[1], Y - ws[k].center[2]) <= c * ws[k].σ, ks)

# Beyond its own disc (t > 1), a pixel inside a NEIGHBOUR's disc is dropped: the
# blobs are packed, so the ring around a blob holds other blobs' ink, which
# would read as the blob spreading and darken the tone curve's paper.
function blob_samples(img, H, bundle; tmax = 1.25)
    Hi = inv(H)
    c = BlobBoards.cutoff(bundle)
    ws = blobs(bundle, :world)
    nbrs = neighbour_lists(ws, c, tmax)
    return map(enumerate(ws)) do (j, w)
        X, Y = Tuple(w.center); R = c * w.σ                       # mm, the cutoff disc
        x0, y0 = apply(H, X, Y)
        ex = tmax * maximum(hypot((apply(H, X + dx, Y + dy) .- (x0, y0))...) for (dx, dy) in ((R, 0), (0, R), (-R, 0), (0, -R))) + 2
        rows = max(1, floor(Int, y0 + 0.5 - ex)):min(size(img, 1), ceil(Int, y0 + 0.5 + ex))
        cols = max(1, floor(Int, x0 + 0.5 - ex)):min(size(img, 2), ceil(Int, x0 + 0.5 + ex))
        t = Float64[]; g = Float64[]; wx = Float64[]; wy = Float64[]
        foreach(v -> sizehint!(v, length(rows) * length(cols)), (t, g, wx, wy))
        for r in rows, q in cols
            Xw, Yw = apply(Hi, q - 0.5, r - 0.5)                  # the pixel centre (COLMAP) on the board, mm
            tt = hypot(Xw - X, Yw - Y) / R
            tt <= tmax || continue
            tt > 1 && in_any(ws, c, nbrs[j], Xw, Yw) && continue
            push!(t, tt); push!(g, Float64(img[r, q])); push!(wx, Xw); push!(wy, Yw)
        end
        (; d_mm = 2R, X, Y, t, g, wx, wy)
    end
end

# The tone curve, inverted: scanned grey → reflectance on the profile's scale
# (0 ink, 1 paper), from the large discs and the paper around them: the median
# grey per profile bin, made monotone. One bin per 8-bit level: coarser bins add
# their own error (40 bins: 1.1 levels RMS).
function tone_inverse(samples, cutoff; dmin = TONE_FROM_MM, nbins = 255)
    f = Float64[]; g = Float64[]
    for s in samples
        s.d_mm >= dmin || continue
        for (t, v) in zip(s.t, s.g)
            push!(f, t <= 1 ? BlobBoards.blob_profile(t, cutoff) : 1.0); push!(g, v)
        end
    end
    bin = min.(nbins, floor.(Int, f .* nbins) .+ 1)                  # each sample's bin, once
    perm = sortperm(bin)
    fc = Float64[]; gc = Float64[]
    i = 1
    while i <= length(perm)
        b = bin[perm[i]]; n = i
        while n < length(perm) && bin[perm[n + 1]] == b
            n += 1
        end
        push!(fc, (b - 0.5) / nbins); push!(gc, median(@view g[perm[i:n]]))
        i = n + 1
    end
    return let fc = fc, gm = accumulate(max, gc)                     # monotone
        function (v)
            v <= gm[1] && return fc[1]
            v >= gm[end] && return fc[end]
            k = searchsortedlast(gm, v)
            return gm[k + 1] == gm[k] ? fc[k] : fc[k] + (fc[k + 1] - fc[k]) * (v - gm[k]) / (gm[k + 1] - gm[k])
        end
    end
end

# -----------------------------------------------------------------------------
# Per blob, then per diameter
# -----------------------------------------------------------------------------

# the radius scale that best fits the profile: 0.05 steps, then 0.01 around the best
function best_scale(r, t, cutoff)
    err(k) = sum(i -> abs2(r[i] - BlobBoards.blob_profile(clamp(t[i] / k, 0, 1), cutoff)), eachindex(t))
    coarse = argmin(err, 0.6:0.05:1.6)
    return argmin(err, max(0.6, coarse - 0.05):0.01:min(1.6, coarse + 0.05))
end

function evaluate(samples, finv, cutoff)
    per = map(samples) do s
        inside = s.t .<= 1
        r = finv.(s.g)
        f = BlobBoards.blob_profile.(clamp.(s.t, 0, 1), cutoff)
        w = max.(0.0, 1 .- r[inside])
        cx, cy = sum(w .* s.wx[inside]) / sum(w), sum(w .* s.wy[inside]) / sum(w)
        (;
            d = round(s.d_mm; digits = 2), profile = 255 * sqrt(mean((r[inside] .- f[inside]) .^ 2)),
            centre = 1000 * hypot(cx - s.X, cy - s.Y), scale = round(best_scale(r, s.t, cutoff); digits = 2),
        )
    end
    return map(sort(unique(p.d for p in per))) do d
        q = [p for p in per if p.d == d]
        (
            d_mm = d, n = length(q), profile = median(p.profile for p in q),
            centre_um = median(p.centre for p in q), scale = median(p.scale for p in q),
        )
    end
end

function report(io, rows)
    println(io, "d_mm,n,profile_rms_levels,centre_um,scale")
    for r in rows
        println(io, join((r.d_mm, r.n, round(r.profile; digits = 2), round(r.centre_um; digits = 1), r.scale), ","))
    end
    return
end

# The board's scale in the scan against the nominal density (print and scan
# together; per-blob errors below are relative to this best fit), and its rotation.
function global_scale(H, scan_dpi)
    o = apply(H, 0.0, 0.0)
    ex = apply(H, 1.0, 0.0) .- o; ey = apply(H, 0.0, 1.0) .- o
    nominal = ustrip(px / mm, scan_dpi * dpi)   # px per mm
    return (
        x_percent = 100 * (hypot(ex...) / nominal - 1), y_percent = 100 * (hypot(ey...) / nominal - 1),
        rotation_deg = atand(ex[2], ex[1]),
    )
end

function run_evaluation(json_path, img, scan_dpi)
    bundle, H, inliers = locate_board(img, scan_dpi, json_path)
    @info "board found" inliers global_scale(H, scan_dpi)
    samples = blob_samples(img, H, bundle)
    cutoff = BlobBoards.cutoff(bundle)
    return evaluate(samples, tone_inverse(samples, cutoff), cutoff)
end

function main(args)
    cmd = isempty(args) ? "" : args[1]
    return if cmd == "generate" && length(args) == 2
        b = blob_board(TEST_BOARD.paper; TEST_BOARD.blob_sizes, TEST_BOARD.seed, dir = args[2])
        @info "test board" blobs = length(b.blobs) dir = args[2]
    elseif cmd == "selfcheck" && length(args) in (2, 3)
        json = args[2]; d = length(args) == 3 ? parse(Int, args[3]) : 600
        pdf = replace(json, r"\.json$" => ".pdf")
        report(stdout, run_evaluation(json, BlobBoards.rasterize_pdf(pdf, d), d))
    elseif cmd == "evaluate" && length(args) in (3, 4)
        json, scan = args[2], args[3]; d = length(args) == 4 ? parse(Int, args[4]) : 1200
        img = Float64.(BlobBoards.Gray.(FileIO.load(scan)))
        rows = run_evaluation(json, img, d)
        report(stdout, rows)
        open(io -> report(io, rows), splitext(scan)[1] * ".print_quality.csv", "w")
    else
        println(stderr, "usage: print_quality.jl generate DIR | selfcheck BOARD.json [DPI] | evaluate BOARD.json SCAN [DPI]  (DPI: 600 / 1200)")
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
