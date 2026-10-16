# The landing page's hero image: one square board built by the package's own
# authoring API, as VECTOR graphics — the print master (each blob a radial
# shading) converted to SVG, so the hero is the same drawing a printer gets, at
# any resolution. The board: a 150 mm square canvas, no ruler, the default
# corner cells (QR code, uid glyphs), no margin and no id label (the board is
# the whole image), blob sizes from the smallest disc that prints well at
# 1200 dpi, 3 mm, to 20% of the short side, and a fixed seed.
#
#   julia --project=authoring docs/make_hero.jl
#
# writes docs/src/assets/landing/hero_board.svg (light) and hero_board_dark.svg,
# plus the board's JSON sidecar (its full configuration and blobs, for
# provenance) at docs/hero_board.json. The seed fixes placement, so the image
# follows from the package version alone. Authoring is CPU-only and headless.
#
# Poppler's pdftocairo writes every blob's gradient with its own ~240 sampled
# stops (5.8 MB for this board). All blobs share one profile — only their
# centre and radius differ — so the stops are written ONCE, resampled, in a
# base gradient that every blob's gradient inherits (`href`). The stops are the
# shading's linear reflectance; they are sRGB-encoded for display.

using BlobBoards

const SIDE = 150mm                     # square canvas
const SEED = 0x0000000b10bb0a2d        # fixed: placement is a function of it
const STOPS = 64                       # the shared profile's stops, evenly spaced
const CORNER = 0.025                   # rounded-corner radius, fraction of the side
const DARK_PAPER = 0.86                # the dark variant's paper tone, a soft light card

const OUT = joinpath(@__DIR__, "src", "assets", "landing")

svg = mktempdir() do dir
    blob_board(SIDE, SIDE;
        blob_sizes = (suggest_blob_sizes(1200dpi, (SIDE, SIDE))[1], 0.2short_side), seed = SEED,
        ruler_width = 0mm, margin = Size2(width = 0mm, height = 0mm), id_label = false,
        dir, output = Pdf())
    cp(only(filter(endswith(".json"), readdir(dir; join = true))), joinpath(@__DIR__, "hero_board.json"); force = true)
    pdf = only(filter(endswith(".pdf"), readdir(dir; join = true)))
    read(`$(BlobBoards.pdftocairo()) -svg $pdf -`, String)
end

srgb(v) = v <= 0.0031308 ? 12.92v : 1.055v^(1 / 2.4) - 0.055

# The profile: the opaque stops of the first gradient (every gradient's are the
# same), as (offset, linear reflectance) — then resampled at STOPS offsets.
const GRADIENT = r"<radialGradient([^>]*)>(.*?)</radialGradient>"s
const STOP = r"<stop offset=\"([^\"]+)\" stop-color=\"rgb\(([^%]+)%[^\"]*\)\" stop-opacity=\"([^\"]+)\"/>"
bodies = [m[2] for m in eachmatch(GRADIENT, svg)]
allequal(bodies) || error("the blobs' gradients differ: one shared profile cannot stand for them")
profile = [(parse(Float64, m[1]), parse(Float64, m[2]) / 100) for m in eachmatch(STOP, first(bodies)) if m[3] == "1"]
function sample(t)
    k = findfirst(p -> p[1] >= t, profile)
    (k === nothing || k == 1) && return profile[something(k, length(profile))][2]
    (t0, v0), (t1, v1) = profile[k - 1], profile[k]
    return t1 == t0 ? v1 : v0 + (v1 - v0) * (t - t0) / (t1 - t0)
end
pct(v) = string(round(100 * srgb(clamp(v, 0, 1)); digits = 3), "%")
stops = join(("<stop offset=\"$(round(t; digits = 5))\" stop-color=\"rgb($(pct(sample(t))), $(pct(sample(t))), $(pct(sample(t))))\"/>"
              for t in range(0, 1; length = STOPS)), "\n")
# beyond the disc the blob is absent, as the PDF's unextended shading
base = "<radialGradient id=\"blob-profile\">\n$stops\n<stop offset=\"1\" stop-color=\"rgb(0%, 0%, 0%)\" stop-opacity=\"0\"/>\n</radialGradient>\n"

# Each blob's gradient keeps its geometry and inherits the profile.
compact = replace(svg, GRADIENT => s -> begin
    attrs = match(GRADIENT, s)[1]
    "<radialGradient$(attrs) href=\"#blob-profile\" xlink:href=\"#blob-profile\"/>"
end)

# The page under everything (the PDF's paper is implicit), clipped to a
# rounded square; the dark variant dims the whole card to DARK_PAPER.
w = parse(Float64, split(match(r"viewBox=\"([^\"]+)\"", compact)[1])[3])
# no fixed size: the page's CSS sizes the image (the viewBox keeps its aspect)
compact = replace(compact, r" width=\"[^\"]*\" height=\"[^\"]*\" viewBox=" => " viewBox="; count = 1)
r = round(CORNER * w; digits = 3)
head, body = split(compact, "</defs>"; limit = 2)
function variant(dim)
    overlay = dim < 1 ? "<rect width=\"100%\" height=\"100%\" fill=\"black\" fill-opacity=\"$(round(1 - dim; digits = 3))\"/>\n" : ""
    content = replace(body, r"</svg>\s*$" => "")
    return head * base * "<clipPath id=\"card\"><rect width=\"$w\" height=\"$w\" rx=\"$r\" ry=\"$r\"/></clipPath>\n</defs>\n" *
           "<g clip-path=\"url(#card)\">\n<rect width=\"$w\" height=\"$w\" fill=\"white\"/>\n" * content * overlay * "</g>\n</svg>\n"
end

mkpath(OUT)
write(joinpath(OUT, "hero_board.svg"), variant(1.0))
write(joinpath(OUT, "hero_board_dark.svg"), variant(DARK_PAPER))
