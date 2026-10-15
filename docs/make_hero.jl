# The landing page's hero image: one square board built by the package's own
# authoring API at the CURRENT defaults of record — `blob_board`'s keyword
# defaults and BoardConfig/PatternConfig's (10 mm margin and frame, ruler
# ticks filling the band, cutoff DEFAULT_CUTOFF, …) — with only the canvas
# (150 mm square), the blob sizes (the smallest disc that prints well at
# 1200 dpi, 3 mm, to 20% of the short side) and the seed given. Exported to a
# 1200 dpi PNG, then reduced to a web image in a light and a dark variant.
#
#   julia --project=authoring docs/make_hero.jl
#
# writes docs/src/assets/landing/hero_board.png (light) and
# hero_board_dark.png, plus the board's JSON sidecar (its full configuration and
# blobs, for provenance) at docs/hero_board.json. The seed fixes placement, so
# the image follows from the package version alone. Authoring is CPU-only
# and headless.

using BlobBoards
using CairoMakie
const FileIO = CairoMakie.Makie.FileIO
const Colors = CairoMakie.Makie.Colors
using .Colors: Gray, GrayA, N0f8, gray

const SIDE = 150mm                     # square canvas
const DENSITY = 1200dpi                # the PNG's: 150 mm @ 1200 dpi = 7087 px
const SEED = 0x0000000b10bb0a2d        # fixed: placement is a function of it
const FACTOR = 7                       # 7087 px sheet -> 1012 px
const CORNER = 0.025                   # rounded-corner radius, fraction of the side
const DARK_PAPER = 0.86                # the dark variant's paper tone (sRGB), a soft light card

const OUT = joinpath(@__DIR__, "src", "assets", "landing")

sheet = mktempdir() do dir
    blob_board(SIDE, SIDE; blob_sizes = (suggest_blob_sizes(DENSITY, (SIDE, SIDE))[1], 0.2short_side), seed = SEED, dir, output = Png(DENSITY))
    cp(only(filter(endswith(".json"), readdir(dir; join = true))),
       joinpath(@__DIR__, "hero_board.json"); force = true)
    png = only(filter(endswith(".png"), readdir(dir; join = true)))
    Float64.(gray.(FileIO.load(png)))  # linear reflectance, 0 = ink, 1 = paper
end

# Box-filter by FACTOR in linear reflectance, then encode sRGB for display.
n = minimum(size(sheet)) ÷ FACTOR
small = [sum(@view sheet[(i - 1) * FACTOR .+ (1:FACTOR), (j - 1) * FACTOR .+ (1:FACTOR)]) / FACTOR^2
         for i in 1:n, j in 1:n]
srgb(v) = v <= 0.0031308 ? 12.92v : 1.055v^(1 / 2.4) - 0.055
display_tone = srgb.(clamp.(small, 0, 1))

# Antialiased rounded-corner alpha (coverage of a pixel centre's distance).
function corner_alpha(n, r)
    a = ones(n, n)
    for i in 1:n, j in 1:n
        x = min(i - 0.5, n - i + 0.5)
        y = min(j - 0.5, n - j + 0.5)
        (x < r && y < r) || continue
        a[i, j] = clamp(r - hypot(r - x, r - y) + 0.5, 0, 1)
    end
    return a
end
alpha = corner_alpha(n, CORNER * n)

image(tone) = GrayA{N0f8}.(N0f8.(clamp.(tone, 0, 1)), N0f8.(alpha))
mkpath(OUT)
FileIO.save(joinpath(OUT, "hero_board.png"), image(display_tone))
FileIO.save(joinpath(OUT, "hero_board_dark.png"), image(DARK_PAPER .* display_tone))
