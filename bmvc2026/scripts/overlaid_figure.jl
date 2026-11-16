#=
overlaid_figure.jl — a photo with a vector overlay, written as the two layers
the paper composites plus a plain backup. Included by the generators that draw
on a photo (make_bmvc2026_intro.jl, make_adapted_panels.jl,
make_asift_detections.jl); they load CairoMakie first.

For an output base `<base>` (a path without extension) it writes

  <base>_photo.jpg   the photo layer: a byte copy of the source JPEG when the
                     photo is shown unmodified, else the shown pixels encoded
                     at the configured JPEG quality
  <base>.pdf         the overlay ONLY, on a transparent page that is exactly
                     the photo's extent (no padding, the axis fills the page)
  <base>.jpg         the backup: the full figure (photo + overlay) rendered at
                     the photo's resolution, a plain JPEG

and the paper composites the first two with `\overlaid` (papers/overlaid.tex):
`\overlaid[width=\linewidth]{<base>_photo.jpg}{<base>.pdf}` stretches the
overlay page over the photo's box, so the photo is embedded as the JPEG itself
(no re-encoding by Cairo, which stores images losslessly — a PDF of the
composite was 5–10x the JPEG) and the overlay stays vector.

Overlay geometry is drawn in the photo's FULL-resolution pixel coordinates,
the axis pinned to (0..W, 0..H); anything reaching past the photo is clipped,
unless `fit_overlays!` grows the page to it (the photo layer is then padded).
=#

using FileIO: save as save_image
using Colors: RGB
using Images: imresize, N0f8

"""
    photo_figure(img; width, photo_px = 0) -> (fig, ax, photo)

A transparent, padding-free figure `width` units wide whose one axis IS the
photo `img`: the image fills it over (0..W, 0..H) in `img`'s pixels, y down,
limits pinned. `photo_px > 0` resizes the shown copy to at most that width
(drawn over the same extent, so overlay coordinates do not change). `photo`
is the image plot; `photo.shown` the pixels it shows; `photo.extent[]` the
(x0, x1, y0, y1) the page covers — the photo's own, until `fit_overlays!`.
"""
function photo_figure(img::AbstractMatrix; width::Integer, photo_px::Integer = 0)
    H, W = size(img)
    s = photo_px > 0 ? photo_px / W : 1.0
    shown = s < 1 ? imresize(img; ratio = s) : img
    fig = Figure(size = (width, round(Int, width * H / W)), figure_padding = 0, backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; yreversed = true, backgroundcolor = :transparent, titlevisible = false,
        xautolimitmargin = (0.0f0, 0.0f0), yautolimitmargin = (0.0f0, 0.0f0))
    # interpolate = true: with false, CairoMakie draws every pixel as its own
    # rectangle in a vector backend (only the backup render sees the photo,
    # but keep it an image either way)
    plt = image!(ax, 0 .. W, 0 .. H, transpose(shown); interpolate = true)
    limits!(ax, 0, W, H, 0)
    hidedecorations!(ax); hidespines!(ax)
    return fig, ax, (; plot = plt, shown, scale = s, width, W, H, extent = Ref{NTuple{4, Float64}}((0, W, 0, H)))
end

"""
    fit_overlays!(fig, ax, photo)

Grow the page past the photo to everything drawn (boards overhanging the frame
stay whole), keeping the figure's units per photo pixel. The extent snaps
outward to whole SHOWN pixels, and the photo layer is padded (`save_overlaid`)
to the same extent, so `\\overlaid` still lines the two up.
"""
function fit_overlays!(fig, ax, photo)
    autolimits!(ax)
    lim = ax.finallimits[]
    (xa, ya), (wa, ha) = lim.origin, lim.widths
    s = photo.scale
    x0, y0 = min(0, floor(xa * s) / s), min(0, floor(ya * s) / s)
    x1, y1 = max(photo.W, ceil((xa + wa) * s) / s), max(photo.H, ceil((ya + ha) * s) / s)
    limits!(ax, x0, x1, y1, y0)
    u = photo.width / photo.W                  # figure units per photo pixel, as before
    resize!(fig.scene, round(Int, u * (x1 - x0)), round(Int, u * (y1 - y0)))
    photo.extent[] = (x0, x1, y0, y1)
    return fig
end

"""
    save_overlaid(base, fig, photo; source = nothing, quality = 90, pt_per_unit = 1.0, pad = RGB(1, 1, 1)) -> Vector{String}

Write `<base>_photo.jpg`, `<base>.pdf` (overlay only) and `<base>.jpg` (backup)
for a figure from `photo_figure`. `source` is the photo's file: when it is a
JPEG and the figure shows it unmodified (not resized, not padded by
`fit_overlays!`; the caller passes `nothing` for a crop or a derived image),
the photo layer is its byte copy. Outside the photo, the photo layer and the
backup are `pad`. Returns the paths written.
"""
function save_overlaid(base::AbstractString, fig, photo; source = nothing, quality::Integer = 90, pt_per_unit = 1.0, pad = RGB(1, 1, 1))
    mkpath(dirname(base))
    photo_jpg, overlay_pdf, backup_jpg = base * "_photo.jpg", base * ".pdf", base * ".jpg"
    x0, x1, y0, y1 = photo.extent[]
    s = photo.scale
    padded = (x0, x1, y0, y1) != (0, photo.W, 0, photo.H)
    # the photo layer, padded to the page's extent in shown pixels
    layer = RGB{N0f8}.(photo.shown)
    if padded
        h, w = size(layer)
        ox, oy = round(Int, -x0 * s), round(Int, -y0 * s)
        canvas = fill(RGB{N0f8}(pad), max(oy + h, round(Int, (y1 - y0) * s)), max(ox + w, round(Int, (x1 - x0) * s)))
        canvas[(oy + 1):(oy + h), (ox + 1):(ox + w)] = layer
        layer = canvas
    end
    if source !== nothing && photo.scale == 1 && !padded && lowercase(splitext(source)[2]) in (".jpg", ".jpeg")
        cp(source, photo_jpg; force = true, follow_symlinks = true)
        chmod(photo_jpg, 0o644)   # not the artifact store's read-only, executable mode
    else
        save_image(photo_jpg, layer; quality)
    end
    # the backup: the composite on `pad`, at the photo layer's resolution
    ppu = size(layer, 2) / fig.scene.viewport[].widths[1]
    bg = fig.scene.backgroundcolor[]
    fig.scene.backgroundcolor[] = Makie.to_color(pad)
    try
        save_image(backup_jpg, RGB{N0f8}.(Makie.colorbuffer(fig; px_per_unit = ppu)); quality)
    finally
        fig.scene.backgroundcolor[] = bg
    end
    # the overlay: the same figure without the photo
    photo.plot.visible = false
    try
        Makie.save(overlay_pdf, fig; pt_per_unit)
    finally
        photo.plot.visible = true
    end
    return [photo_jpg, overlay_pdf, backup_jpg]
end
