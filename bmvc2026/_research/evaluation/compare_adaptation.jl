#=
Compare Adaptation Methods: GN vs Posterior vs Moments

Side-by-side comparison of three blob adaptation methods on the same
detections. Each panel shows fitted ellipses from one method.

1. GNMethod       — Gauss-Newton in intensity space (default)
2. PosteriorMethod — Posterior IRLS in log-intensity space
3. fit_moments     — Intensity-weighted second moments (from VGC)

Usage:
  include("examples/detect/compare_adaptation.jl")
=#

using BlobBoards
using BlobBoards: adapt, detect_blobs, filter_blobs   # advanced API (unexported)
using VisualGeometryToolkitCore: Circle, Ellipse, Disc, imshow, boundary, centroid
using VisualGeometryToolkitCore.Features: fit_moments, Detection
using GeometryBasics: Point2
using GLMakie
using FileIO: load as load_image
using Colors: Gray
using VisualGeometryToolkitCore: ustrip

# Load image and convert to grayscale
image_path = joinpath(data_dir(),
    "dataset_sonyfx3_tamron20mm_f8/jpeg/DSC02797.JPG")
img = Gray.(load_image(image_path))

H, W = size(img)
println("Image: $image_path")
println("Size:  $W x $H")

# --- Detect ---
println("\nDetecting blobs...")
detections = detect_blobs(img;
    device = KA.CPU(),
    peak_threshold = 0.02,
    polarity = :dark)
println("Detections: $(length(detections))")

# --- Method 1: GNMethod (default) ---
println("\nAdapting with GNMethod...")
frames_gn = adapt(detections, img; method = GNMethod())
frames_gn = filter_blobs(frames_gn, detections;
    cutoff = DEFAULT_CUTOFF, image = img)
println("  GN: $(length(frames_gn)) after filter")

# --- Method 2: PosteriorMethod ---
println("Adapting with PosteriorMethod...")
frames_post = adapt(detections, img; method = PosteriorMethod())
frames_post = filter_blobs(frames_post, detections;
    cutoff = DEFAULT_CUTOFF, image = img)
println("  Posterior: $(length(frames_post)) after filter")

# --- Method 3: fit_moments (manual loop) ---
println("Adapting with fit_moments...")
img_f = Float64.(img)
moment_results = map(detections) do d
    disc = Disc(Point2(ustrip.(d.center)...), ustrip(d.σ))
    fit_moments(img_f, disc; extent = DEFAULT_CUTOFF, robust_method = :geman)
end
# Wrap successful fits as BlobDetections for filter_blobs
frames_mom = BlobDetection{Float64}[]
for (i, r) in enumerate(moment_results)
    r.success || continue
    push!(frames_mom, Detection(r.value, (; id = i)))
end
frames_mom = filter_blobs(frames_mom, detections;
    cutoff = DEFAULT_CUTOFF, image = img)
println("  Moments: $(length(frames_mom)) after filter")

# --- Build boundary ellipses ---
ellipses_gn   = [boundary(f.frame, DEFAULT_CUTOFF) for f in frames_gn]
ellipses_post = [boundary(f.frame, DEFAULT_CUTOFF) for f in frames_post]
ellipses_mom  = [boundary(f.frame, DEFAULT_CUTOFF) for f in frames_mom]

# --- 3-panel figure ---
fig = Figure(size = (2100, 700))

panels = [
    ("GNMethod ($(length(ellipses_gn)))",        ellipses_gn,   :magenta),
    ("PosteriorMethod ($(length(ellipses_post)))", ellipses_post, :cyan),
    ("Moments ($(length(ellipses_mom)))",         ellipses_mom,  :orange),
]

axes = map(enumerate(panels)) do (i, (title, shapes, color))
    ax, _ = imshow(fig[1, i], img; title)
    Makie.poly!(ax, shapes;
        strokecolor = color, strokewidth = 2.0, color = :transparent)
    ax
end
linkaxes!(axes...)

display(fig)
println("\nDone!")
