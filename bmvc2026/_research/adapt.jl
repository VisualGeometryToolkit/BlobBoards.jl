#=
Blob Detection + Affine Adaptation Pipeline (CPU)

Detects blobs, fits anisotropic ellipses via IRLS adaptation,
filters bad fits, and visualizes detections vs fitted ellipses.

Uses the same defaults as find_boards / extract_image_features.

Pipeline: detect_blobs -> adapt -> filter_blobs -> visualize

Usage:
  include("examples/pipeline/adapt.jl")
=#

using BlobBoards
using BlobBoards: adapt, detect_blobs, filter_blobs   # advanced API (unexported)
using VisualGeometryCore: Circle, Ellipse, imshow, boundary, centroid
using GLMakie; GLMakie.activate!()
using FileIO: load as load_image
using Colors: Gray
using VisualGeometryCore: ustrip

# Load image and convert to grayscale
image_path = joinpath(data_dir(),
    "dataset_sonyfx3_tamron20mm_f8/jpeg/DSC02797.JPG")
img = Gray.(load_image(image_path))

H, W = size(img)
println("Image: $image_path")
println("Size:  $W x $H")

device = KA.CPU()

# --- Detect (same defaults as FindBoardsConfig) ---
println("\nDetecting blobs...")
detections = detect_blobs(img;
    backend,
    peak_threshold = 0.01,
    max_detections = 2048)
println("Detections: $(length(detections))")

# --- Adapt (same defaults as extract_image_features) ---
adapt_config = AdaptConfig(extent = DEFAULT_CUTOFF)
println("\nAdapting with $(adapt_config.solve)...")
frames = adapt(detections, img; config = adapt_config, backend)
n_fitted = length(frames)

# --- Filter ---
frames = filter_blobs(frames, detections;
    cutoff = DEFAULT_CUTOFF, image = img)
println("Fitted: $n_fitted/$(length(detections)), after filter: $(length(frames))")

# --- Build shapes for plotting ---
det_circles = [Circle(ustrip.(d.center), DEFAULT_CUTOFF * ustrip(d.σ))
               for d in detections]
ellipses = [boundary(f.frame, DEFAULT_CUTOFF) for f in frames]

# --- Visualize: detections (left) vs fitted ellipses (right) ---
fig = Figure(size = (1600, 800))

ax1, _ = imshow(fig[1, 1], img;
    title = "Detections ($(length(det_circles)))")
Makie.poly!(ax1, det_circles;
    strokecolor = :red, strokewidth = 1.5, color = :transparent)

ax2, _ = imshow(fig[1, 2], img;
    title = "Adapted [$(adapt_config.solve)] ($(length(ellipses)))")
Makie.poly!(ax2, ellipses;
    strokecolor = :magenta, strokewidth = 2.0, color = :transparent)

linkaxes!(ax1, ax2)
display(fig)
