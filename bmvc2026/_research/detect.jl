#=
Minimal Blob Detection Example

Detects dark blobs on a board image from test/data/ and visualizes
the detected circles.

Usage:
  include("examples/pipeline/detect.jl")
=#

using BlobBoards
using BlobBoards: detect_blobs   # advanced API (unexported)
using VisualGeometryCore: Circle, imshow!
using GLMakie; GLMakie.activate!()
using FileIO: load as load_image
using Colors: Gray
using VisualGeometryCore: ustrip
using CUDA, NNlib, cuDNN  # triggers GPU extensions

# Load a board image from test/data/
image_path = joinpath(@__DIR__, "..", "..", "test", "data", "blob_board_4e3_e0e40.png")
img = Gray.(load_image(image_path))

H, W = size(img)
println("Image: $image_path")
println("Size:  $W x $H")

# Detect dark blobs on GPU
detections = detect_blobs(img;
    device = CUDABackend(),
    peak_threshold = 0.04,
    polarity = :dark)
println("Detections: $(length(detections))")

# Build circles for visualization (radius = cutoff * sigma)
circles = [Circle(ustrip.(d.center), DEFAULT_CUTOFF * ustrip(d.σ)) for d in detections]

# Plot
fig = Figure(size = (1200, 800))
ax = Axis(fig[1, 1];
    aspect = DataAspect(),
    yreversed = true,
    title = "$(length(detections)) dark blobs")
imshow!(ax, img)
Makie.poly!.(Ref(ax), circles;
    strokecolor = :green, strokewidth = 1.5, color = :transparent)

display(fig)
