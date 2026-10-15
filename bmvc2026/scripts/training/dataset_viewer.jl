#=
Debug Viewer for Training Data — inspect what was annotated for training

Usage:
  include("dataset_viewer.jl")

Or programmatically:
  data_path = "/path/to/truth_dir/"   # directory of per-image JLD2s, OR
  data_path = "/path/to/train.jld2"   # split file produced by split_training_data
  view_training_data(data_path; n_samples=10, visualize=true)

INTERACTIVE: needs GLMakie (a display), which the paper environment does not
carry — bmvc2026 is headless-reproducible (test/paper_project_tests.jl). Run
it from an interactive environment that adds GLMakie beside this project, e.g.
`julia --project=bmvc2026 -i` then `]activate --temp`, `]dev .` (the
repository root) and `]add GLMakie`, on a machine with a display.
=#

using BlobBoards
using Images, FileIO
using Statistics
using LinearAlgebra
using GLMakie
using Makie: Axis, Figure, DataAspect, hist!, hidedecorations!, image!
import FileIO.save
GLMakie.activate!()

"""
    inspect_training_data(path; n_samples=10, visualize=true)

Load and inspect training data from one or more annotation directories
or split JLD2 files.

If `path` is a string, loads a single directory or split file.
If `path` is a vector of strings, loads and combines multiple sources.

Shows:
- Dataset statistics (total patches, blobs, etc.)
- Sample patches from each blob
- Label distribution
- Confuser patches (if any)
"""
function inspect_training_data(
        paths::Union{AbstractString, Vector{<:AbstractString}};
        n_samples::Int = 10,
        visualize::Bool = true
    )
    # Convert single path to vector for uniform handling
    paths_list = paths isa String ? [paths] : paths

    @info "Loading $(length(paths_list)) file(s)..."

    # Load and combine all files
    all_patches = []
    all_board_uids = String[]
    all_blob_ids = Int[]
    all_confusers = []
    patch_size = nothing

    for path in paths_list
        @info "  Processing $path..."
        data = load_annotations(path)
        push!(all_patches, data.patches)
        append!(all_board_uids, data.board_uids)
        append!(all_blob_ids, Int.(data.blob_ids))
        push!(all_confusers, data.confusers)

        if isnothing(patch_size)
            patch_size = size(data.patches, 1)
        end
    end

    # Concatenate all patches along the third dimension
    patches = cat(all_patches...; dims = 3)
    confusers = cat(all_confusers...; dims = 3)
    board_uids = all_board_uids
    blob_ids = all_blob_ids

    P = size(patches, 1)
    N = size(patches, 3)
    C = size(confusers, 3)

    # Build label mapping (board_uid, blob_id) -> label
    key_to_label = Dict{Tuple{String, Int}, Int32}()
    labels = Vector{Int32}(undef, N)
    next_label = Int32(1)
    for i in 1:N
        key = (board_uids[i], blob_ids[i])
        label = get!(key_to_label, key, next_label)
        if label == next_label
            next_label += Int32(1)
        end
        labels[i] = label
    end

    # Statistics
    unique_labels = unique(labels)
    n_blobs = length(unique_labels)

    @info "=== Dataset Statistics ==="
    @info "Total patches: $N ($(P)×$(P))"
    @info "Unique blobs: $n_blobs"
    @info "Confuser patches: $C"
    @info "Boards: $(length(unique(board_uids)))"
    @info ""

    # Label distribution
    @info "=== Label Distribution ==="
    label_counts = Dict{Int32, Int}()
    for label in labels
        label_counts[label] = get(label_counts, label, 0) + 1
    end

    counts = collect(values(label_counts))
    @info "  Min patches/blob: $(minimum(counts))"
    @info "  Max patches/blob: $(maximum(counts))"
    @info "  Mean patches/blob: $(round(mean(counts); digits = 1))"
    @info "  Median patches/blob: $(median(counts))"
    @info ""

    # Show sample patches from first few blobs
    if visualize
        @info "=== Sampling Patches ==="
        for (i, label) in enumerate(unique_labels[1:min(n_samples, length(unique_labels))])
            indices = findall(==(label), labels)
            board_uid = board_uids[indices[1]]
            blob_id = blob_ids[indices[1]]

            @info "  Blob $label: board=$board_uid, blob_id=$blob_id, n_observations=$(length(indices))"

            # Show first patch
            patch = patches[:, :, indices[1]]
            @info "    First patch stats: min=$(round(minimum(patch); digits = 3)), max=$(round(maximum(patch); digits = 3)), mean=$(round(mean(patch); digits = 3))"
        end
        @info ""
    end

    # Confuser info
    if C > 0
        @info "=== Confuser Patches ==="
        @info "  Count: $C"
        conf_sample = confusers[:, :, 1]
        @info "  Stats (sample): min=$(round(minimum(conf_sample); digits = 3)), max=$(round(maximum(conf_sample); digits = 3)), mean=$(round(mean(conf_sample); digits = 3))"
    end

    return (;
        patches, labels, board_uids, blob_ids, confusers,
        n_blobs, n_patches = N, n_confusers = C,
        label_counts, unique_labels,
    )
end

"""
    export_sample_patches(data, output_dir; n_per_blob=5, max_blobs=nothing)

Export sample patches to disk for visual inspection.

Args:
  - data: Data structure from inspect_training_data()
  - output_dir: Directory to save patches (all files saved with blob info as prefix)
  - n_per_blob: Number of patches per blob to export
  - max_blobs: Limit number of blobs (nothing = all)
"""
function export_sample_patches(data, output_dir::String; n_per_blob::Int = 5, max_blobs = nothing)
    mkpath(output_dir)

    label_to_indices = Dict{Int32, Vector{Int}}()
    for (i, label) in enumerate(data.labels)
        push!(get!(Vector{Int}, label_to_indices, label), i)
    end

    sorted_labels = sort(collect(keys(label_to_indices)))
    if !isnothing(max_blobs)
        sorted_labels = sorted_labels[1:min(max_blobs, length(sorted_labels))]
    end

    total_exported = 0
    for label in sorted_labels
        indices = label_to_indices[label]
        board_uid = data.board_uids[indices[1]]
        blob_id = data.blob_ids[indices[1]]

        # Create subdirectory for this board
        board_dir = joinpath(output_dir, board_uid)
        mkpath(board_dir)

        # Use blob info as filename prefix
        prefix = "blob_$(label)_$(blob_id)"

        # Export first n_per_blob patches
        for (j, idx) in enumerate(indices[1:min(n_per_blob, length(indices))])
            patch = data.patches[:, :, idx]
            # Normalize to [0, 1] range
            patch_min, patch_max = extrema(patch)
            patch_norm = (patch .- patch_min) ./ (patch_max - patch_min + eps(Float32))
            patch_img = Gray.(patch_norm)
            filename = joinpath(board_dir, "$(prefix)_patch_$(j).png")
            save(filename, patch_img)
            total_exported += 1
        end
    end

    return @info "Exported $total_exported patches from $(length(sorted_labels)) blobs to $output_dir"
end

"""
    plot_label_distribution(label_counts; output_path=nothing)

Plot histogram of patches per blob using CairoMakie.
"""
function plot_label_distribution(label_counts::Dict; output_path::Union{Nothing, String} = nothing)
    counts = collect(values(label_counts))
    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1]; xlabel = "Patches per Blob", ylabel = "Count", title = "Label Distribution")

    hist!(ax, counts; bins = 50, color = :steelblue, alpha = 0.8)

    if !isnothing(output_path)
        save(output_path, fig)
        @info "Saved plot to $output_path"
    end

    return fig
end

"""
    board_statistics(data)

Print statistics about boards in the dataset.
"""
function board_statistics(data)
    println("=== BOARD STATISTICS ===")
    println()

    # Get unique boards from labeled patches
    label_to_indices = Dict{Int32, Vector{Int}}()
    for (i, label) in enumerate(data.labels)
        push!(get!(Vector{Int}, label_to_indices, label), i)
    end

    unique_boards = Set{String}()
    for label in data.unique_labels
        indices = label_to_indices[label]
        board_uid = data.board_uids[indices[1]]
        push!(unique_boards, board_uid)
    end

    println("Overall number of boards that have blobs and patches: $(length(unique_boards))")
    println()

    println("Board breakdown:")
    for board in sort(collect(unique_boards))
        mask = data.board_uids .== board
        patch_count = sum(mask)
        blob_indices = findall(mask)
        blob_ids = unique(data.blob_ids[blob_indices])
        blob_count = length(blob_ids)
        println("  Board $board: $patch_count patches, $blob_count blobs")
    end
    return println()
end

"""
    plot_sample_patches(patches, labels; n_per_label=3, output_path=nothing)

Visualize sample patches from different blobs using CairoMakie.
"""
function plot_sample_patches(patches::Array, labels::Vector; n_per_label::Int = 3, output_path::Union{Nothing, String} = nothing)
    unique_labels = unique(labels)
    n_show = min(6, length(unique_labels))

    fig = Figure(size = (900, 600))

    for (i, label) in enumerate(unique_labels[1:n_show])
        indices = findall(==(label), labels)

        for (j, idx) in enumerate(indices[1:min(n_per_label, length(indices))])
            row = div(i - 1, 3) + 1
            col = mod(i - 1, 3) + 1
            ax = Axis(fig[row, col]; aspect = DataAspect(), title = "Blob $label (patch $j)")
            hidedecorations!(ax)

            patch = patches[:, :, idx]
            # Normalize to [0, 1] range
            patch_min, patch_max = extrema(patch)
            patch_norm = (patch .- patch_min) ./ (patch_max - patch_min + eps(Float32))
            image!(ax, rotr90(patch_norm))
        end
    end

    if !isnothing(output_path)
        save(output_path, fig)
        @info "Saved patch visualization to $output_path"
    end

    return fig
end

# =============================================================================
# Run inspection
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Override via env vars (handy for non-default split locations).
    splits_dir = get(() -> error("set BB_SPLITS_DIR to the splits directory (train.jld2, …)"), ENV, "BB_SPLITS_DIR")
    splits = split(get(ENV, "BB_SPLITS", "train"), ',')   # e.g. "train,val,test"

    data_paths = filter(
        ispath,
        [joinpath(splits_dir, "$(strip(s)).jld2") for s in splits]
    )

    if isempty(data_paths)
        @warn "No split files found under $splits_dir for splits=$splits"
        @info "Expected files like $(joinpath(splits_dir, "train.jld2"))"
        @info "Override with BB_SPLITS_DIR / BB_SPLITS env vars."
    else
        @info "Processing $(length(data_paths)) split file(s) from $splits_dir"
        data = inspect_training_data(data_paths; n_samples = 20)
        board_statistics(data)

        # Open two interactive GLMakie windows.
        fig_dist = plot_label_distribution(data.label_counts)
        display(GLMakie.Screen(), fig_dist)

        fig_patches = plot_sample_patches(data.patches, data.labels; n_per_label = 3)
        display(GLMakie.Screen(), fig_patches)

        @info "Two windows open: label distribution + sample patches. Ctrl-D to exit Julia."
    end
end
