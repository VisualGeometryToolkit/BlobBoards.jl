#=
Flatten --group SRC|CAM specs into a flat sources file (one
"abs/source/path|abs/camera/path" per line). For each --group:
  - SRC may be a directory (all supported files inside) or a single file
  - CAM may be empty (uncalibrated)

Usage:
  julia examples/training/slurm/build_source_list.jl <output_sources_file> \
        --group SRC|CAM   --group SRC|CAM   ...
=#

const _IMG_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp")
const _VID_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")
_is_supported(p) = lowercase(splitext(p)[2]) in (_IMG_EXTS..., _VID_EXTS...)
_is_video(p) = lowercase(splitext(p)[2]) in _VID_EXTS

function expand_one(src_path::AbstractString)
    if isdir(src_path)
        return sort!(filter(_is_supported, readdir(src_path; join = true)))
    elseif isfile(src_path)
        return [src_path]
    else
        return String[]
    end
end

length(ARGS) >= 1 || error("usage: build_source_list.jl <output_file> --group SRC|CAM ...")
output_file = ARGS[1]
function parse_groups(args)
    groups = Tuple{String, String}[]
    i = 2
    while i <= length(args)
        if args[i] == "--group"
            i + 1 <= length(args) || error("--group needs a value")
            spec = args[i + 1]
            parts = split(spec, '|'; limit = 2)
            length(parts) == 2 || error("--group must be SRC|CAM, got $spec")
            push!(groups, (String(parts[1]), String(parts[2])))
            i += 2
        else
            error("unknown arg: $(args[i])")
        end
    end
    return groups
end
groups = parse_groups(ARGS)
isempty(groups) && error("at least one --group required")

function write_sources(output_file, groups)
    n_total = 0
    n_videos = 0
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        for (src_spec, cam) in groups
            cam_abs = isempty(cam) ? "" : abspath(cam)
            n_in_group = 0
            for src in expand_one(src_spec)
                println(io, "$src|$cam_abs")
                n_in_group += 1
                n_total += 1
                _is_video(src) && (n_videos += 1)
            end
            cam_label = isempty(cam) ? "(uncalibrated)" : basename(cam)
            println(stderr, "  $n_in_group  $(basename(src_spec))  ← $cam_label")
        end
    end
    return n_total, n_videos
end
n_total, n_videos = write_sources(output_file, groups)
println(stderr, "Wrote $n_total sources ($(n_total - n_videos) images + $n_videos videos) → $output_file")
