# =============================================================================
# The annotation workers' state and their two functions, ONE copy for both
# Slurm drivers (driver.jl: addprocs-style; driver_scm.jl: SlurmClusterManager).
# Each driver loads it on every worker:
#   @everywhere include(joinpath(@__DIR__, "annotate_worker.jl"))
# after its `@everywhere begin using BlobBoards … end` block.
# =============================================================================

const _WORKER = Ref{NamedTuple}()

function _init_worker!(model_path, gallery_path, boards_dir)
    descriptor_network = DescriptorNetwork(model_path)
    gallery = BoardGallery(gallery_path, descriptor_network)
    board_paths = filter(endswith(".json"), readdir(boards_dir; join = true))
    bfd = Dict(
        uid => f for (uid, f) in
            [insert!(gallery, p, descriptor_network) for p in board_paths]
    )
    bb = Dict(
        let (b, _, ch) = BlobBoards.load_bundle(p)
                uid_from_hash(ch) => b
        end
            for p in board_paths
    )
    uid_to_path = Dict(
        let (_, _, ch) = BlobBoards.load_bundle(p)
                uid_from_hash(ch) => p
        end
            for p in board_paths
    )
    _WORKER[] = (
        descriptor_network = descriptor_network,
        board_feature_dict = bfd,
        board_bundles = bb,
        uid_to_path = uid_to_path,
        board_paths = board_paths,
        camera_cache = Dict{String, Any}(),
        find_boards_config = FindBoardsConfig(geometry = HomographyGeometry(min_inliers = 10)),
    )
    "ready on worker $(myid())  GPU=$(get(ENV, "CUDA_VISIBLE_DEVICES", "?"))  host=$(gethostname())"
end

function _process_source(
        source::String, camera_path::String,
        output_dir::String, n_frames::Int
    )
    # Self-init guard: the master's @sync init loop has been observed to
    # silently leave workers uninitialized (likely an @async/@sync swallow
    # of an init exception). Worker env vars are inherited from srun, so
    # we can recover by initializing on first source.
    if !isassigned(_WORKER)
        _init_worker!(ENV["BB_MODEL_PATH"], ENV["BB_GALLERY_PATH"], ENV["BB_BOARDS_DIR"])
    end
    s = _WORKER[]
    BlobBoards.is_source_processed(output_dir, source) && return :skipped
    camera = if isempty(camera_path)
        nothing
    else
        get!(s.camera_cache, camera_path) do;
            VgtkIO.load(CameraModel, camera_path)
        end
    end
    # Camera lives in config.geometry — PoseGeometry(camera) for calibrated
    # sources, HomographyGeometry otherwise. Reuse the worker's detect config.
    base = s.find_boards_config
    geom = isnothing(camera) ? HomographyGeometry(min_inliers = 10) :
        PoseGeometry(camera; min_inliers = 10)
    config = FindBoardsConfig(geometry = geom, detect = base.detect)
    if BlobBoards.is_video(source)
        annotate_blobs(
            s.board_feature_dict, s.descriptor_network, source;
            output_dir, board_bundles = s.board_bundles, board_paths = s.board_paths,
            config, device = CUDABackend(), n_frames
        )
        return :video_done
    else
        img = Gray{Float32}.(FileIO.load(source))
        fp = BlobBoards.image_path(output_dir, basename(source))
        meta = BlobBoards.annotate_frame!(
            BlobBoards.existing_ref_uids(output_dir), output_dir,
            s.board_feature_dict, s.descriptor_network, s.uid_to_path,
            fp, basename(source), img;
            board_bundles = s.board_bundles,
            config, cutoff = BlobBoards.DEFAULT_CUTOFF,
            patch_size = 64, outer_extent = 96.0, min_iou = 0.5,
            device = CUDABackend()
        )
        BlobBoards.append_log(output_dir, meta)
        return meta
    end
end
