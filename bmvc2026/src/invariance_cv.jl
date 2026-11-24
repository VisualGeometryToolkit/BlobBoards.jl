# =============================================================================
# The inter-board invariance cross-validation's shared machinery — the one
# implementation the tuning drivers read (scripts/detection_cv.jl: the
# operating point; scripts/detector_levels.jl: the detector's scale ladder)
# =============================================================================
#
# The validation frames are the ONE validation scene of a config's registered
# scene split (VisualGeometryToolkitEval `CrossValidation`), read from its
# `[inputs.images]`; the run config of record (`[runs] bb`) is the pipeline
# every cell runs, the search overriding only its axes. A detection run (the
# features of every validation frame) runs the GPU detector once per frame;
# the post-detection axes reuse it. The criterion's errors are VisualGeometryToolkitEval
# `rigid_invariance_errors`: the translation of every (frame, board pair)
# relative pose against its leave-one-out reference, mm, an undetected pair a
# failure (Inf).
# =============================================================================

using VisualGeometryToolkitCore: BoundedCamera
import VisualGeometryToolkitIO as _VgtkIO
using VisualGeometryToolkitEval.Tuning: CrossValidation, validation_units, validation_scenes
using VisualGeometryToolkitEval.Evaluation: rigid_invariance_errors
using BlobBoards: FindBoardsConfig, DescriptorNetwork, BoardGallery, find_boards, prepare_image_features
using Accessors: @set
import VisualGeometryToolkitFeatures as _VgtkFeatures
using Colors: Gray
using FileIO: load as _load_image

"""
    InvarianceCV

What every cell of an invariance cross-validation shares: the validation
frames (`image_dir`, `frames`, their `stems`), the run config of record
deserialized against the validation session's camera (`base`), the network,
the panel's board features and bundles (`feats`, `bundles`, sorted `uids`),
`min_obs` and the device.
"""
struct InvarianceCV{D}
    image_dir::String
    frames::Vector{String}
    stems::Vector{String}
    base::FindBoardsConfig
    net::DescriptorNetwork
    feats::Dict
    bundles::Dict
    uids::Vector
    min_obs::Int
    device::D
    run_config::String
end

"""
    validation_frames(cfg, cv::CrossValidation) -> (dir, frames)

The ONE validation scene (`<session>/<subset>`) of `cv`'s registered split,
under the config's `[inputs.images]` (the session's `images/` directory),
sampled by its validation sampling.
"""
function validation_frames(cfg::AbstractDict, cv::CrossValidation)
    scene = only(validation_scenes(cv))
    session, subset = String.(split(scene, '/'))
    dir = joinpath(input_path(cfg, "images"), subset)
    # [inputs.images] is the session's images/ directory: <session>/images/<subset>
    basename(dirname(dirname(dir))) == session || error("the validation scene $(scene) is not in [inputs.images] $(dir)")
    all_images = sort(filter(_is_image, readdir(dir)))   # eval_images.jl
    return dir, only(validation_units(cv, Dict(scene => all_images))).second
end

"""
    invariance_cv(cfg, cv::CrossValidation; device) -> InvarianceCV

The shared state of a cross-validation config `cfg` (`[runs] bb`, `[panel]`,
`[gallery]`, `[inputs.images|camera|boards]`).
"""
function invariance_cv(cfg::AbstractDict, cv::CrossValidation; device)
    image_dir, frames = validation_frames(cfg, cv)
    camera = _VgtkIO.load(BoundedCamera, input_path(cfg, "camera"))   # pixel operations take a BoundedCamera
    run_config = config_path(cfg["runs"]["bb"])
    base = FindBoardsConfig(run_config; camera)
    net = DescriptorNetwork(input_path(load_config(cfg["runs"]["bb"]), "network"))   # the run config's [inputs.network]
    gallery_file = joinpath(results_root(), cfg["gallery"]["path"]); mkpath(dirname(gallery_file))
    gallery = BoardGallery(gallery_file, net)
    feats, bundles = load_panel(gallery, net, input_path(cfg, "boards"))   # board_panel.jl
    uids = sort!(collect(keys(bundles)))
    panel = cfg["panel"]
    length(uids) == panel["n_boards"] || error("$(length(uids)) boards in [inputs.boards], the config states $(panel["n_boards"])")
    return InvarianceCV(image_dir, frames, [splitext(f)[1] for f in frames], base, net, feats, bundles, uids,
                        Int(panel["min_obs"]), device, run_config)
end

"""
    detection_run(ctx, cfg::FindBoardsConfig; on_frame = (stem, seconds) -> nothing) -> Dict{String,Any}

One detection run: the image features of every validation frame under `cfg`
(its detection, adaptation and description), by frame stem; the frames'
pixels are not kept. `on_frame` is called with each frame's stem and the wall
time (s) of its `prepare_image_features` (the image's decoding excluded).
"""
function detection_run(ctx::InvarianceCV, cfg::FindBoardsConfig; on_frame = (stem, seconds) -> nothing)
    features = Dict{String,Any}()
    for (stem, f) in zip(ctx.stems, ctx.frames)
        img = Gray.(_load_image(joinpath(ctx.image_dir, f)))
        t = @elapsed features[stem] = prepare_image_features(ctx.net, img; board_bundles = ctx.bundles, config = cfg,
                                                             device = ctx.device).image_features
        on_frame(stem, t)
    end
    return features
end

"What a run on `ctx` read: the validation frames, the camera store, the boards and the run config."
run_inputs(cfg::AbstractDict, ctx::InvarianceCV) =
    [ctx.image_dir, input_path(cfg, "camera"), input_path(cfg, "boards"), ctx.run_config]

"""
    invariance_errors(ctx, features, cfg::FindBoardsConfig) -> (; translation, pairs, poses)

The boards found in every frame of a detection run (`features`) under `cfg`,
and their inter-board invariance errors (`rigid_invariance_errors`): the
(frame × board pair) translation errors (mm; Inf = a failure) and their
pairs; `poses` the boards found per frame.
"""
function invariance_errors(ctx::InvarianceCV, features, cfg::FindBoardsConfig)
    poses = Dict(s => Dict(uid => m for (uid, m, _, _) in find_boards(ctx.feats, features[s];
                                                                      config = cfg, device = ctx.device,
                                                                      board_bundles = ctx.bundles))
                 for s in ctx.stems)
    r = rigid_invariance_errors(poses; targets = ctx.uids, units = ctx.stems, min_obs = ctx.min_obs)
    return (; translation = r.translation, pairs = r.pairs, poses)
end

# ---- the detector's scale ladder (scripts/detector_levels.jl) -------------------------------------------------------

"""
    with_ladder(cfg::FindBoardsConfig, setting) -> FindBoardsConfig

`cfg` detecting with the scale ladder `setting` (a `[[settings]]` table of
configs/detector_levels.toml: `upsample`, `nb_scales`, `nb_octaves`).
"""
function with_ladder(c::FindBoardsConfig, s::AbstractDict)
    c = @set c.detect.upsample = s["upsample"]
    c = @set c.detect.nb_scales = s["nb_scales"]
    return @set c.detect.nb_octaves = s["nb_octaves"]
end

"""
    ladder_sigma_range(setting, W, H; seed_sigma = 1.6) -> (lo, hi)

The scales a ladder `setting` searches in a `W`×`H` image, input pixels: the
interior DoG slices 1 and S of the finest and coarsest octave, in
VisualGeometryToolkitFeatures' own spelling (`_slice_sigma`, and `_nb_octaves`
for the octave count: `nb_octaves = 0` is its default `sift_nb_octaves`, and
an inadmissible count is refused as `AsiftPlan` refuses it). `seed_sigma` is
`AsiftPlan`'s default, the one BlobBoards detects with. Features has no public
accessor for the searched range; these are its internals until it has one.
"""
function ladder_sigma_range(s::AbstractDict, W::Integer, H::Integer; seed_sigma::Real = 1.6)
    u, S = s["upsample"], s["nb_scales"]
    n = _VgtkFeatures._nb_octaves("ladder_sigma_range", W, H, u, s["nb_octaves"] == 0 ? nothing : s["nb_octaves"])
    return (_VgtkFeatures._slice_sigma(1.0, seed_sigma, S, exp2(-u)),
            _VgtkFeatures._slice_sigma(Float64(S), seed_sigma, S, exp2(n - 1 - u)))
end
