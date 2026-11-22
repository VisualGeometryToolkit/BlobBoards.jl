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
# features of every validation frame) costs a GPU pass; the post-detection
# axes reuse it. The criterion's errors are VisualGeometryToolkitEval
# `rigid_invariance_errors`: the translation of every (frame, board pair)
# relative pose against its leave-one-out reference, mm, an undetected pair a
# failure (Inf).
# =============================================================================

using VisualGeometryToolkitCore: BoundedCamera
import VisualGeometryToolkitIO as _VgtkIO
using VisualGeometryToolkitEval.Tuning: CrossValidation, validation_units, validation_scenes
using VisualGeometryToolkitEval.Evaluation: rigid_invariance_errors
using BlobBoards: FindBoardsConfig, DescriptorNetwork, BoardGallery, uid_from_hash, find_boards, load_bundle,
    prepare_image_features
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
    all_images = sort(filter(f -> lowercase(splitext(f)[2]) in (".jpg", ".jpeg", ".png"), readdir(dir)))
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
    board_paths = filter(endswith(".json"), readdir(input_path(cfg, "boards"); join = true))
    feats = Dict(uid => f for (uid, f) in [insert!(gallery, bp, net) for bp in board_paths])
    bundles = Dict(let (b, _, ch) = load_bundle(bp); uid_from_hash(ch) => b end for bp in board_paths)
    uids = sort!(collect(keys(bundles)))
    panel = cfg["panel"]
    length(uids) == panel["n_boards"] || error("$(length(uids)) boards in [inputs.boards], the config states $(panel["n_boards"])")
    return InvarianceCV(image_dir, frames, [splitext(f)[1] for f in frames], base, net, feats, bundles, uids,
                        Int(panel["min_obs"]), device, run_config)
end

"""
    detection_run(ctx, cfg::FindBoardsConfig; on_frame = (stem, seconds) -> nothing) -> Dict{String,Any}

One detection run: the prepared image features of every validation frame
under `cfg` (its detection and adaptation), by frame stem. `on_frame` is
called with each frame's stem and its wall time (s).
"""
function detection_run(ctx::InvarianceCV, cfg::FindBoardsConfig; on_frame = (stem, seconds) -> nothing)
    prepared = Dict{String,Any}()
    for (stem, f) in zip(ctx.stems, ctx.frames)
        t = @elapsed prepared[stem] = prepare_image_features(ctx.net, Gray.(_load_image(joinpath(ctx.image_dir, f)));
                                                            board_bundles = ctx.bundles, config = cfg, device = ctx.device)
        on_frame(stem, t)
    end
    return prepared
end

"""
    invariance_errors(ctx, prepared, cfg::FindBoardsConfig) -> (; translation, pairs, poses)

The boards found in every frame of a detection run (`prepared`) under `cfg`,
and their inter-board invariance errors (`rigid_invariance_errors`): the
(frame × board pair) translation errors (mm; Inf = a failure) and their
pairs; `poses` the boards found per frame.
"""
function invariance_errors(ctx::InvarianceCV, prepared, cfg::FindBoardsConfig)
    poses = Dict(s => Dict(uid => m for (uid, m, _, _) in find_boards(ctx.feats, prepared[s].image_features;
                                                                      config = cfg, device = ctx.device,
                                                                      board_bundles = ctx.bundles))
                 for s in ctx.stems)
    r = rigid_invariance_errors(poses; targets = ctx.uids, units = ctx.stems, min_obs = ctx.min_obs)
    return (; translation = r.translation, pairs = r.pairs, poses)
end

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
interior DoG slices of every octave, `seed_sigma·2^(1.5/S)·2^(−upsample)` …
`seed_sigma·2^((S + 0.5)/S)·2^(nb_octaves − 1 − upsample)`
(VisualGeometryToolkitFeatures `AsiftPlan`; `nb_octaves = 0` is its default,
`sift_nb_octaves`).
"""
function ladder_sigma_range(s::AbstractDict, W::Integer, H::Integer; seed_sigma::Real = 1.6)
    u, S = s["upsample"], s["nb_scales"]
    n = s["nb_octaves"] == 0 ? _VgtkFeatures.sift_nb_octaves(W, H; upsample = u) : s["nb_octaves"]
    return seed_sigma * exp2(1.5 / S - u), seed_sigma * exp2((S + 0.5) / S + n - 1 - u)
end
