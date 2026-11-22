# =============================================================================
# The board panel of a run — ONE source, two modules
# =============================================================================
#
# Every board definition of a directory, its gallery features and its bundle,
# by uid. Included by BOTH BoardSweep (sweep.jl) and BlobBoardsEval (the
# invariance CVs, invariance_cv.jl), as eval_images.jl is.
# =============================================================================

"""
    load_panel(gallery, network, boards_dir) -> (; features, bundles)

The boards of `boards_dir` (every `*.json`): their gallery features
(`insert!(gallery, path, network)`, cached in the gallery) and their bundles,
each by board uid.
"""
function load_panel(gallery, network, boards_dir::AbstractString)
    paths = filter(endswith(".json"), readdir(boards_dir; join = true))
    features = Dict(uid => f for (uid, f) in [insert!(gallery, bp, network) for bp in paths])
    bundles = Dict(let (b, _, ch) = BlobBoards.load_bundle(bp); BlobBoards.uid_from_hash(ch) => b end for bp in paths)
    return (; features, bundles)
end
