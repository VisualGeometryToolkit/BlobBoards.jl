# =============================================================================
# The evaluation set's image predicate — ONE source, two modules
# =============================================================================
#
# Deliberately NARROW (jpg/jpeg/png), and deliberately NOT BlobBoards' broad
# image predicate: mocap files pair with images POSITIONALLY (rig_sequence),
# so a stray .tif in a capture directory would silently misalign an entire
# sequence. Included by BOTH BlobBoardsEval (rig.jl) and BoardSweep (sweep.jl),
# so the sweep can load without the evaluation module — BlobBoardsEval asserts
# the bmvc2026 project, and a project of its own can run the sweep alone.
const _EVAL_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_is_image(f) = lowercase(splitext(f)[2]) in _EVAL_IMAGE_EXTS
