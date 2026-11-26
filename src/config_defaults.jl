# =============================================================================
# Default configuration values — read from configs/default.toml
# =============================================================================
#
# Included BEFORE every config struct, because their `@kwdef` defaults read
# from here. That makes `configs/default.toml` the SINGLE source of truth for
# "the default": there is no second set of Julia literals to drift from it, and
# no sync test standing in for a design.
#
# Consequences worth knowing:
#
#   * `FindBoardsConfig()` IS the file. Every entry point that takes
#     `config = FindBoardsConfig()` — find_boards, annotate_blobs,
#     track_boards — therefore gets TOML configuration without threading a
#     path, and partial keyword construction inherits file values for
#     everything it does not name (`FindBoardsConfig(min_similarity = 0.7)`
#     means "the file, with min_similarity overridden").
#   * The loader (`FindBoardsConfig(path)`) still overrides per key via
#     `present_kwargs`, so a run config that sets only some keys inherits the
#     rest from the file rather than from a parallel set of literals.
#   * Values are baked at PRECOMPILE time, so editing the TOML needs a
#     recompile to take effect. That is the intended trade: these are
#     code-level defaults, and per-run variation goes through `read_run_config`.
#
# `include_dependency` is load-bearing: Julia's staleness check tracks
# `include`d files, not arbitrary files read at top level. Without it, editing
# default.toml would leave stale values baked into the .ji and nothing would
# notice — the exact invisible-drift class this file exists to remove.
# =============================================================================

using VisualGeometryToolkitCore: toml_default, parse_defaults

# Every default is `toml_default(_TOML_DEFAULTS, table, key)` (`table = ""` for a key of `[find_boards]` itself)
# or `toml_default(_AUTHORING_DEFAULTS, table, key)`: Core's one parser (`parse_defaults`: the file, and its
# precompilation dependency) and its one dotted lookup, which throws on a missing key BY DESIGN — a struct field
# whose default is not in the file would be a silent second source of truth, which is what this mechanism exists
# to prevent. There is no package-local lookup wrapper. Fields that are deliberately NOT defaults — `detect.polarity`
# (a board property), `patch_size` (declared by the ONNX graph), and `PoseGeometry`'s thresholds (tuned one below
# the homography arm; the geometry table is homography-scoped) — keep Julia literals and never read the file.
# (`adapt.cutoff` DOES read the file: it is the adaptation scan window, distinct from a board's rendering cutoff.)

"""
    run_config_path() -> String

Path to BlobBoards' `configs/default.toml`: the run configuration of record (its `[find_boards]` tree and its
`[descriptor]` network) for every entry point that is not handed an explicit config.
"""
run_config_path() = normpath(joinpath(@__DIR__, "..", "configs", "default.toml"))

# configs/default.toml's `[find_boards]` tree (`_TOML_DEFAULTS`, the name `parse_defaults` prescribes) and
# configs/authoring.toml: `[blob_board]` (layout, printing) and `[blob_pattern]` (the placement ladder, shared by
# boards and patterns) hold `BoardConfig`'s and `PatternConfig`'s keyword defaults, `[board_raster]` the image a
# board is converted to when no density is named.
const _TOML_DEFAULTS = parse_defaults(run_config_path())["find_boards"]
const _AUTHORING_DEFAULTS = parse_defaults(normpath(joinpath(@__DIR__, "..", "configs", "authoring.toml")))

"""
    DEFAULT_CUTOFF

A blob's default truncation radius, in σ: its printed disc's radius is
`cutoff·σ`, so a disc's diameter is `2·cutoff·σ`. Read from `cutoff` under
`[blob_pattern]` in configs/authoring.toml (2.0); a pattern or board records
the value it was made with.
"""
const DEFAULT_CUTOFF = Float64(toml_default(_AUTHORING_DEFAULTS, "blob_pattern", "cutoff"))

# ---- conversions the struct defaults need -----------------------------------
# These mirror the loader's converters in config_toml.jl. They live here
# because the structs are defined before that file is included.

# "unbounded" ⇒ no cap (TOML integers are 64-bit signed; the literal
# round-trips confusingly, so the file spells it). ONE converter for the
# @kwdef defaults AND the loader — the pair had drifted (any string was
# uncapped here, strict there; a typo'd default.toml now fails loudly).
_max_detections(v::AbstractString) = v == "unbounded" ? typemax(Int) :
    throw(ArgumentError("max_detections must be an integer or \"unbounded\", got $(repr(v))"))
_max_detections(v::Integer) = Int(v)
_dflt_max_detections() = _max_detections(toml_default(_TOML_DEFAULTS, "", "max_detections"))

# Empty list ⇒ the IMAS-25 schedule, and the EMPTY VECTOR carries that meaning
# all the way through: the TOML already spells it `tilts = []`, so translating
# it to `nothing` in the struct only made the field a union for no gain.
# `detect_blobs` maps empty to VGF's own name for it, `tilts = :imas25`, at
# the call boundary. Shared with the loader (they were
# expression-identical twins).
_tilts_from_toml(v) = Tuple{Float64, Float64}[(Float64(t[1]), Float64(t[2])) for t in v]
_dflt_tilts() = _tilts_from_toml(toml_default(_TOML_DEFAULTS, "detect", "tilts"))


# The RANSAC local-optimization strategy, selected by TOML tag through
# VisualGeometryToolkitEstimation's loader over its strategies and LocalOptimizationDoneRight's
# (`student_t_uniform`, `hq`, … are LocalOptimizationDoneRight's):
# `{ type = "student_t_uniform" }`, or e.g.
# `{ type = "hq", loss = "welsch", efficiency_constant = "cross_validated" }`, where
# `efficiency_constant` resolves per `class` and loss through LocalOptimizationDoneRight's
# `tuning_constant` registry. ONE converter
# for the geometry structs' defaults AND config_toml.jl's loader. `class` is
# the RVG pipeline class of the arm (:homography, :absolute_pose,
# :radial_homography); `section` only names the table in error messages.
using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation, optimization_strategy_from_toml
import LocalOptimizationDoneRight
const _OPTIMIZATION_STRATEGY_TAGS = merge(VgtkEstimation.OPTIMIZATION_STRATEGY_TAGS, LocalOptimizationDoneRight.LODR_STRATEGY_TAGS)
"""
    geometry_local_optimization_strategy(d, class::Symbol) -> AbstractOptimizationStrategy

The RANSAC local-optimization strategy a `[find_boards.geometry]`
`local_optimization_strategy` table `d` names. It is read by VisualGeometryToolkitEstimation's
loader over its strategies and LocalOptimizationDoneRight's (`class` is the
arm's pipeline class: `:homography`, `:absolute_pose`, `:radial_homography`, …).
The one reader for every geometry arm's TOML, downstream arms' included.
"""
geometry_local_optimization_strategy(d, class::Symbol) = optimization_strategy_from_toml(
    d; class, section = "find_boards.geometry.local_optimization_strategy", tags = _OPTIMIZATION_STRATEGY_TAGS
)
_dflt_local_optimization_strategy(class::Symbol) =
    geometry_local_optimization_strategy(toml_default(_TOML_DEFAULTS, "geometry", "local_optimization_strategy"), class)
