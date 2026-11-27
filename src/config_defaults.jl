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

using VisualGeometryToolkitCore: toml_default, parse_defaults, ConfigVersions, ConfigMigration, migrate_config, unbounded_from_toml

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

"""
    BB_CONFIG_VERSIONS

BlobBoards' run-configuration converter (VisualGeometryToolkitCore's `ConfigVersions`): a run config's
`[descriptor]` and `[find_boards]` tables are version 3 (configs/default.toml states `config_version = 3`), and
[`read_run_config`](@ref) reads a stored file of an earlier version through `migrate_config`, so the code sees the
current tables only. Version 1 → 2 (BlobBoards 0.8): the arm's find is one `Ransac`, its table
`[find_boards.geometry.ransac]` — `[find_boards.ransac]` moves there whole, the geometry's
`local_optimization_strategy` joins it, and `outlier_halfwidth = h` becomes its
`scoring = { type = "student_t_uniform", outlier_halfwidth = h }`; the pose arm's `resolve_pose_ambiguity = true`
becomes `pose_disambiguation = { type = "planar_pose_disambiguation" }` (`false`: `"none"`). Version 2 → 3
(BlobBoards 0.8): `[find_boards.adapt]` is `[find_boards.blob_adaptation]` ([`BlobAdaptation`](@ref)'s table), its
`max_irls_iter` and `rtol` the solve's stops, `[find_boards.blob_adaptation.stops]` `max_iter` and `rtol`.
"""
const BB_CONFIG_VERSIONS = ConfigVersions("BlobBoards",
    # version 1 → 2 (BlobBoards 0.8): an arm's algorithms are values, its find one `Ransac`
    ConfigMigration(; tables = ["descriptor", "find_boards"],
        renames = [
            "find_boards.ransac" => "find_boards.geometry.ransac",
            "find_boards.geometry.local_optimization_strategy" => "find_boards.geometry.ransac.local_optimization_strategy",
        ],
        values = [
            "find_boards.geometry.outlier_halfwidth" => h -> ("find_boards.geometry.ransac.scoring" =>
                Dict{String, Any}("type" => "student_t_uniform", "outlier_halfwidth" => h)),
            "find_boards.geometry.resolve_pose_ambiguity" => b -> ("find_boards.geometry.pose_disambiguation" =>
                b ? Dict{String, Any}("type" => "planar_pose_disambiguation") : "none"),
        ]),
    # version 2 → 3 (BlobBoards 0.8): the adaptation's table is named after its algorithm (`BlobAdaptation`), its
    # iteration cap and tolerance the solve's stops (`[find_boards.blob_adaptation.stops]`, decision 11)
    ConfigMigration(; tables = ["descriptor", "find_boards"],
        renames = [
            "find_boards.adapt" => "find_boards.blob_adaptation",
            "find_boards.blob_adaptation.max_irls_iter" => "find_boards.blob_adaptation.stops.max_iter",
            "find_boards.blob_adaptation.rtol" => "find_boards.blob_adaptation.stops.rtol",
        ]);
    tables = ["descriptor", "find_boards"])

# configs/default.toml's `[find_boards]` tree (`_TOML_DEFAULTS`, the name `parse_defaults` prescribes) and
# configs/authoring.toml: `[blob_board]` (layout, printing) and `[blob_pattern]` (the placement ladder, shared by
# boards and patterns) hold `BoardConfig`'s and `PatternConfig`'s keyword defaults, `[board_raster]` the image a
# board is converted to when no density is named.
const _TOML_DEFAULTS = parse_defaults(run_config_path(); versions = BB_CONFIG_VERSIONS)["find_boards"]
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

# "unbounded" ⇒ no cap: VisualGeometryToolkitCore's one TOML encoding of an uncapped count (`toml_unbounded`,
# `unbounded_from_toml`), for the @kwdef default AND the loader (a typo'd default.toml fails loudly, naming the key).
_dflt_max_detections() = unbounded_from_toml(toml_default(_TOML_DEFAULTS, "", "max_detections"), "max_detections")

# Empty list ⇒ the IMAS-25 schedule, and the EMPTY VECTOR carries that meaning
# all the way through: the TOML already spells it `tilts = []`, so translating
# it to `nothing` in the struct only made the field a union for no gain.
# `detect_blobs` maps empty to VGF's own name for it, `tilts = :imas25`, at
# the call boundary. Shared with the loader (they were
# expression-identical twins).
_tilts_from_toml(v) = Tuple{Float64, Float64}[(Float64(t[1]), Float64(t[2])) for t in v]
_dflt_tilts() = _tilts_from_toml(toml_default(_TOML_DEFAULTS, "detect", "tilts"))


# A geometry arm's RANSAC (`[find_boards.geometry.ransac]`): its scorer, selected by TOML tag through
# VisualGeometryToolkitEstimation's scorer reader (`scoring = { type = "student_t_uniform", outlier_halfwidth = h }`,
# the priors the stack's `[scoring.student_t_uniform]`), its local-optimization strategy through Estimation's loader
# over its strategies and LocalOptimizationDoneRight's (`student_t_uniform`, `hq`, … are LocalOptimizationDoneRight's:
# `{ type = "student_t_uniform" }`, or e.g. `{ type = "hq", loss = "welsch", efficiency_constant = "cross_validated" }`,
# where `efficiency_constant` resolves per `class` and loss through LocalOptimizationDoneRight's `tuning_constant`
# registry) and the `RansacConfig` fields beside them. ONE reader for the geometry structs' defaults AND
# config_toml.jl's loader, downstream arms' included.
using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation, optimization_strategy_from_toml, scoring_from_toml,
    Ransac, RansacConfig
using VisualGeometryToolkitCore: check_keys
import LocalOptimizationDoneRight
const _OPTIMIZATION_STRATEGY_TAGS = merge(VgtkEstimation.OPTIMIZATION_STRATEGY_TAGS, LocalOptimizationDoneRight.LODR_STRATEGY_TAGS)
const _BOARD_RANSAC_KEYS = vcat(["scoring", "local_optimization_strategy"], String.(collect(fieldnames(RansacConfig))))
"""
    board_ransac_from_toml(P, class::Symbol, d; defaults = Dict(), section = "find_boards.geometry.ransac") -> Ransac

A geometry arm's `Ransac` from its `[find_boards.geometry.ransac]` table `d`: `scoring` (a scorer table, read by
VisualGeometryToolkitEstimation's `scoring_from_toml` for the arm's problem type `P`), `local_optimization_strategy`
(a strategy table, read by Estimation's loader over its strategies and LocalOptimizationDoneRight's for the arm's
pipeline class `class`: `:homography`, `:absolute_pose`, `:radial_homography`, …) and the `RansacConfig` fields
(`min_trials`, `max_trials`, `success_prob`). A key `d` leaves out is `defaults`' (a downstream arm's own
defaults of record, its file's table), and one both leave out is configs/default.toml's `[find_boards.geometry.ransac]`;
an unknown key is an `ArgumentError`. The one reader for every geometry arm's TOML, downstream arms' included.
"""
function board_ransac_from_toml(::Type{P}, class::Symbol, d::AbstractDict;
        defaults::AbstractDict = Dict{String, Any}(),
        section::AbstractString = "find_boards.geometry.ransac") where {P}
    check_keys(defaults, _BOARD_RANSAC_KEYS, section)
    check_keys(d, _BOARD_RANSAC_KEYS, section)
    t = merge(toml_default(_TOML_DEFAULTS, "geometry", "ransac"), defaults, d)
    return Ransac(;
        scoring = scoring_from_toml(t["scoring"]; problem = P, section = "$section.scoring"),
        local_optimization_strategy = optimization_strategy_from_toml(t["local_optimization_strategy"];
            class, section = "$section.local_optimization_strategy", tags = _OPTIMIZATION_STRATEGY_TAGS),
        config = RansacConfig(; (f => convert(fieldtype(RansacConfig, f), t[String(f)]) for f in fieldnames(RansacConfig))...),
    )
end
