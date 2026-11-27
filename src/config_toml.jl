# =============================================================================
# FindBoardsConfig from TOML — the single source of truth for a run
# =============================================================================
#
# POLICY: configuration is TOML, data is JSON (cau-tooling EVAL-CONVENTIONS.md).
#
# WHY THIS EXISTS. Before it, a run's configuration was assembled from four
# unrelated mechanisms: a `DEFAULT_DESCRIPTOR_MODEL` const in this package, a
# `LocalPreferences` key, `ENV["BLOBBOARDS_DESCRIPTOR_MODEL"]`, and per-driver
# `BB_*` env vars — with the descriptor model reachable three different ways
# and the gallery chosen independently of it. Nothing recorded what was used,
# so "which network produced these poses?" was unanswerable after the fact,
# and a stale `BB_LIMIT` in a shell silently truncated a sweep to 40 of 56
# images while looking perfectly clean.
#
# One TOML names everything. `write_config` round-trips it, so the exact
# configuration can be written beside the artifact it produced.
#
# CONSTRUCTION, NOT FIELD-FILLING. `FindBoardsConfig` is parametric and every
# field is concrete on purpose — it is read inside the per-board fitting loop,
# where an abstract field would box and dispatch dynamically. So the loader
# cannot fill a mutable struct; it dispatches a TAG to a constructor. Each
# registry below is exhaustive over its type's subtypes, and an unknown tag
# throws with the valid set rather than silently falling back to a default.

using TOML: TOML

# -----------------------------------------------------------------------------
# Tag registries — the only place a TOML string maps to a type
# -----------------------------------------------------------------------------

# ONE default authority: `configs/default.toml`. Loaders supply only the keys the
# TOML actually states (`present_kwargs`); an absent key falls through to the
# struct's @kwdef default — which READS that same file (src/config_defaults.jl).
# So a run config that sets some keys inherits the rest from the defaults file,
# not from a parallel set of literals.
#
# Two layers of the restated-literal pattern have been removed this way. First
# the loaders' own fallbacks, which is how the geometry min_inliers values
# drifted from the structs on 2026-08-11 — 26 sites, each a silent divergence.
# Then the struct literals themselves: they duplicated the file and were kept
# in step only by a test, so "the default" still had two spellings and the test
# was the safety net rather than the design.
# The generic loader/writer machinery lives in VisualGeometryToolkitCore since
# 2026-09-07 (hoisted from this file and VisualGeometryToolkitEstimation's fork of it):
# present-keys-only keyword overrides, refuse-unknown-keys with retirement, the
# fieldnames-derived table walk with its per-type hooks, the dotted default
# lookup and the sorted writer. What stays here is BlobBoards': the retired-key
# set, the tag registries and the hook methods for its own types.
using VisualGeometryToolkitCore: present_kwargs, check_keys, toml_table, read_table,
    allowed_keys, write_toml, TagRegistry, tag_of, CONFIG_VERSION_KEY, CONFIG_VERSIONS_KEY, current_config_version, ConfigVersions,
    toml_unbounded
import VisualGeometryToolkitCore: toml_skip_fields, toml_field
# A run configuration's TOML table is VisualGeometryToolkitEstimation's `config_dict`
# (one of its CONTRACT_GENERICS): BlobBoards adds the methods of its configs.
import VisualGeometryToolkitEstimation: config_dict

# Absent inner NMS keys inherit configs/default.toml's `[find_boards.detect]
# nms` table when it configures the SAME strategy (the ONE default authority —
# `nms = { type = "ellipse" }` with no `cutoff` must load the file's cutoff,
# not `EllipseNms()`'s literal). When the file configures the OTHER strategy
# it carries no values for this one, and the struct's @kwdef literals are the
# only authority (documented at the structs).
function _nms_kwargs(d, type::AbstractString, spec::Pair{String}...)
    fd = toml_default(_TOML_DEFAULTS, "detect", "nms")
    return present_kwargs(d, spec...; fallback = fd["type"] == type ? fd : nothing)
end

const NMS_TAGS = TagRegistry(
    :nms, "Nms", "none" => nothing, "ellipse" => EllipseNms, "mxm" => MxmNms;
    paths = [r"find_boards\.detect\.nms\.type"],
)

_nms_from_toml(::Nothing, _) = nothing
_nms_from_toml(::Type{EllipseNms}, d) = EllipseNms(;
    _nms_kwargs(d, "ellipse", "cutoff" => Float32, "band_octaves" => Float32, "band_reach" => Int)...
)
_nms_from_toml(::Type{MxmNms}, d) = MxmNms(;
    _nms_kwargs(d, "mxm", "window_half" => Int, "band_octaves" => Float32, "band_reach" => Int)...
)

function _nms_table(d, section)
    t = get(d, "type", nothing)
    t isa AbstractString || throw(ArgumentError("[$section] needs a `type` key; one of $(keys(NMS_TAGS))"))
    v = _nms_from_toml(NMS_TAGS[t], d)   # an unknown tag: the registry's ArgumentError, naming the known ones
    # Validate the INNER keys too. `_check_run_keys` upstream only sees the parent
    # table, so `nms = { type = "ellipse", cutof = 2.0 }` used to construct an
    # EllipseNms at the DEFAULT cutoff and report success — the typo'd-knob
    # failure this loader exists to prevent, one nesting level down.
    _check_run_keys(d, v === nothing ? ["type"] : allowed_keys(typeof(v), :type), section)   # `none` too
    return v
end

# Reject unknown keys rather than ignoring them. A typo'd knob that silently
# does nothing is the failure this file exists to prevent — a config that
# looks like it set `peak_threshold` but spelled it `peak_thresh` would
# otherwise run at the default and report success.
# RETIRED keys: knobs that were removed because nothing ever read them, but
# that recorded run sidecars (immutable provenance) legitimately contain.
# Exact names only — a typo still fails loudly; a retired key loads with a
# warning. One entry per retirement, with the audit that removed it:
# Retirement is (section, key)-scoped: `patch_size` is retired from
# [descriptor], not from every table — `[find_boards.detect] patch_size`
# is a typo and must still fail loudly.
#   ("find_boards.blob_adaptation", "init") — the adaptation's init (:disc/:ls); no reader
#                  ever existed (package-cleanup audit 2026-08-31).
#   ("descriptor", "patch_size") — removed when the patch size became the
#                  ONNX graph's declaration (src/descriptor.jl); recorded
#                  sidecars still carry it (package-cleanup audit 2026-08-31).
#   ("find_boards.blob_adaptation", "store") — per-pixel storage precision; inert since
#                  the KA port (598f535 dropped the reader), deleted 2026-09.
const _RETIRED_RUN_KEYS = Set(
    [
        ("find_boards.blob_adaptation", "init"), ("descriptor", "patch_size"),
        ("find_boards.blob_adaptation", "store"),
    ]
)

# RENAMED keys: refused with the new name (no alias; the owner's rulings).
#   ("find_boards.geometry", "local_optimization") — the find_* strategy keyword
#                  is `local_optimization_strategy` (the 0.9 strategy rename).
const _RENAMED_RUN_KEYS = Dict(
    ("find_boards.geometry", "local_optimization") => "local_optimization_strategy",
)

# BlobBoards' run tables, checked by Core's `check_keys` with BlobBoards' own
# retired and renamed keys (Core refuses a renamed key by name, no alias).
_check_run_keys(d, allowed, section) =
    check_keys(d, allowed, section; retired = _RETIRED_RUN_KEYS, renamed = _RENAMED_RUN_KEYS)

# -----------------------------------------------------------------------------
# Section readers
# -----------------------------------------------------------------------------

function _detect_from_toml(d)
    # No `polarity`: it is a property of the boards being sought, recorded in
    # each bundle, and derived per run (see _board_settings). Asserting it here
    # could only agree — or disagree and silently filter away every blob.
    _check_run_keys(d, allowed_keys(DetectConfig), "find_boards.detect")
    kw = present_kwargs(
        d,
        "peak_threshold" => Float64,
        "edge_threshold" => Float64,
        "upsample" => Int,
        "nb_scales" => Int,
        "nb_octaves" => Int
    )
    # ASIFT tilt schedule as a list of [tilt, angle] pairs; absent OR empty ⇒
    # the struct default (nothing = IMAS-25).
    haskey(d, "tilts") && !isempty(d["tilts"]) &&
        push!(kw, :tilts => _tilts_from_toml(d["tilts"]))
    haskey(d, "nms") &&
        push!(kw, :nms => _nms_table(d["nms"], "find_boards.detect.nms"))
    return DetectConfig(; kw...)
end

function _blob_adaptation_from_toml(d)
    # NO `cutoff` KEY. It belongs to the BOARD, like polarity: it is the printed
    # blob's boundary, and it is simultaneously the adaptation scan window, the
    # clip test's extent and the canonicalization inner extent. A configured
    # value can disagree with the gallery's and make the two sides' descriptors
    # incomparable — SILENTLY, which is what happened while this key existed
    # (the example board set spans 1.75..1.995, so the adaptation scanned at 2.0 while
    # filter_blobs and describe used the 1.8725 mean).
    #
    # This key was dropped ONCE BEFORE and that removal WAS a bug: nothing fed
    # the field afterwards and the CUDA kernel kept reading it. It is removed
    # again now only because `adapt_detections` feeds it from the gallery
    # (`_shared_image_cutoff`). Do not restore the key — oversize or undersize
    # the window with `scan_padding`, which is the lever for exactly that and
    # keeps the extent anchored to the print.
    _check_run_keys(d, allowed_keys(BlobAdaptation, :stops), "find_boards.blob_adaptation")
    return BlobAdaptation(;
        present_kwargs(
            d,
            "min_intensity" => Float64,
            "scan_padding" => Float64,
            "weighting" => _weighting_from_toml,
            "max_scale_fraction" => Float64,
            "max_scale_ratio" => Float64,
            "max_clip_ratio" => Float64,
            "max_aspect_ratio" => Float64
        )...
    )
end

# `[find_boards.blob_adaptation.stops]`: `max_iter` and `rtol` (absent: configs/default.toml's), the solve's
# `StopWhenIncrementLess(rtol) | StopAfterIteration(max_iter)`.
function _blob_adaptation_stops_from_toml(d)
    _check_run_keys(d, ["max_iter", "rtol"], "find_boards.blob_adaptation.stops")
    return blob_adaptation_stops(; present_kwargs(d, "max_iter" => Int, "rtol" => Float64)...)
end
# Its inverse: the one stops form a run config holds.
_blob_adaptation_stops_toml(s::StopWhenAny{Tuple{StopWhenIncrementLess{Float64}, StopAfterIteration}}) =
    Dict{String, Any}("max_iter" => s.criteria[2].n, "rtol" => s.criteria[1].rtol)



# The camera is DATA, not configuration — it is calibrated, not chosen — so it
# arrives as a keyword rather than a path in the file. `type = "pose"` without
# one is an error, not a silent downgrade to the uncalibrated path: quietly
# estimating a homography where the caller asked for metric pose is exactly
# the kind of wrong answer that looks right.
function _geometry_from_toml(d, camera, image_size = nothing)
    t = get(d, "type", nothing)
    t isa AbstractString || throw(
        ArgumentError(
            "[find_boards.geometry] needs a `type`: \"pose\", \"homography\", " *
                "\"radial_homography\", or a tag a loaded package registers, got $(repr(t))"
        )
    )
    # A tag this package does not own goes to the extension point; its
    # fallback refuses with the valid set.
    haskey(GEOMETRY_TAGS, t) || return geometry_from_toml(Val(Symbol(t)), d; camera, image_size)
    return _geometry_arm(GEOMETRY_TAGS[t], d, camera, image_size)
end

# An arm's `ransac` table (absent: every key the defaults').
_ransac_table(d) = get(d, "ransac", Dict{String, Any}())

_geometry_arm(::Type{RadialHomographyGeometry}, d, camera, image_size) =
    radial_geometry_from_toml(RadialHomographyGeometry, _RadialProblem5, :radial_homography, d; camera, image_size)

"""
    radial_geometry_from_toml(G, P, tag::Symbol, d; camera = nothing, image_size, ransac_defaults = Dict()) -> G

A radial arm's geometry from its `[find_boards.geometry]` table `d` (`type = "<tag>"`):
the table's keys checked against `G`'s fields, the image size REQUIRED (data, like the
pose arm's camera: the sensor extent that fixes the distortion centre, which differs per
capture session — a value in the file would drift), a `camera` ignored with a warning, and
`G(image_size; ransac, min_inliers)` built from the keys present (the `[find_boards.geometry.ransac]` table read by
[`board_ransac_from_toml`](@ref) for the arm's problem type `P` and pipeline class `tag`, its absent keys
`ransac_defaults`', a downstream arm's own defaults of record, then the package's). BlobBoards' `RadialHomographyGeometry` reads through it, and so does a radial
arm another package registers (its [`geometry_from_toml`](@ref) method is one call).
"""
function radial_geometry_from_toml(::Type{G}, ::Type{P}, tag::Symbol, d; camera = nothing, image_size,
        ransac_defaults::AbstractDict = Dict{String, Any}()) where {G, P}
    _check_run_keys(d, allowed_keys(G, :type), "find_boards.geometry")
    image_size === nothing && throw(
        ArgumentError("[find_boards.geometry] type = \"$(tag)\" needs the image size: `image_size = (width, height)` beside the config")
    )
    camera === nothing || @warn "geometry type = \"$(tag)\" ignores the `camera` keyword"
    return G(
        image_size;
        ransac = board_ransac_from_toml(P, tag, _ransac_table(d); defaults = ransac_defaults),
        present_kwargs(d, "min_inliers" => Int)...
    )
end

function _geometry_arm(::Type{HomographyGeometry}, d, camera, _)
    # The pose arm's keys are not merely unknown here, they are
    # inapplicable, and "unknown key" would send the reader hunting for a
    # typo. The plane-induced H is a unique 8-DOF map; the two-fold
    # ambiguity belongs to its decomposition into (R, t, n) given K, so
    # with no camera there is no second branch to select.
    haskey(d, "pose_disambiguation") && throw(
        ArgumentError(
            "[find_boards.geometry] pose_disambiguation applies only to " *
                "type = \"pose\" — the planar pose ambiguity arises when a " *
                "homography is DECOMPOSED into a pose, which needs a camera; the " *
                "uncalibrated arm estimates H directly and has no second branch"
        )
    )
    _check_run_keys(d, allowed_keys(HomographyGeometry, :type), "find_boards.geometry")
    camera === nothing || @warn "geometry type = \"homography\" ignores the `camera` keyword"
    return HomographyGeometry(;
        ransac = board_ransac_from_toml(HomographyProblem{Float64}, :homography, _ransac_table(d)),
        present_kwargs(d, "min_inliers" => Int)...
    )
end

function _geometry_arm(::Type{PoseGeometry}, d, camera, _)
    _check_run_keys(d, allowed_keys(PoseGeometry, :type), "find_boards.geometry")
    camera === nothing && throw(
        ArgumentError(
            "[find_boards.geometry] type = \"pose\" needs a camera: " *
                "FindBoardsConfig(path; camera = VisualGeometryToolkitIO.load(BoundedCamera, ...))"
        )
    )
    ransac = board_ransac_from_toml(AbsolutePoseProblem, :absolute_pose, _ransac_table(d))
    return PoseGeometry(
        camera; ransac,
        pose_disambiguation = _pose_disambiguation_from_toml(get(d, "pose_disambiguation", "none"), ransac),
        undistortion = _undistortion_from_toml(UNDISTORTION_TAGS[get(d, "undistortion", "none")]),
        present_kwargs(d, "min_inliers" => Int)...
    )
end

# The tags this package's own arms are written and read under. An arm defined
# outside the package is loaded through the extension point below (a registry
# is closed).
const GEOMETRY_TAGS = TagRegistry(
    :geometry, "Geometry",
    "pose" => PoseGeometry, "homography" => HomographyGeometry, "radial_homography" => RadialHomographyGeometry;
    paths = [r"find_boards\.geometry\.type"],
)

# The pose arm's `undistortion` policy (a tag that selects nothing is `none`: the raw image).
const UNDISTORTION_TAGS = TagRegistry(
    :undistortion, "Undistortion", "none" => nothing, "global" => GlobalUndistortion;
    paths = [r"find_boards\.geometry\.undistortion"],
)
_undistortion_from_toml(::Nothing) = nothing
_undistortion_from_toml(::Type{GlobalUndistortion}) = GlobalUndistortion()
# The pose arm's disambiguation: `"none"`, or `{ type = "planar_pose_disambiguation", min_separation, same_basin }`
# (radians), its refine the arm's own seeded local optimization.
const POSE_DISAMBIGUATION_TAGS = TagRegistry(
    :pose_disambiguation, "", "none" => nothing, "planar_pose_disambiguation" => PlanarPoseDisambiguation;
    paths = [r"find_boards\.geometry\.pose_disambiguation(\.type)?"],
)
_pose_disambiguation_from_toml(t::AbstractString, _) = POSE_DISAMBIGUATION_TAGS[t] === nothing ? nothing : throw(ArgumentError(
    "[find_boards.geometry] pose_disambiguation = $(repr(t)): a disambiguation is a table, `{ type = $(repr(t)) }`"))
function _pose_disambiguation_from_toml(d::AbstractDict, ransac)
    t = get(d, "type", nothing)
    t isa AbstractString || throw(ArgumentError("[find_boards.geometry.pose_disambiguation] needs a `type`; one of $(keys(POSE_DISAMBIGUATION_TAGS))"))
    POSE_DISAMBIGUATION_TAGS[t] === nothing && throw(ArgumentError("[find_boards.geometry] pose_disambiguation = \"none\" is a string, not a table"))
    _check_run_keys(d, ["type", "min_separation", "same_basin"], "find_boards.geometry.pose_disambiguation")
    return PlanarPoseDisambiguation(
        _board_local_optimization(ransac);
        present_kwargs(d, "min_separation" => Float64, "same_basin" => Float64)...
    )
end

"""
    geometry_from_toml(::Val{tag}, table; camera, image_size) -> GeometryConfig

Build the geometry arm a `[find_boards.geometry]` table names by its `type`
tag, for an arm defined OUTSIDE this package. [`FindBoardsConfig`](@ref)'s
loader dispatches every tag it does not own (`"pose"`, `"homography"`,
`"radial_homography"`) here, so a package that defines its own
[`GeometryConfig`](@ref) subtype makes it loadable by adding a method for its
tag, and writable by a [`geometry_tag`](@ref) method:

```julia
BlobBoards.geometry_from_toml(::Val{:my_arm}, d; camera, image_size) = MyArm(...)
BlobBoards.geometry_tag(::MyArm) = "my_arm"
```

`camera` and `image_size` are the loader's data keywords, handed on unchanged.
An unknown tag throws `ArgumentError`.
"""
geometry_from_toml(::Val{T}, d; camera = nothing, image_size = nothing) where {T} = throw(
    ArgumentError(
        "unknown [find_boards.geometry] type $(repr(String(T))); expected one of " *
            "$(join(repr.(keys(GEOMETRY_TAGS)), ", ")), or a tag a loaded package registers " *
            "(`BlobBoards.geometry_from_toml`)"
    )
)

"""
    geometry_tag(geometry::GeometryConfig) -> String

The `type` tag `geometry` is written under in `[find_boards.geometry]` — the
inverse of [`geometry_from_toml`](@ref) for an arm defined outside this package.
"""
geometry_tag(g::GeometryConfig) = throw(
    ArgumentError("$(nameof(typeof(g))) has no TOML tag; define `BlobBoards.geometry_tag` for it")
)
geometry_tag(g::Union{HomographyGeometry, PoseGeometry, RadialHomographyGeometry}) = tag_of(GEOMETRY_TAGS, g)

# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------

"""
    FindBoardsConfig(path::AbstractString; camera = nothing, image_size = nothing) -> FindBoardsConfig

`image_size = (width, height)` is REQUIRED for `type = "radial_homography"`
(the sensor extent that fixes the radial arm's distortion centre, the image
middle); ignored otherwise.

Build a [`FindBoardsConfig`](@ref) from the `[find_boards]` table of a TOML
file. Other top-level tables (`[experiment]`, `[descriptor]`, …) are ignored
here — they are the driver's business — so one file can configure a whole run
and still be the single thing to record.

`camera` is a keyword because it is data, not configuration: it comes from a
calibration, and `type = "pose"` without one raises rather than silently
falling back to the uncalibrated homography path.

The practical consequence is that ONE config serves many cameras. Each capture
session carries its own intrinsics beside its images (260521 was shot at f/9.0,
260527 at f/22), so `go_bmvc2026.jl` and `detection_cv.jl` both discover the
camera from the dataset directory and reuse the same config file across
sessions. A `camera` path in the file would force one near-identical config per
session, differing in a single line — and they would drift.

`type` stays in the file even though the Julia types already dispatch on the
camera (`PoseGeometry{C}` carries it). It records the run's INTENT: inferring
the arm from `camera === nothing` would turn a forgotten keyword into a silent
downgrade from metric pose to homography, and would leave the config written
beside an artifact unable to say which arm produced it.

Unknown keys in any section are an error. A typo'd knob that silently runs at
its default is the failure mode this loader exists to prevent.

```toml
config_version = 4

[find_boards]
min_similarity = 0.6
verify_similarity = 0.6
min_appearance_support = 0.5

[find_boards.detect]
# NB no `polarity` — it is derived from the board bundles and this loader
# REFUSES the key (stating it could only agree, or disagree and silently
# filter away every blob).
peak_threshold = 0.01
nms = { type = "ellipse", cutoff = 2.0 }

[find_boards.geometry]
type = "pose"
min_inliers = 9
pose_disambiguation = { type = "planar_pose_disambiguation" }

[find_boards.geometry.ransac]
scoring = { type = "student_t_uniform", outlier_halfwidth = 1.716 }
min_trials = 100
max_trials = 10_000
```

See also [`write_config`](@ref), [`read_run_config`](@ref).
"""
FindBoardsConfig(path::AbstractString; camera = nothing, image_size = nothing) =
    _find_boards_from_toml(read_find_boards_table(path); camera, image_size)

"""
    read_find_boards_table(path; tables = ()) -> AbstractDict

The `[find_boards]` table of the run config at `path`. It is read by [`read_run_config`](@ref)
(`tables`: the further top-level tables a driver's configs carry beside BlobBoards' own),
an `ArgumentError` when the file has none. The ONE door of every file loader of a
`[find_boards]` config — `FindBoardsConfig(path)`, and a driver's config built on it — so a
typo'd table (`[descripter]`) or key (`[descriptor] modle`) fails exactly as it does for a run config.
"""
function read_find_boards_table(path::AbstractString; tables = ())
    rc = read_run_config(path; tables)
    haskey(rc.config_toml, "find_boards") || throw(ArgumentError("$path has no [find_boards] table"))
    return rc.find_boards
end

"""
    FindBoardsConfig(table::AbstractDict; camera = nothing, image_size = nothing) -> FindBoardsConfig

Build a [`FindBoardsConfig`](@ref) from a parsed `[find_boards]` table — the
inverse of [`config_dict`](@ref), with the same key checking as the file
loader. A table of the run configuration's version 1 (a stored driver's, with the arm's
`outlier_halfwidth` and `local_optimization_strategy` and a `ransac` table) is migrated
([`BB_CONFIG_VERSIONS`](@ref)); a current one is read as it is. For a driver that reads its own run config and hands the package its
`[find_boards]` table.
"""
FindBoardsConfig(table::AbstractDict; camera = nothing, image_size = nothing) =
    _find_boards_from_toml(_current_find_boards(table); camera, image_size)

# A `[find_boards]` table handed over on its own carries no `config_version` (the version is its file's, a key of the
# file's top level): it is read through BB_CONFIG_VERSIONS from version 1, whose steps leave a current table as it is,
# so a stored table of either version loads (a driver's own file, RadiometricCamera's experiments' `[poses.find_boards]`).
_current_find_boards(table::AbstractDict) =
    migrate_config(Dict{String, Any}("find_boards" => table), BB_CONFIG_VERSIONS; source = "a [find_boards] table")["find_boards"]

function _track_from_toml(d::AbstractDict)
    _check_run_keys(d, allowed_keys(TrackConfig), "find_boards.track")
    return TrackConfig(; read_table(TrackConfig, d)...)   # Core's reader: each present key in its field's type
end

function _find_boards_from_toml(fb::AbstractDict; camera = nothing, image_size = nothing)
    _check_run_keys(fb, allowed_keys(FindBoardsConfig), "find_boards")
    return FindBoardsConfig(;
        detect = _detect_from_toml(get(fb, "detect", Dict{String, Any}())),
        blob_adaptation = _blob_adaptation_from_toml(get(fb, "blob_adaptation", Dict{String, Any}())),
        blob_adaptation_stops = _blob_adaptation_stops_from_toml(
            get(get(fb, "blob_adaptation", Dict{String, Any}()), "stops", Dict{String, Any}())),
        track = _track_from_toml(get(fb, "track", Dict{String, Any}())),
        # The arm and its find (`[find_boards.geometry.ransac]`): absent keys inherit configs/default.toml (the ONE
        # default authority, as the arms' own constructors do).
        # An absent table, or a table without a `type` (a version-1 file that tuned only `[find_boards.ransac]`,
        # migrated to `[find_boards.geometry.ransac]`), inherits the file's default arm — not a restated literal
        # that could drift from it.
        geometry = _geometry_from_toml(
            merge(Dict{String, Any}("type" => toml_default(_TOML_DEFAULTS, "geometry", "type")),
                  get(fb, "geometry", Dict{String, Any}())), camera, image_size
        ),
        max_detections = haskey(fb, "max_detections") ?
            unbounded_from_toml(fb["max_detections"], "max_detections") : _dflt_max_detections(),
        # Scalar knobs via present_kwargs: an absent key falls through to the
        # struct's @kwdef default (the ONE default authority — see above).
        present_kwargs(
            fb,
            "min_similarity" => Float64,
            "verify_similarity" => Float64,
            "min_appearance_support" => Float64,
            "ransac_seed" => Int
        )...
    )
end

# --- the inverse: ONE source of truth ------------------------------------------
#
# `config_dict` is DERIVED from the structs, not hand-listed. It used to be a
# literal naming every field, and nothing kept it exhaustive: the adaptation's
# cutoff/min_intensity/scan_padding and DetectConfig's tilts
# were consumed by the pipeline (the CUDA kernel reads `cfg.cutoff`) and
# emitted by nothing, so `write_config` produced sidecars that reloaded to
# different values. Julia's TOML stdlib serialises Dicts only, so the walk is
# ours — but it walks `fieldnames`, which means a new field appears in the file
# with no edit here and cannot be forgotten.
#
# Four things are not mechanical, and each is one small method rather than a
# reason to hand-write the whole table:
#   * TAGGED UNIONS  — nms and geometry need a `type =` discriminator to read back
#   * SENTINELS      — max_detections == typemax(Int) writes as "unbounded"
#   * NON-TOML TYPES — tilts is a Vector{Tuple}
#   * EXCLUSIONS     — the camera is DATA, polarity is DERIVED from the boards

# Fields the walker must not emit. Declared per type so the reason lives next
# to the type it applies to.
toml_skip_fields(::Type{<:PoseGeometry}) = (:camera,)    # data, supplied at load
toml_skip_fields(::Type{<:RadialHomographyGeometry}) = (:image_size,)  # data, supplied at load
toml_skip_fields(::Type{<:DetectConfig}) = (:polarity,)  # derived from the bundles
toml_skip_fields(::Type{<:BlobAdaptation}) = (:cutoff,)  # data, from the gallery
toml_skip_fields(::Type{<:FindBoardsConfig}) = (:blob_adaptation_stops,)   # written inside [find_boards.blob_adaptation]

# Per-field overrides for the four non-mechanical cases.
toml_field(c::FindBoardsConfig, ::Val{:max_detections}) = toml_unbounded(c.max_detections)
toml_field(c::FindBoardsConfig, ::Val{:detect}) = toml_table(c.detect)
toml_field(c::FindBoardsConfig, ::Val{:blob_adaptation}) =
    merge(toml_table(c.blob_adaptation), Dict{String, Any}("stops" => _blob_adaptation_stops_toml(c.blob_adaptation_stops)))
toml_field(c::FindBoardsConfig, ::Val{:geometry}) = _geometry_toml(c.geometry)
toml_field(a::BlobAdaptation, ::Val{:weighting}) = tag_of(BLOB_WEIGHTING_TAGS, a.weighting)
toml_field(c::FindBoardsConfig, ::Val{:track}) = toml_table(c.track)
toml_field(c::DetectConfig, ::Val{:nms}) = _nms_toml(c.nms)
toml_field(c::DetectConfig, ::Val{:tilts}) = [[t[1], t[2]] for t in c.tilts]
# An arm's find writes as its `[find_boards.geometry.ransac]` table: the scorer and the LO strategy as
# Estimation's tagged tables (`{ type = ..., ... }`, the same shape its loaders read — Estimation owns that round
# trip), beside the RansacConfig fields. A board arm's find carries no hooks and no score guard (`find_board`'s
# own algorithms), so the table holds every field it has.
toml_field(g::GeometryConfig, ::Val{:ransac}) = board_ransac_toml(g.ransac)
"""
    board_ransac_toml(r::Ransac) -> Dict{String, Any}

The TOML table of a board fit's `Ransac`, [`board_ransac_from_toml`](@ref)'s inverse: its scorer and its
local-optimization strategy as VisualGeometryToolkitEstimation's tagged tables (`{ type = ..., ... }`, the shape its
readers take), beside the `RansacConfig` fields. A geometry arm's `[find_boards.geometry.ransac]` is written by it, and so
is a downstream package's board fit (CaliBlobs' pose fits). A `Ransac` with a score guard or a trace hook has no table
(an `ArgumentError`).
"""
function board_ransac_toml(r::Ransac)
    (r.score_guard || r.on_improvement !== nothing || r.on_hypothesis !== nothing) && throw(ArgumentError(
        "a board arm's `ransac` with a score guard or a trace hook has no TOML table"))
    return merge(
        Dict{String, Any}(
            "scoring" => VgtkEstimation.config_dict(r.scoring),
            "local_optimization_strategy" => VgtkEstimation.config_dict(r.local_optimization_strategy),
        ),
        VgtkEstimation.config_dict(r.config)
    )
end
toml_field(g::PoseGeometry, ::Val{:undistortion}) = tag_of(UNDISTORTION_TAGS, g.undistortion)
toml_field(g::PoseGeometry, ::Val{:pose_disambiguation}) = _pose_disambiguation_toml(g.pose_disambiguation)
# The gates only: the loader's refine is the arm's own seeded local optimization (`_pose_disambiguation_from_toml`).
_pose_disambiguation_toml(::Nothing) = tag_of(POSE_DISAMBIGUATION_TAGS, nothing)
_pose_disambiguation_toml(d::PlanarPoseDisambiguation) = Dict{String, Any}(
    "type" => tag_of(POSE_DISAMBIGUATION_TAGS, d), "min_separation" => d.min_separation, "same_basin" => d.same_basin)

# Tagged unions: the discriminator plus the type's own fields, so a new field
# on either NMS strategy or either geometry is picked up automatically.
_nms_toml(::Nothing) = Dict{String, Any}("type" => tag_of(NMS_TAGS, nothing))
_nms_toml(n) = merge(Dict{String, Any}("type" => tag_of(NMS_TAGS, n)), toml_table(n))

_geometry_toml(g::GeometryConfig) =
    merge(Dict{String, Any}("type" => geometry_tag(g)), toml_table(g))

"""
    config_dict(config::FindBoardsConfig) -> Dict

The `[find_boards]` table for `config`, as a plain `Dict` — the form
[`write_config`](@ref) emits and provenance sidecars embed.

Derived from the structs by walking `fieldnames`, so it cannot fall out of
sync with them: add a field to any config type and it appears here.
"""
config_dict(c::FindBoardsConfig) = toml_table(c)

"""
    write_config(path, config; extra = Dict(), versions = BB_CONFIG_VERSIONS,
                 tables = Dict("find_boards" => config_dict(config)))

Write `config` — a [`FindBoardsConfig`](@ref), or any config with a [`config_dict`](@ref)
method (a driver's config built on it) — as a TOML file of the current `config_version` with a `[find_boards]` table
(`config_dict(config)`), merging any `extra` top-level tables (`experiment`, `descriptor`,
…); the header names the config's type as its reloading constructor.

A host package whose run config holds BlobBoards' tables beside its own (CaliBlobs) hands its converter over as
`versions` (a VisualGeometryToolkitCore `ConfigVersions` that delegates `find_boards` and `descriptor` to
[`BB_CONFIG_VERSIONS`](@ref)) and its file's top-level tables as `tables`: the file states the host's `config_version`
and each delegated owner's under `[config_versions]`, as [`read_run_config`](@ref)`(path; versions)` reads it. A `FindBoardsConfig`
round-trips through [`FindBoardsConfig(path)`](@ref) — which the test suite asserts — so a
config recorded beside an artifact reproduces the run that made it.

`max_detections = typemax(Int)` is emitted as the string `"unbounded"` (VisualGeometryToolkitCore's `toml_unbounded`), since
TOML integers are 64-bit signed and the literal round-trips confusingly.
"""
function write_config(path::AbstractString, config; extra::AbstractDict = Dict{String, Any}(),
        versions::ConfigVersions = BB_CONFIG_VERSIONS,
        tables::AbstractDict = Dict{String, Any}("find_boards" => config_dict(config)))
    _check_host(versions)
    stamps = Dict{String, Any}(CONFIG_VERSION_KEY => current_config_version(versions))
    versions === BB_CONFIG_VERSIONS || (stamps[CONFIG_VERSIONS_KEY] = Dict{String, Any}(
        o.package => current_config_version(o) for o in unique(last.(versions.delegates))))
    body = merge(Dict{String, Any}(string(k) => v for (k, v) in extra), Dict{String, Any}(string(k) => v for (k, v) in tables), stamps)
    T = nameof(typeof(config))
    return write_toml(
        path, body;
        header = "# Generated by BlobBoards.write_config ($(parentmodule(typeof(config))).$(T)) — the configuration of record\n" *
            "# for this run. Reload with $(T)(path; camera=..., image_size=...)."
    )
end


"""
    descriptor_network_path(run_config) -> String
    descriptor_network_path(path::AbstractString) -> String
    descriptor_network_path() -> String

Absolute path to the descriptor checkpoint named by a run config's
`[descriptor] model` key, resolved against [`data_dir`](@ref).

This is the ONLY way a run should name its network. It replaced a
`DEFAULT_DESCRIPTOR_MODEL` constant in this package plus a
`LocalPreferences` key plus `ENV["BLOBBOARDS_DESCRIPTOR_MODEL"]` plus
per-driver `BB_ONNX` — four mechanisms, of which the library-level constant
was the worst: a package asserting the filename of one lab's checkpoint,
which meant anyone installing BlobBoards got a default pointing at a file
they had never had, and two paper artifacts silently ran a different network
from the paper's main sweep.

`data_dir()` stays in `LocalPreferences` because *where the data lives* is
genuinely per-machine. *Which network to use* is a property of the
experiment, not the machine — which is exactly why the preference mechanism
was the wrong scope for it.

The key is required. There is no default: a run that does not say which
network it used is a run whose results cannot be reproduced.

TWO source forms, selected by the TOML value's type:

  model = "models/x.onnx"                     # STRING: data_dir-relative —
      the FROZEN form. Bytes come from the local primary copy or the DOI'd
      Zenodo `assets` tier; experiment configs use this so the paper's
      network is pinned to an immutable record.

  model = { artifact = "x" }                  # TABLE: Pkg artifact — the
      DISTRIBUTION form. `artifact` names an entry in the package-root
      `Artifacts.toml`, a catalog of every published network (lazy entries,
      tarballs in a numbered version of the BlobBoards Zenodo record, sha256 + git-tree-sha1
      verified). The first run fetches the tarball into Julia's shared
      content-addressed artifact store; every later run resolves locally.
      The catalog cannot float: an entry IS a pinned hash, so a config
      always means the same bytes — newer weights are published as a NEW
      catalog entry and adopted by editing the config's `artifact` name,
      a git-tracked one-line change on each side.
"""
function descriptor_network_path(rc::NamedTuple)
    haskey(rc.descriptor, "model") || throw(
        ArgumentError(
            "config has no [descriptor] model key — a run must name its network " *
                "explicitly; there is no default"
        )
    )
    return _resolve_model(rc.descriptor["model"])
end

# STRING: data_dir-relative (the frozen / Zenodo-record form).
_resolve_model(model::AbstractString) = joinpath(data_dir(), model)

# The package-root artifact catalog: one lazy entry per published descriptor
# network (and example dataset), each pinned by tarball sha256 + git-tree-sha1
# (the URLs point at a numbered version of the BlobBoards Zenodo record). The
# catalog chooses NOTHING — which entry a run uses is named by its config; the
# catalog only makes every
# published network fetchable and locally hosted through Pkg's
# content-addressed artifact store.
_artifacts_toml() = joinpath(dirname(@__DIR__), "Artifacts.toml")

# TABLE: Pkg-artifact by catalog name (the distribution form).
function _resolve_model(model::AbstractDict)
    unknown = setdiff(keys(model), ("artifact",))
    isempty(unknown) || throw(
        ArgumentError(
            "[descriptor] model table has unknown key(s) " *
                "$(sort!(collect(String.(unknown)))); allowed: artifact"
        )
    )
    haskey(model, "artifact") || throw(
        ArgumentError("[descriptor] model table is missing \"artifact\"")
    )
    name = String(model["artifact"])
    toml = _artifacts_toml()
    catalog = TOML.parsefile(toml)
    haskey(catalog, name) || throw(
        ArgumentError(
            "[descriptor] model artifact \"$name\" is not in the package " *
                "catalog ($toml); published networks: $(sort!(collect(String.(keys(catalog)))))"
        )
    )
    dir = VgtkCore.install_artifact(name, toml)   # Core's one install of a published artifact
    onnx = filter(f -> endswith(f, ".onnx"), readdir(dir))
    length(onnx) == 1 || error(
        "descriptor artifact \"$name\" must hold exactly one .onnx file; " *
            "found $(length(onnx)) in $dir"
    )
    return joinpath(dir, only(onnx))
end
"""
    example_data(name::Symbol) -> String

Return the directory of one of the package's example-data artifacts, fetched
on first use.

`name` is `:boards` (50 board-definition JSONs), `:still` (one calibrated
4240x2832 image + `cameras.json`) or `:video` (one calibrated 1080p clip +
`cameras.json`). Each is fetched from the BlobBoards Zenodo record through
the package catalog (Artifacts.toml) into the shared artifact store. The
pipeline examples run entirely on these: a fresh `] add BlobBoards` needs
no dataset, no `data_dir`, no configuration.
"""
function example_data(name::Symbol)
    haskey(_EXAMPLE_ARTIFACTS, name) ||
        throw(ArgumentError("unknown example artifact $(repr(name)); expected one of :boards, :still, :video"))
    return VgtkCore.install_artifact(_EXAMPLE_ARTIFACTS[name], _artifacts_toml())   # Core's one install
end
# The catalog entry of each example (Artifacts.toml is add-only: a changed
# example is a new entry, and this map moves to it): the boards as
# blobboards/board/v5, the still's and clip's cameras.json as vgio/cameras/v4.
const _EXAMPLE_ARTIFACTS = Dict(
    :boards => "bb_example_boards_v5", :still => "bb_example_still_v4", :video => "bb_example_video_v4",
)

descriptor_network_path(path::AbstractString) = descriptor_network_path(read_run_config(path))
descriptor_network_path() = descriptor_network_path(run_config_path())

"""
    read_run_config(path; tables = (), versions = BB_CONFIG_VERSIONS)
        -> (; config_toml, tables, find_boards, experiment, gallery, descriptor)

Read the whole run config, not just the `[find_boards]` table: `config_toml`
is the raw parsed file (for a provenance sidecar), and the named fields are
its top-level tables, defaulting to empty. Drivers use this to get the
dataset, subsets and descriptor model from the same file that configures
`find_boards`, so there is one thing to supply and one thing to record.

OWNERSHIP AND VALIDATION: unknown TOP-LEVEL tables are refused (a typo'd
`[descripter]` used to make every knob in it a silent no-op), and the
package-owned `[descriptor]` table is key-checked (`model` only; the retired
`patch_size` warns — the graph declares it). `[experiment]` and `[gallery]`
are DRIVER-owned free-form tables: the research drivers define their keys,
and the package deliberately has no opinion on them. So is `[inputs]`, the
paper projects' `[inputs.<id>]` statement of what a run reads
(VisualGeometryToolkitEval's convention: tier, path, provenance, sha256). A driver that keeps
further top-level tables of its own names them in `tables`; they are admitted,
not checked, and read from `config_toml`. BlobBoards' own tables are read at the current
version: a file stating an earlier `config_version` (or none: version 1, the files before
versioning) is migrated by [`BB_CONFIG_VERSIONS`](@ref), and `find_boards`/`descriptor` are the
migrated tables (`config_toml` stays the file as written; `tables` is the whole file at the current
version). (The board and pattern authoring defaults are configs/authoring.toml's, never a run config's:
`[blob_board]` there is refused.)

A host package's run config (CaliBlobs': its own tables beside BlobBoards') is read with the host's converter as
`versions`, a VisualGeometryToolkitCore `ConfigVersions` that delegates `find_boards` and `descriptor` to
[`BB_CONFIG_VERSIONS`](@ref): the whole file is migrated through it (its `config_version` the host's, BlobBoards'
tables from the version `[config_versions] BlobBoards` states), its tables are the host's to admit (`tables` is
refused beside `versions`), and `[descriptor]` is key-checked as above.
"""
function read_run_config(path::AbstractString; tables = (), versions::ConfigVersions = BB_CONFIG_VERSIONS)
    isfile(path) || throw(ArgumentError("no such config file: $path"))
    root = TOML.parsefile(path)
    _check_host(versions)
    if versions === BB_CONFIG_VERSIONS
        _check_run_keys(
            root,
            vcat([CONFIG_VERSION_KEY, "find_boards", "experiment", "gallery", "descriptor", "inputs"], String[String(t) for t in tables]),
            "top level"
        )
        # BlobBoards' own tables at the current version (`BB_CONFIG_VERSIONS`): a stored file of an earlier one reads
        # through its version steps.
        own = migrate_config(
            Dict{String, Any}(k => root[k] for k in (CONFIG_VERSION_KEY, "descriptor", "find_boards") if haskey(root, k)),
            BB_CONFIG_VERSIONS; source = path
        )
        current = merge(Dict{String, Any}(root), own)
    else
        isempty(tables) || throw(ArgumentError(
            "read_run_config: a host's tables are its converter's ($(versions.package)'s ConfigVersions), not `tables`"))
        current = migrate_config(root, versions; source = path)   # BlobBoards' tables by BB_CONFIG_VERSIONS, delegated
    end
    descriptor = get(current, "descriptor", Dict{String, Any}())
    _check_run_keys(descriptor, ["model"], "descriptor")
    empty_ = Dict{String, Any}()
    return (;
        config_toml = root,
        tables = current,
        find_boards = get(current, "find_boards", empty_),
        experiment = get(current, "experiment", empty_),
        gallery = get(current, "gallery", empty_),
        descriptor,
    )
end

# A host's converter delegates BlobBoards' tables to BlobBoards' own (`BB_CONFIG_VERSIONS`), or it cannot read them.
function _check_host(versions::ConfigVersions)
    versions === BB_CONFIG_VERSIONS && return nothing
    for t in ("find_boards", "descriptor")
        any(d -> first(d) == t && last(d) === BB_CONFIG_VERSIONS, versions.delegates) || throw(ArgumentError(
            "$(versions.package)'s ConfigVersions does not delegate [$t] to BlobBoards' BB_CONFIG_VERSIONS"))
    end
    return nothing
end
