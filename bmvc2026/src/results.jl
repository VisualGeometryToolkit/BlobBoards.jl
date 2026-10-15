# =============================================================================
# Results layer — per-cell artifacts + `collect_results` into one DataFrame
# =============================================================================
#
# Replaces the CSV round-trip that used to sit between `evaluate_dataset` and
# the plots. `evaluate_dataset` already returns a DataFrame per marker system;
# the CSV was only how it was persisted, and persisting it that way cost us:
#
#   * the tables carried NO parameters. Answering "which run produced this
#     row?" meant opening `_provenance.toml` by hand — which is how a published
#     figure set came to describe a superseded detection run unnoticed.
#   * reaching the plots was a DIRECTORY COPY, which twice skipped tables
#     silently. Not possible here: `collect_results` walks and reads
#     CONTENTS, so layout is not load-bearing.
#   * a CSV has no schema. Readers re-parsed "missing" by hand and guessed at
#     element types.
#
# THE UNIT IS ONE CELL = (dataset, system, occlusion level) — 3 systems x 10
# levels + 3 unoccluded = 33 files. That granularity is deliberate and is the
# one rule to keep: `collect_results` LOADS EVERY FILE to build the frame, so a
# per-image cell (600+ frames x 3 systems) would make it the wrong tool. Keep
# cells coarse; keep per-observation data as vectors INSIDE a cell.
#
# Storage is plain column vectors rather than a nested DataFrame: JLD2 would
# otherwise have to reconstruct a DataFrames type on load, which breaks across
# version bumps for no benefit — `collect_results` reassembles the frame anyway.
#
# Params ride INSIDE each file, so the hash-named artifacts stay opaque on disk
# while the collected frame carries every identifying field as a column. That is
# the property `savename` could not give us: our configs are nested, and
# `savename`'s type filter silently drops nested tables (see cache.jl).

using DrWatson: collect_results
using DataFrames: DataFrame, flatten, groupby, nrow, select, Not

# The per-observation columns `pose_error_table` produces. Asserted on collect
# so a cell written by older code cannot silently contribute `missing` columns —
# `collect_results` UNIONS columns across files and fills gaps, which is
# convenient and is exactly how schema drift goes unnoticed. Same failure family
# as `savename` dropping nested fields, and as `dict_list` expanding four grid
# axes while the applier handled three: both shipped here, neither errored.
const OBS_COLUMNS = [
    :error_angles, :error_distances, :marker_ids, :rig_distances, :rig_angles,
    :marker_scales, :dot_product, :rig_angles_bin, :rig_distances_bin,
    :error_angles_deg, :abs_error_angles, :abs_error_angles_deg,
]

# Identifying columns every cell must carry.
const CELL_PARAMS = [:dataset, :system, :level]

# The run keys a pose-error cell depends on: the motion-capture records and the
# detections of the three marker systems (and their occlusion sweep).
const CELL_SOURCES = ["records", "baselines", "bb", "occlusion"]

"""
    results_dir(src) -> String

Where the pose-error cells of one evaluation's sources live:
`pose_results-<key>/` under results_root(), keyed on the run keys of every
upstream it reads ([`derived_dir`](@ref)). A new detection run is a new
directory, so `collect_results` can enumerate everything in it: there is
nothing older to select against.
"""
results_dir(src::Sources) = derived_dir(src, "pose_results", CELL_SOURCES)

"""
    write_pose_results(src; occlusion_level = nothing, force = false)

Compute one cell per marker system (optionally for one occlusion level) and
persist each through `cached` — so every artifact is `tag!`ed with the commit
and dirty patch that produced it, and re-running is a no-op unless `force`.

Returns the written file paths.
"""
function write_pose_results(
        src::Sources;
        occlusion_level::Union{Nothing, AbstractString} = nothing,
        force::Bool = false
    )
    results = evaluate_dataset(src; occlusion_level)
    level = something(occlusion_level, "none")
    files = String[]
    for (system, r) in results
        cfg = Dict{String, Any}("dataset" => src.dataset, "system" => system, "level" => level)
        _, file = cached(
            _ -> merge(
                copy(cfg),
                Dict{String, Any}(String(c) => r.table[!, c] for c in names(r.table)),
                # `script` is NOT set by `tag = true`: `produce_or_load` calls
                # `tag!`, which records gitcommit and gitpatch but leaves the
                # source alone. Record the entry point explicitly.
                Dict{String, Any}(
                    "meta" => r.meta,
                    "sources" => copy(src.keys),
                    "script" => isempty(PROGRAM_FILE) ? "src/results.jl" : basename(PROGRAM_FILE),
                ),
            ),
            "pose", cfg, relpath(results_dir(src), results_root()); force
        )
        push!(files, file)
        println("cell $(system)@$(level): $(nrow(r.table)) observations -> $file")
    end
    return files
end

"""
    pose_results(src; long = true) -> DataFrame

`collect_results` over every cell of the sources' results directory.

`long = true` (default) returns ONE ROW PER OBSERVATION, with `system` and
`level` as ordinary columns — which is the shape every analysis actually wants:
comparing systems, or sweeping occlusion, becomes `groupby`/`combine` instead of
a bespoke script per question. `long = false` returns the raw one-row-per-cell
frame with vector columns.
"""
function pose_results(src::Sources; long::Bool = true)
    dir = results_dir(src)
    isdir(dir) || error("no pose results at $dir — run make_pose_results.jl first")
    # `rexclude` drops the backups a forced regeneration leaves behind
    # (DrWatson safesave's `_#N`).
    df = collect_results(dir; rexclude = [r"#\d+\.jld2$"])
    isempty(df) && error("no cells in $dir")
    have = Set(Symbol.(names(df)))
    gap = [c for c in vcat(CELL_PARAMS, OBS_COLUMNS) if !(c in have)]
    isempty(gap) || error("cells in $dir are missing expected columns: $(gap) — regenerate with write_pose_results(...; force = true)")
    # ONE cell per (system, level): the key fixes the sources, so a second
    # cell can only be a parameter the key does not see (a code change under
    # `force`'s backup is excluded above). Refused, never resolved silently.
    dup = [(g.system[1], g.level[1]) for g in groupby(df, [:system, :level]) if nrow(g) > 1]
    isempty(dup) || error("$dir holds several cells for $(dup)")

    long || return df
    # Drop DrWatson/JLD2 bookkeeping and the nested meta before flattening; they
    # are per-cell scalars that would otherwise be repeated per observation.
    drop = intersect(Symbol.(names(df)), [:path, :meta, :sources, :gitcommit, :gitpatch, :script])
    out = flatten(select(df, Not(drop)), OBS_COLUMNS)

    # Canonicalise the missing sentinel. `pose_error_table` uses `nothing` for
    # an undetected board, and the CSV round-trip used to convert that to
    # `missing` on the way back in — silently, as a side effect of serialising
    # through a format that has only one sentinel. Reading the artifacts
    # directly preserves BOTH, and a column typed
    # `Union{Missing, Nothing, Float64}` makes Makie's `float_type` ambiguous
    # and every `skipmissing` wrong (it skips `missing` but keeps `nothing`).
    # Normalise here so the frame has one sentinel, and `identity.` narrows the
    # eltype back down from the `Any` that `flatten` leaves behind.
    for c in OBS_COLUMNS
        hasproperty(out, c) || continue
        out[!, c] = identity.(replace(out[!, c], nothing => missing))
    end
    return out
end
