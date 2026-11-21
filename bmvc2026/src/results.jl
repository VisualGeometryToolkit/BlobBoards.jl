# =============================================================================
# Results layer — per-cell products, collected into one DataFrame
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
#     silently. Not possible here: `pose_results` reads the cells'
#     CONTENTS, so layout is not load-bearing.
#   * a CSV has no schema. Readers re-parsed "missing" by hand and guessed at
#     element types.
#
# THE UNIT IS ONE CELL = (dataset, system, occlusion level) — 3 systems x 10
# levels + 3 unoccluded = 33 files. That granularity is deliberate and is the
# one rule to keep: `pose_results` LOADS EVERY CELL to build the frame, so a
# per-image cell (600+ frames x 3 systems) would make it the wrong tool. Keep
# cells coarse; keep per-observation data as vectors INSIDE a cell.
#
# Storage is plain column vectors rather than a nested DataFrame: JLD2 would
# otherwise have to reconstruct a DataFrames type on load, which breaks across
# version bumps for no benefit — `pose_results` reassembles the frame anyway.
#
# Params ride INSIDE each cell's record, so the keyed directories stay opaque on
# disk while the collected frame carries every identifying field as a column.
# That is the property `savename` could not give us: our configs are nested,
# and `savename`'s type filter silently drops nested tables.

using DataFrames: DataFrame, flatten, groupby, nrow, select, Not

# The per-observation columns `pose_error_table` produces. Asserted on collect
# so a cell written by older code cannot silently contribute `missing` columns —
# `vcat(…; cols = :union)` UNIONS columns across cells and fills gaps, which is
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

# The record of one cell: the dataset, the marker system and the occlusion
# level (`"none"` unoccluded); its product folds in the sources' products.
_cell_config(src::Sources, system, level) =
    Dict{String, Any}("dataset" => src.dataset, "system" => String(system), "level" => String(level))

"""
    cell_product(src, system, level) -> Product

The pose-error cell of one marker system at one occlusion level (`"none"`
unoccluded): Eval's `Product` `pose_results` of the cell's record over the
evaluation's [experiment] layout, keyed on the upstream products of every
source it reads ([`derived_product`](@ref)); its output `pose.jld2` holds the
cell's record (JLD2's object `"record"`). A new detection run is a new key.
"""
cell_product(src::Sources, system, level) =
    derived_product(src, "pose_results", CELL_SOURCES; output = "pose.jld2", record = _cell_config(src, system, level))

"""
    write_pose_results(src; occlusion_level = nothing, force = false)

One cell per marker system (optionally for one occlusion level), each its own
product ([`cell_product`](@ref)), whose provenance records the code that
produced it. The products are asked FIRST: the evaluation runs once, only when a
cell is not produced (or `force`), so a rerun with every cell produced
evaluates nothing.

Returns the cells' output paths.
"""
function write_pose_results(
        src::Sources;
        occlusion_level::Union{Nothing, AbstractString} = nothing,
        force::Bool = false
    )
    level = something(occlusion_level, "none")
    evaluated = Ref{Any}(nothing)
    evaluation() = (evaluated[] === nothing && (evaluated[] = evaluate_dataset(src; occlusion_level)); evaluated[])
    files = String[]
    for bt in BOARD_TYPES
        product = cell_product(src, bt.system, level)
        produce(product; driver = "bmvc2026/scripts/make_pose_results.jl",
                config_source = "bmvc2026/configs/figures/evaluation.toml", force) do _
            r = evaluation()[bt.system]
            merge(
                _cell_config(src, bt.system, level),
                Dict{String, Any}(String(n) => r.table[!, n] for n in names(r.table)),
                Dict{String, Any}("meta" => r.meta, "sources" => Dict(k => p.key for (k, p) in src.products)),
            )
        end
        file = product_output(product)
        push!(files, file)
        println("cell $(bt.system)@$(level) -> $file")
    end
    return files
end

"""
    pose_results(src; long = true) -> DataFrame

Every cell of the sources (every system at every level), collected into one frame.

`long = true` (default) returns ONE ROW PER OBSERVATION, with `system` and
`level` as ordinary columns — which is the shape every analysis actually wants:
comparing systems, or sweeping occlusion, becomes `groupby`/`combine` instead of
a bespoke script per question. `long = false` returns the raw one-row-per-cell
frame with vector columns.
"""
function pose_results(src::Sources; long::Bool = true)
    cells = [cell_product(src, bt.system, level) for level in ["none"; src.levels] for bt in BOARD_TYPES]
    missing_cells = [c for c in cells if !produced(c)]
    isempty(missing_cells) || error("pose-error cells not produced: $(join(string.(missing_cells), ", ")) — run make_pose_results.jl first")
    rows = [DataFrame(Dict{Symbol, Any}(Symbol(k) => [v] for (k, v) in load_output(Dict{String, Any}, c))) for c in cells]
    df = vcat(rows...; cols = :union)
    have = Set(Symbol.(names(df)))
    gap = [c for c in vcat(CELL_PARAMS, OBS_COLUMNS) if !(c in have)]
    isempty(gap) || error("the pose-error cells are missing expected columns: $(gap) — regenerate with write_pose_results(...; force = true)")
    # ONE cell per (system, level): each is its own product, so a second can only be a bug.
    dup = [(g.system[1], g.level[1]) for g in groupby(df, [:system, :level]) if nrow(g) > 1]
    isempty(dup) || error("several cells for $(dup)")

    long || return df
    # Drop DrWatson/JLD2 bookkeeping and the nested meta before flattening; they
    # are per-cell scalars that would otherwise be repeated per observation.
    drop = intersect(Symbol.(names(df)), [:meta, :sources])
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
