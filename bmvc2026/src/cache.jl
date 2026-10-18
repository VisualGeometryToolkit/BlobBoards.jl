#=
cache.jl — derived artifacts through the stack's keyed result cache
(VisualGeometryToolkitEval.Experiments `cached_result`: the file name DrWatson's
`produce_or_load(…; filename = hash, prefix)` builds, loaded with `wload`,
saved with `tagsave` — but a save error PROPAGATES, where `produce_or_load`
swallows it, DrWatson saving_files.jl:107-116).

This file is deliberately thin. It fixes the project's conventions in ONE place
(where artifacts land) and adds no naming machinery of its own.

WHY `filename = hash` AND NOT `savename`. `savename` filters values by TYPE:
`default_allowed` is `(Real, String, SubString, Symbol, TimeType)`, so nested
tables and arrays are silently DROPPED from the name. Our configs are almost
entirely nested (`detect`, `adapt`, `geometry`, `ransac`), so two configs
differing only in `detect.peak_threshold` would `savename` to the same string —
and the cache would then load the first run's result and return it as
the second's. That is the exact silent-stale failure keying exists to prevent,
reintroduced by the keying. `hash` sees the whole container: verified
order-independent, sensitive to nested differences, and identical across
separate Julia processes.

`hash` carries no cross-VERSION stability guarantee. Accepted deliberately: if
it ever moved, the cache would compute a new name, fail to find it, and
recompute — orphaned files and wasted time, never a wrong answer. A custom
version-stable digest was written and removed; it bought protection against a
rare, recoverable, non-silent failure at the cost of code to maintain forever.

`prefix` keeps the artifacts greppable, since `hash` alone yields a bare
integer: `handeye_bb_14300935657883058264.jld2`.

WHAT IS NOT CACHED HERE. The detections (`<run>/poses/<system>/`): each is a
whole keyed run directory of its own (src/runs.jl), the same layout for all
three marker systems.
=#

using VisualGeometryToolkitEval.Experiments: cached_result

"""
    cached(f, prefix, cfg, subdir; force = false) -> (data, file)

The keyed result cache (VisualGeometryToolkitEval.Experiments `cached_result`) with
this project's conventions: artifacts under `results_root()/<subdir>` (the
add-only outputs tree, never `data/exp_raw`), named `<prefix>_<hash(cfg)>.jld2`
(the LEGACY Base.hash key, `legacy_hash_key`, until the next full
regeneration), and `tag`ged so gitcommit and the patch of a dirty tree ride
along. `force` keeps the outgoing version as DrWatson `safesave`'s `_#<n>`
(older backups carry this project's former `#<n>` suffix).

`f(cfg)` must return a string-keyed `Dict` — JLD2 is the store.
"""
cached(f, prefix::AbstractString, cfg::AbstractDict, subdir::AbstractString; force::Bool = false) =
    cached_result(f, prefix, cfg, joinpath(results_root(), subdir); force)
