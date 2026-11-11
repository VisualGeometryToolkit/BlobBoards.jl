#=
cache.jl — derived artifacts through the stack's keyed result cache
(VisualGeometryToolkitEval.Experiments `cached_result`: `<dir>/<prefix>_<key>.jld2`,
loaded with `wload`, saved tagged like `tagsave` — a save error PROPAGATES,
where DrWatson's `produce_or_load` swallows it, saving_files.jl:107-116).

This file is deliberately thin. It fixes the project's conventions in ONE place
(where artifacts land) and adds no naming machinery of its own.

WHY `key = run_key` AND NOT `savename` OR `hash`. `savename` filters values
by TYPE (`default_allowed` is `(Real, String, SubString, Symbol, TimeType)`), so
nested tables and arrays are silently DROPPED from the name — two configs
differing only in a nested value would share one artifact. `hash` sees the whole
container but is not stable across Julia versions (Eval's `legacy_hash_key`,
which `cached_result` falls back to, is that). `run_key` is the stack's one run
identity: the SHA-256 of the sorted TOML of the config, stable across key order,
processes and Julia versions. Every call here passes `key = run_key`.

`prefix` keeps the artifacts greppable: `pose_<run_key>.jld2`.

WHAT IS NOT CACHED HERE. The detections (`<run>/poses/<system>/`): each is a
whole keyed run directory of its own (src/runs.jl), the same layout for all
three marker systems.
=#

using VisualGeometryToolkitEval.Experiments: cached_result, run_key

# The cache is called directly (`cached_result(f, prefix, cfg, dir; key = run_key, force)`,
# `dir` under `results_root()`): a project wrapper around it was a second copy
# of the same one line in each paper project, and is deleted.
