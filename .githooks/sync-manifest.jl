# Driven by .githooks/sync-manifest; the env is already selected via --project.
# ARGS[1] is "update" (untracked Manifest: full re-resolve — targeted updates
# fail on an illegal Manifest) or "instantiate" (tracked Manifest: download only).
using Pkg

mode = get(ARGS, 1, "update")
direct = Set(keys(Pkg.project().dependencies))

snapshot() = try
    Dict(i.name => i.version for (_, i) in Pkg.dependencies() if i.name in direct)
catch
    Dict{String, Any}()
end

before = snapshot()
t = @elapsed (mode == "instantiate" ? Pkg.instantiate(; io = devnull) : Pkg.update(; io = devnull))
after = snapshot()

moved = sort([(n, get(before, n, nothing), v) for (n, v) in after if get(before, n, nothing) != v])
secs = round(t; digits = 1)
if isempty(before)
    println("Manifest created: $(length(after)) direct dependencies resolved ($secs s)")
elseif isempty(moved)
    println("Manifest legal; no direct dependency moved ($secs s)")
else
    println("$(length(moved)) direct dependenc$(length(moved) == 1 ? "y" : "ies") moved ($secs s):")
    for (n, a, b) in moved
        println("  $n  $(a === nothing ? "new" : a) → $b")
    end
end
