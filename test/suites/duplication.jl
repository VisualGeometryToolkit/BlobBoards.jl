# The stack's no-duplication rule (VisualGeometryToolkitCore's `check_duplication`,
# its Test extension): no name an upstream stack package owns, no function body,
# file or retired pattern restated. Run over the package and its repository (the
# stack repositories named in VGTK_STACK_PATHS too, when set).
#
# Names an upstream package owns for ANOTHER concept are DUPLICATION.toml entries,
# each with its proof (the owner's ruling). OPEN: the findings below wait for a hoist into the stack or an owner's API
# decision, each an open item of .release/duplication-checklist.md. The list only
# shrinks: a finding not on it fails, and so does an entry that no longer finds
# anything (delete it when its item closes). The release leg refuses while the
# list is not empty.
using Test
import VisualGeometryToolkitCore as VgtkCore

const OPEN = Set([
    "name _Record",               # hoist: IO's public Record
    "name _snap_integer",         # hoist: IO's public snap_integer
    "pattern docs/make_hero.jl",  # hoist: Features' linear_to_srgb (plane-maps)
    "file .githooks/sync-manifest.jl",              # hoist: one sync-manifest (cau-tooling)
    "similar test/suites/readme_rows.jl",           # hoist: Core's README row runner
])

_key(v) = v.check === :name ? "name $(v.text)" :
    v.check === :pattern ? "pattern $(v.file)" : "$(v.check) $(v.file)"

found = VgtkCore.check_duplication(BlobBoards)
keys_found = Set(_key(v) for v in found)
foreach(v -> _key(v) in OPEN || println(stderr, v), found)
@test keys_found ⊆ OPEN
@test isempty(setdiff(OPEN, keys_found)) || isempty(get(ENV, "VGTK_STACK_PATHS", ""))   # file/similar need the stack paths
@test_broken isempty(found)
