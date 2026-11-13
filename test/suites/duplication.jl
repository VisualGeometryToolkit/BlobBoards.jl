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
    # Owner-granted exception for 0.6.3 only (2026-11-13): pre-existing in 0.6.2, found once Core 0.9.5's idiom cap counts
    # this repository's definitions only; fixed in the wave (BlobBoards/CaliBlobs 0.7, Core 0.10). Cross-repository pairs
    # with CaliBlobs, so they are found only with the stack paths set.
    "similar src/config_toml.jl",    # _geometry_arm, _track_from_toml ⇔ CaliBlobs' config.jl restatements
    "similar test/suites/readme.jl", # page_julia, install_lines ⇔ CaliBlobs' copy of the README-test template
])

_key(v) = v.check === :name ? "name $(v.text)" :
    v.check === :pattern ? "pattern $(v.file)" : "$(v.check) $(v.file)"

found = VgtkCore.check_duplication(BlobBoards)
keys_found = Set(_key(v) for v in found)
foreach(v -> _key(v) in OPEN || println(stderr, v), found)
@test keys_found ⊆ OPEN
@test isempty(setdiff(OPEN, keys_found)) || isempty(get(ENV, "VGTK_STACK_PATHS", ""))   # file/similar need the stack paths
isempty(OPEN) && @test isempty(found)    # with open items, the two tests above hold the list exactly
