using Documenter
using DocumenterLandingPage   # LandingPage(): the VitePress-style home page (index.md frontmatter)
using DocumenterCodeBlocks    # CodeBlocks(): line numbers, docstring links and tooltips in code blocks
using BlobBoards

# The QR encoder vendored from QRCoders.jl (src/authoring/qrencode/, kept
# verbatim) carries its upstream doctests, written against its own unqualified
# names: run them with the module's names in scope.
DocMeta.setdocmeta!(BlobBoards.Authoring.QREncode, :DocTestSetup,
    :(using BlobBoards.Authoring.QREncode; using BlobBoards.Authoring.QREncode: getmode, getversion); recursive = true)

# The guide's figures are drawn from the package itself (docs/diagrams.jl), so
# they follow the code: regenerated on every build, never committed.
include(joinpath(@__DIR__, "diagrams.jl"))
write_diagrams(joinpath(@__DIR__, "src", "assets", "generated"))

makedocs(;
    modules=[BlobBoards],
    authors="James Pritts <jbpritts@gmail.com>",
    sitename="BlobBoards.jl",
    format=Documenter.HTML(;
        assets=["assets/landing_tagline.css", "assets/landing_tagline.js"],
        prettyurls=get(ENV, "CI", "false") == "true" || get(ENV, "BB_DOCS_DEPLOY", "0") == "1",
        canonical="https://blobboards.github.io",
        edit_link="main",
        # The two API pages measured 137 KiB (api/public.md) and 125 KiB
        # (api/more.md) on 2026-11-04, over the 100 KiB default warning and
        # under the 200 KiB hard limit; split a page before it nears that.
        size_threshold_warn=150 * 2^10,
    ),
    plugins=[LandingPage(), CodeBlocks()],
    # Every public docstring is in the manual, the doctests run, the links
    # resolve, and no warning is tolerated (the stack's docs gates). A public name with
    # no docstring at all is the docs test item's to find (test/suites/docs.jl).
    checkdocs=:public,
    # The one exception: the QR encoder vendored verbatim from QRCoders.jl 1.4.5
    # (src/authoring/qrencode/, VENDORED.toml). Its docstrings are upstream's and
    # documented there; its doctests still run (the DocTestSetup above). It is
    # named through its parent, BlobBoards.Authoring, because Documenter 1.19
    # applies the ignore list to the top module's direct submodules only;
    # Authoring has no public name or docstring of its own, so nothing else is skipped.
    checkdocs_ignored_modules=[BlobBoards.Authoring],
    doctest=true,
    linkcheck=true,
    pages=[
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Quick Reference" => "guide/reference.md",
        "Install for your task" => "guide/install.md",
        "Getting Started" => "guide/getting-started.md",
        "User Guide" => [
            "Patterns" => "guide/patterns.md",
            "Boards" => "guide/boards.md",
            "Detection and Pose" => "guide/detection.md",
            "Coordinate Systems" => "guide/coordinates.md",
            "Board Training Tracks" => "guide/board-tracks.md",
            "Python Interface" => "guide/python.md",
        ],
        "API Reference" => [
            "Public API" => "api/public.md",
            "Types" => "api/types.md",
            "Patterns" => "api/patterns.md",
            "Boards" => "api/boards.md",
            "I/O" => "api/io.md",
            "Extensions, Tooling and Internals" => "api/more.md",
        ],
    ],
)

# The site is the blobboards organization page (https://blobboards.github.io):
# unversioned, at the root of that repository's `main` branch. CI deploys it
# (.github/workflows/docs.yml, with a deploy key for that repository).
# Documenter deploys only when the building repository's name occurs in `repo`;
# the site is another repository, so the deploy config names it as the target
# (authentication is the DOCUMENTER_KEY deploy key either way).
const SITE_REPO = "blobboards/blobboards.github.io"
deploydocs(;
    repo="github.com/$(SITE_REPO).git",
    branch="main",
    devbranch="main",
    versions=nothing,
    deploy_config=Documenter.GitHubActions(SITE_REPO, get(ENV, "GITHUB_EVENT_NAME", ""), get(ENV, "GITHUB_REF", "")),
)
