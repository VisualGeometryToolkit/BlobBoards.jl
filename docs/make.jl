using Documenter
using DocumenterLandingPage   # LandingPage(): the VitePress-style home page (index.md frontmatter)
using DocumenterCodeBlocks    # CodeBlocks(): line numbers, docstring links and tooltips in code blocks
using BlobBoards

# The QR encoder vendored from QRCoders.jl (src/authoring/qrencode/, kept
# verbatim) carries its upstream doctests, written against its own unqualified
# names: run them with the module's names in scope.
DocMeta.setdocmeta!(BlobBoards.Authoring.QREncode, :DocTestSetup,
    :(using BlobBoards.Authoring.QREncode; using BlobBoards.Authoring.QREncode: getmode, getversion); recursive = true)

makedocs(;
    modules=[BlobBoards],
    authors="James Pritts <jbpritts@gmail.com>",
    sitename="BlobBoards.jl",
    format=Documenter.HTML(;
        assets=["assets/landing_tagline.css", "assets/landing_tagline.js"],
        prettyurls=get(ENV, "CI", "false") == "true" || get(ENV, "BB_DOCS_DEPLOY", "0") == "1",
        canonical="https://blobboards.github.io",
        edit_link="main",
    ),
    plugins=[LandingPage(), CodeBlocks()],
    warnonly=[:missing_docs, :cross_references, :docs_block],
    pages=[
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Quick Reference" => "guide/reference.md",
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
