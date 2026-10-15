# papers/images — the paper's `\includegraphics` assets

Every file here that a document includes is indexed in `bmvc2026/REPRODUCE.md`
(label → asset → kind → generator → data): generated assets name the script
that writes them, static ones carry their provenance there. `make check`
fails on an included file the index does not cover. Regenerate with
`make figures` / `make figures-gpu` (scripts/regenerate_paper.jl); to review a
regeneration before it replaces a submitted asset, point `BB_PAPER_IMAGES` at
a scratch directory first.
