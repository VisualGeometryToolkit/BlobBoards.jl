# papers/images — the paper's `\includegraphics` assets

Every file here that a document includes is indexed in `bmvc2026/REPRODUCE.md`
(label → asset → kind → generator → data): generated assets name the script
that writes them, static ones carry their provenance there. `make check`
fails on an included file the index does not cover. Regenerate with
`make figures` (the stage graph of record, configs/reproduce.toml); a
regeneration is reviewed as the working tree's diff against the submitted
assets before it is committed.
