# Manuscript

The camera-ready sources of the BlobBoards BMVC 2026 paper and its
supplementary material, plus two technical notes.

## Build

```bash
cd bmvc2026/papers
make            # the supplement, then the paper
make paper      # blob_boards_bmvc.pdf (builds the supplement first for its labels)
make supp       # bmvc_supp.pdf
make watch      # continuous rebuild of the paper on save
make notes      # notes/dog_scale_bias.pdf
make clean      # aux files only, keeps PDFs
```

The paper cross-references the supplement through `xr`, reading
`supp_labels.aux` (the supplement's `\newlabel` lines, extracted by `make`
after every supplement build), so build the supplement first when calling
`latexmk` by hand.

## What is here

| | |
|---|---|
| `blob_boards_bmvc.tex` | **the paper** (camera-ready) |
| `bmvc_supp.tex` | the supplementary material (S-numbered) |
| `bmvc2k.cls`, `bmvc2k*.sty`, `bmvcabbrv.sty` | conference class + style files |
| `egbib.bib` | bibliography |
| `figures/floorplan*/` | TikZ panels, assembly tables and macro files, GENERATED — see below |
| `images/` | figure assets the `.tex` files reference (generated or declared static) |
| `notes/dog_scale_bias.tex` | technical note: why the detected DoG σ underestimates the true blob scale |
| `notes/whitened_linear_derivation.tex` | draft appendix fragment (not standalone): the whitened log-space linear solve equals intensity-space Gauss–Newton |

The notes are not part of the camera-ready set: the reproducible-figures
contract below covers only the paper and the supplement, and the numbers in
the notes are typed by hand.

## What is tracked

Sources and the referenced images are tracked, so the paper builds from a fresh
clone. Everything latexmk regenerates — `.aux .bbl .blg .fdb_latexmk .fls .log
.out .toc` and the PDFs — is gitignored; the built paper is ~50 MB and fully
reproducible.

## Regenerating figures, tables and numbers

Every asset the two documents read is indexed in `bmvc2026/REPRODUCE.md`
(label → asset → kind → generator → data) and the index is enforced by
`make check`. One entry point regenerates all of them from the published
inputs alone (every result on the way is a keyed run under `results_root()`):

```bash
make figures       # scripts/regenerate_paper.jl: tables, macros, floor plans, overlays, picks
make figures-gpu   # ... plus the CUDA figures (teaser, adapted and pipeline panels)
make check         # the reproducible-figures contract
```

Drivers write intermediate `.tex` to DrWatson's `plotsdir()` (`bmvc2026/plots`)
and the regeneration copies them into `figures/`; image generators write into
`images/` directly. Nothing is written into the dataset: `data/exp_raw` is
immutable input.
