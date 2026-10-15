# Makefile for the BlobBoards BMVC paper (camera-ready)
# Usage:
#   make            # build the supplement, then the paper (default)
#   make paper      # build the main paper (builds the supplement first)
#   make supp       # build the supplement only
#   make watch      # continuous rebuild of the paper on save
#   make view       # build then open the paper
#   make figures    # regenerate every generated asset (scripts/regenerate_paper.jl)
#   make figures-gpu  # ... plus the CUDA figures (teaser, adapted and pipeline panels)
#   make check      # the reproducible-figures contract (REPRODUCE.md) + config tests
#   make clean      # remove LaTeX aux files (keeps PDFs)
#   make cleanall   # remove aux files and PDFs

# The two documents. The supplement builds first: the paper reads its labels.
DOCS := bmvc_supp blob_boards_bmvc

LATEXMK := latexmk
LFLAGS  := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

# Open the PDF with the system default viewer
ifeq ($(shell uname),Darwin)
  OPEN := open
else
  OPEN := xdg-open
endif

.PHONY: all $(DOCS) paper supp watch view clean cleanall \
        figures figures-gpu check figcheck

# Each document target is phony and always calls latexmk, which is
# dependency-aware and idempotent: it rebuilds only what changed.
all: $(DOCS)

$(DOCS):
	$(LATEXMK) $(LFLAGS) $@

# The main paper cross-references the supplement through xr, reading
# supp_labels.aux: the \newlabel lines of bmvc_supp.aux, extracted after every
# supplement build.
blob_boards_bmvc: supp_labels.aux
supp_labels.aux: bmvc_supp
	grep '^\\newlabel' bmvc_supp.aux > $@
.PHONY: supp_labels.aux

# Friendly aliases
paper: blob_boards_bmvc
supp:  bmvc_supp

watch:
	$(LATEXMK) $(LFLAGS) -pvc blob_boards_bmvc

view: blob_boards_bmvc
	$(OPEN) blob_boards_bmvc.pdf

# ------------------------------------------------------------------------------
# Figure regeneration — ONE entry point. `make figures` runs
# scripts/regenerate_paper.jl, which regenerates every result FROM THE INPUTS
# (the detections, the motion-capture records, the hand-eye fits, the cells:
# keyed runs under results_root(), src/runs.jl) and every generated asset the
# paper reads (REPRODUCE.md indexes each: label -> asset -> generator -> data);
# `make figures-gpu` adds the CUDA figures. Nothing is selected by hand: the
# configs of record name the runs.
BB_REPO  := ../..
BB_JL    := julia --startup-file=no --project=bmvc2026

# Staleness guard for the doc targets: warn when result-affecting code or a
# config of record changed in a commit after the last one that regenerated the
# figures (bb_stats.tex is the representative marker — it is regenerated in the
# same `make figures` batch as every other panel). WARN, never fail.
figcheck:
	@marker=$$(git -C $(BB_REPO) log -1 --format=%H -- bmvc2026/papers/figures/floorplan/bb_stats.tex 2>/dev/null); \
	if [ -n "$$marker" ] && ! git -C $(BB_REPO) diff --quiet "$$marker"..HEAD -- src ext configs bmvc2026/configs bmvc2026/src 2>/dev/null; then \
	  echo "WARNING: result-affecting code or a config of record (src/ ext/ configs/"; \
	  echo "         bmvc2026/configs/ bmvc2026/src/) changed since the figures were last"; \
	  echo "         regenerated ($$marker) — run 'make figures' before trusting the compiled numbers"; \
	fi

$(DOCS): figcheck

figures:
	cd $(BB_REPO) && $(BB_JL) bmvc2026/scripts/regenerate_paper.jl

figures-gpu:
	cd $(BB_REPO) && $(BB_JL) bmvc2026/scripts/regenerate_paper.jl --gpu

# The reproducible-figures contract: every asset the two documents read is
# indexed in ../REPRODUCE.md with an existing generator that writes it (or is
# declared static with provenance), and no generated number is re-typed.
check:
	cd $(BB_REPO) && $(BB_JL) bmvc2026/test/runtests.jl

clean:
	$(LATEXMK) -c $(DOCS)
	rm -f supp_labels.aux

cleanall: clean
	$(LATEXMK) -C $(DOCS)
