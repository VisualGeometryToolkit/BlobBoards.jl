docs-pass: f4bd64b
landing: blobboards/blobboards.github.io a5776d8 (no hand edit: the site is the Docs workflow's build of docs/src; the 0.6.3 change is in the detection guide, which the next Docs deploy publishes)

# 0.6.3 docs pass (julia-docs-pass) — Julia 1.13

BlobBoards 0.6.3 is a patch (review BB#1): a blob of 10–19 samples fits the isotropic paraboloid instead of being
refused at its seed.
- API diff since 0.6.2: none (no public name added, removed or re-signatured; `_solved_theta` is internal).
- Docstrings: `adapt` now states how the model follows the sample count (6 parameters from 20 samples, the isotropic
  4 from 10 to 19, fewer dropped); `count_rejected`/`BlobDetection` unchanged and still true.
- Guides: docs/src/guide/detection.md "Blob outcomes" states the same; README and landing: nothing about the
  adaptation model (no change).
- CHANGELOG: the 0.6.3 section names the result change; 0.6.2's header dated.
- Docs build and the `docs`/`readme` items: run by the fresh-clone gate on Julia 1.13.3 against the registered stack.
- Re-recorded after the compat floors were raised to Core 0.9.6, IO 0.9.4, Features 0.8.3, Estimation 0.10.2, Eval 0.7.3
  (Project.toml [compat] and the CHANGELOG only; the README/guide install lines name no version: no docs change).
- Re-recorded after DUPLICATION.toml gained six `[[similar]]` entries (the quality allow-list only; no docs change).

---

# 0.6.2 docs pass (julia-docs-pass) — Julia 1.13

BlobBoards 0.6.2 is a patch whose only change is the Julia requirement: `julia = "1.13"` in the `[compat]` of every
Project.toml in the repository, the CI workflows on Julia 1.13 only, and the CHANGELOG section.
- API diff since the last release: none (no source file under src/ or ext/ changed).
- Docstrings: unchanged; the suite's docs item runs in the fresh-clone gate.
- README / landing / guides: every statement of the required Julia version says 1.13 now: README.md, docs/src/index.md, overview.md, guide/getting-started.md, guide/install.md, guide/python.md and bmvc2026/REPRODUCE.md.
  Mentions of Julia 1.12 left in test comments record why a bound or a workaround was set (LLVM 18 vs 20 rounding,
  Julia 1.12's docstring rules, which 1.13 keeps); they are history, not instructions.
- Re-recorded after the Downgrade workflow became on-demand only (CI and CHANGELOG only; no docs change).
- Re-recorded after merging GitHub main (CompatHelper #114: VisualGeometryToolkitIO "0.9" in gpu/'s [compat]; no docs change).
- Re-recorded after the compat floors were raised to the gated versions and the Downgrade workflow was deleted (Project.toml
  [compat] and CI only; the README/guide install lines name no version: no docs change).
- Docs build: run by the fresh-clone gate on Julia 1.13.3 against the registered stack (General from GitHub).

---

docs-pass: 9cd89f9

# 0.6.1 docs pass (julia-docs-pass) — on VisualGeometryToolkitEstimation 0.10

BlobBoards 0.6.1 is a patch: `[compat]` VisualGeometryToolkitEstimation "0.10" in every member that declares it, and the
CHANGELOG section. No public name added, removed or re-signatured.
- API diff since the last release: none.
- stale mentions: nothing here names the SplineLens image-space refine Estimation 0.10.0 retired (its free route,
  `penalty`, the GCV and window metadata): checked by grep over src, docs, README and the paper projects.
- README / install lines: unchanged (no version is named); the README rows resolve the registry and pass once
  Estimation 0.10.0 is registered.

---

Re-pointed to bee3b43: the change since adcba13 is the require_extension hoist (owner ruling, 2026-11-12):
BlobBoards' private `_require_extension` and `_extension_loaded` are deleted, the EXTENSION_FUNCTIONS fallbacks and
the board/board! forwarder call VisualGeometryToolkitCore's public `require_extension` (Core 0.9.3) with BlobBoards'
own message, [compat] Core "0.9.3" in every member that states it, and the CHANGELOG's stack line and entry. Checked:
no public name added or removed; EXTENSION_FUNCTIONS' docstring and the extensions.jl/api.jl comments still say what
the fallbacks do (the same install-line ErrorException, the same MethodError when loaded); no docs page or README
mentions the deleted helpers; the README CPU row's missing-extension assertions are unchanged. Suite on the 0.9.3 work
tree (temporary [sources], branch bb-leg-060-tmp093, Slurm 10457): duplication clean, docs/readme/naming/quality items
pass; the one README row that adds VisualGeometryToolkitCore from the registry waits for Core 0.9.3's registration.

Re-pointed to adcba1347fd6ad360b7e1138efaec4e39e2339a1: the change since 89b260e is VisualGeometryToolkitFeatures 0.8 (csponds imported from
VisualGeometryToolkitCore in src/pipeline.jl, [compat] Features "0.8", the CHANGELOG's stack line); no docstring,
docs page or README moved.

Re-pointed to 89b260e98443c8b833d328a36312a8a30335eb86: the change since 153f622 is [compat] LocalOptimizationDoneRight 0.6 (Project.toml, test/Project.toml);
no docstring, docs page, README or source moved.

Recorded on 153f622cecaf858863f56b83a23e07785aa3b81a: the only change since the docs build on 3bc37ac (below) is the CHANGELOG's known-issue note
(bmvc2026's make_param_variations.jl); no docstring, docs page, README or source moved.

BlobBoards 0.6.0 (the VisualGeometryToolkit 0.9 stack). Release tree: paper-09-bb, which carries every blobboards-09
change. Builds on the 0.9 docs pass (docs-09-bb, 2957f24/c4ca56a/26367aa: Blob outcomes, local_optimization_strategy,
the one kernel set, the five public helpers in the manual).

- Public API since that pass: no export or public name added or removed (src/api.jl and every `public`/`export`
  line unchanged since e38e2ee). Changed: `BlobDetection`'s metadata has no `converged` field (a blob's convergence is
  `successful_retcode(retcode)`), and its converged stop rule is `IncrementSize` (VisualGeometryToolkitEstimation's
  `irls_termination`); types.jl's docstring, docs/src/guide/detection.md's outcome table and the CHANGELOG say so.
  The pose arm's `BoundedCamera` is documented in the guides, the README and the Python docs (6e0daab).
- Docstrings: board_fit_keywords, board_endpoints, detection_rtree, geometry_local_optimization_strategy,
  load_descriptor_network and count_rejected now open with one summary sentence (fixed in 3bc37acc229276deb3b9d3e0d83a91dbd8656d34).
- Gates on 3bc37acc229276deb3b9d3e0d83a91dbd8656d34 (Slurm 10063; the temporary [sources] put back in a scratch worktree, since LocalOptimizationDoneRight
  0.6 is not registered yet): Documenter build with checkdocs = :public, doctest, linkcheck and no warnonly: rc 0,
  0 warnings. The suite's docs, readme, agent docs, naming, duplication and quality-gate items: 84 pass, 2 broken
  (the quality gates' accepted report-mode findings). The README rows run from the registry in the fresh-clone gate
  (leg step 8), once LocalOptimizationDoneRight 0.6 is registered.
