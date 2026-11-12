docs-pass: 89b260e98443c8b833d328a36312a8a30335eb86

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
