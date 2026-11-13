docs-pass: 856802a
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
- Re-recorded after DUPLICATION.toml gained six `[[similar]]` entries and the duplication test the owner's 0.6.3 exception (the quality allow-list and its test only; no docs change).

---

# 0.7.0 docs pass (julia-docs-pass) — the wave-10 stack

API diff since 0.6.2, and where the docs say it:
- Terminology (correspondence, observation; `load_correspondence_sets`, `BoardObservation`, Python `n_correspondences`): every
  docstring, guide and README example renamed (the result's fourth element is `scored` in the examples: Core's `Correspondences`);
  docs/src/api/more.md lists `load_correspondence_sets`; CLAUDE.md's nomenclature section is the stack's taxonomy.
- Board v7.0 / pattern v5.0 (IO 0.10's envelope): docs/src/guide/reference.md's board JSON shows the envelope and the units object; the
  migration paragraph says old files load through their steps.
- The posed-camera transfer (`_world_to_image`, Core's `plane_map`): internal; guided_search's docstring states the three transfers.
- adapt's isotropic 10–19-sample model: the `adapt` docstring and docs/src/guide/detection.md ("Blob outcomes", from the 0.6.3 patch).
- find_boards' cap: FindBoardsConfig's `max_detections` docstring says it is detect_blobs' (`spread_strongest`).
- One TIFF codec: docs/src/guide/boards.md names IO's codec; `BlobBoards.write_tiff` is gone from docs/src/api/more.md.
- Python: blobboards.correspondences' module docstring and functions describe IO's export and converter.
- CHANGELOG `## 0.7.0` lists every breaking change.
Docstrings build in the suite's `docs` item (strict) and the gate's docs task.
