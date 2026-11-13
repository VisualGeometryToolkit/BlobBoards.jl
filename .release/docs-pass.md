docs-pass: a0077e0

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
