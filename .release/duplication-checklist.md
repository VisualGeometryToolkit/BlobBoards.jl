# BlobBoards 0.7.0: duplicate removals for the wave-10 leg

The wave's rule (BRIEF "ZERO duplication, checked by CONCEPT"): every concept has one implementation in the stack; a
BlobBoards copy of an upstream concept is deleted and the owner's called. An item is ticked with the commit that
closes it; an open item says what it waits for. The release leg refuses while any `- [ ]` line is left.

## Hoisted into, or replaced by, their stack owner

- [x] the posed camera's plane map: BlobBoards' pinhole/lens `_model_to_homography` methods → Core's `plane_map` /
  `transfer_ellipse(::PosedCamera, _)` through ONE `_world_to_image` method for a pose (e565238)
- [x] `_project_blob_centers`' per-transfer copies → one method over `_plane_point_map` (Core `plane_map`) (e565238)
- [x] `_guided_transfer(::HomographyMap)` / `(::PosedCamera)` identical bodies → one method (e565238);
  `_guided_detell` likewise (b186ad2)
- [x] find_boards' top-N detection cap (a copy of Features' retired `strongest`) → detect_blobs' `_cap_detections`,
  Features' `spread_strongest` (982c2b3; catch-up review BB#2)
- [x] the isotropic model's hand-written θ → Core's `Paraboloid3D{T,4}` via `coeffs` (`_solved_theta`, b3b0b0c)
- [x] `_overlay_shape(::Disc)` → Core's hoisted `shape_factor` (8218071)
- [x] BlobBoards' libtiff TIFF writer (src/authoring/tiff.jl) and reader (`_read_grey_tiff`) → VisualGeometryToolkitIO's
  `write_tiff` / `read_tiff`, the stack's one TIFF codec (5d385ef; owner ruling, io095's port map)
- [x] the board file's own `units` block (IO's retired UnitContract layout via `json_value`/`from_json`) → IO 0.10's
  envelope (`write_record`'s units object; `VgioStyle(record)`; old files via `migrate_units`) (1b248ad)
- [x] python/blobboards/correspondences.py's copies of IO's formats (the rig and observations readers, the cameras and
  transforms writers, `opencv_lens_record`) → IO's `save_correspondences` export and `convert_opencv_calibration` (4ba8237)

## Open, each waiting for its owner

- [ ] ONNX Runtime runs outside Features' arena accounting (`describe`'s `model.session(…)`, the CUDA extension's
  `CAPI.Run`; `reserve_workspace!`): waits for Features' public `onnx_run` (handoffs.md "bb → feat09", review BB#3)
- [x] `_iou_below` (src/pipeline.jl), a generic IoU bound → Core's `iou_below` beside `iou` (core10 0bcd5559; BB calls it, 06c3d5c;
  verified once the shared snapshot carries core10 ≥ 0bcd5559)
- [x] `test/suites/readme.jl` `page_julia` / `install_lines` (⇔ CaliBlobs') → Core's `page_examples` / `install_lines` (core10 0bcd5559; 06c3d5c)
- [x] `_track_from_toml` ⇔ CaliBlobs' `_calibrate_from_toml` → Core's `read_table` (2766502); `_file_length_unit` (an IO name) → `_exact_length_unit` (2766502)

## check_duplication

Pasted from the gate's prep (`check_duplication(BlobBoards)`, stack VGTK_STACK_PATHS = w10-lodr, w10-rsdr, w10-cb, w10-ps)
at the release commit.
