# Extensions, Tooling and Internals

```@meta
CurrentModule = BlobBoards
```

The rest of the public surface: the functions the package extensions provide
(plots, the Python interface, the Lux backend, training), the pipeline's
configuration and stages, traces, and the training-data tooling. Each name is
reached as `BlobBoards.name` unless it is exported.

```@docs
BlobBoards
```

## Plots (BlobBoardsMakieExt)

Loaded by any Makie backend: `pkg> add BlobBoards CairoMakie`, then
`using BlobBoards, CairoMakie`. Without one, each raises an error naming that
line.

```@docs
board
board!
BlobBoards.plot_boards
BlobBoards.plot_board_poses
BlobBoards.board_overlay!
BlobBoards.inliers!
BlobBoards.board_panel!
BlobBoards.plot_comparison
BlobBoards.replay
BlobBoards.inspect_pipeline
BlobBoards.plot_annotations
BlobBoards.view_angle_figure
BlobBoards.save_view_angle_figure
BlobBoards.board_palette
BlobBoards.board_quad
```

## Python interface (BlobBoardsPythonExt)

The Julia side of the `blobboards` Python package, loaded with PythonCall.

```@docs
blob_pattern_python
blob_board_python
blobs_python
detect_blobs_python
find_boards_python
BlobBoards.load_board_python
BlobBoards.render_board_python
BlobBoards.track_boards_python
```

## Descriptor networks

```@docs
AbstractDescriptorNetwork
LuxDescriptorNetwork
BlobBoards.network_hash
BlobBoards.patch_size
BlobBoards.reserve_workspace!
descriptor_network_path
DEFAULT_CUTOFF
BlobBoards.DEFAULT_OUTER_EXTENT
```

## Pipeline configuration and stages

```@docs
GeometryConfig
BlobBoards.geometry_from_toml
BlobBoards.geometry_tag
BlobBoards.CONTRACT_GENERICS
BlobBoards.TrackConfig
config_dict
write_config
read_run_config
default_config_path
data_dir
BlobDetection
BlobBoards.BoardObservation
BlobBoards.filter_blobs
BlobBoards.find_board
BlobBoards.board_fit_keywords
BlobBoards.board_endpoints
BlobBoards.distortion_center
BlobBoards.geometry_local_optimization_strategy
BlobBoards.load_descriptor_network
BlobBoards.match_boards
BlobBoards.guided_search
BlobBoards.detection_rtree
BlobBoards.fit_rng
extract_board_features
extract_image_features
BlobBoards.Telemetry
BlobBoards.with_tag
```

## Boards, bundles and the design resolver

```@docs
pattern_config
board_config
cutoff
uid_from_hash
board_library
board_points
board_resolver
board_target
BlobBoards.hruler
BlobBoards.vruler
BlobBoards.rasterize_pdf
BlobBoards.write_tiff
```

## Traces

```@docs
TrackTrace
BlobBoards.TrackFrame
save_trace
load_trace
BlobBoards.trace_pipeline
BlobBoards.PipelineTrace
BlobBoards.PipelineCapture
```

## Training data and tracks

```@docs
BlobBoards.train_descriptor
BlobBoards.annotate_frame!
BlobBoards.append_log
BlobBoards.check_annotation_hashes
BlobBoards.existing_ref_uids
BlobBoards.is_source_processed
BlobBoards.image_path
BlobBoards.config_hash
BlobBoards.build_annotations
BlobBoards.load_annotations
BlobBoards.load_correspondence_sets
BlobBoards.split_training_data
BlobBoards.summarize_annotations
BlobBoards.collect_media
BlobBoards.load_board_gallery
BlobBoards.merge_track_files
BlobBoards.save_board_anchors
BlobBoards.symmetric_iou_match
BlobBoards.TrainingPair
BlobBoards.view_angle_profile
BlobBoards.print_view_angle_profile
BlobBoards.ViewAngleStat
```
