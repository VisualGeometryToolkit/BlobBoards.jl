# =============================================================================
# Default compute device — provided by VisualGeometryCore
# =============================================================================
#
# `default_device()` (an accelerator when one is loaded AND functional, else
# `KA.CPU()`) and its extension-registered resolver were hoisted to
# VisualGeometryCore on 2026-09-07 (`src/base/device.jl` there;
# `VisualGeometryCoreCUDAExt` registers the CUDA thunk). BlobBoards re-exports
# the name so every entry point's `device = default_device()` default is
# unchanged, and so is the file header's argument for it: the host path is
# the one that can take the machine down.

using VisualGeometryCore: default_device
