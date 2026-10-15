# Board API

## Main Functions

See [Public API](public.md) for `blob_board`.

## Coordinate Functions

Coordinate transforms are extensions of the `CoordinateTransformations` constructors:

```julia
# Reference-frame translations (within the board layout)
T = Translation(board_config, :pattern, :canvas)   # BoardConfig: :pattern/:board/:canvas

# Coordinate-system transforms (pixel ↔ physical ↔ world)
H = AffineMap(bundle, :pixel, :world; density = 300dpi)   # :pixel/:physical/:world

# Image-origin conventions (from VisualGeometryCore)
T_origin = Translation(:colmap, :julia)            # :colmap/:julia/:opencv

# Compose freely
T_full = Translation(:colmap, :julia) ∘ Translation(board_config(bundle), :pattern, :canvas)
```

See also [Public API](public.md) for `blobs` (including `blobs(bundle, :world)`).

## Internal Functions

### QR Code Generation

- `make_qr_code(preamble, config_hash)` - the QR code matrix a board carries: its board code, or its stored identity
