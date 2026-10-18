# Board API

## Main Functions

See [Public API](public.md) for `blob_board`.

## Coordinate Functions

Coordinate transforms are extensions of the `CoordinateTransformations` constructors:

```julia
# Reference-frame translations (within the board layout)
T = BlobBoards.Translation(board_config, :pattern, :canvas)   # BoardConfig: :pattern/:board/:canvas

# Coordinate-system transforms (pixel ↔ physical ↔ world)
H = BlobBoards.AffineMap(bundle, :pixel, :world; density = 300dpi)   # :pixel/:physical/:world

# Image-origin conventions (from VisualGeometryToolkitCore), on plain pixel coordinates
to_julia = BlobBoards.EuclideanMap(BlobBoards.PixelFrame{:colmap}, BlobBoards.PixelFrame{:julia})   # :colmap/:julia/:opencv
```

See also [Public API](public.md) for `blobs` (including `blobs(bundle, :world)`).

## Internal Functions

### QR Code Generation

- `make_qr_code(text)` - the QR code matrix a board carries, for `text`: its board code, or its stored identity
