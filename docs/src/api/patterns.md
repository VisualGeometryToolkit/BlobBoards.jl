# Pattern API

## Main Functions

See [Public API](public.md) for `blob_pattern`.

## Internal Functions

### Blob Placement

Blobs are placed largest scale first, each at a random centre that does not
overlap a placed blob (per-level uniform grids hold the placed discs), so placement is
reproducible from the seed alone.

### Scale Calculation

The following internal functions handle scale distribution:

- `calc_blob_scales(min_scale, max_scale; scales_per_octave = 3)` - The scale ladder from the smallest to the largest σ (px), both exact, evenly spaced in log
- `calc_scale_counts(pattern_size, scales; ...)` - Calculate blob counts per scale

### Rendering

- `render_blob!(pattern, blob; cutoff)` - Render a single Gaussian blob to the pattern image
