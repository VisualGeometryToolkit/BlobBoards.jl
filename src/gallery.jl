# =============================================================================
# BoardGallery — persistent board descriptor store backed by JLD2
# =============================================================================
#
# Thin wrapper around FileCache (JLD2) for managing board descriptors.
#
#   gallery = BoardGallery("boards.jld2", "model.onnx")
#   insert!(gallery, "board.json")          # load + extract + store
#
# The gallery pins the descriptor model hash so that a model change
# invalidates all cached features. Each board's cutoff is read from
# its bundle metadata — the gallery does not impose a global cutoff.
#
# =============================================================================

using JLD2: JLD2  # Triggers VisualGeometryFeaturesJLD2Ext for open_cache
using Colors: RGB, distinguishable_colors
using VisualGeometryFeatures:
    open_cache, close_cache, cache_key    # the get-or-compute is `Base.get!(f, cache, key)` (VGF's JLD2 ext)
using VisualGeometryFeatures:
    described_features

# v4: invalidates every entry extracted while `_load_image` rasterized at
# px/mm instead of dpi (the bare-ustrip density bug, 2026-07-29) — those
# features came from ~47 dpi renders with off-image canvas coordinates.
# v5 (2026-10-10): `rasterize_pdf` moved from pdftoppm, which altered ~36 % of a
# board's pattern pixels (up to 22 levels), to pdftocairo (pixel-exact); and
# `Translation(:pattern, :canvas)` now takes the layout's margins, moving the
# ground-truth blobs 1 px in y on boards with an odd vertical slack (4 of the
# 136 known boards, all in the example-data artifact).
# v6 (2026-10-14): every board PNG is read linearly (`_load_raster`, gamma 1.0)
# — a stored v3 board's was read display-gamma converted (127 → 186) — and a v4
# board's image is the board with its marks (its pattern is the whole board).
const _GALLERY_VERSION = 6

"""
    BoardGallery

Persistent store of board descriptors keyed by board UID.

Wraps a JLD2 file with extraction parameters baked into the cache key,
so changing the model or extraction settings produces fresh entries.
Each board's Gaussian cutoff is read from its bundle metadata.

# Fields
- `path::String` — JLD2 file path
- `network_hash::String` — SHA-256 of the ONNX descriptor model
- `patch_size::Int` — canonical patch size
- `outer_extent::Float64` — outer sampling radius

# Constructor
```julia
gallery = BoardGallery("boards.jld2", "model.onnx")
gallery = BoardGallery("boards.jld2", model::AbstractDescriptorNetwork)
```
"""
struct BoardGallery
    path::String
    network_hash::String
    patch_size::Int
    outer_extent::Float64
end

# `patch_size` comes from the MODEL, not the caller: it is fixed by the
# network (the ONNX graph declares it; a Lux checkpoint carries it in `cfg`),
# and it is part of the cache key below. A caller-supplied value could only
# ever agree — or silently key the cache at a resolution the network will then
# refuse, after re-extracting the whole board set.
# `outer_extent` is NOT a keyword either (2026-09-07): the query side samples at
# `DEFAULT_OUTER_EXTENT` and nothing per-call could follow a different gallery
# value, so a caller-supplied one could only make gallery and query
# descriptors silently incomparable. The field stays in the cache key.
function BoardGallery(path::AbstractString, model::AbstractDescriptorNetwork)
    return BoardGallery(
        String(path), network_hash(model),
        patch_size(model), Float64(DEFAULT_OUTER_EXTENT)
    )
end

BoardGallery(path::AbstractString, onnx_path::AbstractString) =
    BoardGallery(path, DescriptorNetwork(onnx_path))

# Cache key: the board's full identity (`config_hash`; its few-character uid
# collides across board sets sharing one gallery file), its own cutoff and the
# density its descriptors come from — a v3 board's identity already names its
# density; a v4 board's does not (one layout at any density), so the density it
# is rasterized at joins the key.
_gallery_key(g::BoardGallery, board) = _gallery_key(g, board, board.bundle.preamble.board_config.layout_grid)
_gallery_key(g::BoardGallery, board, ::PixelGrid) = cache_key(
    _GALLERY_VERSION, board.config_hash, g.network_hash, Float64(cutoff(board.bundle)), g.patch_size, g.outer_extent
)
_gallery_key(g::BoardGallery, board, ::PointGrid) = cache_key(
    _GALLERY_VERSION, board.config_hash, g.network_hash, Float64(cutoff(board.bundle)), g.patch_size, g.outer_extent,
    ustrip(dpi, board.density)
)

# =============================================================================
# insert! — load board, extract features, store
# =============================================================================

"""
    insert!(gallery::BoardGallery, board_path, model::AbstractDescriptorNetwork;
            device = default_device()) -> (uid, features)
    insert!(gallery::BoardGallery, board_paths::AbstractVector, model;
            device = default_device(), progress = true) -> Vector{(uid, features)}

Load a board from `board_path`, extract descriptors using `model`, and store
in the gallery. The board's own cutoff (from bundle metadata) is used for
canonicalization and baked into the cache key. Supply a GPU `device` (e.g.
`CUDABackend()`) to run canonicalize + describe on the GPU (same
`_canon_describe` stage as the query path).

The vector form builds the whole gallery with a single-line progress bar on
stderr (per-board uid, count, elapsed/ETA — cache hits fly through);
`progress = false` silences it for scripted runs.

Returns `(uid, features)` where features is a `StructArray{DescribedFeature}`
pairing world-coordinate blobs with descriptors.

No-op if the board is already present (returns existing features).
"""
function Base.insert!(
        gallery::BoardGallery, board_path::AbstractString,
        model::AbstractDescriptorNetwork;
        device::KA.Backend = default_device()
    )
    return _insert_board!(gallery, load_board(board_path), model; device)
end

# The one insertion path, on an already-loaded board (`load_board`): the batch
# method below loads each board once and reads its cutoff from the same
# object, instead of re-parsing every bundle a second time for the warning.
function _insert_board!(
        gallery::BoardGallery, board, model::AbstractDescriptorNetwork;
        device::KA.Backend, image = nothing
    )
    cache = open_cache(gallery.path)
    isnothing(cache) && error("Cannot open gallery at $(gallery.path)")
    try
        key = _gallery_key(gallery, board)
        features = get!(cache, key) do
            img = image === nothing ? _board_raster(board.image_path, board.bundle, board.config_hash; density = board.density) : image
            extract_board_features(
                model, board;
                patch_size = gallery.patch_size,
                outer_extent = gallery.outer_extent,
                device, image = img
            )
        end
        return (board.uid, features)
    finally
        close_cache(cache)
    end
end

# The keys of `boards` the gallery file does not hold yet. Reads the JLD2 cache's
# key set (VGF's `FileCache.db`, a `JLDFile`) — the same test its `get!` makes.
function _missing_gallery_keys(gallery::BoardGallery, boards)
    keys_wanted = [_gallery_key(gallery, b) for b in boards]
    cache = open_cache(gallery.path)
    isnothing(cache) && return Set(keys_wanted)
    try
        return Set(k for k in keys_wanted if !haskey(cache.db, k))
    finally
        close_cache(cache)
    end
end

# Single-line stderr progress: `gallery [#####....] 24/50 6b0  61s, ~78s left`.
# In-place (\r) so a TTY shows one live line; each update overwrites the last.
function _gallery_progress(i::Int, n::Int, t0::Float64, uid::AbstractString)
    w = 24
    filled = round(Int, w * i / n)
    elapsed = time() - t0
    eta = i == n ? 0.0 : elapsed / i * (n - i)
    bar = string("#"^filled, "."^(w - filled))
    tail = i == n ? "" : ", ~$(round(Int, eta))s left"
    msg = "\rgallery [$bar] $i/$n $uid  $(round(Int, elapsed))s elapsed$tail"
    print(stderr, rpad(msg, 80))
    i == n && print(stderr, "\n")
    return nothing
end

function Base.insert!(
        gallery::BoardGallery, board_paths::AbstractVector{<:AbstractString},
        model::AbstractDescriptorNetwork;
        device::KA.Backend = default_device(), progress::Bool = true
    )
    n = length(board_paths)
    t0 = time()
    # Load every board and, for those the gallery does not hold yet, rasterize
    # them — on all threads (pdftocairo subprocesses, PNG decodes; `_board_raster`
    # caches). Describing stays serial on the one device. `fetch` in path
    # order keeps the output order; each result depends on its own board only.
    boards = map(fetch, [Threads.@spawn(load_board(bp)) for bp in board_paths])
    missing_keys = _missing_gallery_keys(gallery, boards)
    images = Union{Nothing, AbstractMatrix}[
        fetch(t) for t in [
            Threads.@spawn(
                _gallery_key(gallery, b) in missing_keys ?
                    _board_raster(b.image_path, b.bundle, b.config_hash; density = b.density) : nothing
            ) for b in boards
        ]
    ]
    cutoffs = Float64[]
    out = map(enumerate(boards)) do (i, board)
        push!(cutoffs, Float64(cutoff(board.bundle)))
        r = _insert_board!(gallery, board, model; device, image = images[i])
        images[i] = nothing   # a 50 MB raster per board: release it once described
        progress && _gallery_progress(i, n, t0, r[1])
        r
    end
    # A gallery is matched against ONE image-side descriptor extent
    # (`_shared_image_cutoff`): a uniform board cutoff makes that extent
    # exact for every board, mixed cutoffs leave a bounded annulus mismatch
    # (find_boards falls back to the mean). Surface it at build time, where
    # regenerating the boards is still cheap.
    #
    # SCOPE: this check sees only THIS call's batch. Boards inserted one at
    # a time (the scalar `insert!` above) or across separate batch calls can
    # still assemble a mixed-cutoff gallery unwarned here — the runtime
    # backstop is `_shared_image_cutoff`, which re-derives the cutoff set
    # from the boards actually queried and warns on the mean fallback.
    cs = unique(cutoffs)
    length(cs) > 1 && @warn "gallery boards carry $(length(cs)) distinct rendering " *
        "cutoffs (span $(minimum(cs))..$(maximum(cs))); find_boards will describe " *
        "the image at their mean — prefer a uniform cutoff across the board set"
    return out
end

"""
    board_palette(uids) -> Dict{String, Color}

ONE color per board id, deterministic over the SORTED unique id list — the
same board gets the same color in every image, every sweep, every call site
(sweep overlays, `plot_board_poses`, training overlays; this replaced three
palettes with three different stability conventions). Seeded against white
AND black so nothing produced is invisible on a light or a dark image.
"""
function board_palette(uids)
    ids = sort!(unique!([String(u) for u in uids]))
    palette = distinguishable_colors(length(ids), [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed = true)
    return Dict(id => c for (id, c) in zip(ids, palette))
end
