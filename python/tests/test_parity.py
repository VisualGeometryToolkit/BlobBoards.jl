"""Each user-facing Python function against the Julia call it wraps, on the same inputs."""
import os

import numpy as np
import pytest

import blobboards
from blobboards import (PRINT_SIZES, blob_board, blob_pattern, blobs, descriptor_network_path,
                        example_data, find_boards, load_board, render_board,
                        suggest_blob_sizes, short_side, track_boards)


def _jl_blobs_px(jl, expr, density):
    """Julia blobs (pattern frame, px at `density` — a Julia expression of a dpi
    quantity —, COLMAP) as an (N, 3) array."""
    return np.asarray(jl.seval(f"""let b = {expr}
        d = BlobBoards.blobs(b, :pattern; density = {density})
        V = BlobBoards.VisualGeometryCore
        [Float64(V.ustrip(V.px, x)) for x in vcat(([q.center[1] q.center[2] q.σ] for q in d)...)]
    end"""))


def test_blob_board_matches_julia(jl):
    b = "BlobBoards.blob_board(\"A5\"; seed = 0x51, blob_sizes = (4.0BlobBoards.mm, 0.2BlobBoards.short_side))"
    jl.seval("using CairoMakie")
    # pixel blobs at a density the caller names (a PDF's file has none)
    py = blob_board("A5", blob_sizes=(4.0, 0.2 * short_side), seed=0x51, density=100)
    assert py.config_hash == str(jl.seval(f"BlobBoards.hash_config(({b}).preamble)"))
    np.testing.assert_array_equal(py.blobs, _jl_blobs_px(jl, b, "100BlobBoards.dpi"))
    bc = py.preamble.board_config
    assert py.density_dpi == 100.0
    assert (bc.major_tick_width_mm, bc.minor_tick_width_mm) == (
        bc.ruler_width_mm - 2 * bc.frame_padding_mm, (bc.ruler_width_mm - 2 * bc.frame_padding_mm) / 2)
    # ... and, with none named, at the board's own raster density (`b` is the board inside the helper's `let`)
    own = blob_board("A5", blob_sizes=(4.0, 0.2 * short_side), seed=0x51)
    assert own.config_hash == py.config_hash
    np.testing.assert_array_equal(own.blobs, _jl_blobs_px(jl, b, "BlobBoards._default_density(b)"))
    assert own.density_dpi == float(jl.seval(
        f"BlobBoards.VisualGeometryCore.ustrip(BlobBoards.dpi, BlobBoards._default_density({b}))"))


def test_suggest_blob_sizes_matches_julia(jl):
    for dpi, extent, jl_extent in ((300, "A4", '"A4"'), (600, "A0", '"A0"'),
                                   (1200, (300, 200), "(300BlobBoards.mm, 200BlobBoards.mm)")):
        jl_sizes = jl.seval(f"Tuple(Float64(BlobBoards.VisualGeometryCore.ustrip(BlobBoards.mm, s)) for s in "
                            f"BlobBoards.suggest_blob_sizes({dpi}BlobBoards.dpi, {jl_extent}))")
        assert suggest_blob_sizes(dpi, extent) == (float(jl_sizes[0]), float(jl_sizes[1]))
    assert suggest_blob_sizes(600, "A4") == pytest.approx((6.0, 38.0))     # 20% of the 190 mm board side
    assert suggest_blob_sizes(1200, (841, 1189)) == pytest.approx((3.0, 64.0))   # ... at most 64 mm
    pint = pytest.importorskip("pint")
    cm = pint.UnitRegistry().cm
    assert suggest_blob_sizes(600, (19 * cm, 27.7 * cm)) == pytest.approx((6.0, 38.0))
    with pytest.raises(Exception, match="unknown paper size"):
        suggest_blob_sizes(600, "A9")


def test_blob_board_none_max_raises():
    # a board states both ends of blob_sizes: a length or a short_side fraction
    with pytest.raises(Exception, match="suggest_blob_sizes"):
        blob_board("A5", blob_sizes=(12, None), seed=1)
    py = blob_board("A5", blob_sizes=suggest_blob_sizes(300, "A5"), seed=1)
    assert py.preamble.pattern_config.max_scale > py.preamble.pattern_config.min_scale


@pytest.mark.parametrize("kw", [{"blob_measure": "diameter"}, {"frame_width": 5.0}, {"frame_style": "none"}, {"raster_aligned": True},
                                {"pdf_standard": "x4"}, {"pdf_standard": "x3"}, {"print_density": 300}])
def test_blob_board_removed_keywords_raise(kw):
    with pytest.raises(TypeError):
        blob_board("A5", blob_sizes=(12, 0.2 * short_side), seed=1, **kw)


def test_blob_board_signature():
    with pytest.raises(TypeError):
        blob_board("A5", seed=1)                              # blob_sizes is required
    with pytest.raises(TypeError):
        blob_board("A5", 300, blob_sizes=(12, 0.2 * short_side), seed=1)  # the positional density is gone


def test_blob_board_raster_needs_a_density(tmp_path):
    with pytest.raises(Exception, match="density"):
        blob_board("A5", blob_sizes=(12, 0.2 * short_side), seed=1, dir=str(tmp_path), format="png")
    py = blob_board("A5", blob_sizes=(12, 0.2 * short_side), seed=1, dir=str(tmp_path), format="png", density=300)
    assert py.density_dpi == 300.0
    assert any(f.endswith(".png") for f in os.listdir(tmp_path))


def test_blob_pattern_matches_julia(jl):
    py = blob_pattern(400, 300, blob_sizes=(12, None), seed=0x52)
    expr = "BlobBoards.blob_pattern(400, 300; seed = 0x52, blob_sizes = (12BlobBoards.px, nothing))"
    assert py.config_hash == str(jl.seval(f"BlobBoards.hash_config(({expr})[3])"))
    np.testing.assert_array_equal(py.pattern, np.asarray(jl.seval(f"Float64.(({expr})[1])")))
    with pytest.raises(TypeError):
        blob_pattern(400, 300, blob_measure="diameter")


def test_load_board_matches_julia(jl, board_paths):
    p = board_paths[0]
    py = load_board(p)
    assert py.config_hash == str(jl.seval(f'BlobBoards.load_bundle("{p}")[3]'))
    # the density of the board's image (an example board, v3: its own)
    density = f'(let (b, i, _) = BlobBoards.load_bundle("{p}"); BlobBoards._image_density(i, b) end)'
    np.testing.assert_array_equal(py.blobs, _jl_blobs_px(jl, f'BlobBoards.load_bundle("{p}")[1]', density))
    assert py.uid == os.path.basename(p)[11:14]
    assert py.density_dpi == float(jl.seval(
        f"BlobBoards.VisualGeometryCore.ustrip(BlobBoards.dpi, {density})"))
    with pytest.raises(TypeError):
        load_board(p, print_density=300)


def test_render_board_keeps_identity(tmp_path, board_paths):
    p = board_paths[1]
    stored = load_board(p)
    # a raster names its density; a stored board converts at its own
    py = render_board(p, dir=str(tmp_path), format="png",
                      density=stored.density_dpi)
    stem = os.path.basename(p)[:-5]
    assert {stem + ".png", stem + ".json"} <= set(os.listdir(tmp_path))
    assert py.config_hash == stored.config_hash
    np.testing.assert_array_equal(py.blobs, stored.blobs)
    with pytest.raises(Exception, match="density"):
        render_board(p, dir=str(tmp_path), format="png")
    for kw in ({"pdf_standard": "x4"}, {"print_density": 300}):
        with pytest.raises(TypeError):
            render_board(p, dir=str(tmp_path), **kw)


def test_blobs_world_matches_julia(jl, board_paths):
    p = board_paths[2]
    w = blobs(p, reference_frame="world", units="mm")
    jw = np.asarray(jl.seval(f"""let d = BlobBoards.blobs(BlobBoards.load_bundle("{p}")[1], :world)
        [Float64(x) for x in vcat(([q.center[1] q.center[2]] for q in d)...)]
    end"""))
    np.testing.assert_allclose(w[:, :2], jw, rtol=0, atol=1e-12)


def test_constants_and_paths_match_julia(jl):
    assert PRINT_SIZES["A4"] == (210.0, 297.0)
    assert len(PRINT_SIZES) == int(jl.seval("length(BlobBoards.PRINT_SIZES)"))
    assert example_data("boards") == str(jl.seval("BlobBoards.example_data(:boards)"))
    assert descriptor_network_path() == str(jl.seval("BlobBoards.descriptor_network_path()"))
    with pytest.raises(ValueError):
        example_data("nope")


def test_device_selection(has_gpu):
    assert blobboards.device() == ("cuda" if has_gpu else "cpu")
    assert blobboards.device("cpu") == "cpu"
    with pytest.raises(ValueError):
        blobboards.device("tpu")
    if not has_gpu:
        with pytest.raises(RuntimeError):
            blobboards.device("cuda")


def _julia_find(jl, board_paths, model_path, gallery_path, img_expr, camera=None):
    paths = "[" + ", ".join(f'"{p}"' for p in board_paths) + "]"
    geo = "BlobBoards.HomographyGeometry()" if camera is None else \
        f'BlobBoards.PoseGeometry(BlobBoards.VisualGeometryIo.load(BlobBoards.VisualGeometryCore.CameraModel, "{camera}"))'
    return jl.seval(f"""let net = BlobBoards.DescriptorNetwork("{model_path}"),
            g = BlobBoards.BoardGallery("{gallery_path}.jl", net),
            feats = Dict(insert!(g, {paths}, net; progress = false)),
            bundles = Dict(b.uid => b.bundle for b in BlobBoards.load_board.({paths}))
        r = BlobBoards.find_boards(feats, net, {img_expr}; board_bundles = bundles,
                                   config = BlobBoards.FindBoardsConfig(geometry = {geo}))
        [(u, count(BlobBoards.VisualGeometryCore.is_map_inlier, s.weights)) for (u, m, q, s) in r]
    end""")


def test_find_boards_matches_julia_gpu(jl, has_gpu, board_paths, model_path, gallery_path, still_img):
    if not has_gpu:
        pytest.skip("no usable GPU")
    still = os.path.join(example_data("still"), "DSC02847.JPG")
    cam = os.path.join(example_data("still"), "cameras.json")
    py = find_boards(board_paths, model_path, still_img, gallery_path=gallery_path, camera_json=cam)
    assert all(b["device"] == "cuda" and b["execution_provider"] == "cuda" for b in py)
    ref = _julia_find(jl, board_paths, model_path, gallery_path,
                      f'BlobBoards.Colors.Gray{{Float32}}.(BlobBoards.FileIO.load("{still}"))', camera=cam)
    assert [(b["uid"], b["n_inliers"]) for b in py] == [(str(u), int(n)) for (u, n) in ref]
    assert py[0]["model"]["R"].shape == (3, 3)


def test_track_boards_gpu(jl, has_gpu, board_paths, model_path, gallery_path):
    if not has_gpu:
        pytest.skip("no usable GPU")
    clip = os.path.join(example_data("video"), "blobboards_handheld.mov")
    frames = np.asarray(jl.seval(f"""let r = BlobBoards.VideoIO.openvideo("{clip}")
        fs = [Float64.(BlobBoards.Colors.Gray.(read(r))) for _ in 1:3]; close(r); cat(fs...; dims = 3)
    end"""))
    frames = [frames[:, :, i] for i in range(frames.shape[2])]
    py = track_boards(board_paths, model_path, frames, gallery_path=gallery_path)
    assert py and all(len(d) == 3 for d in py.values())
    assert all(d[0]["frame"] == 1 and d[0]["source"] in ("detect", "init", "track") for d in py.values())
