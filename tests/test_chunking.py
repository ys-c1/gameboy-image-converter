"""Tests for map chunking: chunk index geometry, load order, validation,
streaming load/unload with preload radius and LRU eviction, and
pixel-identical reassembly against a whole-image render.

All fixtures are tiny and built in-file (no external maps, no new deps).
"""

import time

import numpy as np
import pytest
from PIL import Image

import gb_pipeline
from gb_pipeline import (
    ChunkIndex,
    ChunkStreamer,
    build_chunk_index,
    convert_for_hardware,
    stitch_chunks,
)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _ramp_map(width: int, height: int) -> np.ndarray:
    """Deterministic RGB map: every pixel encodes its own (x, y) coordinates.

    R = x, G = y, B = (x + y) mod 256, so any misplaced or duplicated pixel
    is detected immediately by an equality comparison.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    return np.stack(
        [xx.astype(np.uint8), yy.astype(np.uint8), ((xx + yy) % 256).astype(np.uint8)],
        axis=-1,
    )


class _RecordingLoader:
    """Loader slicing an in-RAM map while counting calls and adding latency.

    Stands in for a disk/network backend: the ChunkStreamer only ever sees a
    callable returning ndarrays, and this one lets tests assert how often a
    chunk was actually fetched versus served from the resident set.
    """

    def __init__(self, source: np.ndarray, delay: float = 0.0):
        self.source = source
        self.delay = delay
        self.calls: list = []

    def __call__(self, info) -> np.ndarray:
        self.calls.append((info.row, info.col))
        if self.delay:
            time.sleep(self.delay)
        return self.source[info.y0:info.y1, info.x0:info.x1].copy()


# --------------------------------------------------------------------------
# build_chunk_index: geometry and load order
# --------------------------------------------------------------------------

def test_index_grid_covers_map_exactly_with_partial_edge_chunks():
    idx = build_chunk_index(20, 12, chunk_width=8, chunk_height=8)
    assert (idx.cols, idx.rows) == (3, 2)
    assert len(idx.chunks) == 6

    # Full-size interior chunks, partial right/bottom edges.
    first = idx.chunk_at(0, 0)
    assert (first.x0, first.y0, first.x1, first.y1) == (0, 0, 8, 8)
    assert (first.width, first.height) == (8, 8)
    right = idx.chunk_at(0, 2)
    assert (right.x0, right.x1, right.width) == (16, 20, 4)
    bottom = idx.chunk_at(1, 0)
    assert (bottom.y0, bottom.y1, bottom.height) == (8, 12, 4)

    # The boxes tile the map without gaps or overlaps.
    covered = np.zeros((12, 20), dtype=bool)
    for info in idx.chunks:
        assert not covered[info.y0:info.y1, info.x0:info.x1].any()
        covered[info.y0:info.y1, info.x0:info.x1] = True
    assert covered.all()


def test_index_exact_division_has_no_partial_chunks():
    idx = build_chunk_index(16, 16, chunk_width=8, chunk_height=8)
    assert (idx.cols, idx.rows) == (2, 2)
    assert all(info.width == 8 and info.height == 8 for info in idx.chunks)
    assert {k for k in idx.load_order} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_load_order_is_center_out_and_deterministic():
    # 5x3 grid: center chunk is (1, 2); rings expand by Chebyshev distance,
    # ties broken top-to-bottom then left-to-right.
    idx = build_chunk_index(40, 24, chunk_width=8, chunk_height=8)
    assert idx.load_order[0] == (1, 2)
    first_ring = set(idx.load_order[1:8])  # ring distance 1 holds 7 chunks
    assert first_ring == {
        (0, 1), (0, 2), (0, 3),
        (1, 1), (1, 3),
        (2, 1), (2, 2),
    }
    assert len(first_ring) == 7
    # Every chunk appears exactly once and Chebyshev distances are nondecreasing.
    assert len(idx.load_order) == len(set(idx.load_order)) == 15
    dists = [
        max(abs(r - 1), abs(c - 2)) for r, c in idx.load_order
    ]
    assert dists == sorted(dists)
    # Tie-break within a ring is row-major.
    ring1_ordered = [k for k in idx.load_order if max(abs(k[0] - 1), abs(k[1] - 2)) == 1]
    assert ring1_ordered == sorted(ring1_ordered)
    # Building twice yields the identical order (no set/dict ordering leakage).
    assert build_chunk_index(40, 24, 8, 8).load_order == idx.load_order


def test_chunk_at_rejects_out_of_grid_keys():
    idx = build_chunk_index(16, 16, 8, 8)
    with pytest.raises(IndexError):
        idx.chunk_at(2, 0)
    with pytest.raises(IndexError):
        idx.chunk_at(0, -1)


# --------------------------------------------------------------------------
# Illegal chunk sizes / arguments must raise
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(map_width=0, map_height=16, chunk_width=8, chunk_height=8),
        dict(map_width=16, map_height=-4, chunk_width=8, chunk_height=8),
        dict(map_width=16, map_height=16, chunk_width=0, chunk_height=8),
        dict(map_width=16, map_height=16, chunk_width=8, chunk_height=-8),
        dict(map_width=16, map_height=16, chunk_width=32, chunk_height=8),
        dict(map_width=16, map_height=16, chunk_width=8, chunk_height=32),
    ],
)
def test_build_chunk_index_rejects_illegal_sizes(kwargs):
    with pytest.raises(ValueError):
        build_chunk_index(**kwargs)


@pytest.mark.parametrize("bad", [1.5, "8", None, True, False])
def test_build_chunk_index_rejects_non_integer_arguments(bad):
    with pytest.raises(TypeError):
        build_chunk_index(bad, 16, 8, 8)
    with pytest.raises(TypeError):
        build_chunk_index(16, 16, bad, 8)


def test_streamer_rejects_illegal_constructor_arguments():
    idx = build_chunk_index(16, 16, 8, 8)
    loader = _RecordingLoader(_ramp_map(16, 16))
    with pytest.raises(TypeError):
        ChunkStreamer("not an index", loader)
    with pytest.raises(TypeError):
        ChunkStreamer(idx, object())
    with pytest.raises(ValueError):
        ChunkStreamer(idx, loader, preload_radius=-1)
    with pytest.raises(ValueError):
        ChunkStreamer(idx, loader, max_resident=0)
    with pytest.raises(TypeError):
        ChunkStreamer(idx, loader, preload_radius=1.0)
    with pytest.raises(TypeError):
        ChunkStreamer(idx, loader, preload_radius=True)


def test_viewport_queries_reject_bad_geometry():
    idx = build_chunk_index(16, 16, 8, 8)
    with pytest.raises(ValueError):
        idx.chunks_for_viewport(0, 0, 0, 8)
    with pytest.raises(ValueError):
        idx.chunks_for_viewport(0, 0, 8, -1)
    loader = _RecordingLoader(_ramp_map(16, 16))
    s = ChunkStreamer(idx, loader)
    with pytest.raises(ValueError):
        s.get_viewport_pixels(0, 0, 17, 8)
    with pytest.raises(ValueError):
        s.get_viewport_pixels(0, 0, 8, 0)


# --------------------------------------------------------------------------
# Viewport -> chunk overlap
# --------------------------------------------------------------------------

def test_chunks_for_viewport_overlap_and_clipping():
    idx = build_chunk_index(24, 24, chunk_width=8, chunk_height=8)
    # Single chunk interior.
    assert idx.chunks_for_viewport(2, 3, 4, 4) == ((0, 0),)
    # Straddling a vertical and horizontal chunk boundary.
    assert idx.chunks_for_viewport(6, 6, 8, 8) == (
        (0, 0), (0, 1), (1, 0), (1, 1)
    )
    # Viewport hanging off the bottom-right is clipped, not expanded.
    assert idx.chunks_for_viewport(20, 20, 80, 80) == ((2, 2),)
    # Fully outside the map yields no chunks rather than a bogus key.
    assert idx.chunks_for_viewport(100, 100, 8, 8) == ()
    assert idx.chunks_for_viewport(-80, -80, 4, 4) == ()


# --------------------------------------------------------------------------
# Streaming: load, cache hit, unload, preload ring, LRU cap, metrics
# --------------------------------------------------------------------------

def test_load_chunk_caches_and_unload_chunk_frees():
    source = _ramp_map(16, 16)
    loader = _RecordingLoader(source)
    s = ChunkStreamer(build_chunk_index(16, 16, 8, 8), loader)

    pixels = s.load_chunk(0, 0)
    assert np.array_equal(pixels, source[0:8, 0:8])
    assert loader.calls == [(0, 0)]
    # Second fetch is a cache hit: loader not invoked again.
    assert np.array_equal(s.load_chunk(0, 0), pixels)
    assert loader.calls == [(0, 0)]
    assert s.metrics["loads"] == 1

    assert s.unload_chunk(0, 0) is True
    assert s.resident_keys == ()
    assert s.unload_chunk(0, 0) is False  # already gone
    assert s.metrics["unloads"] == 1
    # Resident byte accounting tracks actual ndarray memory.
    s.load_chunk(0, 0)
    assert s.resident_bytes == pixels.nbytes


def test_focus_viewport_preloads_ring_and_evicts_far_chunks():
    source = _ramp_map(40, 24)  # 5 cols x 3 rows of 8px chunks
    idx = build_chunk_index(40, 24, 8, 8)
    loader = _RecordingLoader(source)
    s = ChunkStreamer(idx, loader, preload_radius=1)

    # Viewport entirely inside chunk (1, 2); ring 1 around it = 9 chunks total.
    loaded, unloaded = s.focus_viewport(17, 9, 2, 2)
    assert set(loaded) == {
        (r, c) for r in range(0, 3) for c in range(1, 4)
    }
    assert unloaded == ()
    assert len(s.resident_keys) == 9
    # The viewport's own chunk must be loaded before its preload neighbours.
    assert loaded[0] == (1, 2)
    assert s.metrics["peak_resident_chunks"] == 9

    # Re-focusing the same area loads nothing.
    assert s.focus_viewport(17, 9, 2, 2) == ((), ())

    # Move to the top-left corner: only the clipped 2x2 ring survives.
    loaded, unloaded = s.focus_viewport(1, 1, 2, 2)
    assert set(s.resident_keys) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    # The four survivors never left; everything else was unloaded once.
    assert set(unloaded) == {
        (r, c) for r in range(0, 3) for c in range(1, 4)
    } - {(0, 1), (1, 1)}
    assert set(loaded).isdisjoint(set(unloaded))


def test_preload_radius_zero_keeps_only_viewport_chunks():
    source = _ramp_map(32, 32)
    idx = build_chunk_index(32, 32, 8, 8)
    s = ChunkStreamer(idx, _RecordingLoader(source), preload_radius=0)
    # A 1px viewport at (7,7) lives wholly inside chunk (0,0).
    s.focus_viewport(7, 7, 1, 1)
    assert set(s.resident_keys) == {(0, 0)}
    # Crossing both chunk boundaries pulls exactly the four adjacent chunks.
    s.focus_viewport(7, 7, 2, 2)
    assert set(s.resident_keys) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_max_resident_lru_eviction_bounds_working_set():
    source = _ramp_map(32, 32)
    idx = build_chunk_index(32, 32, 8, 8)
    s = ChunkStreamer(idx, _RecordingLoader(source), preload_radius=0, max_resident=2)

    s.load_chunk(0, 0)
    s.load_chunk(0, 1)
    s.load_chunk(1, 0)  # evicts LRU (0,0)
    assert set(s.resident_keys) == {(0, 1), (1, 0)}
    assert s.metrics["peak_resident_chunks"] == 2

    # Touching (0,1) makes it most-recently-used; eviction falls on (1,0).
    s.load_chunk(0, 1)
    s.load_chunk(1, 1)
    assert set(s.resident_keys) == {(0, 1), (1, 1)}
    assert s.resident_bytes == 2 * source[0:8, 0:8].nbytes


def test_metrics_record_timing_and_peak_memory():
    source = _ramp_map(24, 24)
    idx = build_chunk_index(24, 24, 8, 8)
    loader = _RecordingLoader(source, delay=0.001)
    s = ChunkStreamer(idx, loader)
    s.load_chunk(0, 0)
    s.load_chunk(0, 1)
    m = s.metrics
    assert m["loads"] == 2 and m["unloads"] == 0
    assert m["total_chunks"] == 9
    assert m["peak_resident_bytes"] == m["resident_bytes"]
    assert m["load_seconds"] >= 0.002  # both simulated I/O sleeps counted
    s.unload_all()
    assert s.resident_keys == () and s.resident_bytes == 0


def test_loader_returning_wrong_shape_is_rejected():
    idx = build_chunk_index(20, 16, 8, 8)  # right-edge chunk is only 4 wide

    def bad_loader(info):
        return np.zeros((8, 8, 3), dtype=np.uint8)  # ignores partial edge

    s = ChunkStreamer(idx, bad_loader)
    with pytest.raises(ValueError, match="loader returned shape"):
        s.load_chunk(0, 2)


# --------------------------------------------------------------------------
# Pixel-identical reassembly and viewport crops
# --------------------------------------------------------------------------

def test_stitched_chunks_equal_whole_map_pixel_for_pixel():
    source = _ramp_map(20, 12)  # deliberately non-divisible on both axes
    idx = build_chunk_index(20, 12, chunk_width=8, chunk_height=8)
    chunks = {}
    for info in idx.chunks:
        chunks[(info.row, info.col)] = source[info.y0:info.y1, info.x0:info.x1]
    rebuilt = stitch_chunks(idx, chunks)
    assert rebuilt.shape == source.shape
    assert np.array_equal(rebuilt, source)


def test_stitch_requires_every_chunk():
    idx = build_chunk_index(16, 16, 8, 8)
    with pytest.raises(ValueError):
        stitch_chunks(idx, {(0, 0): np.zeros((8, 8, 3), dtype=np.uint8)})


def test_streamed_viewport_crop_matches_direct_map_crop():
    source = _ramp_map(40, 24)
    idx = build_chunk_index(40, 24, 8, 8)
    s = ChunkStreamer(idx, _RecordingLoader(source), preload_radius=1)

    # Viewport crossing four chunks and straddling the partial right edge.
    crop = s.get_viewport_pixels(13, 11, 22, 10)
    assert crop.shape == (10, 22, 3)
    assert np.array_equal(crop, source[11:21, 13:35])

    # Top-left corner crop, including a zero origin.
    crop0 = s.get_viewport_pixels(0, 0, 9, 9)
    assert np.array_equal(crop0, source[0:9, 0:9])


def test_full_streaming_walk_visits_every_chunk_and_stays_consistent():
    """Simulate a viewport sweeping the whole map; every observed crop must
    equal the corresponding direct slice, and after the walk all chunks have
    been fetched at least once."""
    source = _ramp_map(48, 32)
    idx = build_chunk_index(48, 32, 8, 8)
    loader = _RecordingLoader(source)
    s = ChunkStreamer(idx, loader, preload_radius=1, max_resident=6)

    vw, vh = 12, 12
    for y in range(0, 32 - vh + 1, 4):
        for x in range(0, 48 - vw + 1, 4):
            crop = s.get_viewport_pixels(x, y, vw, vh)
            assert np.array_equal(crop, source[y:y + vh, x:x + vw])
            assert len(s.resident_keys) <= 6
    assert set(loader.calls) == {(info.row, info.col) for info in idx.chunks}
    assert s.metrics["peak_resident_chunks"] <= 6


# --------------------------------------------------------------------------
# End-to-end: chunking a real convert_for_hardware render changes no pixels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["color_only", "mono"])
def test_convert_render_chunked_and_stitched_is_pixel_identical(preset):
    # 96x80 stays 96x80 (only logo presets resize); 32px chunks -> 3x3 grid
    # with a 16px partial bottom row.
    img = Image.fromarray(_ramp_map(96, 80), mode="RGB")
    result = convert_for_hardware(img, preset, dither="bayer")
    whole = np.asarray(result.image, dtype=np.uint8)

    idx = build_chunk_index(whole.shape[1], whole.shape[0], 32, 32)
    assert (idx.cols, idx.rows) == (3, 3)
    assert idx.chunk_at(2, 0).height == 16
    streamer = ChunkStreamer(idx, _RecordingLoader(whole), preload_radius=1)

    # Streaming path: every chunk fetched through the loader/cache machinery.
    chunks = {}
    for info in idx.chunks:
        chunks[(info.row, info.col)] = streamer.load_chunk(info.row, info.col)
    rebuilt = stitch_chunks(idx, chunks)
    assert np.array_equal(rebuilt, whole)

    # And a streamed viewport crop equals the same window of the full render.
    crop = streamer.get_viewport_pixels(30, 28, 40, 30)
    assert np.array_equal(crop, whole[28:58, 30:70])
