"""Tests for gb_pipeline.py exported-tileset auto-trim (trim_tiles/rebuild_tiles).

Covers the three stages -- content bbox from pixel-vs-background difference,
crop + offset recording, and the tile table -- plus the configurables
(background threshold, per-tile vs union crop), all-empty tile reporting, and
pixel-identical rebuild at threshold 0.
"""

import numpy as np
import pytest

from gb_pipeline import (
    GBImage,
    find_content_bbox,
    rebuild_tiles,
    trim_gb_image,
    trim_tiles,
)

BG = (8, 24, 32)  # DMG darkest shade, a typical export background


def _rgb_tiles(n, h=8, w=8, bg=BG):
    tiles = np.zeros((n, h, w, 3), dtype=np.uint8)
    tiles[...] = bg
    return tiles


def test_bbox_finds_content_position():
    tile = np.zeros((8, 8, 3), dtype=np.uint8)
    tile[...] = BG
    tile[2:5, 3:7] = (224, 248, 208)
    assert find_content_bbox(tile, BG) == (2, 3, 5, 7)


def test_bbox_all_background_returns_none():
    tile = np.zeros((8, 8, 3), dtype=np.uint8)
    tile[...] = BG
    assert find_content_bbox(tile, BG) is None


def test_bbox_threshold_hides_near_background_pixels():
    tile = np.zeros((8, 8, 3), dtype=np.uint8)
    tile[...] = BG
    near = tuple(c + 3 for c in BG)  # within threshold 5
    far = tuple(c + 40 for c in BG)  # beyond threshold 5
    tile[1, 1] = near
    tile[6, 6] = far
    assert find_content_bbox(tile, BG, threshold=5) == (6, 6, 7, 7)
    assert find_content_bbox(tile, BG, threshold=0) == (1, 1, 7, 7)


def test_bbox_threshold_uses_max_channel_difference():
    tile = np.zeros((4, 4, 3), dtype=np.uint8)
    tile[...] = (100, 100, 100)
    tile[2, 2] = (100, 100, 110)  # only blue differs, by 10
    assert find_content_bbox(tile, (100, 100, 100), threshold=9) == (2, 2, 3, 3)
    assert find_content_bbox(tile, (100, 100, 100), threshold=10) is None


def test_trim_per_tile_records_sizes_and_offsets():
    tiles = _rgb_tiles(3)
    tiles[0, 1:4, 2:6] = (200, 200, 200)  # bbox (1,2)-(4,6)
    tiles[1, 5:, 0:2] = (90, 90, 90)      # bbox (5,0)-(8,2)
    tiles[2] = BG                          # all background
    atlas = trim_tiles(tiles, background=BG)

    assert atlas.per_tile
    assert atlas.tile_size == (8, 8)
    shapes = sorted(t.shape for t in atlas.tiles)
    assert (1, 1, 3) in shapes            # empty placeholder, never 0-sized
    assert (3, 4, 3) in shapes
    assert (3, 2, 3) in shapes
    # Offsets recorded per source tile in the table.
    tid0, oy0, ox0 = atlas.table[0]
    assert (oy0, ox0) == (1, 2)
    assert atlas.tiles[int(tid0)].shape == (3, 4, 3)
    tid1, oy1, ox1 = atlas.table[1]
    assert (oy1, ox1) == (5, 0)
    tid2, oy2, ox2 = atlas.table[2]
    assert (oy2, ox2) == (0, 0)
    assert atlas.tiles[int(tid2)].shape == (1, 1, 3)


def test_trim_reports_empty_tiles_not_zero_sized():
    tiles = _rgb_tiles(4)
    tiles[1, 0, 0] = (255, 255, 255)
    atlas = trim_tiles(tiles, background=BG)
    assert atlas.empty.tolist() == [0, 2, 3]
    for i in atlas.empty:
        tile = atlas.tiles[int(atlas.table[i, 0])]
        assert tile.size > 0, "empty tile must not be cropped to 0 size"
        assert tile.shape == (1, 1, 3)
        assert tuple(tile[0, 0]) == BG


def test_trim_union_mode_shares_one_bbox():
    tiles = _rgb_tiles(3)
    tiles[0, 1:3, 1:3] = (200, 0, 0)
    tiles[1, 4:7, 5:8] = (0, 200, 0)
    tiles[2] = BG
    atlas = trim_tiles(tiles, background=BG, per_tile=False)
    # Union bbox: rows 1..7, cols 1..8 -> 6x7 for every non-empty tile.
    for i in (0, 1):
        tid, oy, ox = atlas.table[i]
        assert (oy, ox) == (1, 1)
        assert atlas.tiles[int(tid)].shape == (6, 7, 3)


def test_trim_auto_background_from_border_mode():
    tiles = _rgb_tiles(2, bg=(16, 32, 48))
    tiles[0, 3, 3] = (250, 250, 250)
    tiles[1, 4, 4] = (10, 200, 10)
    atlas = trim_tiles(tiles)  # background="auto"
    assert atlas.background == (16, 32, 48)
    assert np.array_equal(rebuild_tiles(atlas), tiles)


def test_trim_dedups_identical_content_at_same_offset():
    tiles = _rgb_tiles(4)
    block = (np.array([[1, 2], [3, 4]], dtype=np.uint8) * 40)[..., None] * np.ones(3, dtype=np.uint8)
    for i in (0, 1, 2):
        tiles[i, 2:4, 2:4] = block  # same content, same offset
    tiles[3, 0:2, 0:2] = block      # same content, different offset
    atlas = trim_tiles(tiles, background=BG)
    # 2 unique tiles: the 2x2 block + the 1x1 background placeholder is NOT
    # present (no empty tiles here) -- so just the block... plus nothing else.
    assert len(atlas.tiles) == 1
    assert atlas.tiles[0].shape == (2, 2, 3)
    # Tile count change: 4 source tiles -> 1 atlas tile.
    assert atlas.table.shape == (4, 3)
    assert set(atlas.table[:, 0].tolist()) == {0}
    # Offsets differ per source tile even though the tile is shared.
    assert tuple(atlas.table[0, 1:]) == (2, 2)
    assert tuple(atlas.table[3, 1:]) == (0, 0)


def test_rebuild_is_pixel_identical_at_threshold_zero():
    rng = np.random.RandomState(7)
    tiles = _rgb_tiles(12)
    # Random content blobs at random offsets, plus two all-empty tiles.
    for i in range(10):
        rh, rw = rng.randint(1, 5, size=2)
        r0, c0 = rng.randint(0, 8 - rh + 1), rng.randint(0, 8 - rw + 1)
        tiles[i, r0 : r0 + rh, c0 : c0 + rw] = rng.randint(
            0, 256, size=(rh, rw, 3)
        ).astype(np.uint8)
    atlas = trim_tiles(tiles, background=BG)
    rebuilt = rebuild_tiles(atlas)
    assert np.array_equal(rebuilt, tiles)
    # Offsets are correct: each cropped tile equals the source slice at its
    # recorded offset.
    for i in range(12):
        tid, oy, ox = (int(v) for v in atlas.table[i])
        crop = atlas.tiles[tid]
        th, tw = crop.shape[:2]
        assert np.array_equal(crop, tiles[i, oy : oy + th, ox : ox + tw])


def test_rebuild_snaps_near_background_at_positive_threshold():
    tiles = _rgb_tiles(1)
    tiles[0, 2, 2] = tuple(c + 2 for c in BG)   # within threshold -> cropped
    tiles[0, 5, 5] = tuple(c + 100 for c in BG)  # content
    atlas = trim_tiles(tiles, background=BG, threshold=4)
    rebuilt = rebuild_tiles(atlas)
    # The near-background pixel is rebuilt as the exact background color.
    assert tuple(rebuilt[0, 2, 2]) == BG
    assert tuple(rebuilt[0, 5, 5]) == tuple(c + 100 for c in BG)


def test_trim_index_tiles_2d_stack():
    tiles = np.zeros((3, 8, 8), dtype=np.uint8)  # background index 0
    tiles[0, 1:5, 1:5] = 3
    tiles[1, 6, 6] = 1
    atlas = trim_tiles(tiles, background=0)
    assert atlas.background == (0,)
    assert atlas.empty.tolist() == [2]
    assert np.array_equal(rebuild_tiles(atlas), tiles)


def test_trim_rejects_bad_input():
    with pytest.raises(ValueError):
        trim_tiles(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        trim_tiles(_rgb_tiles(1), background="bogus")
    with pytest.raises(ValueError):
        trim_tiles(_rgb_tiles(1), threshold=-1)


def _dummy_palettes():
    return np.array(
        [[[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]]],
        dtype=np.uint8,
    )


def _gb(patterns, tilemap):
    th, tw = tilemap.shape
    return GBImage(
        patterns=patterns,
        tilemap=tilemap,
        attrs_palette=np.zeros((th, tw), dtype=np.uint8),
        attrs_hflip=np.zeros((th, tw), dtype=bool),
        attrs_vflip=np.zeros((th, tw), dtype=bool),
        palettes=_dummy_palettes(),
    )


def test_trim_gb_image_writes_offsets_into_tile_table():
    p_sparse = np.zeros((8, 8), dtype=np.uint8)
    p_sparse[2:6, 3:5] = 2                       # content at (2,3), size 4x2
    p_full = (np.arange(64).reshape(8, 8) % 4).astype(np.uint8)
    p_blank = np.zeros((8, 8), dtype=np.uint8)
    patterns = np.stack([p_sparse, p_full, p_blank])
    tilemap = np.array([[0, 1], [2, 0]], dtype=np.int32)  # reuses pattern 0
    gb = _gb(patterns, tilemap)

    atlas, cell_table = trim_gb_image(gb, background=0)

    assert atlas.empty.tolist() == [2]
    assert cell_table.shape == (2, 2, 3)
    # Cell (0,0) references pattern 0's trimmed tile at its recorded offset.
    tid, oy, ox = (int(v) for v in cell_table[0, 0])
    assert (oy, ox) == (2, 3)
    assert atlas.tiles[tid].shape == (4, 2)
    # Cells referencing the same pattern share the tile-table entry.
    assert np.array_equal(cell_table[0, 0], cell_table[1, 1])
    # Rebuild reproduces the per-cell patterns exactly (threshold 0).
    rebuilt = rebuild_tiles(atlas)
    assert np.array_equal(rebuilt[gb.tilemap.reshape(-1)].reshape(2, 2, 8, 8),
                          patterns[tilemap])


def test_trim_gb_image_union_mode_uniform_size():
    p1 = np.zeros((8, 8), dtype=np.uint8)
    p1[0, 0] = 1
    p2 = np.zeros((8, 8), dtype=np.uint8)
    p2[7, 7] = 2
    gb = _gb(np.stack([p1, p2]), np.array([[0, 1]], dtype=np.int32))
    atlas, cell_table = trim_gb_image(gb, background=0, per_tile=False)
    # Union bbox spans the whole 8x8, so no cropping happens but offsets are 0.
    assert all(t.shape == (8, 8) for t in atlas.tiles)
    assert np.all(cell_table[:, :, 1:] == 0)
