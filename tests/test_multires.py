"""Tests for the Stage 5b multi-resolution export (multires_export).

Covers the delivery contract:
  * four target-size groups (exact-aligned, pad on both axes, crop on both
    axes, and an odd small target) -- tile counts, alignment-padding pixel
    ratio, map offset table, and tileset/tilemap round-trip reconstruction;
  * the three alignment strategies (pad / crop / reject), edge trimming, and
  * every illegal target-size / option input raising before pixel work;
  * the fixed-size hardware outputs (logo presets) staying pixel-identical,
    since the new stage is additive and never touches that path.

All fixtures are tiny (max 40x36) and pure numpy/PIL, so the whole module
runs in well under a second.
"""

import numpy as np
import pytest
from PIL import Image

import gb_pipeline as gp
from gb_pipeline import (
    TILE,
    MultiresResult,
    align_to_tile_grid,
    crop_to_content,
    multires_export,
    reconstruct_multires,
)


# ---------------------------------------------------------------------------
# Tiny deterministic fixtures
# ---------------------------------------------------------------------------

def _bordered_image(width: int, height: int, border=(0, 0, 0), seed: int = 7):
    """An RGB image with a 1-px uniform border and a noisy interior.

    The interior is a per-pixel seeded pattern, so no two 8x8 tiles collide
    by accident (tile counts stay predictable), while the border stays one
    exact color for edge-trim tests.
    """
    rng = np.random.RandomState(seed)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = border
    if height > 2 and width > 2:
        interior = rng.randint(0, 256, size=(height - 2, width - 2, 3)).astype(np.uint8)
        arr[1:-1, 1:-1] = interior
    return Image.fromarray(arr, "RGB")


def _flat_image(width: int, height: int, color=(10, 20, 30)):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr, "RGB")


# Four target-size groups requested for delivery:
#   (width, height, expected align behavior under strategy="pad")
SIZE_GROUPS = [
    (32, 24, "exact"),       # already 8-multiples -> no padding at all
    (30, 22, "pad_both"),    # both axes non-aligned -> pad to 32x24
    (35, 27, "crop_both"),   # pad strategy grows it; crop strategy shrinks it
    (13, 11, "odd_small"),   # tiny odd target, large relative padding share
]


# ---------------------------------------------------------------------------
# Input validation -- illegal sizes/options must raise before pixel work
# ---------------------------------------------------------------------------

ILLEGAL_TARGETS = [
    None,
    160,
    "160x144",
    (160,),
    (160, 144, 2),
    (0, 24),
    (32, 0),
    (-8, 24),
    (32, -1),
    (32.0, 24),
    (32, 24.5),
    (True, 24),
    ("32", "24"),
]


@pytest.mark.parametrize("bad", ILLEGAL_TARGETS)
def test_illegal_target_size_raises(bad):
    img = _flat_image(16, 16)
    with pytest.raises((TypeError, ValueError)):
        multires_export(img, bad)


def test_illegal_alignment_strategy_raises():
    img = _flat_image(16, 16)
    with pytest.raises(ValueError, match="alignment strategy"):
        multires_export(img, (16, 16), align="stretch")


def test_illegal_resample_filter_raises():
    img = _flat_image(16, 16)
    with pytest.raises(ValueError, match="resample filter"):
        multires_export(img, (16, 16), resample="spline64")


def test_reject_strategy_raises_on_misaligned_target():
    img = _flat_image(17, 9)
    with pytest.raises(ValueError, match="multiple of 8"):
        multires_export(img, (17, 9), align="reject")
    # ...but accepts an already-aligned target unchanged.
    res = multires_export(img, (16, 8), align="reject")
    assert res.canvas_size == (16, 8)
    assert res.pad_ratio == 0.0


# ---------------------------------------------------------------------------
# Alignment primitives
# ---------------------------------------------------------------------------

def test_align_pad_grows_to_next_multiple_of_8_and_centers():
    img = _flat_image(13, 11, color=(200, 100, 50))
    aligned, box = align_to_tile_grid(img, "pad", pad_color=(0, 0, 0))
    # 13 -> 16 (3 extra), 11 -> 16 (5 extra); content stays centered.
    assert aligned.size == (16, 16)
    assert box == (1, 2, 14, 13)  # left=3//2, top=5//2
    arr = np.asarray(aligned)
    # Pasted content retains its color; the padding takes pad_color.
    assert tuple(arr[2, 1]) == (200, 100, 50)
    assert tuple(arr[0, 0]) == (0, 0, 0)
    assert tuple(arr[15, 15]) == (0, 0, 0)


def test_align_crop_shrinks_to_largest_fitting_multiple():
    img = _flat_image(13, 11, color=(1, 2, 3))
    aligned, box = align_to_tile_grid(img, "crop")
    assert aligned.size == (8, 8)
    assert box == (0, 0, 8, 8)
    assert np.asarray(aligned).shape == (8, 8, 3)


def test_align_reject_passes_aligned_canvas_through():
    img = _flat_image(24, 16)
    aligned, box = align_to_tile_grid(img, "reject")
    assert aligned.size == img.size
    assert np.array_equal(np.asarray(aligned), np.asarray(img))
    assert box == (0, 0, 24, 16)


def test_crop_to_content_trims_uniform_border():
    img = _bordered_image(20, 16)
    box = crop_to_content(img)
    assert box == (1, 1, 19, 15)
    # Fully uniform image collapses to a zero-area box.
    assert crop_to_content(_flat_image(12, 12)) == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Four size groups: tile counts, padding ratio, offset table, reconstruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,height,label", SIZE_GROUPS,
                         ids=[g[2] for g in SIZE_GROUPS])
def test_four_size_groups_pad_strategy(width, height, label):
    img = _bordered_image(40, 36)
    res = multires_export(img, (width, height), align="pad", resample="nearest")

    assert isinstance(res, MultiresResult)
    cw, ch = res.canvas_size
    assert cw % TILE == 0 and ch % TILE == 0
    assert cw >= width and ch >= height

    th, tw = ch // TILE, cw // TILE
    n_cells = th * tw

    # Tile/tilemap/tileset shape contract.
    assert res.tilemap.shape == (th, tw)
    assert res.tileset.ndim == 4 and res.tileset.shape[1:] == (TILE, TILE, 3)
    assert res.tileset.shape[0] <= n_cells
    assert set(np.unique(res.tilemap).tolist()) == set(range(res.tileset.shape[0]))

    # Map offset table: one row per cell, raster order, correct coordinates.
    assert len(res.map_offsets) == n_cells
    for i, (tr, tc, x, y, padded) in enumerate(res.map_offsets):
        assert (tr, tc) == (i // tw, i % tw)
        assert (x, y) == (tc * TILE, tr * TILE)
        assert padded in (0, 1)

    # Padding ratio equals the exact fraction of canvas pixels outside the
    # content box (partial cells count their real padding pixels, not 64).
    n_pad_cells = sum(row[4] for row in res.map_offsets)
    left, top, right, bottom = res.content_box
    content_pixels = max(0, right - left) * max(0, bottom - top)
    expected_ratio = (cw * ch - content_pixels) / (cw * ch)
    assert res.pad_ratio == pytest.approx(expected_ratio)
    assert 0.0 <= res.pad_ratio <= 1.0
    assert 0 <= left <= right <= cw and 0 <= top <= bottom <= ch
    if label == "exact":
        assert n_pad_cells == 0 and res.pad_ratio == 0.0
        assert res.content_box == (0, 0, width, height)
    else:
        assert n_pad_cells > 0
        assert res.pad_ratio > 0.0

    # Round-trip: tileset + tilemap rebuild the delivered canvas pixel-exact.
    rebuilt = reconstruct_multires(res)
    assert np.array_equal(rebuilt, np.asarray(res.image))


@pytest.mark.parametrize("width,height,label", SIZE_GROUPS,
                         ids=[g[2] for g in SIZE_GROUPS])
def test_four_size_groups_crop_strategy_never_pads(width, height, label):
    img = _bordered_image(40, 36)
    res = multires_export(img, (width, height), align="crop", resample="nearest")

    cw, ch = res.canvas_size
    assert cw % TILE == 0 and ch % TILE == 0
    assert cw <= width and ch <= height and cw >= TILE and ch >= TILE
    # Cropping cannot introduce padding: every cell is real content.
    assert res.pad_ratio == 0.0
    assert all(row[4] == 0 for row in res.map_offsets)
    assert res.content_box == (0, 0, cw, ch)

    rebuilt = reconstruct_multires(res)
    assert np.array_equal(rebuilt, np.asarray(res.image))


def test_pad_ratio_matches_hand_computed_geometry():
    # 13x11 target -> 16x16 canvas; content box (1,2)-(14,13), i.e. 143 real
    # pixels and 256-143 = 113 padding pixels -> 113/256. Every cell touches
    # the border on at least one axis, so all four cells carry the padded flag.
    res = multires_export(_flat_image(13, 11), (13, 11), align="pad")
    assert res.canvas_size == (16, 16)
    assert res.content_box == (1, 2, 14, 13)
    assert res.pad_ratio == pytest.approx(113 / 256)
    padded_cells = [(r, c) for r, c, x, y, p in res.map_offsets if p]
    assert padded_cells == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # No padding case stays exactly zero.
    exact = multires_export(_flat_image(16, 16), (16, 16), align="pad")
    assert exact.pad_ratio == 0.0
    assert all(row[4] == 0 for row in exact.map_offsets)


def test_unique_tiles_count_and_dedup_on_repeated_content():
    # A 2x2 tile grid where every tile is identical -> exactly one unique tile.
    img = _flat_image(16, 16, color=(7, 77, 177))
    res = multires_export(img, (16, 16), align="pad")
    assert res.tileset.shape[0] == 1
    assert np.all(res.tilemap == 0)
    assert len(res.map_offsets) == 4

    # A per-tile checker of four distinct solid tiles -> four unique tiles.
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    colors = [(10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40)]
    for tr in range(2):
        for tc in range(2):
            arr[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8] = colors[tr * 2 + tc]
    res2 = multires_export(Image.fromarray(arr, "RGB"), (16, 16))
    assert res2.tileset.shape[0] == 4
    # Tilemap walks the tileset in first-appearance raster order.
    assert res2.tilemap.tolist() == [[0, 1], [2, 3]]


def test_reconstruction_independence_ignores_delivered_image():
    # Corrupt the delivered image after the fact: reconstruction must still
    # reproduce the tileset content (it never reads result.image).
    res = multires_export(_bordered_image(24, 16), (24, 16), resample="nearest")
    honest = reconstruct_multires(res)
    # Mutate the PIL image in place; the round-trip output is untouched.
    arr = np.asarray(res.image).copy()
    arr[0, 0] = (255, 255, 255)
    res.image = Image.fromarray(arr, "RGB")
    assert np.array_equal(reconstruct_multires(res), honest)


# ---------------------------------------------------------------------------
# Optional edge trimming + resample filter configuration
# ---------------------------------------------------------------------------

def test_crop_edges_removes_border_before_resampling():
    img = _bordered_image(18, 18, border=(0, 0, 0))
    trimmed = multires_export(img, (16, 16), crop_edges=True, resample="nearest")
    plain = multires_export(img, (16, 16), crop_edges=False, resample="nearest")
    # The trim changes the pixels fed into the resize, so the canvases differ.
    assert not np.array_equal(np.asarray(trimmed.image), np.asarray(plain.image))
    assert trimmed.canvas_size == (16, 16)
    assert np.array_equal(reconstruct_multires(trimmed), np.asarray(trimmed.image))


@pytest.mark.parametrize("filt", gp.RESAMPLE_FILTERS)
def test_all_resample_filters_are_accepted_and_reconstruct(filt):
    res = multires_export(_bordered_image(20, 20), (24, 16), resample=filt)
    assert res.canvas_size == (24, 16)
    assert np.array_equal(reconstruct_multires(res), np.asarray(res.image))


def test_nearest_upscale_is_block_exact():
    # Nearest-neighbor 2x upscale of an 8x8 image reproduces each source
    # pixel as a 2x2 block -- no interpolation ambiguity.
    src = _flat_image(8, 8, color=(5, 55, 105))
    res = multires_export(src, (16, 16), resample="nearest")
    arr = np.asarray(res.image)
    assert np.all(arr == (5, 55, 105))


# ---------------------------------------------------------------------------
# Fixed-size hardware output must remain pixel-identical (additive stage)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["logo_color", "logo_mono"])
def test_fixed_size_logo_output_unchanged(preset):
    img = _bordered_image(40, 36)
    result = gp.convert_for_hardware(img, preset)
    assert result.image.size == gp.PRESETS[preset].fixed_size
    # Hardware invariants still hold: RGB555-only colors, exact 8-grid size.
    arr = np.asarray(result.image)
    w, h = result.image.size
    assert w % TILE == 0 and h % TILE == 0
    assert np.array_equal(gp.snap_rgb555(arr), arr)


def test_multires_does_not_mutate_source_image():
    img = _bordered_image(17, 13)
    before = np.asarray(img).copy()
    multires_export(img, (32, 24), align="pad")
    multires_export(img, (8, 8), align="crop")
    assert np.array_equal(np.asarray(img), before)
