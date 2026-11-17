"""Tests for the multi-image shared 4-color palette solver.

The solver fits ONE 4-color palette over the pooled colors of several images
(weighted by usage frequency, Euclidean CIELAB distances), hands back each
image's source-color -> shared-color mapping table and the total/individual
perceptual error, and compares the shared palette against every image's own
independently fitted palette. These tests use only small hand-built images.
"""

import numpy as np
import pytest
from PIL import Image

import gb_pipeline as gbp

# --------------------------------------------------------------------------
# Tiny hand-built image sets
# --------------------------------------------------------------------------

def _solid(color, size=(8, 8)):
    arr = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def _conflicting_pair():
    """Two 8x8 images whose own 4-color palettes overlap only conceptually.

    Image A uses saturated R/G/B + white; image B uses slightly shifted
    R/G/B + black, so independent palettes produce 8 conflicting entries a
    shared palette must reconcile.
    """
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    a[:4, :4] = (220, 30, 30)
    a[:4, 4:] = (30, 220, 30)
    a[4:, :4] = (30, 30, 220)
    a[4:, 4:] = (230, 230, 230)
    b = np.zeros((8, 8, 3), dtype=np.uint8)
    b[:4, :4] = (200, 60, 60)
    b[:4, 4:] = (60, 200, 60)
    b[4:, :4] = (60, 60, 200)
    b[4:, 4:] = (40, 40, 40)
    return a, b


def _four_color_quadrants(colors):
    """8x8 image whose four 4x4 quadrants are filled from 4 RGB colors."""
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:4, :4] = colors[0]
    arr[:4, 4:] = colors[1]
    arr[4:, :4] = colors[2]
    arr[4:, 4:] = colors[3]
    return arr


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None])
def test_empty_image_set_raises(bad):
    with pytest.raises(ValueError, match="empty"):
        gbp.solve_shared_palette(bad)


def test_zero_pixel_image_raises():
    with pytest.raises(ValueError, match="zero pixels"):
        gbp.solve_shared_palette([np.zeros((0, 8, 3), dtype=np.uint8)])


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        gbp.solve_shared_palette([np.zeros((8, 8), dtype=np.uint8)])


def test_unknown_algorithm_raises():
    a, _ = _conflicting_pair()
    with pytest.raises(ValueError, match="algorithm"):
        gbp.solve_shared_palette([a], algorithm="kmeans")


def test_negative_iterations_raises():
    a, _ = _conflicting_pair()
    with pytest.raises(ValueError, match="n_iter"):
        gbp.solve_shared_palette([a], n_iter=-1)


# --------------------------------------------------------------------------
# Result structure and hardware invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ["greedy", "mediancut"])
def test_palette_is_4_colors_snapped_and_luminance_sorted(algorithm):
    a, b = _conflicting_pair()
    res = gbp.solve_shared_palette([a, b], algorithm=algorithm, n_iter=5)
    assert res.palette.shape == (4, 3)
    assert res.palette.dtype == np.uint8
    assert np.array_equal(gbp.snap_rgb555(res.palette), res.palette)
    assert np.array_equal(gbp.luminance_sort(res.palette), res.palette)
    assert len(np.unique(res.palette, axis=0)) == 4
    assert res.own_palettes.shape == (2, 4, 3)
    assert len(res.mappings) == 2
    assert len(res.per_image_error) == 2
    assert len(res.palette_diff) == 2
    assert res.solve_seconds >= 0.0


@pytest.mark.parametrize("algorithm", ["greedy", "mediancut"])
def test_mapping_tables_cover_every_source_color(algorithm):
    a, b = _conflicting_pair()
    res = gbp.solve_shared_palette([a, b], algorithm=algorithm)
    for img, table in zip((a, b), res.mappings):
        src_colors = {tuple(c) for c in np.unique(img.reshape(-1, 3), axis=0)}
        assert len(table) == len(src_colors)
        assert {m.source_rgb for m in table} == src_colors
        for m in table:
            # Every target is the palette row named by target_index.
            assert m.target_rgb == tuple(int(v) for v in res.palette[m.target_index])
            assert 0 <= m.target_index < 4
            assert m.delta_e >= 0.0
            assert m.pixel_count > 0
        # Pixel counts partition the image.
        assert sum(m.pixel_count for m in table) == img.shape[0] * img.shape[1]


@pytest.mark.parametrize("algorithm", ["greedy", "mediancut"])
def test_total_error_is_pixel_weighted_sum_of_mapping_errors(algorithm):
    a, b = _conflicting_pair()
    res = gbp.solve_shared_palette([a, b], algorithm=algorithm, n_iter=3)
    rebuilt = 0.0
    for table in res.mappings:
        rebuilt += sum(m.delta_e * m.pixel_count for m in table)
    assert rebuilt == pytest.approx(res.total_error, rel=1e-12, abs=1e-9)
    assert res.total_error == pytest.approx(sum(res.per_image_error), abs=1e-9)
    assert res.total_error >= 0.0


@pytest.mark.parametrize("algorithm", ["greedy", "mediancut"])
def test_mapping_matches_independent_nearest_lab(algorithm):
    """Each mapping's target must genuinely be the nearest palette entry."""
    a, b = _conflicting_pair()
    res = gbp.solve_shared_palette([a, b], algorithm=algorithm)
    pal_lab = gbp._rgb_to_lab(res.palette)
    for img, table in zip((a, b), res.mappings):
        cols = np.unique(img.reshape(-1, 3), axis=0)
        expected = gbp._nearest_indices(gbp._rgb_to_lab(cols), pal_lab)
        got = {m.source_rgb: m.target_index for m in table}
        for col, idx in zip(cols, expected):
            assert got[tuple(int(v) for v in col)] == int(idx)


# --------------------------------------------------------------------------
# Algorithm behavior
# --------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ["greedy", "mediancut"])
def test_deterministic_bit_identical(algorithm):
    a, b = _conflicting_pair()
    r1 = gbp.solve_shared_palette([a, b], algorithm=algorithm, n_iter=7)
    r2 = gbp.solve_shared_palette([a, b], algorithm=algorithm, n_iter=7)
    assert np.array_equal(r1.palette, r2.palette)
    assert r1.total_error == r2.total_error
    assert [[(m.source_rgb, m.target_rgb) for m in t] for t in r1.mappings] == \
           [[(m.source_rgb, m.target_rgb) for m in t] for t in r2.mappings]


def test_greedy_first_center_is_most_frequent_color():
    # One dominant color (48 px) plus three rare ones (4 px each).
    img = _four_color_quadrants(
        [(10, 10, 10), (200, 20, 20), (20, 200, 20), (20, 20, 200)]
    )
    img[:6, :] = (10, 10, 10)  # 48 of 64 pixels are the dark color
    cols, cnts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    lab = gbp._rgb_to_lab(cols)
    centers = gbp._greedy_seed_lab(lab, cnts.astype(np.float64), 4)
    # First seed sits exactly on the heaviest color's Lab value.
    assert np.allclose(centers[0], lab[int(np.argmax(cnts))])


def test_mediancut_seed_returns_k_centers_and_splits_flat_boxes():
    # Degenerate input: colors identical along two Lab axes must not crash.
    cols = gbp.snap_rgb555(
        np.array([[i * 16, 0, 0] for i in range(6)], dtype=np.uint8)
    )
    weights = np.full(len(cols), 3.0)
    centers = gbp._mediancut_seed_lab(gbp._rgb_to_lab(cols), weights, 4)
    assert centers.shape == (4, 3)
    # All-singleton flat set (n <= k) returns the points unchanged.
    few = gbp._rgb_to_lab(cols[:3])
    assert np.array_equal(gbp._mediancut_seed_lab(few, weights[:3], 4), few)


def test_refinement_never_increases_error_on_seed():
    a, b = _conflicting_pair()
    seed = gbp.solve_shared_palette([a, b], algorithm="greedy", n_iter=0)
    refined = gbp.solve_shared_palette([a, b], algorithm="greedy", n_iter=20)
    # Lloyd iterations minimize the same weighted nearest-center objective the
    # total error measures (up to RGB555 snapping), so it cannot grow.
    assert refined.total_error <= seed.total_error + 1e-6


def test_iteration_zero_is_a_valid_seed():
    a, b = _conflicting_pair()
    res = gbp.solve_shared_palette([a, b], algorithm="mediancut", n_iter=0)
    assert res.palette.shape == (4, 3)
    assert np.unique(res.palette, axis=0).shape[0] == 4


def test_identical_color_sets_share_with_zero_error():
    colors = gbp.snap_rgb555(
        np.array([(220, 30, 30), (30, 220, 30), (30, 30, 220), (230, 230, 230)],
                 dtype=np.uint8)
    )
    a = _four_color_quadrants(colors)
    b = _four_color_quadrants(colors)
    res = gbp.solve_shared_palette([a, b])
    assert res.total_error == pytest.approx(0.0, abs=1e-12)
    assert all(e == pytest.approx(0.0, abs=1e-12) for e in res.per_image_error)
    assert all(d == pytest.approx(0.0, abs=1e-12) for d in res.palette_diff)
    # All four source colors appear verbatim in the shared palette.
    pal = {tuple(c) for c in res.palette}
    assert {tuple(c) for c in colors} <= pal


def test_single_flat_image_palette_contains_its_color():
    color = gbp.snap_rgb555(np.array([(96, 128, 160)], dtype=np.uint8))[0]
    res = gbp.solve_shared_palette([_solid(tuple(int(v) for v in color))])
    assert tuple(int(v) for v in color) in {tuple(c) for c in res.palette}
    assert res.total_error == pytest.approx(0.0, abs=1e-12)
    assert len(res.mappings[0]) == 1


def test_snap_collision_is_repaired_to_four_distinct_colors():
    # Colors close enough that 4 fitted centers can snap onto fewer RGB555
    # cells; the repair path must still return 4 distinct palette rows.
    base = np.array([(100, 100, 100), (104, 100, 100), (108, 100, 100),
                     (112, 100, 100), (200, 50, 50), (50, 200, 50)],
                    dtype=np.uint8)
    img = np.zeros((12, 8, 3), dtype=np.uint8)
    for i, c in enumerate(base):
        img[2 * i:2 * i + 2] = c
    res = gbp.solve_shared_palette([img], algorithm="mediancut", n_iter=3)
    assert np.unique(res.palette, axis=0).shape[0] == 4


# --------------------------------------------------------------------------
# Own-palette difference and input handling
# --------------------------------------------------------------------------

def test_palette_diff_zero_when_own_palette_matches_shared():
    colors = gbp.snap_rgb555(
        np.array([(10, 10, 10), (90, 90, 90), (170, 170, 170), (240, 240, 240)],
                 dtype=np.uint8)
    )
    img = _four_color_quadrants(colors)
    res = gbp.solve_shared_palette([img])
    # A single image's own 4-color palette IS the shared palette's color set.
    assert set(map(tuple, res.own_palettes[0].tolist())) == \
           set(map(tuple, res.palette.tolist()))
    assert res.palette_diff[0] == pytest.approx(0.0, abs=1e-12)


def test_pil_and_array_inputs_give_identical_results():
    a, b = _conflicting_pair()
    r_arr = gbp.solve_shared_palette([a, b])
    r_pil = gbp.solve_shared_palette([Image.fromarray(a), Image.fromarray(b)])
    assert np.array_equal(r_arr.palette, r_pil.palette)
    assert r_arr.total_error == r_pil.total_error


def test_input_arrays_are_not_mutated():
    a, b = _conflicting_pair()
    a_before, b_before = a.copy(), b.copy()
    gbp.solve_shared_palette([a, b])
    assert np.array_equal(a, a_before)
    assert np.array_equal(b, b_before)


def test_greedy_and_mediancut_both_valid_on_many_color_image():
    rng = np.random.RandomState(5)
    img = rng.randint(0, 256, size=(24, 24, 3)).astype(np.uint8)
    for algorithm in ("greedy", "mediancut"):
        res = gbp.solve_shared_palette([img], algorithm=algorithm, n_iter=5)
        assert np.unique(res.palette, axis=0).shape[0] == 4
        assert res.total_error > 0.0
        assert sum(len(t) for t in res.mappings) == len(
            np.unique(img.reshape(-1, 3), axis=0)
        )
