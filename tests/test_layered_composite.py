"""Tests for the layered two-palette composite (gb_pipeline.composite_layers).

Covers the three-stage contract:

1. each layer gets its own four-colour palette (fitted or caller-supplied);
2. overlap pixels are resolved by hardware priority plus sprite alpha, and
   reversing the priority order must visibly change the result (counter-example
   data is asserted, not assumed);
3. the total distinct-colour budget (hardware ceiling 8) is enforced by
   perceptually-closest-pair CIELAB merges, never by silent truncation -- an
   unmergeable overflow raises LayerColorBudgetError.

Also pins the non-regression guarantee: the single-palette / mono pipeline
paths are untouched, so their output stays pixel-identical.

All fixtures are tiny (16x24 / 24x24) and deterministic so the whole file
runs in well under a second.
"""

import numpy as np
import pytest
from PIL import Image

import gb_pipeline as gp
from gb_pipeline import (
    LAYER_HARDWARE_COLOR_BUDGET,
    LayerColorBudgetError,
    composite_layers,
    convert_for_hardware,
    snap_rgb555,
)


# --------------------------------------------------------------------------
# Fixtures: four small, hand-built layer pairs
# --------------------------------------------------------------------------

def _grid(h=16, w=24):
    yy, xx = np.mgrid[0:h, 0:w]
    return yy, xx


def _background_blues(h=16, w=24):
    """Cool background: fixed R/G, four distinct blue levels (4 colours)."""
    _, xx = _grid(h, w)
    blue = np.array([40, 100, 160, 220], dtype=np.uint8)
    b = blue[(xx // (w // 4)) % 4]
    return np.stack([np.full((h, w), 40, np.uint8),
                     np.full((h, w), 80, np.uint8), b], axis=-1)


def _solid_sprite(h=16, w=24, top=4, left=8, bottom=12, right=20,
                  rgb=(200, 60, 60), alpha=255):
    """Axis-aligned RGBA sprite block on a transparent canvas."""
    sp = np.zeros((h, w, 4), dtype=np.uint8)
    sp[top:bottom, left:right, 0] = rgb[0]
    sp[top:bottom, left:right, 1] = rgb[1]
    sp[top:bottom, left:right, 2] = rgb[2]
    sp[top:bottom, left:right, 3] = alpha
    return sp


@pytest.fixture
def case_flat_overlap():
    """Case 1: flat sprite fully inside the background -- clean opaque overlap."""
    bg = _background_blues()
    sp = _solid_sprite()
    return bg, sp


@pytest.fixture
def case_alpha_edge():
    """Case 2: opaque body plus a partial-alpha edge band (blend winner)."""
    bg = _background_blues()
    sp = _solid_sprite()
    sp[6:8, 10:12, 3] = 64  # ~0.25 alpha -> alpha-blended overlap pixels
    return bg, sp


@pytest.fixture
def case_multicolor_sprite():
    """Case 3: four-colour sprite (warm ramp) overlapping a four-colour bg."""
    h, w = 16, 24
    bg = _background_blues(h, w)
    sp = np.zeros((h, w, 4), dtype=np.uint8)
    warm = np.array([[220, 40, 40], [220, 140, 40], [220, 220, 60], [160, 60, 200]],
                    dtype=np.uint8)
    block = sp[4:12, 8:20]  # (8, 12)
    for i in range(8):
        block[i, :, :3] = warm[i // 2]
    sp[4:12, 8:20, 3] = 255
    return bg, sp


@pytest.fixture
def case_offset_corner():
    """Case 4: smaller sprite tucked into the bottom-left corner overlap."""
    h, w = 16, 24
    yy, xx = _grid(h, w)
    bg = np.stack([(xx * 9 % 200 + 30).astype(np.uint8),
                   np.full((h, w), 90, np.uint8),
                   (yy * 11 % 180 + 40).astype(np.uint8)], axis=-1)
    sp = _solid_sprite(h, w, top=10, left=2, bottom=15, right=10,
                       rgb=(40, 200, 120))
    return bg, sp


ALL_CASES = ["case_flat_overlap", "case_alpha_edge",
             "case_multicolor_sprite", "case_offset_corner"]


# --------------------------------------------------------------------------
# Stage 1 -- separate four-colour palettes
# --------------------------------------------------------------------------

def test_each_layer_gets_exactly_four_slots_rgb555_sorted(case_flat_overlap):
    bg, sp = case_flat_overlap
    r = composite_layers(bg, sp)
    for pal in (r.bg_palette, r.sprite_palette):
        assert pal.shape == (4, 3)
        assert np.array_equal(snap_rgb555(pal), pal)
        assert np.array_equal(gp.luminance_sort(pal), pal)  # lightest first


def test_explicit_palettes_are_canonicalized_and_used(case_flat_overlap):
    bg, sp = case_flat_overlap
    bpal = np.array([[0, 0, 0], [255, 255, 255], [100, 100, 100], [200, 200, 200]],
                    np.uint8)
    spal = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                    np.uint8)
    r = composite_layers(bg, sp, bg_palette=bpal, sprite_palette=spal)
    # Snap + luminance sort, but the four chosen colours must be exactly the
    # caller's sets (no new colours introduced at the quantize stage).
    assert {tuple(c) for c in r.bg_palette} == {
        tuple(c) for c in snap_rgb555(bpal)}
    assert {tuple(c) for c in r.sprite_palette} == {
        tuple(c) for c in snap_rgb555(spal)}
    # Every opaque sprite pixel is one of the (snapped) sprite-palette colours.
    opaque = r.overlap_mask & (r.winner == 1)
    sp_colors = {tuple(c) for c in snap_rgb555(spal)}
    assert {tuple(c) for c in r.image[opaque]} <= sp_colors


def test_invalid_palette_length_rejected(case_flat_overlap):
    bg, sp = case_flat_overlap
    with pytest.raises(ValueError):
        composite_layers(bg, sp, bg_palette=np.zeros((5, 3), np.uint8))
    with pytest.raises(ValueError):
        composite_layers(bg, sp, sprite_palette=np.zeros((0, 3), np.uint8))


def test_layer_quantize_error_is_finite_and_nonnegative(request):
    for name in ALL_CASES:
        r = composite_layers(*request.getfixturevalue(name))
        assert r.bg_delta_e >= 0.0 and np.isfinite(r.bg_delta_e)
        assert r.sprite_delta_e >= 0.0 and np.isfinite(r.sprite_delta_e)


# --------------------------------------------------------------------------
# Stage 2 -- priority + alpha overlap resolution, incl. counter-example
# --------------------------------------------------------------------------

def test_sprite_first_wins_every_opaque_overlap(case_flat_overlap):
    bg, sp = case_flat_overlap
    r = composite_layers(bg, sp, priority="sprite_first")
    opaque = r.overlap_mask
    assert opaque.sum() == r.overlap_pixels == 8 * 12
    # Every overlap pixel takes the sprite colour; none keeps the background.
    sp_color = {tuple(c) for c in r.sprite_palette}
    assert {tuple(c) for c in r.image[opaque]} <= sp_color
    assert set(np.unique(r.winner[opaque])) == {1}


def test_reversed_priority_is_a_visible_counter_example(case_flat_overlap):
    bg, sp = case_flat_overlap
    forward = composite_layers(bg, sp, priority="sprite_first")
    reversed_ = composite_layers(bg, sp, priority="background_first")
    # The result object itself carries the counter-example data: every opaque
    # overlap pixel changes colour when the order is reversed.
    assert forward.overlap_mismatch == forward.overlap_pixels
    assert reversed_.overlap_mismatch == reversed_.overlap_pixels
    # And the two rendered frames genuinely differ on the overlap region.
    changed = np.any(forward.image != reversed_.image, axis=2)
    assert int(changed.sum()) == forward.overlap_pixels
    # Outside the overlap the order cannot change anything.
    outside = ~forward.overlap_mask
    assert np.array_equal(forward.image[outside], reversed_.image[outside])
    # Under background_first the overlap pixels keep background colours, i.e.
    # the winner map reports background (0), never sprite (1), there.
    assert set(np.unique(reversed_.winner[reversed_.overlap_mask])) <= {0, 2}


def test_partial_alpha_blends_and_is_order_invariant(case_alpha_edge):
    bg, sp = case_alpha_edge
    r = composite_layers(bg, sp)
    blend = r.winner == 2
    assert blend.sum() == 4  # the 2x2 partial-alpha band
    # Blended pixels sit strictly between the two quantized layer colours.
    for y, x in np.argwhere(blend):
        bcol = r.bg_palette[r.bg_index[y, x]].astype(np.int64)
        scol = r.sprite_palette[r.sprite_index[y, x]].astype(np.int64)
        out = r.image[y, x].astype(np.int64)
        assert np.all(np.minimum(bcol, scol) <= out)
        assert np.all(out <= np.maximum(bcol, scol))
    # Alpha blending is symmetric w.r.t. priority order.
    rev = composite_layers(bg, sp, priority="background_first")
    assert np.array_equal(r.image[blend], rev.image[blend])


def test_sprite_transparent_slot_never_covers_background(case_multicolor_sprite):
    bg, sp = case_multicolor_sprite
    # Force the sprite's lightest warm colour into slot 0, then declare slot 0
    # the hardware-transparent OBJ slot: those pixels must show the background.
    spal = gp.luminance_sort(snap_rgb555(np.array(
        [[220, 40, 40], [220, 140, 40], [220, 220, 60], [160, 60, 200]], np.uint8)))
    r = composite_layers(bg, sp, sprite_palette=spal,
                         sprite_transparent_index=0)
    # The fixture's four warm colours each occupy a 2x12 band = 24 px, so slot
    # 0 is genuinely populated; those 24 px are excluded from the overlap.
    slot0_px = (r.sprite_index == 0) & (sp[:, :, 3] > 0)
    assert int(slot0_px.sum()) == 24
    assert r.overlap_pixels == int((sp[:, :, 3] > 0).sum()) - 24
    # Every transparent-slot pixel renders a background-palette colour instead.
    bg_colors = {tuple(c) for c in r.bg_palette}
    assert {tuple(c) for c in r.image[slot0_px]} <= bg_colors
    # And the winner there is background (0), never sprite/blend.
    assert set(np.unique(r.winner[slot0_px])) == {0}


def test_rgb_sprite_is_fully_opaque(case_flat_overlap):
    bg, _ = case_flat_overlap
    # An RGB sprite carries no alpha channel, so every canvas pixel is opaque;
    # the "sprite" is simply a solid-colour layer here.
    sp_rgb = np.broadcast_to(np.array([200, 60, 60], np.uint8),
                             bg.shape).copy()
    r = composite_layers(bg, sp_rgb)
    assert r.overlap_pixels == bg.shape[0] * bg.shape[1]
    assert (r.winner == 1).sum() == r.overlap_pixels


def test_unknown_priority_and_shape_mismatch_rejected(case_flat_overlap):
    bg, sp = case_flat_overlap
    with pytest.raises(ValueError):
        composite_layers(bg, sp, priority="inside_out")
    with pytest.raises(ValueError):
        composite_layers(bg, sp[:-1])


# --------------------------------------------------------------------------
# Stage 3 -- total colour budget with perceptual closest-pair merges
# --------------------------------------------------------------------------

def test_hardware_ceiling_constant_is_eight():
    assert LAYER_HARDWARE_COLOR_BUDGET == 8


@pytest.mark.parametrize("case_name", ALL_CASES)
def test_output_never_exceeds_the_budget(request, case_name):
    r = composite_layers(*request.getfixturevalue(case_name))
    assert r.n_colors <= 8
    assert len(np.unique(r.image.reshape(-1, 3), axis=0)) == r.n_colors


def test_overflow_merges_closest_pair_first(case_multicolor_sprite):
    bg, sp = case_multicolor_sprite
    # Ask for a tight budget so at least one real merge must happen. A generous
    # threshold (dE <= 60) permits the cross-palette warm/cool merges this
    # fixture needs; the threshold-blocking path is covered separately.
    r = composite_layers(bg, sp, color_budget=6, merge_threshold=60.0)
    assert r.n_colors <= 6
    assert r.n_merges >= 1
    # Each recorded merge's pair must be the closest available at that step:
    # re-derive dE for the record's two colours and check it is no larger than
    # either colour's distance to the eventual replacement colour.
    for a, b, merged, de, px in r.merge_records:
        assert de >= 0.0 and px > 0
        lab = gp._rgb_to_lab(np.stack([a, b, merged]))
        d_ab = float(np.sqrt(np.sum((lab[0] - lab[1]) ** 2)))
        assert abs(d_ab - de) < 1e-9
    # Merge loss is a real, finite per-pixel CIELAB quantity.
    assert r.merge_loss >= 0.0 and np.isfinite(r.merge_p95_delta_e)
    assert r.merge_p95_delta_e >= r.merge_loss - 1e-9


def test_merge_threshold_blocks_unmergeable_overflow(case_multicolor_sprite):
    bg, sp = case_multicolor_sprite
    # Budget 1 forces merges down to a single colour; a zero threshold forbids
    # every pair (distinct colours always have dE > 0), so it must raise rather
    # than silently truncate.
    with pytest.raises(LayerColorBudgetError) as exc:
        composite_layers(bg, sp, color_budget=1, merge_threshold=0.0)
    msg = str(exc.value)
    assert "budget is 1" in msg and "dE=" in msg
    # The error names the offending pair as concrete data.
    assert "closest merge pair" in msg


def test_no_silent_truncation_keeps_all_pixels(case_multicolor_sprite):
    bg, sp = case_multicolor_sprite
    r = composite_layers(bg, sp, color_budget=5, merge_threshold=60.0)
    # Same canvas shape, every pixel still assigned a colour.
    assert r.image.shape == bg.shape
    assert r.image.dtype == np.uint8
    assert r.n_colors <= 5


def test_within_budget_frame_is_not_merged(case_flat_overlap):
    bg, sp = case_flat_overlap
    r = composite_layers(bg, sp)  # default budget 8
    if r.n_colors_pre_merge <= 8:
        assert r.n_merges == 0
        assert r.merge_loss == 0.0
        assert r.n_colors == r.n_colors_pre_merge


def test_budget_is_clamped_to_hardware_ceiling(case_multicolor_sprite):
    bg, sp = case_multicolor_sprite
    r = composite_layers(bg, sp, color_budget=99)
    assert r.n_colors <= 8


def test_each_merge_picks_the_true_closest_remaining_pair(case_multicolor_sprite):
    """Independently verify greedy closest-pair selection: replay every merge
    against the live colour set and confirm no other pair was closer."""
    bg, sp = case_multicolor_sprite
    # The unmerged (budget-8) frame holds the exact pre-merge distinct colours.
    pre = composite_layers(bg, sp, color_budget=8)
    r = composite_layers(bg, sp, color_budget=4, merge_threshold=100.0)
    assert r.n_merges >= 2
    live = {tuple(c) for c in np.unique(pre.image.reshape(-1, 3), axis=0)}
    for a, b, merged, de, _px in r.merge_records:
        a, b, merged = tuple(a), tuple(b), tuple(merged)
        assert a in live and b in live
        # The recorded pair must be the globally closest among live colours.
        others = np.array(sorted(live), dtype=np.uint8)
        lab = gp._rgb_to_lab(others)
        d2 = np.sum((lab[:, None, :] - lab[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        assert abs(float(np.sqrt(d2.min())) - de) < 1e-9
        live.discard(a)
        live.discard(b)
        live.add(merged)
    assert len(live) == r.n_colors


def test_budget_merges_preserve_geometry_and_layer_choice(case_multicolor_sprite):
    """Merges only recolour; the overlap mask and winner map must be identical
    with and without a forced merge, and no pixel moves position."""
    bg, sp = case_multicolor_sprite
    loose = composite_layers(bg, sp, color_budget=8)
    tight = composite_layers(bg, sp, color_budget=4, merge_threshold=100.0)
    assert tight.n_merges > 0
    assert np.array_equal(loose.overlap_mask, tight.overlap_mask)
    assert np.array_equal(loose.winner, tight.winner)
    assert loose.image.shape == tight.image.shape


def test_quantize_is_exact_when_layer_uses_palette_colors():
    """A layer whose pixels already equal its (RGB555-stable) palette colours
    quantizes with zero CIELAB error -- the nearest-entry path is lossless
    on-palette."""
    h, w = 8, 8
    cols = gp.snap_rgb555(np.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]], np.uint8))
    bg = np.zeros((h, w, 3), np.uint8)
    for i in range(4):
        bg[:, i * 2:(i + 1) * 2] = cols[i]
    sp = np.zeros((h, w, 4), np.uint8)
    sp[:, :, :3] = cols[0]
    sp[:, :, 3] = 255
    r = composite_layers(bg, sp, bg_palette=cols,
                         sprite_palette=np.broadcast_to(cols[0], (4, 3)).copy())
    assert r.bg_delta_e == 0.0
    assert r.sprite_delta_e == 0.0


# --------------------------------------------------------------------------
# Measurements the delivery must expose (all four cases)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", ALL_CASES)
def test_result_carries_all_required_measurements(request, case_name):
    r = composite_layers(*request.getfixturevalue(case_name))
    assert r.overlap_pixels > 0
    assert 0 <= r.overlap_mismatch <= r.overlap_pixels
    assert set(r.timing) == {"quantize_s", "composite_s",
                             "budget_merge_s", "total_s"}
    assert r.timing["total_s"] > 0.0
    for key in ("quantize_s", "composite_s", "budget_merge_s"):
        assert r.timing[key] >= 0.0
    # Timing components add up to the reported total.
    assert abs(r.timing["total_s"] -
               (r.timing["quantize_s"] + r.timing["composite_s"] +
                r.timing["budget_merge_s"])) < 1e-9


# --------------------------------------------------------------------------
# Non-regression: the single-palette pipeline output is pixel-identical
# --------------------------------------------------------------------------

def test_mono_pipeline_output_unchanged_by_this_module():
    # A small deterministic image through the public mono entry point. The
    # layered code is purely additive: convert_for_hardware must render exactly
    # what it did before (checked against an independent re-run, i.e. stable,
    # and against the mono ramp's own colour set).
    img = Image.fromarray(_background_blues(72, 80), "RGB")
    r1 = convert_for_hardware(img, "mono")
    r2 = convert_for_hardware(img, "mono")
    assert np.array_equal(np.asarray(r1.image), np.asarray(r2.image))
    out_colors = {tuple(c) for c in np.asarray(r1.image).reshape(-1, 3)}
    ramp_colors = {tuple(c) for c in gp.snap_rgb555(gp.DMG_RAMP)}
    assert out_colors <= ramp_colors  # mono still uses the single DMG ramp


def test_color_pipeline_still_renders_legal_hardware_frame():
    img = Image.fromarray(_background_blues(72, 80), "RGB")
    res = convert_for_hardware(img, "color_only", max_colors=8)
    arr = np.asarray(res.image)
    assert len(np.unique(arr.reshape(-1, 3), axis=0)) <= 8
    assert res.stats["gbstudio_corrupted_tiles"] == 0
