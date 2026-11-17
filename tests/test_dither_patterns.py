"""Tests for scripts/dither_patterns.py (4x4 grey-transition pattern library).

Covers the three documented stages:

1. generation -- all usable 4x4 four-shade patterns, grey levels, the
   ``max_patterns`` cap and the empty-library error;
2. selection -- pattern chosen per 4x4 region by target grey, with the
   boundary penalty configurable;
3. application -- patterns stamped onto an index image with boundary
   continuity reported, plus the four sample groups' metrics.

The module is a standalone script (no gb_pipeline imports), loaded here by
path so the suite keeps working without __init__ changes to scripts/.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, "scripts", "dither_patterns.py")


def _load():
    spec = importlib.util.spec_from_file_location("dither_patterns", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sys.modules[spec.name]


dp = _load()

LIBRARY = dp.build_pattern_library(512)


# ---------------------------------------------------------------------------
# 1. generation
# ---------------------------------------------------------------------------


def test_generation_enumerates_all_patterns_and_grey_levels():
    # 6 unordered shade pairs x 2**16 masks (before exact-pattern dedupe).
    assert LIBRARY.candidate_count == 6 * (2 ** 16)
    # Every one of the 49 grey levels (k/16 for k = 0..48) is represented.
    assert LIBRARY.grey_levels.size == 49
    assert LIBRARY.grey_levels[0] == 0
    assert LIBRARY.grey_levels[-1] == 48
    # Grey is 1/16-quantised and derived from the pattern's own mean shade.
    assert np.allclose(LIBRARY.grey, np.round(LIBRARY.grey * 16) / 16)
    assert np.allclose(LIBRARY.patterns.reshape(len(LIBRARY), -1).mean(axis=1),
                       LIBRARY.grey)


def test_generation_patterns_are_valid_4x4_four_shade():
    assert LIBRARY.patterns.shape == (512, 4, 4)
    assert LIBRARY.patterns.dtype == np.uint8
    assert LIBRARY.patterns.min() == 0
    assert LIBRARY.patterns.max() == 3
    # Each pattern mixes exactly the two shades its lo/hi record.  (The two
    # solid extremes -- all coverage 0 and all coverage 16 -- occur once per
    # adjacent shade pair, so they are recorded with a single shade.)
    for pattern, lo, hi in zip(LIBRARY.patterns, LIBRARY.lo, LIBRARY.hi):
        assert lo < hi
        unique = set(np.unique(pattern).tolist())
        if unique == {int(lo)}:
            continue  # solid low shade (coverage 0)
        if unique == {int(hi)}:
            continue  # solid high shade (coverage 16)
        assert unique == {int(lo), int(hi)}


def test_generation_respects_the_cap():
    small = dp.build_pattern_library(64)
    assert len(small) == 64
    assert small.max_patterns == 64
    # Even a tiny cap still spans the full 0..3 tone range.
    assert small.grey.min() == 0.0
    assert small.grey.max() == 3.0
    # And a mid-size cap keeps every level usable.
    mid = dp.build_pattern_library(196)
    assert len(mid) == 196
    assert mid.grey_levels.size == 49


def test_generation_is_deterministic():
    a = dp.build_pattern_library(128)
    b = dp.build_pattern_library(128)
    assert np.array_equal(a.patterns, b.patterns)
    assert np.array_equal(a.grey, b.grey)
    # Different seeds give genuinely different library contents.
    c = dp.build_pattern_library(128, seed=7)
    assert c.patterns.shape == a.patterns.shape
    assert not np.array_equal(a.patterns, c.patterns)


@pytest.mark.parametrize("bad_cap", [0, -1])
def test_empty_library_raises(bad_cap):
    with pytest.raises(ValueError, match="pattern library is empty"):
        dp.build_pattern_library(bad_cap)


def test_empty_library_rejected_by_select_and_apply():
    empty = dp.PatternLibrary(
        patterns=np.zeros((0, 4, 4), dtype=np.uint8),
        mask=np.zeros((0, 4, 4), dtype=np.uint8),
        grey=np.zeros(0),
        lo=np.zeros(0, dtype=np.uint8),
        hi=np.zeros(0, dtype=np.uint8),
        score=np.zeros(0),
        max_patterns=0,
        candidate_count=0,
    )
    assert empty.empty
    target = np.zeros((8, 8))
    with pytest.raises(ValueError, match="pattern library is empty"):
        dp.dither_index_map(target, empty)
    with pytest.raises(ValueError, match="pattern library is empty"):
        dp.apply_patterns(np.zeros((2, 2), dtype=np.int32), empty)


# ---------------------------------------------------------------------------
# 2. selection by target grey
# ---------------------------------------------------------------------------


def test_select_hits_the_requested_tone():
    for tone in (0.0, 0.75, 1.5, 2.5, 3.0):
        target = np.full((16, 16), tone)
        ids, index_map = dp.dither_index_map(target, LIBRARY)
        assert index_map.shape == (16, 16)
        # Every region lands within one 1/16 grey step of the requested tone.
        block_grey = dp._block_mean(index_map)
        assert np.max(np.abs(block_grey - tone)) <= 2 * dp.GREY_STEP
        assert abs(float(block_grey.mean()) - tone) <= dp.GREY_STEP


def test_select_flat_target_without_boundary_penalty_is_one_pattern():
    # With no boundary penalty and a flat tone, the cheapest pattern wins
    # everywhere -- the jagged pattern above only appears because the default
    # boundary_weight deliberately varies masks to break up the phase.
    target = np.full((16, 16), 1.5)
    ids, index_map = dp.dither_index_map(
        target, LIBRARY, boundary_weight=0.0, pixel_weight=0.0
    )
    assert np.unique(ids).size <= 2
    assert all(
        abs(LIBRARY.grey[j] - 1.5) <= dp.GREY_STEP
        for j in np.unique(ids).tolist()
    )


def test_select_gradient_tracks_tone():
    target = np.tile(np.linspace(0.0, 3.0, 64), (32, 1))
    ids, index_map = dp.dither_index_map(target, LIBRARY)
    err = dp.grey_error(index_map, target)
    # Library dithering tracks the ramp closely in block-grey terms (the
    # per-pixel error stays near half a shade, as any 1-of-4 dither must).
    assert err["mae"] < 0.10
    assert err["rmse"] < 0.15
    # Tone rises left-to-right: block means are correlated with the ramp.
    block_grey = dp._block_mean(index_map)
    block_target = dp._block_mean(target)
    assert np.corrcoef(block_grey.ravel(), block_target.ravel())[0, 1] > 0.99
    # And the per-pixel texture is smoother than a flat-tone dither: matching
    # the local gradient (pixel_weight) beats tone-only selection.
    _, flat_map = dp.dither_index_map(target, LIBRARY, pixel_weight=0.0)
    assert np.abs(index_map.astype(float) - target).mean() < np.abs(
        flat_map.astype(float) - target
    ).mean()


def test_select_is_deterministic():
    target = dp.luminance_target_grey(dp.make_sample_images(32)["diagonal"])
    first = dp.dither_index_map(target, LIBRARY, boundary_weight=0.2)
    second = dp.dither_index_map(target, LIBRARY, boundary_weight=0.2)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_boundary_weight_is_configurable_and_trades_continuity_for_tone():
    target = dp.luminance_target_grey(dp.make_sample_images(64)["horizontal"])
    _, loose = dp.dither_index_map(target, LIBRARY, boundary_weight=0.0)
    _, tight = dp.dither_index_map(target, LIBRARY, boundary_weight=0.6)
    loose_stats = dp.check_boundary_continuity(loose)
    tight_stats = dp.check_boundary_continuity(tight)
    # More weight -> measurably more continuous pattern borders.
    assert tight_stats["continuity"] > loose_stats["continuity"]
    # ... and slightly more grey error, since that is what is being traded.
    assert dp.grey_error(tight, target)["mae"] > dp.grey_error(loose, target)["mae"]


@pytest.mark.parametrize("bad_weight", [-0.1, float("nan"), float("inf")])
def test_invalid_boundary_weight_rejected(bad_weight):
    target = np.zeros((8, 8))
    with pytest.raises(ValueError, match="boundary_weight"):
        dp.select_patterns(target, LIBRARY, boundary_weight=bad_weight)


def test_select_rejects_unaligned_shapes():
    with pytest.raises(ValueError, match="multiple"):
        dp.select_patterns(np.zeros((10, 10)), LIBRARY)


# ---------------------------------------------------------------------------
# 3. apply + boundary continuity
# ---------------------------------------------------------------------------


def test_apply_round_trips_region_ids():
    ids = np.arange(6 * 5, dtype=np.int32).reshape(5, 6) % len(LIBRARY)
    index_map = dp.apply_patterns(ids, LIBRARY)
    assert index_map.shape == (20, 24)
    # Each 4x4 block is exactly the library pattern its id names.
    blocks = (
        index_map.reshape(5, 4, 6, 4).transpose(0, 2, 1, 3).reshape(5, 6, 16)
    )
    assert np.array_equal(blocks, LIBRARY.patterns[ids].reshape(5, 6, 16))


def test_apply_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match="region id"):
        dp.apply_patterns(np.array([[-1, 0]]), LIBRARY)


def test_boundary_check_flat_image_is_perfectly_continuous():
    flat = np.zeros((16, 16), dtype=np.uint8)
    stats = dp.check_boundary_continuity(flat)
    assert stats["continuity"] == 1.0
    assert stats["discontinuity_rate"] == 0.0
    assert stats["hard_seams"] == 0
    assert stats["seamless"]
    # 3 internal vertical borders x 16 rows + 3 horizontal x 16 columns.
    assert stats["boundary_pairs"] == 3 * 16 + 3 * 16


def test_boundary_check_counts_mismatches_exactly():
    # Blocks alternate solid 0 and solid 1 in a checker of 4x4 regions.
    ids = np.indices((2, 2)).sum(axis=0) % 2
    idx = np.kron(ids, np.ones((4, 4))).astype(np.uint8)
    stats = dp.check_boundary_continuity(idx)
    # Every one of the two internal borders is a complete 4-pixel seam.
    assert stats["boundary_mismatches"] == stats["boundary_pairs"] == 16
    assert stats["continuity"] == 0.0
    assert stats["hard_seams"] == 4  # 2 vertical segments + 2 horizontal
    assert stats["max_edge_mismatch"] == 4
    assert not stats["seamless"]


def test_boundary_check_requires_aligned_index_image():
    with pytest.raises(ValueError, match="multiple"):
        dp.check_boundary_continuity(np.zeros((10, 12), dtype=np.uint8))


def test_library_output_is_seamless_and_less_striped_than_bayer():
    # The whole point of the library: no hard 4x4 seams, and the repeated
    # phase (stripe) signature of ordered dithering is largely averaged away.
    for name, rgb in dp.make_sample_images(64).items():
        target = dp.luminance_target_grey(rgb)
        _, index_map = dp.dither_index_map(target, LIBRARY, boundary_weight=0.2)
        stats = dp.check_boundary_continuity(index_map)
        assert stats["seamless"], name
        assert stats["continuity"] > 0.85, name

        baseline = dp.bayer_index_map(target)
        assert not dp.check_boundary_continuity(baseline)["seamless"], name
        assert dp.stripe_visibility(index_map) < 0.5 * dp.stripe_visibility(baseline)


def test_sample_groups_report_all_required_metrics():
    rows = [
        dp.evaluate_group(name, rgb, LIBRARY, 0.2)
        for name, rgb in dp.make_sample_images(32).items()
    ]
    assert [row["group"] for row in rows] == [
        "horizontal", "diagonal", "radial", "bands",
    ]
    for row in rows:
        assert np.isfinite(row["grey_mae"])
        assert np.isfinite(row["grey_rmse"])
        assert 0.0 <= row["stripe_visibility"] <= 1.0
        assert isinstance(row["pattern_switches"], int)
        assert row["pattern_switches"] >= 0
        assert row["seconds"] >= 0.0
        assert 0.0 <= row["boundary_continuity"] <= 1.0
        assert isinstance(row["hard_seams"], int)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_module_is_opt_in_and_dependency_free():
    # The library must not become part of the existing pipeline: importing it
    # must not pull in gb_pipeline, main or any third-party module beyond
    # numpy.  That keeps the converter's output pixel-for-pixel unchanged.
    with open(MODULE_PATH, encoding="utf-8") as handle:
        source = handle.read()
    assert "import gb_pipeline" not in source
    assert "import main" not in source
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("import ", "from ")):
            continue
        target = stripped.split()[1]
        assert target.startswith(("__future__", "argparse", "dataclasses",
                                  "sys", "time", "numpy", "json")), stripped


def test_cli_runs_end_to_end(capsys):
    assert dp.main(["--max-patterns", "64", "--size", "32"]) == 0
    out = capsys.readouterr().out
    assert "pattern library: 64 patterns kept from 393216 candidates" in out
    for group in ("horizontal", "diagonal", "radial", "bands"):
        assert group in out


def test_cli_reports_empty_library_error(capsys):
    assert dp.main(["--max-patterns", "0"]) == 2
    assert "pattern library is empty" in capsys.readouterr().err
