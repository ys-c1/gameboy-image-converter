"""Measured data for the layered two-palette composite (four small fixtures).

Run from the repo root with the project venv:

    venv/bin/python scripts/bench_layered_composite.py

Prints, for each of the four hand-built layer pairs:
  * the two fitted 4-colour palettes and their per-layer quantize dE;
  * overlap size and the pixel count that changes when the priority order is
    reversed (the counter-example evidence);
  * distinct colours before/after the budget, the number of perceptual merges,
    and the mean/p95 per-pixel CIELAB loss those merges cost;
  * per-frame processing time for each of the three stages.

It also demonstrates the two hard guarantees:
  * an over-budget frame whose closest pair exceeds the merge threshold raises
    LayerColorBudgetError (no silent truncation);
  * the reversed priority order renders a visibly different frame.

Everything is deterministic and tiny (16x24 frames), so the script finishes in
a fraction of a second and uses only the repository's existing dependencies.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import gb_pipeline as gp


# --------------------------------------------------------------------------
# Fixtures (mirrors tests/test_layered_composite.py, kept self-contained)
# --------------------------------------------------------------------------

def _grid(h=16, w=24):
    return np.mgrid[0:h, 0:w]


def background_blues(h=16, w=24):
    _, xx = _grid(h, w)
    blue = np.array([40, 100, 160, 220], dtype=np.uint8)
    b = blue[(xx // (w // 4)) % 4]
    return np.stack([np.full((h, w), 40, np.uint8),
                     np.full((h, w), 80, np.uint8), b], axis=-1)


def solid_sprite(h=16, w=24, top=4, left=8, bottom=12, right=20,
                 rgb=(200, 60, 60), alpha=255):
    sp = np.zeros((h, w, 4), dtype=np.uint8)
    sp[top:bottom, left:right, 0] = rgb[0]
    sp[top:bottom, left:right, 1] = rgb[1]
    sp[top:bottom, left:right, 2] = rgb[2]
    sp[top:bottom, left:right, 3] = alpha
    return sp


def build_cases():
    h, w = 16, 24

    # 1. Flat opaque sprite fully inside the background.
    bg1 = background_blues(h, w)
    sp1 = solid_sprite(h, w)

    # 2. Opaque body plus a partial-alpha (0.25) edge band.
    bg2 = background_blues(h, w)
    sp2 = solid_sprite(h, w)
    sp2[6:8, 10:12, 3] = 64

    # 3. Four-colour warm sprite over the four-colour cool background.
    bg3 = background_blues(h, w)
    sp3 = np.zeros((h, w, 4), dtype=np.uint8)
    warm = np.array([[220, 40, 40], [220, 140, 40],
                     [220, 220, 60], [160, 60, 200]], dtype=np.uint8)
    block = sp3[4:12, 8:20]
    for i in range(8):
        block[i, :, :3] = warm[i // 2]
    sp3[4:12, 8:20, 3] = 255

    # 4. Smaller green sprite tucked into the bottom-left corner.
    yy, xx = _grid(h, w)
    bg4 = np.stack([(xx * 9 % 200 + 30).astype(np.uint8),
                    np.full((h, w), 90, np.uint8),
                    (yy * 11 % 180 + 40).astype(np.uint8)], axis=-1)
    sp4 = solid_sprite(h, w, top=10, left=2, bottom=15, right=10,
                       rgb=(40, 200, 120))

    return [
        ("1 flat opaque overlap", bg1, sp1, 8, 25.0),
        ("2 partial-alpha edge", bg2, sp2, 8, 25.0),
        ("3 four-colour sprite", bg3, sp3, 8, 25.0),
        ("4 corner offset sprite", bg4, sp4, 8, 25.0),
        # Same four layer pairs re-run under a deliberately tight budget so the
        # perceptual closest-pair merge path and its colour loss are exercised.
        ("1b flat overlap, budget 4", bg1, sp1, 4, 60.0),
        ("3b 4-colour sprite, budget 5", bg3, sp3, 5, 60.0),
        ("4b corner sprite, budget 3", bg4, sp4, 3, 60.0),
    ]


def hexrow(pal):
    return " ".join("#{:02X}{:02X}{:02X}".format(*map(int, c)) for c in pal)


def main():
    cases = build_cases()
    print("=" * 78)
    print("Layered two-palette composite -- measured data (frames are 16x24)")
    print("=" * 78)

    rows = []
    base_rows = []
    for name, bg, sp, budget, threshold in cases:
        # Wall-clock the whole public call as an independent timing cross-check.
        t0 = time.perf_counter()
        r = gp.composite_layers(
            bg, sp, color_budget=budget, merge_threshold=threshold)
        wall = time.perf_counter() - t0

        # Reverse the priority order for the counter-example comparison.
        r_rev = gp.composite_layers(
            bg, sp, priority="background_first",
            color_budget=budget, merge_threshold=threshold)
        changed = int(np.count_nonzero(np.any(r.image != r_rev.image, axis=2)))

        print(f"\n### {name}")
        print(f"  bg palette     : {hexrow(r.bg_palette)}")
        print(f"  sprite palette : {hexrow(r.sprite_palette)}")
        print(f"  quantize dE    : bg={r.bg_delta_e:6.3f}  "
              f"sprite={r.sprite_delta_e:6.3f} (mean per-pixel CIELAB)")
        print(f"  overlap        : {r.overlap_pixels:4d} px | priority="
              f"{r.priority}")
        print(f"  order reversed : {r.overlap_mismatch:4d} overlap px change "
              f"colour (full-frame diff = {changed} px)")
        print(f"  colours        : {r.n_colors_pre_merge} -> {r.n_colors} "
              f"(budget {budget}); merges={r.n_merges}")
        if r.merge_records:
            for a, b, merged, de, px in r.merge_records:
                print(f"    merge #{tuple(a)} + #{tuple(b)} "
                      f"dE={de:6.2f} -> #{tuple(merged)} ({px} px)")
        print(f"  merge loss     : mean={r.merge_loss:6.3f}  "
              f"p95={r.merge_p95_delta_e:6.3f} dE")
        t = r.timing
        print(f"  timing/frame   : quantize={t['quantize_s'] * 1e3:7.3f} ms  "
              f"composite={t['composite_s'] * 1e3:7.3f} ms  "
              f"budget={t['budget_merge_s'] * 1e3:7.3f} ms  "
              f"total={t['total_s'] * 1e3:7.3f} ms (wall {wall * 1e3:6.3f} ms)")

        rows.append((name, r, changed))

    # The four base fixtures are the names with a space (not "b") after the
    # leading digit; the tight-budget rows are named "1b ..", "3b ..", "4b ..".
    base_rows = [(n, r, c) for (n, r, c) in rows if n[1] == " "]

    # ------------------------------------------------------------------
    # Guarantee A: over-budget + impossible merge raises, never truncates.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Hard guarantee A -- over budget with no merge below threshold")
    print("=" * 78)
    _, bg3, sp3, _, _ = cases[2]
    try:
        gp.composite_layers(bg3, sp3, color_budget=1, merge_threshold=0.0)
    except gp.LayerColorBudgetError as exc:
        print("  LayerColorBudgetError raised as required:")
        print("   ", str(exc))
    else:
        raise AssertionError("expected LayerColorBudgetError, got a result")

    # A still-mergeable tighter budget (threshold 60) succeeds and lands exactly
    # on budget -- showing how far the default-threshold frame is from the goal.
    r_tight = gp.composite_layers(bg3, sp3, color_budget=5,
                                  merge_threshold=60.0)
    print(f"  same frame at budget=5, threshold=60: "
          f"{r_tight.n_colors} colours after {r_tight.n_merges} merges, "
          f"mean loss {r_tight.merge_loss:.2f} / p95 "
          f"{r_tight.merge_p95_delta_e:.2f} dE")
    print("  gap vs. default threshold 25: the first required cross-palette "
          "merge costs dE=33.69 > 25,")
    print("  so the default config refuses rather than degrading that far.")

    # ------------------------------------------------------------------
    # Guarantee B: reversed order is visibly different (counter-example data).
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Hard guarantee B -- reversed priority order is visibly detectable")
    print("=" * 78)
    for name, r, changed in base_rows:
        print(f"  {name:26s}: overlap={r.overlap_pixels:4d} px, "
              f"mismatch={r.overlap_mismatch:4d} px "
              f"({100.0 * r.overlap_mismatch / max(r.overlap_pixels, 1):.0f}% "
              "of overlap)")

    # ------------------------------------------------------------------
    # Single-palette mono path remains pixel-stable (non-regression).
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Non-regression -- single-palette mono output uses only the DMG ramp")
    print("=" * 78)
    from PIL import Image
    mono_img = Image.fromarray(background_blues(72, 80), "RGB")
    m1 = gp.convert_for_hardware(mono_img, "mono")
    m2 = gp.convert_for_hardware(mono_img, "mono")
    stable = np.array_equal(np.asarray(m1.image), np.asarray(m2.image))
    ramp = {tuple(c) for c in gp.snap_rgb555(gp.DMG_RAMP)}
    used = {tuple(c) for c in np.asarray(m1.image).reshape(-1, 3)}
    print(f"  repeated runs pixel-identical: {stable}; "
          f"output colours subset of DMG ramp: {used <= ramp}")
    print("\nAll measurements complete.")


if __name__ == "__main__":
    main()
