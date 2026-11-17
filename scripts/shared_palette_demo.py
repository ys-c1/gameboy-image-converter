"""Shared 4-color palette solver demo on four small hand-built image sets.

Run (from the repo root):
    python scripts/shared_palette_demo.py

For each image set it prints, for both seeding algorithms and a configurable
iteration count: the pooled total perceptual error (weighted sum of pixel
dE), each image's own error and its mean palette-level difference from the
shared palette, the mapping-table size per image, and the solve time. The
empty-set error path is demonstrated last. No network or third-party data:
every image is synthesized below with numpy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import gb_pipeline as gbp


def _quadrants(colors, size=8):
    """An 8x8 (default) image split into four solid quadrants."""
    h = w = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, : w // 2] = colors[0]
    arr[: h // 2, w // 2:] = colors[1]
    arr[h // 2:, : w // 2] = colors[2]
    arr[h // 2:, w // 2:] = colors[3]
    return arr


def _ramp_image(rgb_ramp, size=8):
    """Horizontal 4-step ramp from four RGB colors."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i, color in enumerate(rgb_ramp):
        lo = i * size // 4
        hi = (i + 1) * size // 4
        arr[:, lo:hi] = color
    return arr


def build_image_sets():
    """Four deliberately small sets that stress different conflict shapes."""
    # Set 1: identical palettes -- the shared fit should be near lossless.
    warm_ramp = gbp.snap_rgb555(np.array(
        [(240, 230, 200), (220, 160, 90), (160, 80, 40), (40, 20, 10)],
        dtype=np.uint8,
    ))
    set1 = [_ramp_image(warm_ramp), _quadrants(warm_ramp)]

    # Set 2: warm vs. cool palettes -- hue conflict across images.
    cool_ramp = gbp.snap_rgb555(np.array(
        [(200, 220, 240), (90, 140, 220), (40, 70, 160), (10, 15, 60)],
        dtype=np.uint8,
    ))
    set2 = [_ramp_image(warm_ramp), _ramp_image(cool_ramp)]

    # Set 3: gray ramp vs. saturated RGB triplet -- lightness vs. chroma fight.
    gray_ramp = gbp.snap_rgb555(np.array(
        [(240, 240, 240), (170, 170, 170), (90, 90, 90), (20, 20, 20)],
        dtype=np.uint8,
    ))
    rgb_quads = gbp.snap_rgb555(np.array(
        [(230, 30, 30), (30, 230, 30), (30, 30, 230), (230, 230, 30)],
        dtype=np.uint8,
    ))
    set3 = [_ramp_image(gray_ramp), _quadrants(rgb_quads)]

    # Set 4: three images, near-duplicate shifted ramps -- the classic
    # "each image got its own 4 colors, merged they collide" case.
    shifted_a = gbp.snap_rgb555(np.array(
        [(220, 60, 60), (180, 100, 40), (120, 60, 90), (40, 20, 30)],
        dtype=np.uint8,
    ))
    shifted_b = gbp.snap_rgb555(np.array(
        [(224, 64, 56), (176, 104, 44), (124, 64, 88), (44, 24, 28)],
        dtype=np.uint8,
    ))
    shifted_c = gbp.snap_rgb555(np.array(
        [(216, 56, 64), (184, 96, 36), (116, 64, 96), (36, 24, 32)],
        dtype=np.uint8,
    ))
    set4 = [_ramp_image(shifted_a), _ramp_image(shifted_b), _quadrants(shifted_c)]

    return [
        ("identical warm ramps", set1),
        ("warm vs cool", set2),
        ("gray vs saturated RGB", set3),
        ("three near-duplicate ramps", set4),
    ]


def _hex(color):
    return "#{:02X}{:02X}{:02X}".format(*(int(v) for v in color))


def report(name, images, algorithm, n_iter):
    res = gbp.solve_shared_palette(images, algorithm=algorithm, n_iter=n_iter)
    print(f"  [{algorithm:9s} n_iter={n_iter:2d}] shared palette: "
          + " ".join(_hex(c) for c in res.palette))
    print(f"    total color difference (weighted pixel dE): {res.total_error:10.3f}")
    for i, (err, diff, table) in enumerate(
        zip(res.per_image_error, res.palette_diff, res.mappings)
    ):
        print(f"    image {i}: pixel dE={err:9.3f}  own-palette diff (mean dE)="
              f"{diff:6.3f}  mapping table size={len(table)}")
    print(f"    solve time: {res.solve_seconds * 1000:7.2f} ms")
    return res


def main():
    n_iter = 10
    print("=" * 78)
    print("Shared 4-color palette solver -- four image sets, both algorithms")
    print("=" * 78)
    for name, images in build_image_sets():
        print(f"\n* image set: {name} ({len(images)} images, "
              f"{images[0].shape[1]}x{images[0].shape[0]} each)")
        for algorithm in ("greedy", "mediancut"):
            report(name, images, algorithm, n_iter)

    print("\n" + "-" * 78)
    print("empty image set must raise:")
    try:
        gbp.solve_shared_palette([])
    except ValueError as exc:
        print(f"  ValueError: {exc}")
    print("-" * 78)
    print("done.")


if __name__ == "__main__":
    main()
