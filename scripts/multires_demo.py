"""Multi-resolution export demo -- one small asset, four target sizes.

Run:  venv/bin/python scripts/multires_demo.py

Builds a single tiny in-memory asset (no files, no network), exports it at
four target sizes with the configurable alignment strategies, and prints the
tile count, alignment-padding pixel ratio and round-trip reconstruction
status for each, followed by the illegal-size error behavior.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

import gb_pipeline as gp

TARGETS = [
    (32, 24, "exact-aligned"),
    (30, 22, "pad both axes"),
    (35, 27, "pad/crop both axes"),
    (13, 11, "odd small target"),
]


def make_asset(width=40, height=36, seed=7):
    """Tiny bordered noisy asset: a uniform 1-px border around seeded noise."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    interior = rng.randint(0, 256, size=(height - 2, width - 2, 3)).astype(np.uint8)
    arr[1:-1, 1:-1] = interior
    return Image.fromarray(arr, "RGB")


def main():
    asset = make_asset()
    print(f"source asset: {asset.size[0]}x{asset.size[1]} px\n")
    print(f"{'target':>9} {'strategy':>8} {'canvas':>9} {'tiles':>6} "
          f"{'cells':>6} {'pad%':>7}  reconstruct")
    for (w, h), note in [(t[:2], t[2]) for t in TARGETS]:
        for strategy in ("pad", "crop"):
            res = gp.multires_export(asset, (w, h), align=strategy,
                                     resample="nearest", crop_edges=False)
            rebuilt = gp.reconstruct_multires(res)
            ok = np.array_equal(rebuilt, np.asarray(res.image))
            cw, ch = res.canvas_size
            print(f"{w:>4}x{h:<4} {strategy:>8} {cw:>4}x{ch:<4} "
                  f"{res.tileset.shape[0]:>6} {len(res.map_offsets):>6} "
                  f"{res.pad_ratio * 100:>6.2f}%  {'OK' if ok else 'FAIL'}"
                  f"   # {note}")

    print("\nillegal target sizes:")
    for bad in [(0, 24), (32.0, 24), (32,), "32x24", (-8, 9), (True, 8)]:
        try:
            gp.multires_export(asset, bad)
        except (TypeError, ValueError) as exc:
            print(f"  {bad!r:>12} -> {type(exc).__name__}: {exc}")
        else:
            raise SystemExit(f"expected an error for {bad!r}")


if __name__ == "__main__":
    main()
