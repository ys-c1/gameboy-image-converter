"""Benchmark the cartridge tile packers against naive byte storage.

Run: /home/ys/.cache/gb-venv/bin/python scripts/bench_packing.py

Builds four small synthetic sample tilesets (flat / checker / gradient /
random), packs each to 2bpp and 1bpp, verifies decode consistency, and
reports byte counts against the naive "one uint8 byte per pixel" storage.

No external data or network access: the samples are generated here.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import gb_pipeline as gp


def samples():
    """Four tiny deterministic (n, 8, 8) index tilesets.

    Each group is a small but non-trivial tileset: flat, checker, gradient and
    random, with different tile counts so the byte totals are not all equal.
    """
    yy, xx = np.mgrid[0:8, 0:8]

    flat = np.repeat(np.array([0, 1, 2, 3], np.uint8), 4)[:, None, None]
    flat = np.broadcast_to(flat, (16, 8, 8)).copy()

    checker = np.empty((24, 8, 8), dtype=np.uint8)
    for i in range(24):
        checker[i] = (((xx + yy + i) % 2) * (i % 4)).astype(np.uint8)

    grad = ((xx // 2) + (yy // 4)).astype(np.uint8) % 4
    gradient = np.stack(
        [
            np.rot90(grad, k) if k else grad.copy()
            for k in range(4) for _ in range(9)
        ]
    )

    rng = np.random.RandomState(0)
    random_tiles = rng.randint(0, 4, size=(64, 8, 8)).astype(np.uint8)

    return [
        ("flat", flat),
        ("checker", checker),
        ("gradient", gradient),
        ("random", random_tiles),
    ]


def time_pack(fn, patterns, repeats=200):
    start = time.perf_counter()
    for _ in range(repeats):
        fn(patterns)
    return (time.perf_counter() - start) / repeats


def main():
    rows = []
    total_2bpp = total_1bpp = total_naive = 0

    print(
        f"{'sample':10s} {'tiles':>5s} {'naive B':>8s} {'2bpp B':>8s} "
        f"{'ratio':>6s} {'1bpp B':>8s} {'2bpp us':>8s} {'1bpp us':>8s} "
        f"{'roundtrip':>9s}"
    )
    for name, patterns in samples():
        n = patterns.shape[0]
        naive = patterns.nbytes  # one uint8 per pixel: 64 bytes per tile
        data2 = gp.pack_tiles_2bpp(patterns)
        data1 = gp.pack_tiles_1bpp((patterns % 2).astype(np.uint8))

        ok2 = np.array_equal(gp.unpack_tiles_2bpp(data2), patterns)
        ok1 = np.array_equal(
            gp.unpack_tiles_1bpp(data1), (patterns % 2).astype(np.uint8)
        )

        us2 = time_pack(gp.pack_tiles_2bpp, patterns) * 1e6
        us1 = time_pack(gp.pack_tiles_1bpp, (patterns % 2).astype(np.uint8)) * 1e6

        total_naive += naive
        total_2bpp += len(data2)
        total_1bpp += len(data1)
        print(
            f"{name:10s} {n:5d} {naive:8d} {len(data2):8d} "
            f"{naive / len(data2):5.1f}x {len(data1):8d} {us2:8.1f} {us1:8.1f} "
            f"{'ok' if ok2 and ok1 else 'FAIL':>9s}"
        )
        rows.append((name, n, naive, len(data2), len(data1), ok2 and ok1, us2, us1))

    print(
        f"{'TOTAL':10s} {'':5s} {total_naive:8d} {total_2bpp:8d} "
        f"{total_naive / total_2bpp:5.1f}x {total_1bpp:8d}"
    )
    print(
        f"2bpp savings vs naive byte storage: "
        f"{100 * (1 - total_2bpp / total_naive):.0f}% "
        f"({total_naive} -> {total_2bpp} bytes)"
    )
    if not all(r[5] for r in rows):
        raise SystemExit("round-trip failure")


if __name__ == "__main__":
    main()
