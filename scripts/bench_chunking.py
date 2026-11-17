"""Map-chunking benchmark: chunk counts, peak resident memory, streaming load
timing, and pixel-level consistency versus loading the whole map at once.

Four small, self-made deterministic maps (no external assets, no network):
two GB-tile-aligned sizes and two sizes with partial edge chunks, each run
through the real convert_for_hardware render and then streamed through a
ChunkStreamer sweeping a viewport across the whole map.

Run:  python scripts/bench_chunking.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

import gb_pipeline
from gb_pipeline import ChunkStreamer, build_chunk_index, stitch_chunks

# Small on purpose: the whole sweep finishes in a couple of seconds and the
# numbers stay directly readable. (w, h, preset, chunk_px, viewport_px, radius)
MAPS = [
    ("small-aligned",   96,  80, "color_only", 32, 24, 0),
    ("wide-partial",   120,  72, "color_only", 32, 24, 0),
    ("tall-partial",    72, 104, "mono",       32, 24, 1),
    ("large-aligned",  160, 144, "mono",       40, 32, 1),
]


def _make_map(width: int, height: int, seed: int) -> Image.Image:
    """Deterministic coordinate-encoded map with regions + mild noise."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    r = (xx * 255 // max(width - 1, 1)).astype(np.int16)
    g = (yy * 255 // max(height - 1, 1)).astype(np.int16)
    b = (((xx + yy) * 2) % 256).astype(np.int16)
    arr = np.stack([r, g, b], axis=-1) + rng.randint(-8, 9, (height, width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


class _MapLoader:
    """Slice chunks out of the in-RAM render, standing in for a disk backend."""

    def __init__(self, source: np.ndarray):
        self.source = source

    def __call__(self, info):
        return self.source[info.y0:info.y1, info.x0:info.x1].copy()


def benchmark_one(name, width, height, preset, chunk_px, viewport_px, radius, seed):
    image = _make_map(width, height, seed)

    t0 = time.perf_counter()
    result = gb_pipeline.convert_for_hardware(image, preset, dither="bayer")
    convert_seconds = time.perf_counter() - t0
    whole = np.asarray(result.image, dtype=np.uint8)
    full_bytes = whole.nbytes

    idx = build_chunk_index(whole.shape[1], whole.shape[0], chunk_px, chunk_px)
    loader = _MapLoader(whole)
    streamer = ChunkStreamer(idx, loader, preload_radius=radius)

    # Whole-image reference: one contiguous load, timed.
    t0 = time.perf_counter()
    direct = whole.copy()
    whole_load_seconds = time.perf_counter() - t0

    # Streaming: sweep the viewport top-to-bottom, left-to-right one chunk at
    # a time, exercising preload, eviction and viewport assembly throughout.
    mismatches = 0
    t0 = time.perf_counter()
    vw = min(viewport_px, whole.shape[1])
    vh = min(viewport_px, whole.shape[0])
    for y in range(0, whole.shape[0] - vh + 1, chunk_px):
        for x in range(0, whole.shape[1] - vw + 1, chunk_px):
            crop = streamer.get_viewport_pixels(x, y, vw, vh)
            if not np.array_equal(crop, whole[y:y + vh, x:x + vw]):
                mismatches += 1
    stream_seconds = time.perf_counter() - t0

    # Final consistency guarantee: every chunk stitched back together equals
    # the whole-image render pixel-for-pixel.
    chunks = {
        (info.row, info.col): loader(info) for info in idx.chunks
    }
    stitched = stitch_chunks(idx, chunks)
    stitched_ok = np.array_equal(stitched, whole)

    m = streamer.metrics
    return {
        "name": name,
        "map": f"{whole.shape[1]}x{whole.shape[0]}",
        "preset": preset,
        "radius": radius,
        "chunk_grid": f"{idx.cols}x{idx.rows}",
        "n_chunks": len(idx.chunks),
        "full_load_bytes": full_bytes,
        "peak_stream_bytes": m["peak_resident_bytes"],
        "memory_saving_pct": 100.0 * (1.0 - m["peak_resident_bytes"] / full_bytes),
        "loads": m["loads"],
        "unloads": m["unloads"],
        "whole_load_ms": whole_load_seconds * 1000.0,
        "stream_total_ms": stream_seconds * 1000.0,
        "stream_load_ms": m["load_seconds"] * 1000.0,
        "convert_ms": convert_seconds * 1000.0,
        "mismatched_viewports": mismatches,
        "stitched_identical": stitched_ok,
    }


def main():
    rows = [
        benchmark_one(name, w, h, preset, cp, vp, radius, seed=100 + i)
        for i, (name, w, h, preset, cp, vp, radius) in enumerate(MAPS)
    ]

    hdr = (
        f"{'map':<14}{'size':>8}{'preset':>12}{'radius':>7}{'chunks':>9}{'grid':>7}"
        f"{'full KB':>10}{'peak KB':>10}{'saved':>8}{'loads':>7}{'unloads':>9}"
        f"{'full ms':>10}{'stream ms':>11}{'mismatch':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['name']:<14}{r['map']:>8}{r['preset']:>12}{r['radius']:>7}"
            f"{r['n_chunks']:>9}{r['chunk_grid']:>7}"
            f"{r['full_load_bytes'] / 1024:>10.1f}"
            f"{r['peak_stream_bytes'] / 1024:>10.1f}"
            f"{r['memory_saving_pct']:>7.1f}%"
            f"{r['loads']:>7}{r['unloads']:>9}"
            f"{r['whole_load_ms']:>10.3f}{r['stream_total_ms']:>11.2f}"
            f"{r['mismatched_viewports']:>10}"
        )

    print()
    for r in rows:
        print(
            f"{r['name']}: convert={r['convert_ms']:.1f}ms "
            f"loader-only time={r['stream_load_ms']:.2f}ms "
            f"stitched==whole: {r['stitched_identical']}"
        )

    assert all(r["mismatched_viewports"] == 0 for r in rows), "viewport crop mismatch"
    assert all(r["stitched_identical"] for r in rows), "stitched map differs from whole render"
    print("\nAll four maps: every streamed viewport and every stitched map "
          "match the whole-image render pixel-for-pixel.")


if __name__ == "__main__":
    main()
