"""Auto-trim report for exported tilesets. Run: python scripts/trim_report.py

Builds four small synthetic asset groups (GB-style 8x8 RGB tiles padded with
blank background margins), runs gb_pipeline.trim_tiles on each, and reports
per group: trimmed sizes, tile-count change (source tiles -> unique atlas
tiles after crop+dedup), rebuild correctness (pixel-identical reconstruction
from tiles + recorded offsets), and wall-clock time. All-empty tiles are
reported, never cropped to 0 size. Everything runs once, in-memory.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gb_pipeline import rebuild_tiles, trim_tiles

BG = (8, 24, 32)  # DMG darkest shade -- the export background


def _canvas(n, h=8, w=8):
    tiles = np.zeros((n, h, w, 3), dtype=np.uint8)
    tiles[...] = BG
    return tiles


def group_centered_sprites():
    """Sprites drawn small in the middle of the tile: big blank margins."""
    rng = np.random.RandomState(1)
    tiles = _canvas(24)
    for i in range(24):
        rh, rw = rng.randint(2, 5, size=2)
        r0 = (8 - rh) // 2
        c0 = (8 - rw) // 2
        tiles[i, r0 : r0 + rh, c0 : c0 + rw] = rng.randint(
            40, 256, size=(rh, rw, 3)
        ).astype(np.uint8)
    return tiles


def group_dense_tiles():
    """Full-tile content: nothing to trim (control group)."""
    rng = np.random.RandomState(2)
    tiles = _canvas(24)
    noise = rng.randint(0, 256, size=(24, 8, 8, 3)).astype(np.uint8)
    # Keep pixels clearly off the background so every pixel is content.
    return np.where(noise == np.array(BG, dtype=np.uint8), noise + 1, noise)


def group_mixed_with_empty():
    """Half content tiles, half all-background tiles (must be reported)."""
    rng = np.random.RandomState(3)
    tiles = _canvas(24)
    for i in range(0, 24, 2):
        rh, rw = rng.randint(1, 7, size=2)
        r0, c0 = rng.randint(0, 8 - rh + 1), rng.randint(0, 8 - rw + 1)
        tiles[i, r0 : r0 + rh, c0 : c0 + rw] = rng.randint(
            0, 256, size=(rh, rw, 3)
        ).astype(np.uint8)
    return tiles


def group_shared_sprites_offset():
    """The same few sprites stamped at different margins: dedup showcase."""
    rng = np.random.RandomState(4)
    sprites = [rng.randint(0, 256, size=(3, 3, 3)).astype(np.uint8) for _ in range(3)]
    tiles = _canvas(24)
    for i in range(24):
        r0, c0 = rng.randint(0, 6, size=2)
        tiles[i, r0 : r0 + 3, c0 : c0 + 3] = sprites[i % 3]
    return tiles


GROUPS = [
    ("centered-sprites", group_centered_sprites(), dict(per_tile=True)),
    ("dense-control", group_dense_tiles(), dict(per_tile=True)),
    ("mixed-with-empty", group_mixed_with_empty(), dict(per_tile=True)),
    ("shared-offset", group_shared_sprites_offset(), dict(per_tile=True)),
]


def _sizes_summary(atlas):
    from collections import Counter

    counts = Counter(f"{t.shape[0]}x{t.shape[1]}" for t in atlas.tiles)
    return ", ".join(f"{size} (x{n})" for size, n in sorted(counts.items()))


def run_group(name, tiles, kw):
    t0 = time.perf_counter()
    atlas = trim_tiles(tiles, background=BG, threshold=0, **kw)
    dt = time.perf_counter() - t0

    rebuilt = rebuild_tiles(atlas)
    identical = np.array_equal(rebuilt, tiles)

    # Offset correctness: every cropped tile must equal the source slice at
    # its recorded offset (and only background was cropped away).
    offsets_ok = True
    for i in range(tiles.shape[0]):
        tid, oy, ox = (int(v) for v in atlas.table[i])
        crop = atlas.tiles[tid]
        th, tw = crop.shape[:2]
        if not np.array_equal(crop, tiles[i, oy : oy + th, ox : ox + tw]):
            offsets_ok = False
            break

    n_src = tiles.shape[0]
    n_out = len(atlas.tiles)
    src_bytes = n_src * 8 * 8 * 3
    out_bytes = sum(int(t.nbytes) for t in atlas.tiles)
    print(f"[{name}]")
    print(f"  trimmed sizes (unique tiles): {_sizes_summary(atlas)}")
    print(f"  tiles: {n_src} source -> {n_out} atlas "
          f"({src_bytes}B -> {out_bytes}B, {100.0 * out_bytes / src_bytes:.1f}%)")
    print(f"  empty tiles reported: {atlas.empty.tolist() or 'none'}")
    print(f"  rebuild pixel-identical: {identical}; offsets correct: {offsets_ok}")
    print(f"  time: {dt * 1000:.2f} ms")
    assert identical, f"{name}: rebuild is not pixel-identical"
    assert offsets_ok, f"{name}: recorded offsets do not rebuild the source"
    for i in atlas.empty:
        assert atlas.tiles[int(atlas.table[i, 0])].size > 0, "empty tile is 0-sized"
    print()


if __name__ == "__main__":
    for name, tiles, kw in GROUPS:
        run_group(name, tiles, kw)
    print("all groups: rebuild pixel-identical, offsets verified")
