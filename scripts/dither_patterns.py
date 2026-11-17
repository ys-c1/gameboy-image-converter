"""4x4 grey-transition dither pattern library for the GB pipeline.

Why this exists
---------------
The pipeline's ordered (Bayer) dither is anchored to absolute image
coordinates.  That is deliberately position-stable for tile dedup, but it
means *every* 4x4 cell at the same tone repeats the exact same mask.  Over a
smooth grey ramp the eye reads that repeated phase as regular stripes /
checkerboards.

This module replaces the fixed Bayer matrix with a small library of 4x4
dither patterns:

1. **Generate** every usable 4x4 pattern over the four GB shades and compute
   each pattern's grey level.  A usable dither pattern is a binary mask
   rendered between two of the four shades:
   ``level = lo + (hi - lo) * coverage / 16`` (equivalently, the mean of the
   pattern's 16 shade indices).  Enumerating the 6 unordered shade pairs x
   2**16 masks gives 393,216 candidates over 49 distinct grey levels in 1/16
   steps from 0 (lightest) to 3 (darkest).  (The unrestricted 4**16 space is
   4.3 billion patterns; a dither cell only ever mixes two shades, so those
   are exactly the usable ones -- flat 1-shade patterns are included as the
   coverage-0 / coverage-16 members of each pair, then de-duplicated.)
2. **Select** a pattern per 4x4 region by target grey.  The cost combines the
   region's tone error, a per-pixel texture term, and a configurable
   ``boundary_weight`` penalty for index mismatches across the already-chosen
   left/top edges (a refinement sweep also aligns the right/bottom edges).
   The library size is capped by ``max_patterns``; the cap is filled
   round-robin across grey levels so every tone keeps candidates.
3. **Apply** the selected patterns to an index image (values 0-3) and
   **check** the continuity of every internal 4x4 pattern boundary, reporting
   the mismatch rate and the number of completely broken 4x4 seams.

The library is opt-in: nothing in ``gb_pipeline``/``main`` imports this
module, so the existing converter's output is pixel-for-pixel unchanged.

Pure numpy -- no new third-party dependencies.

Run
---
    python scripts/dither_patterns.py
    python scripts/dither_patterns.py --max-patterns 256 --boundary-weight 0.3
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import numpy as np

PATTERN_SIZE = 4
N_SHADES = 4
CELLS = PATTERN_SIZE * PATTERN_SIZE
N_MASKS = 1 << CELLS
GREY_STEP = 1.0 / CELLS
MAX_GREY = N_SHADES - 1

# 4x4 ordered Bayer matrix, retained only as the comparison baseline.
BAYER4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
) / 16.0


# ---------------------------------------------------------------------------
# 1. Generate every usable 4x4 pattern + its grey level
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PatternLibrary:
    """A capped, deterministic subset of the 4x4 four-shade pattern space."""

    patterns: np.ndarray  # (n, 4, 4) uint8, values 0-3
    mask: np.ndarray      # (n, 4, 4) uint8, 0/1 (which cells use the high shade)
    grey: np.ndarray      # (n,) float64 perceived grey level, 0.0 (lightest) - 3.0
    lo: np.ndarray        # (n,) uint8 low shade of each pattern
    hi: np.ndarray        # (n,) uint8 high shade of each pattern
    score: np.ndarray     # (n,) float64 intrinsic (isotropy) ranking score
    max_patterns: int     # requested cap
    candidate_count: int  # candidates enumerated before the cap/dedup

    def __len__(self) -> int:
        return int(self.patterns.shape[0])

    @property
    def empty(self) -> bool:
        return len(self) == 0

    def validate(self) -> None:
        if self.empty:
            raise ValueError(
                "pattern library is empty: cannot select or apply patterns"
            )
        if self.patterns.shape[1:] != (PATTERN_SIZE, PATTERN_SIZE):
            raise ValueError("pattern library holds non-4x4 patterns")
        if self.patterns.min() < 0 or self.patterns.max() >= N_SHADES:
            raise ValueError("pattern library values must be in 0..3")

    @property
    def grey_levels(self) -> np.ndarray:
        return np.unique(np.round(self.grey / GREY_STEP).astype(int))


def _all_masks() -> np.ndarray:
    """Every 4x4 binary mask, shape (65536, 4, 4) uint8."""
    ids = np.arange(N_MASKS, dtype=np.uint32)
    bits = ((ids[:, None] >> np.arange(CELLS, dtype=np.uint32)[None, :]) & 1)
    return bits.astype(np.uint8).reshape(N_MASKS, PATTERN_SIZE, PATTERN_SIZE)


def _intrinsic_score(
    masks: np.ndarray, spans: np.ndarray
) -> np.ndarray:
    """Rank patterns by isotropy: high, balanced horizontal/vertical detail.

    Regular stripes have zero transitions in one direction (e.g. horizontal
    bars never change along a row) and near-maximum in the other, so
    ``transitions - 2 * |horizontal - vertical|`` rejects them.  A small span
    penalty prefers dithering between *adjacent* shades at equal grey (same
    average tone, lower contrast/grain).
    """
    horizontal = (masks[:, :, :-1] != masks[:, :, 1:]).sum(axis=(1, 2))
    vertical = (masks[:, :-1, :] != masks[:, 1:, :]).sum(axis=(1, 2))
    total = horizontal + vertical
    balance = np.abs(horizontal.astype(np.int64) - vertical.astype(np.int64))
    return (total - 2 * balance - 4 * (spans - 1)).astype(np.float64)


def _enumerate_candidates(dedupe_identical: bool = True) -> dict:
    """All (mask, lo, hi) candidates and their derived grey/score arrays.

    Enumerating the six unordered shade pairs is what "all four-colour
    patterns" means here in a tractable form: a full 4**16 enumeration is
    4.3 billion patterns, while a dither pattern only ever mixes two shades.
    """
    masks = _all_masks()  # (65536, 4, 4)
    coverage = masks.reshape(N_MASKS, CELLS).sum(axis=1).astype(np.int16)
    flat_masks = masks.reshape(N_MASKS, CELLS)
    mask_score = _intrinsic_score(masks, np.ones(N_MASKS, dtype=np.int64))

    all_patterns = []
    all_grey = []
    all_lo = []
    all_hi = []
    all_span = []
    all_score = []
    for lo in range(N_SHADES):
        for hi in range(lo + 1, N_SHADES):
            span = hi - lo
            all_patterns.append(
                (lo + flat_masks * span).astype(np.uint8)
            )  # (65536, 16)
            all_grey.append(lo + span * coverage * GREY_STEP)
            all_lo.append(np.full(N_MASKS, lo, dtype=np.uint8))
            all_hi.append(np.full(N_MASKS, hi, dtype=np.uint8))
            all_span.append(np.full(N_MASKS, span, dtype=np.int64))
            all_score.append(mask_score - 4 * (span - 1))

    patterns = np.concatenate(all_patterns, axis=0)  # (393216, 16)
    grey = np.concatenate(all_grey).astype(np.float64)
    lo = np.concatenate(all_lo)
    hi = np.concatenate(all_hi)
    span = np.concatenate(all_span)
    score = np.concatenate(all_score).astype(np.float64)
    candidate_count = int(patterns.shape[0])

    # De-duplicate exact patterns (the all-low / all-high masks recur as the
    # solid levels in two adjacent pairs).  Pack the 16 two-bit cells into one
    # uint32 key and keep the first (lowest shade-pair) occurrence.
    if dedupe_identical:
        keys = _pattern_keys(patterns)
        _, first = np.unique(keys, return_index=True)
        first = np.sort(first)
        patterns, grey, lo, hi = patterns[first], grey[first], lo[first], hi[first]
        span = span[first]
        score = score[first]

    mask = (patterns > lo[:, None]).astype(np.uint8)
    return {
        "patterns": patterns,
        "mask": mask,
        "grey": grey,
        "lo": lo,
        "hi": hi,
        "score": score,
        "grey_level": np.round(grey / GREY_STEP).astype(np.int64),
        "candidate_count": candidate_count,
    }


def _pattern_keys(patterns: np.ndarray) -> np.ndarray:
    """Pack each (n, 16) 0-3 pattern into a unique uint64 key."""
    patterns = np.asarray(patterns, dtype=np.uint64)
    keys = np.zeros(patterns.shape[0], dtype=np.uint64)
    for cell in range(CELLS):
        keys |= patterns[:, cell] << np.uint64(2 * cell)
    return keys


def build_pattern_library(
    max_patterns: int = 512,
    *,
    seed: int = 0,
    dedupe_identical: bool = True,
) -> PatternLibrary:
    """Generate all candidates and return a capped, varied pattern library.

    ``max_patterns`` is the library cap.  At least one pattern is kept for as
    many grey levels as the cap allows (levels are spread evenly when the cap
    is smaller than the 49 distinct levels); extra capacity is filled
    round-robin across levels.  Within a level the cap's slots are drawn from
    a deterministically shuffled queue (``seed``), so a tone cycles through
    genuinely different masks instead of the same "most isotropic" handful
    repeated across a ramp.

    ``dedupe_identical`` collapses patterns with identical 0-3 content (the
    same mask can be reached through two adjacent shade pairs); leave it on
    unless you specifically want candidates counted per shade pair.

    Raises ``ValueError`` when the cap is <= 0 or the resulting library is
    empty -- an empty library cannot select or apply anything.
    """
    max_patterns = int(max_patterns)
    if max_patterns <= 0:
        raise ValueError(
            "pattern library is empty: max_patterns must be a positive integer"
        )

    cand = _enumerate_candidates(dedupe_identical=dedupe_identical)
    patterns = cand["patterns"]
    grey = cand["grey"]
    score = cand["score"]
    level = np.round(grey / GREY_STEP).astype(np.int64)
    levels = np.unique(level)

    # Which levels get at least one slot.  With a normal cap every level does;
    # with a tiny cap, spread the slots across the whole 0..3 range.
    if max_patterns < levels.size:
        keep_levels = np.unique(
            np.round(np.linspace(levels[0], levels[-1], max_patterns)).astype(int)
        )
    else:
        keep_levels = levels

    rng = np.random.RandomState(seed)
    queues = []
    for lv in keep_levels:
        idx = np.flatnonzero(level == lv)
        # Deterministic shuffle within the level.  Drawing the cap's slots
        # from a shuffled per-level queue (rather than always taking the same
        # few top-scoring masks) is what keeps the library varied: a tone
        # cycles through genuinely different masks instead of the identical
        # 2-3 "best" ones at every nearby grey.  Intrinsic score is exported
        # on the library for inspection but deliberately does not prune; the
        # selection cost (stage 2) is what actually rejects stripey masks.
        idx = idx[rng.permutation(idx.size)]
        queues.append(idx)
    positions = [0] * len(queues)

    selected: list[int] = []

    def _take(i: int) -> bool:
        queue = queues[i]
        pos = positions[i]
        while pos < len(queue):
            cand = int(queue[pos])
            pos += 1
            selected.append(cand)
            positions[i] = pos
            return True
        positions[i] = pos
        return False

    while len(selected) < max_patterns:
        progressed = False
        for i in range(len(queues)):
            if len(selected) >= max_patterns:
                break
            if _take(i):
                progressed = True
        if not progressed:
            break

    if not selected:
        raise ValueError("pattern library is empty: no usable patterns generated")

    chosen = np.asarray(selected, dtype=np.int64)
    # Deterministic, tone-ordered library layout (stable within a tone).
    chosen = chosen[np.lexsort((chosen, level[chosen]))]
    library = PatternLibrary(
        patterns=patterns[chosen].reshape(-1, PATTERN_SIZE, PATTERN_SIZE).copy(),
        mask=cand["mask"][chosen].copy(),
        grey=grey[chosen].copy(),
        lo=cand["lo"][chosen].copy(),
        hi=cand["hi"][chosen].copy(),
        score=score[chosen].copy(),
        max_patterns=max_patterns,
        candidate_count=cand["candidate_count"],
    )
    library.validate()
    return library


# ---------------------------------------------------------------------------
# 2. Pick a pattern per 4x4 region by target grey
# ---------------------------------------------------------------------------


def _block_mean(arr: np.ndarray) -> np.ndarray:
    """Mean of each non-overlapping 4x4 block -> (H//4, W//4) float64."""
    arr = np.asarray(arr, dtype=np.float64)
    h, w = arr.shape
    gh, gw = h // PATTERN_SIZE, w // PATTERN_SIZE
    blocks = arr[: gh * PATTERN_SIZE, : gw * PATTERN_SIZE].reshape(
        gh, PATTERN_SIZE, gw, PATTERN_SIZE
    )
    return blocks.mean(axis=(1, 3))


def _as_block_target(target, shape: tuple) -> np.ndarray:
    """Normalise a target grey map to (H//4, W//4)."""
    target = np.asarray(target, dtype=np.float64)
    h, w = shape
    gh, gw = h // PATTERN_SIZE, w // PATTERN_SIZE
    if target.shape == (h, w):
        return _block_mean(target)
    if target.shape == (gh, gw):
        return target.copy()
    raise ValueError(
        f"target grey must be {h}x{w} or {gh}x{gw}, got {target.shape}"
    )


def select_patterns(
    target_grey,
    library: PatternLibrary,
    boundary_weight: float = 0.1,
    *,
    pixel_weight: float = 0.03,
    pixel_shape: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one library pattern per 4x4 region for a target grey map.

    ``target_grey`` is either a per-pixel grey map (0 lightest .. 3 darkest)
    or an already block-averaged (H//4, W//4) map.  Pass ``pixel_shape``
    explicitly to disambiguate a small block map (e.g. 4x4) from a per-pixel
    map.  For each region the cost is::

        (pattern_grey - block_target_grey)**2
          + pixel_weight * mean((pattern - per_pixel_target)**2)
          + boundary_weight * edge_mismatch_fraction

    The first term pins the region's *tone*; the second (only when a
    per-pixel map is supplied) keeps the pattern's interior texture faithful
    to the local gradient; the third compares the pattern's left column to
    the chosen left neighbour's right column and its top row to the chosen
    top neighbour's bottom row -- a refinement pass also aligns against the
    right and bottom neighbours once they exist.

    ``boundary_weight`` therefore trades a little grey accuracy for smoother
    pattern borders; 0.0 is pure tone/texture match, and higher values buy
    continuity up to the point where broad flat areas start to drift in
    brightness.

    Returns ``(region_ids (H//4, W//4) int32, index_map (H, W) uint8)``.
    """
    target = np.asarray(target_grey, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("target_grey must be 2-D")
    if pixel_shape is None:
        pixel_shape = target.shape
    return _select_patterns_impl(
        target, library, boundary_weight,
        shape=pixel_shape, pixel_weight=pixel_weight,
    )


def _select_patterns_impl(
    target_grey,
    library: PatternLibrary,
    boundary_weight: float,
    shape: tuple | None = None,
    sweeps: int = 2,
    pixel_weight: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    library.validate()
    boundary_weight = float(boundary_weight)
    if not np.isfinite(boundary_weight) or boundary_weight < 0.0:
        raise ValueError("boundary_weight must be a finite value >= 0")
    pixel_weight = float(pixel_weight)
    if not np.isfinite(pixel_weight) or pixel_weight < 0.0:
        raise ValueError("pixel_weight must be a finite value >= 0")
    sweeps = int(sweeps)
    if sweeps < 1:
        raise ValueError("sweeps must be >= 1")

    target = np.asarray(target_grey, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("target_grey must be 2-D")
    if shape is None:
        shape = target.shape
    h, w = shape
    if h % PATTERN_SIZE or w % PATTERN_SIZE:
        raise ValueError(
            f"index image dimensions must be multiples of {PATTERN_SIZE}, got {h}x{w}"
        )
    block_target = _as_block_target(target, (h, w))
    gh, gw = h // PATTERN_SIZE, w // PATTERN_SIZE

    # Per-pixel target term when the caller supplied a per-pixel map: a
    # tone-only (block-mean) match scatters 1/16-quantisation error across
    # every region and flattens within-block texture.
    per_pixel = target.shape == (h, w)

    flat = library.patterns.reshape(len(library), CELLS).astype(np.float64)
    left_edge = flat[:, 0::PATTERN_SIZE]   # (n, 4) left column (indices 0,4,8,12)
    right_edge = flat[:, 3::PATTERN_SIZE]  # (n, 4) right column
    top_edge = flat[:, :PATTERN_SIZE]      # (n, 4) top row
    bottom_edge = flat[:, 12:16]           # (n, 4) bottom row
    grey = library.grey

    ids = np.empty((gh, gw), dtype=np.int32)
    out = np.zeros((h, w), dtype=np.uint8)

    def _pick(by, bx, forward_only: bool) -> int:
        y0, x0 = by * PATTERN_SIZE, bx * PATTERN_SIZE
        cost = (grey - block_target[by, bx]) ** 2
        if per_pixel and pixel_weight > 0:
            block = target[y0 : y0 + PATTERN_SIZE, x0 : x0 + PATTERN_SIZE]
            cost = cost + pixel_weight * (
                (flat - block.reshape(CELLS)[None, :]) ** 2
            ).mean(axis=1)
        if bx > 0:
            left = out[y0 : y0 + PATTERN_SIZE, x0 - 1]
            cost = cost + boundary_weight * (
                (left_edge != left[None, :]).sum(axis=1) / PATTERN_SIZE
            )
        if by > 0:
            top = out[y0 - 1, x0 : x0 + PATTERN_SIZE]
            cost = cost + boundary_weight * (
                (top_edge != top[None, :]).sum(axis=1) / PATTERN_SIZE
            )
        if not forward_only:
            if bx < gw - 1:
                right = out[y0 : y0 + PATTERN_SIZE, x0 + PATTERN_SIZE]
                cost = cost + boundary_weight * (
                    (right_edge != right[None, :]).sum(axis=1) / PATTERN_SIZE
                )
            if by < gh - 1:
                below = out[y0 + PATTERN_SIZE, x0 : x0 + PATTERN_SIZE]
                cost = cost + boundary_weight * (
                    (bottom_edge != below[None, :]).sum(axis=1) / PATTERN_SIZE
                )
        return int(np.argmin(cost))

    # Forward pass: only the left/top neighbours are populated yet.  (The
    # array starts zero-filled, so a not-yet-chosen right/bottom neighbour
    # would otherwise look like a solid shade-0 edge.)
    for by in range(gh):
        y0 = by * PATTERN_SIZE
        for bx in range(gw):
            x0 = bx * PATTERN_SIZE
            j = _pick(by, bx, forward_only=True)
            ids[by, bx] = j
            out[y0 : y0 + PATTERN_SIZE, x0 : x0 + PATTERN_SIZE] = library.patterns[j]

    # Refinement sweeps: every neighbour is populated, so a cell also aligns
    # its right/bottom edges with cells chosen after it.  A couple of sweeps
    # lifts border continuity at a few percent grey error.
    for _ in range(sweeps - 1):
        for by in range(gh):
            y0 = by * PATTERN_SIZE
            for bx in range(gw):
                x0 = bx * PATTERN_SIZE
                j = _pick(by, bx, forward_only=False)
                ids[by, bx] = j
                out[y0 : y0 + PATTERN_SIZE, x0 : x0 + PATTERN_SIZE] = library.patterns[j]
    return ids, out


def dither_index_map(
    target_grey,
    library: PatternLibrary,
    boundary_weight: float = 0.1,
    *,
    pixel_weight: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Select patterns for a per-pixel grey map and return (ids, index_map)."""
    target = np.asarray(target_grey)
    if target.ndim != 2:
        raise ValueError("target_grey must be 2-D")
    return _select_patterns_impl(
        target, library, boundary_weight,
        shape=target.shape,
        pixel_weight=pixel_weight,
    )


def redither_index_map(
    index_map,
    library: PatternLibrary,
    boundary_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-dither an existing 0-3 index image using the pattern library.

    Each 4x4 region's target grey is the mean of its existing indices, so an
    image already indexed by any dither (e.g. Bayer) keeps its tone while
    swapping its repeated phase for a library pattern.  Returns
    ``(region_ids, new_index_map)``.
    """
    index_map = np.asarray(index_map, dtype=np.uint8)
    if index_map.ndim != 2:
        raise ValueError("index_map must be 2-D")
    if index_map.size and (index_map.min() < 0 or index_map.max() >= N_SHADES):
        raise ValueError("index_map values must be in 0..3")
    target = _block_mean(index_map)
    return _select_patterns_impl(
        target, library, boundary_weight, shape=index_map.shape
    )


# ---------------------------------------------------------------------------
# 3. Apply + check pattern-boundary continuity
# ---------------------------------------------------------------------------


def apply_patterns(region_ids, library: PatternLibrary) -> np.ndarray:
    """Apply (H//4, W//4) region ids -> a (H, W) uint8 index image."""
    library.validate()
    ids = np.asarray(region_ids)
    if ids.ndim != 2:
        raise ValueError("region_ids must be 2-D")
    if ids.size and (ids.min() < 0 or ids.max() >= len(library)):
        raise ValueError("region id out of range for this pattern library")
    gh, gw = ids.shape
    return (
        library.patterns[ids]
        .transpose(0, 2, 1, 3)
        .reshape(gh * PATTERN_SIZE, gw * PATTERN_SIZE)
        .copy()
    )


def check_boundary_continuity(index_map) -> dict:
    """Check every internal 4x4 pattern border in an index image.

    Returns a dict with:

    ``boundary_pairs``
        Number of adjacent pixel pairs straddling internal pattern borders.
    ``boundary_mismatches``
        How many of those pairs carry different indices.
    ``continuity``
        ``1 - mismatches / pairs`` in [0, 1]; 1.0 means every border pixel
        matches its neighbour (impossible for non-flat dithered images).
    ``discontinuity_rate``
        ``mismatches / pairs``.
    ``hard_seams``
        Number of 4-pixel border segments where *all four* pairs mismatch
        (a visually broken patch edge).
    ``max_edge_mismatch``
        Worst per-segment mismatch count (0..4).
    ``seamless``
        True when no border segment is completely broken (``hard_seams == 0``).
    """
    index_map = np.asarray(index_map)
    if index_map.ndim != 2:
        raise ValueError("index_map must be 2-D")
    h, w = index_map.shape
    if h % PATTERN_SIZE or w % PATTERN_SIZE:
        raise ValueError(
            f"index image dimensions must be multiples of {PATTERN_SIZE}, got {h}x{w}"
        )

    matches = []
    edge_mismatches = []

    # Vertical borders at x = 4, 8, ... : (H, n_v) boolean matches.
    if w > PATTERN_SIZE:
        v_match = index_map[:, PATTERN_SIZE::PATTERN_SIZE] == index_map[
            :, PATTERN_SIZE - 1 : -1 : PATTERN_SIZE
        ]
        matches.append(v_match.ravel())
        # Each border segment covers 4 rows of one block row.
        gh = h // PATTERN_SIZE
        n_v = v_match.shape[1]
        per_edge = v_match.reshape(gh, PATTERN_SIZE, n_v).sum(axis=1)
        edge_mismatches.append(PATTERN_SIZE - per_edge)

    # Horizontal borders at y = 4, 8, ... : (n_h, W) matches.
    if h > PATTERN_SIZE:
        h_match = index_map[PATTERN_SIZE::PATTERN_SIZE, :] == index_map[
            PATTERN_SIZE - 1 : -1 : PATTERN_SIZE, :
        ]
        matches.append(h_match.ravel())
        gw = w // PATTERN_SIZE
        n_h = h_match.shape[0]
        per_edge = h_match.reshape(n_h, gw, PATTERN_SIZE).sum(axis=2)
        edge_mismatches.append(PATTERN_SIZE - per_edge)

    if matches:
        pairs = int(sum(m.size for m in matches))
        matched = int(sum(int(m.sum()) for m in matches))
        edge_mismatch = np.concatenate([e.ravel() for e in edge_mismatches])
        max_edge_mismatch = int(edge_mismatch.max())
        hard_seams = int((edge_mismatch == PATTERN_SIZE).sum())
    else:
        pairs = matched = 0
        max_edge_mismatch = 0
        hard_seams = 0

    discontinuity = 0.0 if pairs == 0 else (pairs - matched) / pairs
    return {
        "boundary_pairs": pairs,
        "boundary_mismatches": pairs - matched,
        "continuity": 1.0 - discontinuity,
        "discontinuity_rate": discontinuity,
        "hard_seams": hard_seams,
        "max_edge_mismatch": max_edge_mismatch,
        "seamless": hard_seams == 0,
    }


# ---------------------------------------------------------------------------
# Metrics + the four sample groups
# ---------------------------------------------------------------------------


def stripe_visibility(index_map) -> float:
    """Phase-locked stripe visibility of a 4x4-dithered index image, in [0, 1].

    Split the image into 4x4 blocks, subtract each block's own mean, and
    average the residual at each of the 16 phases across all blocks.  If the
    dither repeats the same mask everywhere the phase mean is as large as the
    residual itself (ratio ~1); a library that varies the mask per region
    averages the phase away (ratio near 0).  This directly measures the
    regular striping the Bayer matrix produces on a grey ramp.
    """
    arr = np.asarray(index_map, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("index_map must be 2-D")
    h, w = arr.shape
    gh, gw = h // PATTERN_SIZE, w // PATTERN_SIZE
    if gh == 0 or gw == 0:
        return 0.0
    blocks = (
        arr[: gh * PATTERN_SIZE, : gw * PATTERN_SIZE]
        .reshape(gh, PATTERN_SIZE, gw, PATTERN_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(gh, gw, CELLS)
    )
    residual = blocks - blocks.mean(axis=2, keepdims=True)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if rms < 1e-12:
        return 0.0
    phase_mean = residual.mean(axis=(0, 1))
    return float(np.sqrt(np.mean(phase_mean ** 2)) / rms)


def pattern_switches(region_ids) -> int:
    """Number of differing 4x4 neighbours (horizontal + vertical)."""
    ids = np.asarray(region_ids)
    if ids.ndim != 2:
        raise ValueError("region_ids must be 2-D")
    switches = 0
    if ids.shape[1] > 1:
        switches += int((ids[:, 1:] != ids[:, :-1]).sum())
    if ids.shape[0] > 1:
        switches += int((ids[1:, :] != ids[:-1, :]).sum())
    return switches


def index_map_region_ids(index_map) -> np.ndarray:
    """Recover per-4x4 pattern ids from an index image (for baseline metrics)."""
    arr = np.asarray(index_map, dtype=np.uint8)
    h, w = arr.shape
    if h % PATTERN_SIZE or w % PATTERN_SIZE:
        raise ValueError("index image dimensions must be multiples of 4")
    gh, gw = h // PATTERN_SIZE, w // PATTERN_SIZE
    blocks = (
        arr.reshape(gh, PATTERN_SIZE, gw, PATTERN_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(gh, gw, CELLS)
    )
    mapping: dict[bytes, int] = {}
    ids = np.empty((gh, gw), dtype=np.int32)
    for by in range(gh):
        for bx in range(gw):
            key = blocks[by, bx].tobytes()
            ids[by, bx] = mapping.setdefault(key, len(mapping))
    return ids


def grey_error(index_map, target_grey) -> dict:
    """Mean-absolute / RMSE grey error, measured per 4x4 region."""
    index_map = np.asarray(index_map, dtype=np.float64)
    target = np.asarray(target_grey, dtype=np.float64)
    h, w = index_map.shape
    if target.shape != (h, w):
        raise ValueError("target_grey must match the index image shape")
    actual_blocks = _block_mean(index_map)
    target_blocks = _block_mean(target)
    err = actual_blocks - target_blocks
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))) if err.size else 0.0,
    }


def bayer_index_map(target_grey) -> np.ndarray:
    """The existing-style ordered-dither baseline for a per-pixel grey map."""
    target = np.clip(np.asarray(target_grey, dtype=np.float64), 0.0, MAX_GREY)
    h, w = target.shape
    base = np.minimum(np.floor(target), MAX_GREY - 1)
    frac = target - base
    threshold = BAYER4[np.arange(h)[:, None] % PATTERN_SIZE,
                       np.arange(w)[None, :] % PATTERN_SIZE]
    return (base + (frac > threshold)).clip(0, MAX_GREY).astype(np.uint8)


def make_sample_images(size: int = 96) -> dict[str, np.ndarray]:
    """Four small, self-contained grey-ramp test images (RGB uint8).

    The four groups deliberately stress different transition shapes:
    horizontal ramp, diagonal ramp, radial ramp, and flat stepped bands.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    span = max(size - 1, 1)
    fields = {
        "horizontal": xx / span,
        "diagonal": (xx + yy) / (2.0 * span),
        "radial": 1.0 - np.hypot(xx - span / 2.0, yy - span / 2.0)
        / np.hypot(span / 2.0, span / 2.0),
        "bands": np.floor(xx / span * 16.0) / 15.0,
    }
    out = {}
    for name, value in fields.items():
        v = np.clip(value, 0.0, 1.0)
        rgb = np.stack([v, v, v], axis=-1)
        out[name] = np.round(rgb * 255.0).astype(np.uint8)
    return out


def luminance_target_grey(rgb) -> np.ndarray:
    """Map an RGB image to target grey levels (0 lightest .. 3 darkest)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB array")
    luminance = (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ) / 255.0
    return (1.0 - luminance) * MAX_GREY


def evaluate_group(
    name: str,
    rgb: np.ndarray,
    library: PatternLibrary,
    boundary_weight: float,
) -> dict:
    """Run one sample group, returning its report row (plus Bayer baseline)."""
    target = luminance_target_grey(rgb)
    baseline = bayer_index_map(target)

    start = time.perf_counter()
    ids, index_map = dither_index_map(
        target, library, boundary_weight=boundary_weight
    )
    elapsed = time.perf_counter() - start

    err = grey_error(index_map, target)
    base_err = grey_error(baseline, target)
    continuity = check_boundary_continuity(index_map)
    base_ids = index_map_region_ids(baseline)
    return {
        "group": name,
        "grey_mae": err["mae"],
        "grey_rmse": err["rmse"],
        "grey_max_abs": err["max_abs"],
        "stripe_visibility": stripe_visibility(index_map),
        "pattern_switches": pattern_switches(ids),
        "unique_patterns": int(np.unique(ids).size),
        "seconds": elapsed,
        "boundary_continuity": continuity["continuity"],
        "boundary_discontinuity_rate": continuity["discontinuity_rate"],
        "hard_seams": continuity["hard_seams"],
        "max_edge_mismatch": continuity["max_edge_mismatch"],
        "seamless": continuity["seamless"],
        "baseline_grey_mae": base_err["mae"],
        "baseline_stripe_visibility": stripe_visibility(baseline),
        "baseline_pattern_switches": pattern_switches(base_ids),
        "index_map": index_map,
        "target_grey": target,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'group':<11} {'grey_mae':>8} {'grey_rmse':>9} {'stripe':>7} "
        f"{'switches':>8} {'patterns':>8} {'time_s':>7} {'boundary':>8} "
        f"{'hard':>4}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['group']:<11} {row['grey_mae']:>8.4f} {row['grey_rmse']:>9.4f} "
            f"{row['stripe_visibility']:>7.4f} {row['pattern_switches']:>8d} "
            f"{row['unique_patterns']:>8d} {row['seconds']:>7.3f} "
            f"{row['boundary_continuity']:>8.3f} {row['hard_seams']:>4d}"
        )
    print()
    print(
        f"{'baseline':<11} {'grey_mae':>8} {'':>9} {'stripe':>7} {'switches':>8}"
    )
    for row in rows:
        print(
            f"{row['group']:<11} {row['baseline_grey_mae']:>8.4f} {'':>9} "
            f"{row['baseline_stripe_visibility']:>7.4f} "
            f"{row['baseline_pattern_switches']:>8d}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="4x4 dither pattern library benchmark (offline, numpy only)"
    )
    parser.add_argument(
        "--max-patterns", type=int, default=512,
        help="pattern library cap (must be > 0; default 512)",
    )
    parser.add_argument(
        "--boundary-weight", type=float, default=0.1,
        help="penalty for mismatched pattern borders (default 0.1)",
    )
    parser.add_argument(
        "--size", type=int, default=96,
        help="sample image size in pixels, multiple of 4 (default 96)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the four report rows as JSON"
    )
    args = parser.parse_args(argv)

    if args.size <= 0 or args.size % PATTERN_SIZE:
        parser.error("--size must be a positive multiple of 4")

    try:
        build_start = time.perf_counter()
        library = build_pattern_library(args.max_patterns)
        build_seconds = time.perf_counter() - build_start
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"pattern library: {len(library)} patterns kept from "
        f"{library.candidate_count} candidates, "
        f"{library.grey_levels.size} grey levels, "
        f"built in {build_seconds:.3f}s"
    )
    print(
        f"boundary_weight={args.boundary_weight}  sample={args.size}x{args.size}\n"
    )

    rows = [
        evaluate_group(name, rgb, library, args.boundary_weight)
        for name, rgb in make_sample_images(args.size).items()
    ]

    if args.json:
        import json

        print(json.dumps(
            [
                {k: v for k, v in row.items()
                 if k not in ("index_map", "target_grey")}
                for row in rows
            ],
            indent=2,
        ))
    else:
        _print_table(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
