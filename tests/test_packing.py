"""Tests for cartridge tile byte packing (gb_pipeline pack/unpack 1bpp/2bpp)."""

import numpy as np
import pytest

import gb_pipeline as gp


def _patterns(seed=7, n=9):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 4, size=(n, 8, 8)).astype(np.uint8)


LAYOUTS = [
    ("msb", True),
    ("msb", False),
    ("lsb", True),
    ("lsb", False),
]


@pytest.mark.parametrize("bit_order,interleaved", LAYOUTS)
def test_2bpp_roundtrip_all_layouts(bit_order, interleaved):
    patterns = _patterns()
    data = gp.pack_tiles_2bpp(
        patterns, bit_order=bit_order, interleaved=interleaved
    )
    assert len(data) == 16 * patterns.shape[0]
    restored = gp.unpack_tiles_2bpp(
        data, bit_order=bit_order, interleaved=interleaved
    )
    assert restored.dtype == np.uint8
    assert np.array_equal(restored, patterns)


def test_2bpp_default_is_gb_hardware_layout():
    patterns = _patterns()
    data = gp.pack_tiles_2bpp(patterns)
    for i, pattern in enumerate(patterns):
        assert data[i * 16 : (i + 1) * 16] == gp.pattern_to_2bpp(pattern)


def test_2bpp_msb_leftmost_pixel_is_bit7_low_plane():
    tile = np.zeros((1, 8, 8), dtype=np.uint8)
    tile[0, 0, 0] = 1
    assert gp.pack_tiles_2bpp(tile)[:2] == bytes([0x80, 0x00])


def test_2bpp_lsb_leftmost_pixel_is_bit0():
    tile = np.zeros((1, 8, 8), dtype=np.uint8)
    tile[0, 0, 0] = 1
    assert gp.pack_tiles_2bpp(tile, bit_order="lsb")[:2] == bytes([0x01, 0x00])


def test_2bpp_planar_is_all_low_rows_then_high_rows():
    # Index 1 in row 0 -> low plane row 0 = 0x80; index 2 in row 7 -> high
    # plane row 7 = 0x80. Planar layout puts low-plane rows first.
    tile = np.zeros((1, 8, 8), dtype=np.uint8)
    tile[0, 0, 0] = 1
    tile[0, 7, 7] = 2
    data = gp.pack_tiles_2bpp(tile, interleaved=False)
    assert data[0] == 0x80
    assert data[8 + 7] == 0x01  # high plane, row 7, rightmost pixel -> bit 0
    assert sum(data) == 0x80 + 0x01


def test_2bpp_all_four_indices_each_plane():
    tile = np.array(
        [[[0, 1, 2, 3] * 2] * 8], dtype=np.uint8
    )  # (1, 8, 8): 0,1,2,3 repeating
    data = gp.pack_tiles_2bpp(tile)
    low, high = data[0], data[1]
    assert low == 0b01010101
    assert high == 0b00110011


def test_1bpp_roundtrip_and_size():
    rng = np.random.RandomState(3)
    patterns = rng.randint(0, 2, size=(5, 8, 8)).astype(np.uint8)
    data = gp.pack_tiles_1bpp(patterns)
    assert len(data) == 8 * patterns.shape[0]
    assert np.array_equal(gp.unpack_tiles_1bpp(data), patterns)


def test_1bpp_known_bytes_and_bit_order():
    tile = np.zeros((1, 8, 8), dtype=np.uint8)
    tile[0, 0, 0] = 1
    assert gp.pack_tiles_1bpp(tile) == bytes([0x80] + [0] * 7)
    assert gp.pack_tiles_1bpp(tile, bit_order="lsb") == bytes([0x01] + [0] * 7)


def test_1bpp_palette_swap_inverts_tile():
    patterns = np.ones((2, 8, 8), dtype=np.uint8)
    data = gp.pack_tiles_1bpp(patterns, palette_swap=(1, 0))
    assert data == bytes(16)
    restored = gp.unpack_tiles_1bpp(data, palette_swap=(1, 0))
    assert np.array_equal(restored, patterns)


def test_2bpp_palette_swap_roundtrip():
    patterns = _patterns()
    swap = [2, 0, 3, 1]
    data = gp.pack_tiles_2bpp(patterns, palette_swap=swap)
    restored = gp.unpack_tiles_2bpp(data, palette_swap=swap)
    assert np.array_equal(restored, patterns)


def test_2bpp_swap_is_palette_interchange():
    """The same packed tiles decoded through a different palette permutation
    display through that palette -- the cartridge palette-swap trick."""
    patterns = _patterns()
    stored_swap = [2, 0, 3, 1]
    other_swap = [3, 1, 0, 2]
    data = gp.pack_tiles_2bpp(patterns, palette_swap=stored_swap)

    via_other = gp.unpack_tiles_2bpp(data, palette_swap=other_swap)
    plain = gp.unpack_tiles_2bpp(data)

    assert np.array_equal(np.asarray(other_swap)[plain], via_other)
    # And it differs from the original indices in general.
    assert not np.array_equal(via_other, patterns)


def test_2bpp_n_tiles_argument_checks_length():
    data = gp.pack_tiles_2bpp(_patterns(n=2))
    assert gp.unpack_tiles_2bpp(data, n_tiles=2).shape == (2, 8, 8)
    with pytest.raises(ValueError):
        gp.unpack_tiles_2bpp(data, n_tiles=3)


def test_2bpp_rejects_out_of_range_indices():
    for bad_value in (4, 5, 255):
        bad = np.zeros((1, 8, 8), dtype=np.uint8)
        bad[0, 0, 0] = bad_value
        with pytest.raises(ValueError, match="0\\.\\.3"):
            gp.pack_tiles_2bpp(bad)


def test_2bpp_rejects_negative_index():
    bad = np.full((1, 8, 8), -1, dtype=np.int8)
    with pytest.raises(ValueError, match="0\\.\\.3"):
        gp.pack_tiles_2bpp(bad)


def test_1bpp_rejects_2bit_indices():
    bad = np.full((1, 8, 8), 2, dtype=np.uint8)
    with pytest.raises(ValueError, match="0\\.\\.1"):
        gp.pack_tiles_1bpp(bad)


def test_pack_rejects_non_integer_and_bad_shape():
    with pytest.raises(TypeError):
        gp.pack_tiles_2bpp(np.zeros((1, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        gp.pack_tiles_2bpp(np.zeros((8, 8), dtype=np.uint8))


def test_pack_rejects_bad_options():
    patterns = _patterns()
    with pytest.raises(ValueError):
        gp.pack_tiles_2bpp(patterns, bit_order="msb-first")
    with pytest.raises(ValueError):
        gp.pack_tiles_2bpp(patterns, palette_swap=[1, 1, 2, 3])
    with pytest.raises(ValueError):
        gp.pack_tiles_1bpp(patterns % 2, palette_swap=[0, 2])


def test_unpack_rejects_bad_buffers():
    with pytest.raises(ValueError):
        gp.unpack_tiles_2bpp(b"\x00" * 15)
    with pytest.raises(ValueError):
        gp.unpack_tiles_1bpp(b"\x00" * 9)
    with pytest.raises(TypeError):
        gp.unpack_tiles_2bpp(np.zeros(16, dtype=np.int16))


def test_pack_accepts_bytes_like_and_memoryview():
    patterns = _patterns(n=1)
    data = gp.pack_tiles_2bpp(patterns)
    assert np.array_equal(gp.unpack_tiles_2bpp(memoryview(data)), patterns)
    assert np.array_equal(gp.unpack_tiles_2bpp(bytearray(data)), patterns)
    assert np.array_equal(
        gp.unpack_tiles_2bpp(np.frombuffer(data, np.uint8)), patterns
    )


def test_cartridge_bytes_render_pixel_identical():
    """Packing a rendered GBImage to cartridge bytes and decoding it back
    reproduces the pipeline's output pixel-for-pixel."""
    from PIL import Image

    rng = np.random.RandomState(11)
    arr = rng.randint(0, 256, size=(24, 32, 3)).astype(np.uint8)
    palettes = gp.snap_rgb555(
        np.array(
            [[[240, 240, 240], [170, 170, 170], [90, 90, 90], [10, 10, 10]]],
            dtype=np.uint8,
        )
    )
    assignment = np.zeros((3, 4), dtype=np.uint8)
    gb = gp.index_tiles(arr, palettes, assignment)
    original = np.asarray(gp.render(gb))

    data = gp.pack_tiles_2bpp(gb.patterns)
    restored = gp.unpack_tiles_2bpp(data)
    rebuilt = gp.GBImage(
        patterns=restored,
        tilemap=gb.tilemap,
        attrs_palette=gb.attrs_palette,
        attrs_hflip=gb.attrs_hflip,
        attrs_vflip=gb.attrs_vflip,
        palettes=gb.palettes,
    )
    assert np.array_equal(np.asarray(gp.render(rebuilt)), original)
    Image.fromarray(original, "RGB")  # sanity: the output is a valid image

