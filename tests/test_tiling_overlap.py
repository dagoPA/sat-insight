"""Overlapping windows: the strided grid and the per-token averaging."""

import numpy as np

from satinsight.encoders import average_overlaps
from satinsight.tiling import TOKEN_SIZE, Tile, grid


def test_strided_grid_overlaps_and_shares_token_positions():
    tiles = grid((448, 336), size=224, stride=112)
    assert len(tiles) == 3 * 2
    assert {(t.y0, t.x0) for t in tiles} == {
        (0, 0),
        (0, 112),
        (112, 0),
        (112, 112),
        (224, 0),
        (224, 112),
    }
    assert all(t.y0 % TOKEN_SIZE == 0 and t.x0 % TOKEN_SIZE == 0 for t in tiles)


def test_unstrided_grid_is_unchanged():
    assert [(t.y0, t.x0) for t in grid((448, 448), size=224)] == [
        (0, 0),
        (0, 224),
        (224, 0),
        (224, 224),
    ]


def test_average_overlaps_means_repeated_positions_and_keeps_first_order():
    tokens = [
        Tile(0, 0, 0, 0, 16),
        Tile(0, 1, 0, 16, 16),
        Tile(0, 0, 0, 16, 16),  # the second window sees the token at (0, 16) again
        Tile(0, 1, 0, 32, 16),
    ]
    matrix = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]], dtype="float32")
    averaged, kept = average_overlaps(matrix, tokens)
    assert [(t.y0, t.x0) for t in kept] == [(0, 0), (0, 16), (0, 32)]
    np.testing.assert_allclose(averaged, [[1.0, 1.0], [3.0, 3.0], [8.0, 8.0]])


def test_average_overlaps_is_identity_without_repeats():
    tokens = [Tile(0, 0, 0, 0, 16), Tile(0, 1, 0, 16, 16)]
    matrix = np.arange(4, dtype="float32").reshape(2, 2)
    averaged, kept = average_overlaps(matrix, tokens)
    assert kept == tokens
    np.testing.assert_array_equal(averaged, matrix)
