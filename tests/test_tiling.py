"""Tests for the patch grid. No network, no imagery."""

import numpy as np
import pytest

from satinsight.grid import grid_from_bbox
from satinsight.tiling import Tile, centers, grid, select, stack, valid_fraction

BBOX = (-93.135, 16.740, -93.095, 16.768)


def bands(height=140, width=200, value=1.0):
    return {"vv": np.full((height, width), value, dtype="float32")}


def test_grid_covers_whole_patches_only():
    tiles = grid((140, 200), size=64)
    assert len(tiles) == 2 * 3
    assert max(t.y0 for t in tiles) + 64 <= 140
    assert max(t.x0 for t in tiles) + 64 <= 200


def test_grid_rejects_an_array_smaller_than_one_patch():
    with pytest.raises(ValueError, match="no whole"):
        grid((30, 200), size=64)


def test_a_window_cuts_the_expected_square():
    crop = np.arange(100 * 100).reshape(100, 100)[Tile(0, 1, 0, 64, 32).window]
    assert crop.shape == (32, 32)
    assert crop[0, 0] == 64


def test_valid_fraction_needs_every_band_present():
    b = bands()
    b["vh"] = np.full((140, 200), np.nan, dtype="float32")
    b["vh"][:, :100] = 1.0
    assert valid_fraction(b, Tile(0, 0, 0, 0, 64)) == 1.0
    assert valid_fraction(b, Tile(0, 2, 0, 128, 64)) == 0.0


def test_select_drops_the_patches_under_the_threshold():
    b = bands()
    b["vv"][:, 128:] = np.nan
    tiles = select(b, size=64, min_valid_fraction=0.9)
    assert {t.col for t in tiles} == {0, 1}


def test_select_rejects_bands_of_different_shape():
    b = bands()
    b["vh"] = np.ones((10, 10), dtype="float32")
    with pytest.raises(ValueError, match="disagree on shape"):
        select(b)


def test_centers_land_inside_the_grid_bounds():
    city_grid = grid_from_bbox(BBOX, "EPSG:32615")
    tiles = grid(city_grid.shape, size=64)
    xy = centers(tiles, city_grid)
    left, bottom, right, top = city_grid.bounds
    assert xy.shape == (len(tiles), 2)
    assert (xy[:, 0] > left).all() and (xy[:, 0] < right).all()
    assert (xy[:, 1] > bottom).all() and (xy[:, 1] < top).all()


def test_stack_respects_the_channel_order_it_is_given():
    b = {"vv": np.zeros((64, 64), "float32"), "vh": np.ones((64, 64), "float32")}
    stacked = stack(b, Tile(0, 0, 0, 0, 64), order=["vh", "vv"])
    assert stacked.shape == (2, 64, 64)
    assert stacked[0].mean() == 1.0 and stacked[1].mean() == 0.0


def test_a_window_splits_into_the_expected_token_grid():
    from satinsight.tiling import tokens

    cells = tokens(Tile(0, 0, 224, 448, 224), token_size=16)
    assert len(cells) == 196
    assert cells[0].y0 == 224 and cells[0].x0 == 448
    # coordinates come in the full image, not relative to the window
    assert cells[-1].y0 == 224 + 13 * 16
    assert cells[-1].x0 == 448 + 13 * 16


def test_tokens_must_divide_the_window():
    from satinsight.tiling import tokens

    with pytest.raises(ValueError, match="does not divide"):
        tokens(Tile(0, 0, 0, 0, 100), token_size=16)


def test_instances_index_points_at_the_right_model_output():
    from satinsight.tiling import instances

    b = bands(224, 448)
    windows = grid((224, 448), size=224)
    b["vv"][:16, :16] = np.nan  # only the first token of the first window
    valid_tokens, indices = instances(windows, b)
    assert len(valid_tokens) == len(indices) == 2 * 196 - 1
    assert indices[0] == 1
    assert indices[-1] == 2 * 196 - 1
