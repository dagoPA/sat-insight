"""Tests for the patch grid. No network, no imagery."""

import numpy as np
import pytest

from satinsight.malla import malla_de_bbox
from satinsight.tiling import Tile, centers, grid, select, stack, valid_fraction

BBOX = (-93.135, 16.740, -93.095, 16.768)


def bandas(alto=140, ancho=200, valor=1.0):
    return {"vv": np.full((alto, ancho), valor, dtype="float32")}


def test_grid_covers_whole_patches_only():
    tiles = grid((140, 200), size=64)
    assert len(tiles) == 2 * 3
    assert max(t.y0 for t in tiles) + 64 <= 140
    assert max(t.x0 for t in tiles) + 64 <= 200


def test_grid_rejects_an_array_smaller_than_one_patch():
    with pytest.raises(ValueError, match="no whole"):
        grid((30, 200), size=64)


def test_a_window_cuts_the_expected_square():
    recorte = np.arange(100 * 100).reshape(100, 100)[Tile(0, 1, 0, 64, 32).window]
    assert recorte.shape == (32, 32)
    assert recorte[0, 0] == 64


def test_valid_fraction_needs_every_band_present():
    b = bandas()
    b["vh"] = np.full((140, 200), np.nan, dtype="float32")
    b["vh"][:, :100] = 1.0
    assert valid_fraction(b, Tile(0, 0, 0, 0, 64)) == 1.0
    assert valid_fraction(b, Tile(0, 2, 0, 128, 64)) == 0.0


def test_select_drops_the_patches_under_the_threshold():
    b = bandas()
    b["vv"][:, 128:] = np.nan
    tiles = select(b, size=64, min_valid_fraction=0.9)
    assert {t.col for t in tiles} == {0, 1}


def test_select_rejects_bands_of_different_shape():
    b = bandas()
    b["vh"] = np.ones((10, 10), dtype="float32")
    with pytest.raises(ValueError, match="disagree on shape"):
        select(b)


def test_centers_land_inside_the_grid_bounds():
    malla = malla_de_bbox(BBOX, "EPSG:32615")
    tiles = grid(malla.forma, size=64)
    xy = centers(tiles, malla)
    izq, abajo, der, arriba = malla.limites
    assert xy.shape == (len(tiles), 2)
    assert (xy[:, 0] > izq).all() and (xy[:, 0] < der).all()
    assert (xy[:, 1] > abajo).all() and (xy[:, 1] < arriba).all()


def test_stack_respects_the_channel_order_it_is_given():
    b = {"vv": np.zeros((64, 64), "float32"), "vh": np.ones((64, 64), "float32")}
    apilado = stack(b, Tile(0, 0, 0, 0, 64), order=["vh", "vv"])
    assert apilado.shape == (2, 64, 64)
    assert apilado[0].mean() == 1.0 and apilado[1].mean() == 0.0
