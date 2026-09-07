"""Tests of the composite cache. They write to tmp_path, never to the network."""

import numpy as np
import pytest

from satinsight.cache import composite_path, exists, load, save
from satinsight.grid import grid_from_bbox

BBOX = (-93.135, 16.740, -93.095, 16.768)


@pytest.fixture
def city_grid():
    return grid_from_bbox(BBOX, "EPSG:32615", resolution_m=100)


def bands_of(city_grid, names=("vv", "vh")):
    rng = np.random.default_rng(0)
    return {n: rng.random(city_grid.shape).astype("float32") for n in names}


def test_round_trip_keeps_the_values(tmp_path, city_grid):
    original = bands_of(city_grid)
    destination = save(original, city_grid, tmp_path / "x.tif")
    recovered, _, _ = load(destination)

    assert set(recovered) == set(original)
    for name, array in original.items():
        np.testing.assert_allclose(recovered[name], array, rtol=1e-6)


def test_round_trip_keeps_the_georeference(tmp_path, city_grid):
    save(bands_of(city_grid), city_grid, tmp_path / "x.tif")
    _, recovered, _ = load(tmp_path / "x.tif")

    assert recovered.shape == city_grid.shape
    assert recovered.crs == city_grid.crs
    assert recovered.transform == pytest.approx(city_grid.transform, abs=1e-6)


def test_band_order_is_kept(tmp_path, city_grid):
    original = bands_of(city_grid, ("B04", "B03", "B02", "B08"))
    save(original, city_grid, tmp_path / "x.tif")
    recovered, _, _ = load(tmp_path / "x.tif")
    assert list(recovered) == list(original)


def test_tags_survive(tmp_path, city_grid):
    save(bands_of(city_grid), city_grid, tmp_path / "x.tif", scenes_used=17, orbit="ascending · 99")
    _, _, tags = load(tmp_path / "x.tif")
    assert tags["scenes_used"] == 17
    assert tags["orbit"] == "ascending · 99"


def test_nans_survive(tmp_path, city_grid):
    bands = bands_of(city_grid, ("vv",))
    bands["vv"][0, 0] = np.nan
    save(bands, city_grid, tmp_path / "x.tif")
    recovered, _, _ = load(tmp_path / "x.tif")
    assert np.isnan(recovered["vv"][0, 0])


def test_saving_without_bands_fails(tmp_path, city_grid):
    with pytest.raises(ValueError, match="no bands to save"):
        save({}, city_grid, tmp_path / "x.tif")


def test_bands_of_different_shapes_fail(tmp_path, city_grid):
    bands = bands_of(city_grid, ("vv",))
    bands["vh"] = np.zeros((3, 3), dtype="float32")
    with pytest.raises(ValueError, match="do not share a shape"):
        save(bands, city_grid, tmp_path / "x.tif")


def test_a_shape_that_does_not_match_the_grid_fails(tmp_path, city_grid):
    bands = {"vv": np.zeros((5, 5), dtype="float32")}
    with pytest.raises(ValueError, match="does not match the grid"):
        save(bands, city_grid, tmp_path / "x.tif")


def test_the_path_tells_city_and_sensor_apart(tmp_path):
    a = composite_path("tuxtla", "s1", tmp_path)
    b = composite_path("tuxtla", "s2", tmp_path)
    assert a != b
    assert a.suffix == ".tif"


def test_exists_reports_absence(tmp_path):
    assert not exists("merida", "s1", tmp_path)
