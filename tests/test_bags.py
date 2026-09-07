"""Tests for bag assembly. Geometry is built by hand, nothing is read from disk."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from satinsight.agebs import GRADES
from satinsight.bags import build, locate, municipal_labels
from satinsight.grid import grid_from_bbox
from satinsight.tiling import grid

BBOX = (-93.135, 16.740, -93.095, 16.768)
CRS = "EPSG:32615"


def grid_and_tiles(size=64):
    city_grid = grid_from_bbox(BBOX, CRS)
    return city_grid, grid(city_grid.shape, size=size)


def fake_agebs(city_grid, cuts=2, municipalities=("07101", "07102")):
    """Splits the box into vertical strips, one AGEB per strip."""
    left, bottom, right, top = city_grid.bounds
    width = (right - left) / cuts
    rows = []
    for i in range(cuts):
        mun = municipalities[i % len(municipalities)]
        rows.append(
            {
                "cvegeo": f"{mun}0001{i:03d}",
                "grade": [GRADES[0], GRADES[3]][i % 2],
                "ordinal": [0, 3][i % 2],
                "population": 1000 * (i + 1),
                "geometry": box(left + i * width, bottom, left + (i + 1) * width, top),
            }
        )
    return gpd.GeoDataFrame(rows, crs=CRS)


def test_every_patch_lands_in_exactly_one_ageb():
    city_grid, tiles = grid_and_tiles()
    table = locate(tiles, city_grid, fake_agebs(city_grid))
    assert len(table) == len(tiles)
    assert table.tile.is_unique
    assert set(table.municipality) == {"07101", "07102"}


def test_patches_outside_every_ageb_are_dropped():
    city_grid, tiles = grid_and_tiles()
    left, bottom, right, top = city_grid.bounds
    # a single AGEB covering the left half
    left_only = gpd.GeoDataFrame(
        [{"cvegeo": "0710100010001", "geometry": box(left, bottom, (left + right) / 2, top)}],
        crs=CRS,
    )
    table = locate(tiles, city_grid, left_only)
    assert 0 < len(table) < len(tiles)


def test_locate_on_no_tiles_returns_an_empty_frame():
    city_grid, _ = grid_and_tiles()
    empty = locate([], city_grid, fake_agebs(city_grid))
    assert empty.empty and "municipality" in empty.columns


def test_the_bag_label_is_weighted_by_population():
    agebs = gpd.GeoDataFrame(
        {
            "cvegeo": ["0710100010001", "0710100010002"],
            "ordinal": [0, 4],
            "population": [1, 999],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        crs=CRS,
    )
    labels = municipal_labels(agebs)
    assert len(labels) == 1
    # the populated AGEB rules: the weighted mean lands very close to 4
    assert labels.ordinal.iloc[0] == 4
    assert labels.ordinal_continuous.iloc[0] > 3.9
    assert labels.grade.iloc[0] == GRADES[4]


def test_a_municipality_with_no_population_still_gets_a_label():
    agebs = gpd.GeoDataFrame(
        {
            "cvegeo": ["0710100010001", "0710100010002"],
            "ordinal": [0, 2],
            "population": [0, 0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        crs=CRS,
    )
    assert municipal_labels(agebs).ordinal.iloc[0] == 1


def test_municipal_labels_demands_its_columns():
    agebs = gpd.GeoDataFrame({"cvegeo": ["0710100010001"], "geometry": [box(0, 0, 1, 1)]}, crs=CRS)
    with pytest.raises(KeyError, match="ordinal"):
        municipal_labels(agebs)


def test_build_returns_matching_instance_and_bag_tables():
    city_grid, tiles = grid_and_tiles()
    instances, bags = build(tiles, city_grid, fake_agebs(city_grid), "test")
    assert set(instances.municipality) == set(bags.municipality)
    assert bags.instances.sum() == len(instances)
    assert (instances.city == "test").all()


def test_bags_below_the_minimum_are_dropped_with_their_instances():
    city_grid, tiles = grid_and_tiles()
    agebs = fake_agebs(city_grid, cuts=8, municipalities=tuple(f"0710{i}" for i in range(8)))
    instances, bags = build(tiles, city_grid, agebs, "test", min_instances=1000)
    assert bags.empty and instances.empty


def test_build_fails_when_nothing_lands_inside():
    city_grid, tiles = grid_and_tiles()
    far_away = gpd.GeoDataFrame(
        [{"cvegeo": "0710100010001", "ordinal": 1, "population": 10, "geometry": box(0, 0, 1, 1)}],
        crs=CRS,
    )
    with pytest.raises(ValueError, match="no patch landed"):
        build(tiles, city_grid, far_away, "test")


def test_the_cumulative_shares_are_population_weighted():
    """The share of population living in AGEB of grade k or above.

    It is the aggregate a municipal figure actually knows, and the only one of the three
    that forces localisation: predicting that a tenth of the population lives in deprived
    AGEB demands identifying which tenth.
    """
    agebs = gpd.GeoDataFrame(
        {
            "cvegeo": ["0710100010001", "0710100010002", "0710100010003"],
            "ordinal": [0, 2, 4],
            "population": [800, 100, 100],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        crs=CRS,
    )
    row = municipal_labels(agebs).iloc[0]
    assert row.p1 == pytest.approx(0.2)
    assert row.p3 == pytest.approx(0.1)
    assert row.p4 == pytest.approx(0.1)
    # rounding flattens that structure to a single class and loses that a tenth of the
    # population lives at the highest grade
    assert row.ordinal == 1


def test_the_shares_fall_as_the_threshold_rises():
    agebs = gpd.GeoDataFrame(
        {
            "cvegeo": [f"071010001000{i}" for i in range(5)],
            "ordinal": [0, 1, 2, 3, 4],
            "population": [100] * 5,
            "geometry": [box(i, 0, i + 1, 1) for i in range(5)],
        },
        crs=CRS,
    )
    row = municipal_labels(agebs).iloc[0]
    assert row.p1 > row.p2 > row.p3 > row.p4
    assert row.p1 == pytest.approx(0.8)
