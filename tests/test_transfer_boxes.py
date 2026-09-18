"""The municipality a seat box was drawn for always enters its bags."""

import geopandas as gpd
from shapely.geometry import box

from satinsight.transfer import municipalities_in_box


def polygons():
    return gpd.GeoDataFrame(
        {
            "municipality": ["rural_seat", "neighbour", "grazing"],
            "name": ["Rural seat", "Neighbour", "Grazing"],
        },
        geometry=[box(0, 0, 1, 1), box(0.9, 0.9, 1.0, 1.0), box(0.99, 0.0, 2.0, 1.0)],
        crs="EPSG:4326",
    )


def test_a_small_seat_box_keeps_its_own_municipality():
    small = (0.95, 0.95, 1.0, 1.0)
    without = municipalities_in_box(polygons(), small)
    assert "rural_seat" not in set(without.municipality)
    kept = municipalities_in_box(polygons(), small, own="rural_seat")
    assert set(kept.municipality) == {"rural_seat", "neighbour"}


def test_neighbours_that_graze_the_box_stay_out():
    kept = municipalities_in_box(polygons(), (0.0, 0.0, 1.0, 1.0), own="rural_seat")
    assert set(kept.municipality) == {"rural_seat", "neighbour"}
