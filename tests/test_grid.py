"""Tests of the analysis grid. No network: the catalogue items are stand-ins."""

from dataclasses import dataclass, field

import pytest

from satinsight.grid import grid_from_bbox, grid_from_scenes, select_crs


@dataclass
class FakeItem:
    """Stand-in for a STAC item, of which only the properties matter."""

    properties: dict = field(default_factory=dict)


def item(epsg=32615, key="proj:epsg"):
    return FakeItem(properties={key: epsg})


BBOX_TUXTLA = (-93.135, 16.740, -93.095, 16.768)


def test_the_grid_covers_the_whole_box():
    city_grid = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615", resolution_m=10)
    left, bottom, right, top = city_grid.bounds
    assert city_grid.width * 10 >= right - left
    assert city_grid.height * 10 >= top - bottom


def test_the_pixel_size_is_the_one_requested():
    city_grid = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615", resolution_m=10)
    assert city_grid.transform.a == pytest.approx(10, abs=0.5)
    assert city_grid.transform.e == pytest.approx(-10, abs=0.5)


def test_lowering_the_resolution_reduces_the_pixels():
    fine = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615", resolution_m=10)
    coarse = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615", resolution_m=20)
    assert coarse.width < fine.width
    assert coarse.megapixels < fine.megapixels


def test_the_top_left_corner_matches_the_bounds():
    city_grid = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615")
    assert city_grid.transform.c == pytest.approx(city_grid.bounds[0])
    assert city_grid.transform.f == pytest.approx(city_grid.bounds[3])


def test_a_single_crs_is_accepted():
    city_grid, scenes = grid_from_scenes(BBOX_TUXTLA, [item(), item(), item()])
    assert city_grid.crs == "EPSG:32615"
    assert len(scenes) == 3


def test_an_already_prefixed_crs_is_not_duplicated():
    city_grid, _ = grid_from_scenes(BBOX_TUXTLA, [item(epsg="EPSG:32615")])
    assert city_grid.crs == "EPSG:32615"


def test_the_modern_key_is_read_too():
    city_grid, _ = grid_from_scenes(BBOX_TUXTLA, [item(epsg="EPSG:32616", key="proj:code")])
    assert city_grid.crs == "EPSG:32616"


def test_with_mixed_zones_the_majority_wins():
    scenes = [item(32615), item(32615), item(32616)]
    crs, selected = select_crs(scenes)
    assert crs == "EPSG:32615"
    assert len(selected) == 2


def test_the_scenes_of_the_discarded_zone_are_not_returned():
    scenes = [item(32616), item(32615), item(32615), item(32615)]
    city_grid, selected = grid_from_scenes(BBOX_TUXTLA, scenes)
    assert city_grid.crs == "EPSG:32615"
    assert len(selected) == 3
    assert all(s.properties["proj:epsg"] == 32615 for s in selected)


def test_scenes_without_a_projection_are_ignored_when_others_exist():
    scenes = [FakeItem(properties={}), item(32615)]
    crs, selected = select_crs(scenes)
    assert crs == "EPSG:32615"
    assert len(selected) == 1


def test_no_declared_projection_fails():
    with pytest.raises(ValueError, match="reference system"):
        grid_from_scenes(BBOX_TUXTLA, [FakeItem(properties={})])


def test_the_zone_is_chosen_by_coverage_and_not_by_count():
    """Guasave's case: the most numerous zone sees half the box.

    Eighty-seven scenes of zone 13 reach half the city and sixty-one of zone 12 cover it
    whole. Choosing by count left the city without a composite.
    """
    many = [item("EPSG:32613", key="proj:code") for _ in range(87)]
    few = [item("EPSG:32612", key="proj:code") for _ in range(61)]
    coverage = {"EPSG:32613": 0.53, "EPSG:32612": 1.0}

    def score(group):
        return coverage[group[0].properties["proj:code"]]

    chosen, selected = select_crs(many + few, score)
    assert chosen == "EPSG:32612"
    assert len(selected) == 61


def test_without_a_score_the_count_still_rules():
    many = [item("EPSG:32613", key="proj:code") for _ in range(87)]
    few = [item("EPSG:32612", key="proj:code") for _ in range(61)]
    assert select_crs(many + few)[0] == "EPSG:32613"


def test_at_equal_coverage_the_count_breaks_the_tie():
    many = [item("EPSG:32613", key="proj:code") for _ in range(87)]
    few = [item("EPSG:32612", key="proj:code") for _ in range(61)]
    chosen, _ = select_crs(many + few, lambda group: 1.0)
    assert chosen == "EPSG:32613"
