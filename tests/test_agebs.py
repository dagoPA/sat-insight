"""Tests of the AGEB layer that touch neither disk nor network."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from satinsight.agebs import (
    CITIES,
    GRADES,
    INDICATORS,
    METRIC_CRS,
    ORDINAL,
    _connected_mass,
    grade_summary,
)


def squares(specs):
    """Metric GeoDataFrame of 1 km square AGEB, given as (key, x_km, y_km)."""
    return gpd.GeoDataFrame(
        {"cvegeo": [c for c, _, _ in specs]},
        geometry=[box(x * 1000, y * 1000, x * 1000 + 1000, y * 1000 + 1000) for _, x, y in specs],
        crs=METRIC_CRS,
    )


def table(grades, populations=None):
    populations = populations or [100] * len(grades)
    return pd.DataFrame(
        {
            "cvegeo": [f"{i:013d}" for i in range(len(grades))],
            "grade": grades,
            "population": populations,
        }
    )


def test_the_ordinal_order_runs_from_least_to_most_deprived():
    assert ORDINAL[GRADES[0]] == 0
    assert ORDINAL[GRADES[-1]] == len(GRADES) - 1
    assert list(ORDINAL.values()) == sorted(ORDINAL.values())


def test_there_are_seventeen_distinct_indicators():
    assert len(INDICATORS) == 17
    assert len(set(INDICATORS)) == 17


def test_every_city_declares_a_municipality_of_its_state():
    for key, city in CITIES.items():
        assert city.key == key
        assert len(city.state) == 2
        assert len(city.municipality) == 5
        assert city.municipality.startswith(city.state)


def test_the_summary_counts_every_class_even_when_absent():
    summary = grade_summary(table([GRADES[1], GRADES[1], GRADES[3]]))
    assert list(summary.index) == list(GRADES)
    assert summary.loc[GRADES[1], "agebs"] == 2
    assert summary.loc[GRADES[3], "agebs"] == 1
    assert summary.loc[GRADES[4], "agebs"] == 0


def test_the_summary_adds_population_per_class():
    summary = grade_summary(table([GRADES[1], GRADES[1], GRADES[3]], [10, 20, 5]))
    assert summary.loc[GRADES[1], "population"] == 30
    assert summary.loc[GRADES[3], "population"] == 5


def test_the_percentages_add_up_to_a_hundred():
    summary = grade_summary(table(list(GRADES[:4])))
    assert summary["pct_agebs"].sum() == pytest.approx(100.0, abs=0.2)


def test_an_empty_table_does_not_divide_by_zero():
    summary = grade_summary(table([]))
    assert summary["agebs"].sum() == 0
    assert summary["pct_agebs"].sum() == 0


def test_the_footprint_drops_the_distant_satellite():
    agebs = squares(
        [
            ("0710100010001", 0, 0),
            ("0710100010002", 1, 0),
            ("0799900010001", 40, 0),  # detached locality forty kilometres away
        ]
    )
    footprint = _connected_mass(agebs, "07101", bridge_m=2500)
    assert set(footprint["cvegeo"]) == {"0710100010001", "0710100010002"}


def test_the_footprint_keeps_the_adjoining_neighbour():
    agebs = squares(
        [
            ("0710100010001", 0, 0),
            ("0710200010001", 2, 0),  # another municipality, but conurbated
        ]
    )
    footprint = _connected_mass(agebs, "07101", bridge_m=2500)
    assert len(footprint) == 2


def test_a_single_footprint_passes_untouched():
    agebs = squares([("0710100010001", 0, 0), ("0710100010002", 1, 0)])
    assert len(_connected_mass(agebs, "07101", bridge_m=2500)) == 2


def test_the_footprint_with_most_core_agebs_is_chosen():
    agebs = squares(
        [
            ("0799900010001", 0, 0),
            ("0799900010002", 1, 0),
            ("0799900010003", 2, 0),
            ("0710100010001", 40, 0),
            ("0710100010002", 41, 0),
        ]
    )
    footprint = _connected_mass(agebs, "07101", bridge_m=2500)
    assert set(footprint["cvegeo"]) == {"0710100010001", "0710100010002"}


def test_a_generous_bridge_joins_what_a_strict_one_separates():
    agebs = squares([("0710100010001", 0, 0), ("0710100010002", 10, 0)])
    assert len(_connected_mass(agebs, "07101", bridge_m=1000)) == 1
    assert len(_connected_mass(agebs, "07101", bridge_m=6000)) == 2
