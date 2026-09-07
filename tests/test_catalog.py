import pytest

from satinsight.catalog import by_cloud_cover, cloud_summary, dominant_orbit, group_by_orbit


class FakeScene:
    """Minimal stand-in for pystac.Item, to test the pure logic of the module."""

    def __init__(self, identifier: str, **properties):
        self.id = identifier
        self.properties = properties


def optical(identifier, cloud_cover):
    return FakeScene(identifier, **{"eo:cloud_cover": cloud_cover})


def sar(identifier, state, relative):
    return FakeScene(
        identifier,
        **{"sat:orbit_state": state, "sat:relative_orbit": relative},
    )


def test_cloud_summary_computes_the_shares():
    scenes = [optical(f"e{i}", n) for i, n in enumerate([0, 10, 55, 60, 85, 90, 95, 99])]
    summary = cloud_summary(scenes)
    assert summary["scenes"] == 8
    assert summary["minimum"] == 0
    assert summary["maximum"] == 99
    assert summary["pct_over_50"] == 75
    assert summary["pct_over_80"] == 50


def test_cloud_summary_without_scenes_fails():
    with pytest.raises(ValueError, match="no scenes to summarise"):
        cloud_summary([])


def test_by_cloud_cover_sorts_ascending():
    scenes = [optical("a", 80), optical("b", 5), optical("c", 40)]
    assert [e.id for e in by_cloud_cover(scenes)] == ["b", "c", "a"]


def test_group_by_orbit_separates_geometries():
    scenes = [
        sar("a", "ascending", 99),
        sar("b", "descending", 99),
        sar("c", "ascending", 99),
        sar("d", "ascending", 143),
    ]
    groups = group_by_orbit(scenes)
    assert len(groups) == 3
    assert len(groups[("ascending", 99)]) == 2


def test_dominant_orbit_picks_the_most_populated():
    scenes = [
        sar("a", "ascending", 99),
        sar("b", "descending", 41),
        sar("c", "descending", 41),
        sar("d", "descending", 41),
    ]
    key, selection = dominant_orbit(scenes)
    assert key == ("descending", 41)
    assert len(selection) == 3


def test_dominant_orbit_without_scenes_fails():
    with pytest.raises(ValueError, match="no SAR scenes to group"):
        dominant_orbit([])
