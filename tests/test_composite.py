"""Tests of the compositing. The remote read is replaced by a stand-in."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from satinsight import composite
from satinsight.composite import _check_failures, composite_s1

BBOX = (-93.135, 16.740, -93.095, 16.768)
SHAPE = (8, 8)


@dataclass
class FakeAsset:
    href: str


@dataclass
class FakeItem:
    """Stand-in for a SAR scene, with the minimum `dominant_orbit` looks at."""

    id: str
    assets: dict = field(default_factory=dict)
    properties: dict = field(default_factory=dict)


def sar_scene(name: str, orbit: int = 99) -> FakeItem:
    return FakeItem(
        id=name,
        assets={"vv": FakeAsset(f"{name}/vv"), "vh": FakeAsset(f"{name}/vh")},
        properties={"sat:orbit_state": "ascending", "sat:relative_orbit": orbit},
    )


def test_a_few_failures_do_not_abort():
    _check_failures("Sentinel-2", failed=2, attempted=20, fraction=0.3)


def test_too_many_failures_abort():
    with pytest.raises(RuntimeError, match="failed to read"):
        _check_failures("Sentinel-2", failed=15, attempted=20, fraction=0.3)


def test_none_disables_the_check():
    _check_failures("Sentinel-2", failed=20, attempted=20, fraction=None)


def test_zero_is_the_strict_extreme_and_not_disabled():
    _check_failures("Sentinel-2", failed=0, attempted=20, fraction=0.0)
    with pytest.raises(RuntimeError, match="failed to read"):
        _check_failures("Sentinel-2", failed=1, attempted=20, fraction=0.0)


def test_no_attempted_scenes_does_not_divide_by_zero():
    _check_failures("Sentinel-1", failed=0, attempted=0, fraction=0.3)


def test_a_broken_read_does_not_desynchronise_the_polarisations(monkeypatch):
    """If VH fails, VV must not stay stacked on its own.

    The medians of the two polarisations would come out computed over different sets of
    scenes, and the ratio between them would stop meaning what it claims to mean.
    """

    def read(href, bbox, shape=None):
        if href == "b/vh":
            raise OSError("broken read")
        return np.ones(SHAPE, dtype="float32")

    monkeypatch.setattr(composite, "read_window", read)
    scenes = [sar_scene(name) for name in "abcde"]  # a single failure, under the threshold
    bands, meta = composite_s1(scenes, BBOX, SHAPE)

    assert bands["vv"].shape == SHAPE
    assert meta["scenes_used"] == 4


def test_it_aborts_when_almost_every_read_fails(monkeypatch):
    def read(href, bbox, shape=None):
        raise OSError("HTTP response code: 403")

    monkeypatch.setattr(composite, "read_window", read)
    scenes = [sar_scene(name) for name in "abcde"]
    with pytest.raises(RuntimeError, match="failed to read"):
        composite_s1(scenes, BBOX, SHAPE)


def test_no_scenes_fails_with_a_clear_message():
    with pytest.raises(ValueError, match="no Sentinel-1 scenes"):
        composite_s1([], BBOX, SHAPE)


def test_the_orbit_is_chosen_by_measured_coverage(monkeypatch):
    """An orbit with fewer passes wins if it is the one that really sees the city.

    Mexicali's case: the most repeated orbit of the catalogue grazes the box at the edge of
    its swath and leaves almost everything unobserved.
    """
    scenes = [sar_scene(f"edge{i}", orbit=166) for i in range(5)]
    scenes += [sar_scene(f"full{i}", orbit=173) for i in range(2)]

    def read(href, bbox, shape):
        array = np.full(shape, np.nan, dtype="float32")
        if href.startswith("full"):
            array[:] = 1.0
        else:
            array[0, 0] = 1.0
        return array

    monkeypatch.setattr(composite, "read_window", read)
    key, selection, coverage = composite.useful_orbit(scenes, BBOX)
    assert key == ("ascending", 173)
    assert len(selection) == 2
    assert coverage == pytest.approx(1.0)


def test_at_equal_coverage_the_orbit_with_more_scenes_wins(monkeypatch):
    scenes = [sar_scene(f"a{i}", orbit=10) for i in range(2)]
    scenes += [sar_scene(f"b{i}", orbit=20) for i in range(6)]
    monkeypatch.setattr(composite, "read_window", lambda h, b, f: np.ones(f, dtype="float32"))
    key, selection, _ = composite.useful_orbit(scenes, BBOX)
    assert key == ("ascending", 20)
    assert len(selection) == 6


def test_an_entirely_unreadable_orbit_counts_as_no_coverage(monkeypatch):
    def read(href, bbox, shape):
        raise OSError("403")

    monkeypatch.setattr(composite, "read_window", read)
    assert composite.useful_coverage([sar_scene("x")], BBOX) == 0.0


def test_a_dropped_read_does_not_sink_its_orbit(monkeypatch):
    """Guasave's case: one lost request knocked out an orbit that covers everything.

    Counting the failed read as zero coverage confuses the orbit not seeing the city with
    the link being cut, and under congestion the second is frequent.
    """
    scenes = [sar_scene(f"good{i}", orbit=20) for i in range(4)]

    def read(href, bbox, shape):
        if href.startswith("good0"):
            raise OSError("connection dropped")
        return np.ones(shape, dtype="float32")

    monkeypatch.setattr(composite, "read_window", read)
    assert composite.useful_coverage(scenes, BBOX) == pytest.approx(1.0)


def test_a_radar_composite_with_zeros_is_rejected():
    """Linear gamma0 is positive: a zero betrays no-data that slipped into the median."""
    array = np.full((8, 8), 0.2, dtype="float32")
    array[0, 0] = 0.0
    with pytest.raises(RuntimeError, match="zero or negative"):
        composite._check_composite_s1({"vv": array})


def test_a_radar_composite_with_an_intermediate_value_is_rejected():
    """The case that hides: the median averages -32768 with a good value."""
    array = np.full((8, 8), -16384.0, dtype="float32")
    with pytest.raises(RuntimeError, match="zero or negative"):
        composite._check_composite_s1({"vv": array})


def test_a_radar_composite_mostly_unobserved_is_rejected():
    array = np.full((10, 10), np.nan, dtype="float32")
    array[:5] = 0.2
    with pytest.raises(RuntimeError, match="no orbit covers"):
        composite._check_composite_s1({"vv": array})


def test_a_healthy_radar_composite_passes():
    array = np.full((10, 10), 0.2, dtype="float32")
    array[0] = np.nan
    assert composite._check_composite_s1({"vv": array, "vh": array}) == pytest.approx(0.9)


def optical_scene(name: str, tile: str, cloud_cover: float = 10.0) -> FakeItem:
    return FakeItem(
        id=name,
        assets={b: FakeAsset(f"{name}/{b}") for b in ("SCL", "B04", "B03", "B02")},
        properties={"s2:mgrs_tile": tile, "eo:cloud_cover": cloud_cover},
    )


def test_the_tile_that_does_not_touch_the_box_is_dropped(monkeypatch):
    """San Pedro Tlaquepaque's case: the clearest scenes belong to the wrong tile.

    Outside its footprint the read arrives as zeros, and SCL zero means no data, so the
    scene contributes not one pixel however clear it comes.
    """
    scenes = [optical_scene(f"far{i}", "13QFD", cloud_cover=1.0) for i in range(19)]
    scenes += [optical_scene("over", "13QFC", cloud_cover=40.0)]

    def read(href, bbox, shape):
        return np.full(shape, 0 if href.startswith("far") else 4, dtype="uint8")

    monkeypatch.setattr(composite, "read_window", read)
    useful = composite.useful_tiles(scenes, BBOX)
    assert [e.id for e in useful] == ["over"]


def test_a_split_box_keeps_both_tiles(monkeypatch):
    scenes = [optical_scene(f"a{i}", "14QKH") for i in range(3)]
    scenes += [optical_scene(f"b{i}", "14QLH") for i in range(3)]

    def read(href, bbox, shape):
        array = np.zeros(shape, dtype="uint8")
        if href.startswith("a"):
            array[:, : shape[1] // 2] = 4
        else:
            array[:, shape[1] // 2 :] = 5
        return array

    monkeypatch.setattr(composite, "read_window", read)
    assert len(composite.useful_tiles(scenes, BBOX)) == 6


def test_no_useful_tile_at_all_fails(monkeypatch):
    monkeypatch.setattr(composite, "read_window", lambda h, b, f: np.zeros(f, dtype="uint8"))
    with pytest.raises(RuntimeError, match="none of the"):
        composite.useful_tiles([optical_scene("x", "14QKH")], BBOX)
