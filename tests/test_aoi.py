import pytest

from satinsight.aoi import AOI, PILOT, get


def test_every_pilot_box_is_valid():
    for key, area in PILOT.items():
        assert area.key == key
        assert area.width_degrees > 0
        assert area.height_degrees > 0


def test_the_pilot_boxes_share_a_size():
    widths = {round(a.width_degrees, 6) for a in PILOT.values()}
    heights = {round(a.height_degrees, 6) for a in PILOT.values()}
    assert len(widths) == 1, "the pilot boxes must be comparable with each other"
    assert len(heights) == 1


def test_approximate_shape_is_plausible():
    height, width = PILOT["tuxtla"].approximate_shape(resolution_m=10)
    assert 250 < height < 350
    assert 350 < width < 480


@pytest.mark.parametrize(
    "bbox",
    [
        (-93.0, 16.7, -93.1, 16.8),  # longitudes inverted
        (-93.1, 16.8, -93.0, 16.7),  # latitudes inverted
        (-200.0, 16.7, -93.0, 16.8),  # longitude out of range
        (-93.1, -95.0, -93.0, 16.8),  # latitude out of range
    ],
)
def test_an_invalid_bbox_fails(bbox):
    with pytest.raises(ValueError):
        AOI(key="x", name="x", state="x", bbox=bbox)


def test_get_on_an_unknown_key_suggests_the_available_ones():
    with pytest.raises(KeyError, match="tuxtla"):
        get("saltillo")


def test_at_least_widens_small_boxes_around_their_centre():
    from satinsight.aoi import AOI

    small = AOI(key="k", name="n", state="s", bbox=(-100.01, 20.0, -99.99, 20.01))
    wide = small.at_least(3360.0)
    height, width = wide.approximate_shape()
    assert 330 <= height <= 340 and 330 <= width <= 340
    assert abs((wide.bbox[0] + wide.bbox[2]) / 2 - (-100.0)) < 1e-9
    assert abs((wide.bbox[1] + wide.bbox[3]) / 2 - 20.005) < 1e-9
    big = AOI(key="k", name="n", state="s", bbox=(-100.1, 20.0, -99.9, 20.1))
    assert big.at_least(3360.0).bbox == big.bbox
