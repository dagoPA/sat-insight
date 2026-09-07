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
