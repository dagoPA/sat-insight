"""Tests of the texture features, over synthetic images of known properties."""

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from satinsight.texture import (
    DISTANCES,
    LEVELS,
    features_of_patch,
    features_per_ageb,
    first_order_features,
    quantise,
    robust_range,
)

TRANSFORM = Affine.translation(0.0, 2000.0) * Affine.scale(10.0, -10.0)
"""Metric 10 m grid with its origin at the top left, like that of a composite."""


def image_with_patch(seed=0):
    """Smooth background with a central patch of high variance."""
    rng = np.random.default_rng(seed)
    band = rng.normal(0.2, 0.01, (200, 200))
    band[50:150, 50:150] += rng.normal(0, 0.15, (100, 100))
    return band


def pixel_box(row_start, row_stop, col_start, col_stop):
    """Translates a range of rows and columns into the polygon covering it."""
    x_min, y_max = TRANSFORM * (col_start, row_start)
    x_max, y_min = TRANSFORM * (col_stop, row_stop)
    return box(x_min, y_min, x_max, y_max)


def test_quantise_respects_the_range_and_reserves_zero():
    band = np.linspace(0.0, 1.0, 100).reshape(10, 10)
    q = quantise(band, (0.0, 1.0), LEVELS)
    assert q.min() == 1
    assert q.max() == LEVELS


def test_non_finite_values_end_at_zero():
    band = np.full((4, 4), 0.5)
    band[0, 0] = np.nan
    q = quantise(band, (0.0, 1.0))
    assert q[0, 0] == 0
    assert (q[1:] > 0).all()


def test_out_of_range_values_are_clipped_without_being_lost():
    band = np.array([[-5.0, 0.5, 9.0]])
    q = quantise(band, (0.0, 1.0))
    assert q[0, 0] == 1
    assert q[0, 2] == LEVELS


def test_a_degenerate_range_fails():
    with pytest.raises(ValueError, match="degenerate range"):
        quantise(np.ones((3, 3)), (1.0, 1.0))


def test_the_robust_range_ignores_non_finite_values():
    band = np.array([[np.nan, 1.0, 2.0, 3.0, np.inf]])
    low, high = robust_range(band, 0, 100)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(3.0)


def test_a_band_without_finite_pixels_fails():
    with pytest.raises(ValueError, match="not one finite pixel"):
        robust_range(np.full((3, 3), np.nan))


def test_a_textured_region_contrasts_more_than_a_smooth_one():
    band = image_with_patch()
    smooth = pixel_box(5, 45, 5, 45)  # 1600 px, outside the patch
    rough = pixel_box(60, 100, 60, 100)  # 1600 px, inside the patch
    table = features_per_ageb(band, TRANSFORM, [smooth, rough], ["smooth", "rough"], prefix="c")
    row = table.set_index("cvegeo")

    assert row.loc["rough", "c_contrast_d1"] > 5 * row.loc["smooth", "c_contrast_d1"]
    assert row.loc["rough", "c_homogeneity_d1"] < row.loc["smooth", "c_homogeneity_d1"]
    assert row.loc["rough", "c_entropy_d1"] > row.loc["smooth", "c_entropy_d1"]


def test_each_distance_comes_out_as_its_own_column():
    band = image_with_patch()
    rough = pixel_box(60, 100, 60, 100)
    table = features_per_ageb(band, TRANSFORM, [rough], ["x"], prefix="c")
    for distance in DISTANCES:
        assert f"c_contrast_d{distance}" in table.columns
    # the wider the separation, the higher the contrast on a surface with no periodic
    # structure
    assert table.loc[0, "c_contrast_d4"] > table.loc[0, "c_contrast_d1"]


def test_a_fixed_range_does_not_depend_on_the_band():
    """Two bands at different levels must quantise alike if the range is fixed.

    It is what holds up the comparability of radar between cities and between countries.
    """
    rng = np.random.default_rng(1)
    pattern = rng.normal(0, 1, (60, 60))
    square = pixel_box(0, 60, 0, 60)
    fixed = (-10.0, 10.0)

    a = features_per_ageb(pattern, TRANSFORM, [square], ["x"], prefix="c", value_range=fixed)
    b = features_per_ageb(pattern + 3.0, TRANSFORM, [square], ["x"], prefix="c", value_range=fixed)
    free_a = features_per_ageb(pattern, TRANSFORM, [square], ["x"], prefix="c")
    free_b = features_per_ageb(pattern + 3.0, TRANSFORM, [square], ["x"], prefix="c")

    assert a.loc[0, "c_contrast_d1"] != pytest.approx(b.loc[0, "c_contrast_d1"])
    assert free_a.loc[0, "c_contrast_d1"] == pytest.approx(free_b.loc[0, "c_contrast_d1"])


def test_a_tiny_polygon_is_reported_without_features():
    band = image_with_patch()
    tiny = pixel_box(10, 12, 10, 12)
    table = features_per_ageb(band, TRANSFORM, [tiny], ["x"], prefix="c", min_pixels=50)
    assert table.loc[0, "c_n_px"] < 50
    assert "c_contrast_d1" not in table.columns


def test_a_polygon_outside_the_raster_does_not_break():
    band = image_with_patch()
    far_away = box(500_000, 500_000, 500_100, 500_100)
    table = features_per_ageb(band, TRANSFORM, [far_away], ["outside"], prefix="c")
    assert table.loc[0, "c_n_px"] == 0


def test_mismatched_geometries_and_keys_fail():
    band = image_with_patch()
    with pytest.raises(ValueError, match="geometries"):
        features_per_ageb(band, TRANSFORM, [pixel_box(0, 10, 0, 10)], ["a", "b"], prefix="c")


def test_a_constant_crop_has_no_entropy():
    constant = np.full((30, 30), 5, dtype=np.uint8)
    features = features_of_patch(constant)
    assert features["entropy_d1"] == pytest.approx(0.0, abs=1e-9)
    assert features["contrast_d1"] == pytest.approx(0.0, abs=1e-9)


def test_an_empty_crop_returns_nan():
    features = features_of_patch(np.zeros((20, 20), dtype=np.uint8))
    assert np.isnan(features["contrast_d1"])


def test_first_order_reproduces_the_mean_and_the_spread():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    features = first_order_features(values)
    assert features["mean"] == pytest.approx(3.0)
    assert features["p50"] == pytest.approx(3.0)
    assert features["std"] == pytest.approx(np.std(values))


def test_first_order_without_data_returns_nan():
    features = first_order_features(np.array([]))
    assert all(np.isnan(v) for v in features.values())
