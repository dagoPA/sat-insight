import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from satinsight.raster import percentiles, read_window, stretch, to_db


def test_stretch_returns_the_full_range():
    band = np.arange(1, 101, dtype="float32").reshape(10, 10)
    output = stretch(band, lower=0, upper=100)
    assert output.dtype == np.uint8
    assert output.min() == 0
    assert output.max() == 255


def test_stretch_without_valid_pixels_returns_zeros():
    output = stretch(np.zeros((4, 4), dtype="float32"))
    assert output.shape == (4, 4)
    assert not output.any()


def test_stretch_treats_zero_as_no_data():
    band = np.array([[0, 0], [10, 20]], dtype="float32")
    output = stretch(band, lower=0, upper=100)
    # the minimum of the stretch comes from 10, not from the fill
    assert output[1, 0] == 0
    assert output[1, 1] == 255


def test_stretch_on_a_constant_band_does_not_divide_by_zero():
    output = stretch(np.full((3, 3), 7.0, dtype="float32"))
    assert np.isfinite(output).all()


def test_to_db_converts_a_known_power():
    power = np.array([[1.0, 0.1, 10.0]], dtype="float32")
    result = to_db(power)
    np.testing.assert_allclose(result, [[0.0, -10.0, 10.0]], atol=1e-4)


def test_to_db_marks_non_positives_as_nan():
    result = to_db(np.array([[0.0, -1.0, 1.0]], dtype="float32"))
    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])
    assert result[0, 2] == 0.0


def test_percentiles_ignore_nan():
    band = np.array([[np.nan, 1.0, 2.0, 3.0, 4.0]], dtype="float32")
    lower, upper = percentiles(band, 0, 100)
    assert lower == 1.0
    assert upper == 4.0


def test_the_scene_sentinel_becomes_nan(tmp_path):
    """The no-data value of a floating point raster must not enter the median.

    Sentinel-1 RTC declares -32768 and writes it outside the swath and in radar shadow.
    Read as a number it sinks the composite of the cities that peek past the scene edge.
    """
    path = tmp_path / "sar.tif"
    data = np.full((8, 8), 0.25, dtype="float32")
    data[:4] = -32768.0
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(4326),
        "transform": from_origin(-93.14, 16.77, 0.005, 0.005),
        "nodata": -32768.0,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(data, 1)

    read = read_window(str(path), (-93.14, 16.73, -93.10, 16.77), (8, 8))
    assert np.isnan(read[:4]).all()
    assert np.allclose(read[4:], 0.25)


def test_an_integer_raster_keeps_the_fill_at_zero(tmp_path):
    """Sentinel-2 and WorldCover arrive as integers, which admit no NaN."""
    path = tmp_path / "optical.tif"
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "uint16",
        "crs": CRS.from_epsg(4326),
        "transform": from_origin(-93.14, 16.77, 0.005, 0.005),
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(np.full((8, 8), 1200, dtype="uint16"), 1)

    read = read_window(str(path), (-93.20, 16.73, -93.10, 16.77), (8, 16))
    assert read.dtype == np.uint16
    assert (read == 0).any()
