"""Network-free tests of the incumbent products layer.

Two pieces carry real risk. The Mollweide tile index decides which files are fetched, and
an off-by-one there downloads the wrong corner of the world without failing. The zonal
mean decides every number in the comparison, and its sub-pixel fallback is the part a
referee would question, so both the pixel-centre rule and the fallback are pinned.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box

from satinsight.incumbents import (
    MOLLWEIDE,
    TILE_ORIGIN_X,
    TILE_SIZE_M,
    tiles_for,
    zonal_mean,
)

GEOGRAPHIC = "EPSG:4326"


def test_tiles_for_matches_the_published_grid():
    """Acámbaro sits in R7_C9, the tile the JRC server actually serves for it."""
    assert tiles_for([(-100.740, 19.987, -100.676, 20.077)]) == [(7, 9)]


def test_tiles_for_covers_a_box_straddling_a_seam():
    """A box laid over a column boundary asks for the tiles on both sides of it."""
    seam_x = TILE_ORIGIN_X + 8 * TILE_SIZE_M
    to_geographic = Transformer.from_crs(MOLLWEIDE, GEOGRAPHIC, always_xy=True)
    west, south = to_geographic.transform(seam_x - 20_000, 2_400_000)
    east, north = to_geographic.transform(seam_x + 20_000, 2_450_000)

    chosen = tiles_for([(west, south, east, north)])
    columns = {col for _, col in chosen}
    assert columns == {8, 9}


def _raster(path, values, crs=GEOGRAPHIC):
    """Writes a small north-up raster whose pixel centres sit at half-integer degrees."""
    height, width = values.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(0, height, 1, 1),
    ) as destination:
        destination.write(values.astype("float32"), 1)
    return path


def _polygons(geometries, keys):
    return gpd.GeoDataFrame({"cvegeo": keys}, geometry=geometries, crs=GEOGRAPHIC)


@pytest.fixture
def grid(tmp_path):
    """A 4x4 raster whose value is the flat index of the pixel."""
    values = np.arange(16, dtype="float32").reshape(4, 4)
    return _raster(tmp_path / "grid.tif", values)


def test_zonal_mean_averages_the_pixels_whose_centre_falls_inside(grid):
    """Only centres count, so a polygon that clips a neighbour does not take its value."""
    polygon = box(0.2, 3.2, 1.8, 3.8)
    result = zonal_mean([grid], _polygons([polygon], ["A"]), ["A"], column="v")

    assert result.loc[0, "v_n_px"] == 2
    assert result.loc[0, "v"] == pytest.approx(0.5)


def test_zonal_mean_falls_back_to_the_centroid_below_one_pixel(grid):
    """A tract smaller than a cell keeps a value and is flagged with a zero pixel count."""
    polygon = box(2.40, 1.40, 2.45, 1.45)
    result = zonal_mean([grid], _polygons([polygon], ["B"]), ["B"], column="v")

    assert result.loc[0, "v_n_px"] == 0
    assert result.loc[0, "v"] == pytest.approx(10.0)


def test_zonal_mean_accumulates_across_tiles(tmp_path):
    """A polygon spanning two rasters averages over both, which is the seam case."""
    left = _raster(tmp_path / "left.tif", np.full((4, 4), 2.0))
    right_values = np.full((4, 4), 6.0)
    with rasterio.open(
        tmp_path / "right.tif",
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs=GEOGRAPHIC,
        transform=from_origin(4, 4, 1, 1),
    ) as destination:
        destination.write(right_values.astype("float32"), 1)

    polygon = box(2.2, 1.2, 5.8, 2.8)
    result = zonal_mean(
        [left, tmp_path / "right.tif"], _polygons([polygon], ["C"]), ["C"], column="v"
    )

    # four centres on the left raster at 2.0 and four on the right at 6.0
    assert result.loc[0, "v_n_px"] == 8
    assert result.loc[0, "v"] == pytest.approx(4.0)


def test_zonal_mean_refuses_mismatched_keys(grid):
    with pytest.raises(ValueError, match="against"):
        zonal_mean([grid], _polygons([box(0, 0, 1, 1)], ["A"]), ["A", "B"], column="v")
