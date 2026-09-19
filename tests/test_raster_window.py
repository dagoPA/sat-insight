"""A window read in another reference system lands on that system's grid."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from satinsight import raster


def test_read_window_can_warp_into_the_neighbouring_zone(tmp_path, monkeypatch):
    # a small scene in zone 19 over a box near the 72W meridian, which zone 18 also maps
    path = tmp_path / "scene.tif"
    left, bottom, right, top = 145000, 1250000, 170000, 1270000
    height, width = 100, 400
    transform = from_bounds(left, bottom, right, top, width, height)
    data = np.linspace(1.0, 2.0, height * width, dtype="float32").reshape(height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32619",
        transform=transform,
        nodata=-32768.0,
    ) as out:
        out.write(data, 1)
    monkeypatch.setattr(raster, "sign", lambda href: href)
    bbox = (-72.2, 11.35, -72.1, 11.41)
    native = raster.read_window(str(path), bbox, (30, 50))
    warped = raster.read_window(str(path), bbox, (30, 50), crs="EPSG:32618")
    assert native.shape == warped.shape == (30, 50)
    assert np.isfinite(warped).mean() > 0.9
    assert abs(float(np.nanmean(warped)) - float(np.nanmean(native))) < 0.1
