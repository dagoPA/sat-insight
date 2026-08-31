"""Reading of remote rasters and intensity transforms.

Reads are always windowed: only the pixels of the box of interest are asked of the
server, which avoids downloading whole scenes of hundreds of megabytes.
"""

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from satinsight.aoi import Bbox
from satinsight.catalog import sign

GEOGRAPHIC_CRS = "EPSG:4326"


def read_window(
    href: str,
    bbox: Bbox,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reads from the remote COG only the window covering the box.

    The box arrives in geographic coordinates and is reprojected into the scene's own
    system. When `shape` is given the window is resampled to those dimensions, which is
    what aligns bands of different resolution.

    The signature is renewed here rather than when the catalogue is queried, because a
    composite takes longer than the token lives.
    """
    with rasterio.open(sign(href)) as source:
        bounds = transform_bounds(GEOGRAPHIC_CRS, source.crs, *bbox)
        window = from_bounds(*bounds, source.transform)
        target = shape or (int(window.height), int(window.width))
        # The boundless read's fill and the scene's sentinel mark the same thing: pixels
        # never observed. In a floating point raster both become NaN, which is what the
        # composite medians know to ignore. Sentinel-1 RTC declares -32768 and writes it
        # in radar shadow and outside the swath; letting it pass as a number sinks the
        # median of every city whose box peeks over the edge of the scene, and a zero fill
        # would read as null backscatter, which is not true either. Integer rasters
        # , Sentinel-2, WorldCover, keep the zero fill because they admit no NaN and their
        # zero already means outside the data.
        floating = np.issubdtype(np.dtype(source.dtypes[0]), np.floating)
        data = source.read(
            1,
            window=window,
            out_shape=target,
            boundless=True,
            fill_value=np.nan if floating else 0,
        )
        if floating and source.nodata is not None:
            data[data == np.float32(source.nodata)] = np.nan
        return data


def stretch(band: np.ndarray, lower: float = 2, upper: float = 98) -> np.ndarray:
    """Takes a band to 0-255, clipping at percentiles.

    Zeros count as missing, which is the fill convention of `read_window`. A band with no
    valid pixels comes back all zeros.
    """
    band = np.asarray(band, dtype="float32")
    valid = band[np.isfinite(band) & (band != 0)]
    if valid.size == 0:
        return np.zeros(band.shape, dtype="uint8")
    floor, ceiling = np.percentile(valid, [lower, upper])
    if ceiling <= floor:
        ceiling = floor + 1e-6
    return np.clip((band - floor) / (ceiling - floor) * 255, 0, 255).astype("uint8")


def to_db(power: np.ndarray) -> np.ndarray:
    """Converts linear backscatter to decibels.

    Null or negative values are left as NaN, which is what a pixel with no measurable
    return deserves.
    """
    power = np.asarray(power, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.where(power > 0, power, np.nan))


def percentiles(band: np.ndarray, lower: float = 5, upper: float = 95) -> tuple[float, float]:
    """A pair of percentiles of a band, ignoring NaN. Useful for legends."""
    return (
        round(float(np.nanpercentile(band, lower)), 1),
        round(float(np.nanpercentile(band, upper)), 1),
    )
