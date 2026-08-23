"""Analysis grid: the georeferencing that composites do not carry with them.

`read_window` reprojects the box into each scene's own system, cuts that window and
resamples it to the requested shape. The array that comes out then sits on a UTM grid
whose affine transform nobody stored, and without it there is no way to say which pixel
falls inside which AGEB.

This module rebuilds that transform in advance. By handing the composites the shape
computed here, the grid they produce matches the one this module describes, and polygons
can be reprojected into the same system and rasterised with no offset.

The reconstruction holds as long as every scene shares a reference system. Sentinel-2 is
delivered in MGRS tiles, and tiles near the edge of a UTM zone are published in both
neighbouring zones; resampling both groups to the same shape and stacking them misaligns
them without anything failing. `select_crs` keeps one group and returns only those scenes,
which is what makes the grid reconstructible.
"""

import logging
from collections.abc import Callable
from math import ceil
from typing import TYPE_CHECKING, NamedTuple

from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

if TYPE_CHECKING:
    from affine import Affine
    from pystac import Item

from satinsight.aoi import Bbox

log = logging.getLogger(__name__)

GEOGRAPHIC_CRS = "EPSG:4326"
RESOLUTION_M = 10
"""Pixel size of the working grid. It is the native resolution of Sentinel-2."""


class Grid(NamedTuple):
    """Grid on which a composite and the AGEB rasterised against it both live."""

    transform: "Affine"
    shape: tuple[int, int]
    crs: str
    bounds: tuple[float, float, float, float]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def megapixels(self) -> float:
        return self.height * self.width / 1e6


def _crs_code(item: "Item") -> str | None:
    """Reads the reference system a scene declares in the STAC."""
    code = item.properties.get("proj:epsg") or item.properties.get("proj:code")
    if code is None:
        return None
    code = str(code)
    return code if code.upper().startswith("EPSG:") else f"EPSG:{code}"


def select_crs(
    items: list["Item"], score: Callable[[list["Item"]], float] | None = None
) -> tuple[str, list["Item"]]:
    """Picks a reference system and returns only the scenes that declare it.

    Sentinel-2 MGRS tiles near the edge of a UTM zone are published in both neighbouring
    zones. Mérida sits some forty kilometres from meridian 90 and gets 148 scenes in zone
    15 against 145 in zone 16, each group covering the whole box on its own. Resampling
    both groups to the same shape and stacking them misaligns them without anything
    failing.

    Keeping the larger group settles that case, and fails when the two zones do not cover
    the same ground. Over Guasave the 87 scenes of zone 13 reach half the box and the 61 of
    zone 12 cover it whole: choosing by count left the city with no radar composite for
    lack of coverage, while two orbits see it entire.

    With `score` the decision moves to measured coverage and the count only breaks ties.
    The function receives the scenes of one zone and returns how much of the box they
    reach; it lives outside this module because measuring it means reading pixels.

    The code is read from the STAC projection extension, which saves the per-scene network
    request that opening every raster would cost.
    """
    groups: dict[str, list[Item]] = {}
    for item in items:
        code = _crs_code(item)
        if code is not None:
            groups.setdefault(code, []).append(item)

    if not groups:
        raise ValueError("no scene declares a reference system in the STAC")

    if score is None or len(groups) == 1:
        chosen = max(groups, key=lambda c: len(groups[c]))
    else:
        coverage = {c: score(v) for c, v in groups.items()}
        chosen = max(groups, key=lambda c: (round(coverage[c], 2), len(groups[c])))
        log.info(
            "coverage per zone: %s",
            ", ".join(f"{c} {100 * v:.0f}%" for c, v in sorted(coverage.items())),
        )
    if len(groups) > 1:
        dropped = ", ".join(
            f"{c}: {len(v)}" for c, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        )
        log.warning(
            "the box crosses a UTM zone and scenes arrive in %d systems (%s); "
            "compositing uses only %s so that misaligned grids are not stacked",
            len(groups),
            dropped,
            chosen,
        )
    return chosen, groups[chosen]


def common_crs(items: list["Item"]) -> str:
    """Reference system dominant among the scenes."""
    return select_crs(items)[0]


def grid_from_bbox(bbox: Bbox, crs: str, resolution_m: int = RESOLUTION_M) -> Grid:
    """Builds the grid `read_window` would produce for this box and system.

    The geographic box is taken into the target system and covered with square pixels of
    the requested resolution, rounding up so the border is not lost.
    """
    bounds = transform_bounds(GEOGRAPHIC_CRS, crs, *bbox)
    left, bottom, right, top = bounds
    width = max(1, ceil((right - left) / resolution_m))
    height = max(1, ceil((top - bottom) / resolution_m))
    transform = from_bounds(left, bottom, right, top, width, height)
    log.info("grid %dx%d px @%d m in %s", width, height, resolution_m, crs)
    return Grid(transform=transform, shape=(height, width), crs=crs, bounds=bounds)


def polygon_window(transform, geometry, shape: tuple[int, int]):
    """Slices and mask of a polygon over the grid, or `None` when it falls outside.

    This is the step common to any zonal statistic: bound the polygon to its enclosing
    window and know which pixels of that window fall inside. It lives here so that texture
    and cover fractions cut in exactly the same way; if each solved it on its own, an
    offset between them would go unnoticed and we would compare them believing they look
    at the same pixels.

    The geometry has to arrive in the reference system of `transform`.
    """
    from rasterio.features import geometry_mask
    from rasterio.transform import rowcol

    height, width = shape
    x_min, y_min, x_max, y_max = geometry.bounds
    row_a, col_a = rowcol(transform, x_min, y_max)
    row_b, col_b = rowcol(transform, x_max, y_min)
    row_start, row_end = max(0, min(row_a, row_b)), min(height, max(row_a, row_b) + 1)
    col_start, col_end = max(0, min(col_a, col_b)), min(width, max(col_a, col_b) + 1)
    if row_end <= row_start or col_end <= col_start:
        return None

    rows = slice(row_start, row_end)
    columns = slice(col_start, col_end)
    inside = ~geometry_mask(
        [geometry],
        out_shape=(row_end - row_start, col_end - col_start),
        transform=transform * transform.translation(col_start, row_start),
        invert=False,
    )
    return rows, columns, inside


def grid_from_scenes(
    bbox: Bbox,
    items: list["Item"],
    resolution_m: int = RESOLUTION_M,
    score: Callable[[list["Item"]], float] | None = None,
) -> tuple[Grid, list["Item"]]:
    """Working grid together with the scenes that live on it.

    Returns the filtered scenes as well as the grid: compositing with those left outside
    the chosen system would produce a misaligned stack.
    """
    crs, selected = select_crs(items, score)
    return grid_from_bbox(bbox, crs, resolution_m), selected
