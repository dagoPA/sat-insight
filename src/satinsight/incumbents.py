"""Downloadable global products the map is measured against.

The RWI comparison answers one referee question: does the map beat the leading gridded
wealth product? It leaves the two cheaper objections standing. A practitioner who wanted
to rank neighborhoods today would not train anything; they would download built-up
density, building height, or nighttime lights, and sort by that. If those already order
census tracts as well as the map does, the supervision-efficiency result loses its point.

Four products are pinned here, all free and unauthenticated:

- GHS-BUILT-S R2023A, built-up surface in square metres per 100 m cell, 2020 epoch.
- GHS-BUILT-H R2023A, average net building height, 2018 epoch, the only year published.
- GHS-POP R2023A, residential population per cell, 2020 epoch.
- The harmonized global nighttime lights of Li et al. (Sci. Data 2020), 2020 layer,
  VIIRS-derived and calibrated onto the DMSP digital-number scale, 30 arcsec.

Nighttime lights come from the harmonized series rather than the EOG annual composite
because EOG puts its files behind an account. The harmonized layer is the same VIIRS
observation reduced to a coarser grid and a compressed scale, and the paper states that
where the number is reported.

All four are development proxies: more built surface, taller buildings, more people per
cell, more light at night all read as less deprived in the way practitioners use them.
`DEPRIVATION_SIGN` fixes that orientation once, before anything is measured, so the
comparison cannot be tuned by flipping a sign after seeing the correlation.

The GHSL tiling is Mollweide, one million metres to a side, indexed from the northwest
corner of the global extent. `tiles_for` turns a set of boxes into the tiles that cover
them, which is what keeps the download to the seven tiles Mexico needs instead of the
global mosaic.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from math import floor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window, from_bounds

from satinsight.download import DATA_ROOT, PLAIN_USER_AGENT, download

log = logging.getLogger(__name__)

GHSL_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
MOLLWEIDE = "ESRI:54009"

TILE_SIZE_M = 1_000_000
"""Side of a GHSL R2023A tile in Mollweide metres."""

TILE_ORIGIN_X = -18_041_000.0
TILE_ORIGIN_Y = 9_000_000.0
"""Northwest corner the R2023A tile index counts from. Column and row both start at one."""


@dataclass(frozen=True)
class Product:
    """One GHSL product at one epoch, enough to build every tile URL it has."""

    key: str
    folder: str
    dataset: str
    label: str
    units: str

    def url(self, row: int, col: int) -> str:
        stem = f"{self.dataset}_V1_0_R{row}_C{col}"
        return f"{GHSL_BASE}/{self.folder}/{self.dataset}/V1-0/tiles/{stem}.zip"

    def tile_name(self, row: int, col: int) -> str:
        return f"{self.dataset}_V1_0_R{row}_C{col}"


GHSL_PRODUCTS: dict[str, Product] = {
    "built_surface": Product(
        key="built_surface",
        folder="GHS_BUILT_S_GLOBE_R2023A",
        dataset="GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100",
        label="GHSL built-up surface",
        units="m2 per 100 m cell",
    ),
    "built_height": Product(
        key="built_height",
        folder="GHS_BUILT_H_GLOBE_R2023A",
        dataset="GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100",
        label="GHSL building height",
        units="average net building height, m",
    ),
    "population": Product(
        key="population",
        folder="GHS_POP_GLOBE_R2023A",
        dataset="GHS_POP_E2020_GLOBE_R2023A_54009_100",
        label="GHSL population",
        units="residents per 100 m cell",
    ),
}

NIGHTLIGHTS_KEY = "nightlights"
NIGHTLIGHTS_URL = "https://ndownloader.figshare.com/files/57065297"
NIGHTLIGHTS_NAME = "Harmonized_DN_NTL_2020_simVIIRS.tif"
NIGHTLIGHTS_LABEL = "Harmonized nighttime lights"

DEPRIVATION_SIGN: dict[str, int] = {
    "built_surface": -1,
    "built_height": -1,
    "population": -1,
    NIGHTLIGHTS_KEY: -1,
}
"""Sign that turns each product into a deprivation score, fixed before measuring.

Every one of the four is used in practice as a development proxy, so deprivation is the
negative of the product in all four cases. Population is the weakest of the four
conventions, and the paper reports it as a density control rather than as a wealth
product.
"""

LABELS: dict[str, str] = {key: p.label for key, p in GHSL_PRODUCTS.items()} | {
    NIGHTLIGHTS_KEY: NIGHTLIGHTS_LABEL
}


def tiles_for(bounds: list[tuple[float, float, float, float]]) -> list[tuple[int, int]]:
    """GHSL tiles covering a list of geographic boxes.

    The corners of each box are projected into Mollweide and the tile index is read off
    the global grid. Corners are enough because the tile grid is coarser by orders of
    magnitude than any city box, and taking the whole rectangle between the corner
    indices covers a box that straddles a seam.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", MOLLWEIDE, always_xy=True)
    chosen: set[tuple[int, int]] = set()
    for west, south, east, north in bounds:
        lons = [west, east, west, east]
        lats = [south, south, north, north]
        xs, ys = transformer.transform(lons, lats)
        cols = [floor((x - TILE_ORIGIN_X) / TILE_SIZE_M) + 1 for x in xs]
        rows = [floor((TILE_ORIGIN_Y - y) / TILE_SIZE_M) + 1 for y in ys]
        for row in range(min(rows), max(rows) + 1):
            for col in range(min(cols), max(cols) + 1):
                chosen.add((row, col))
    return sorted(chosen)


def ensure_ghsl(
    product: str,
    tiles: list[tuple[int, int]],
    root: Path = DATA_ROOT,
    *,
    force: bool = False,
) -> list[Path]:
    """Leaves the requested tiles of one GHSL product on disk and returns their paths.

    Tiles are downloaded as zips and unpacked next to them. A tile the server does not
    hold is skipped with a warning: the tile grid is global and covers ocean, so a box on
    the coast legitimately asks for tiles that were never published.
    """
    spec = GHSL_PRODUCTS[product]
    folder = root / "externos" / "ghsl" / product
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for row, col in tiles:
        name = spec.tile_name(row, col)
        tif = folder / f"{name}.tif"
        if tif.exists() and not force:
            paths.append(tif)
            continue
        try:
            archive = download(spec.url(row, col), folder / f"{name}.zip", force=force)
        except RuntimeError as error:
            log.warning("%s: tile R%d_C%d unavailable (%s)", product, row, col, error)
            continue
        import zipfile

        with zipfile.ZipFile(archive) as z:
            members = [n for n in z.namelist() if n.endswith(".tif")]
            if not members:
                raise FileNotFoundError(f"{archive.name} carries no GeoTIFF")
            z.extract(members[0], folder)
        extracted = folder / members[0]
        if extracted != tif:
            extracted.replace(tif)
        archive.unlink(missing_ok=True)
        paths.append(tif)
    if not paths:
        raise RuntimeError(f"no {product} tile could be obtained for {tiles}")
    log.info("%s: %d tiles on disk", product, len(paths))
    return paths


def ensure_nightlights(root: Path = DATA_ROOT, *, force: bool = False) -> list[Path]:
    """Leaves the 2020 harmonized nighttime lights layer on disk.

    Figshare is served from behind a WAF that challenges browser user agents, so this one
    download announces itself as a script.
    """
    folder = root / "externos" / "nightlights"
    path = download(
        NIGHTLIGHTS_URL,
        folder / NIGHTLIGHTS_NAME,
        force=force,
        user_agent=PLAIN_USER_AGENT,
    )
    return [path]


def _window_of(source, bounds: tuple[float, float, float, float]) -> Window | None:
    """Window of a raster covering a box, clipped to the raster, or None if disjoint.

    The clipping is done on the offsets rather than through `Window.intersection`, which
    raises on a window of zero area. A box smaller than one pixel rounds to exactly that,
    and it is a legitimate input here: the caller asks for the extent of a set of polygons,
    and a single tract below the cell size produces one.

    The box is grown by a pixel on each side before rounding, so a polygon whose centre
    pixel sits just outside the rounded window is still read.
    """
    window = from_bounds(*bounds, source.transform)
    col_off = int(np.floor(window.col_off)) - 1
    row_off = int(np.floor(window.row_off)) - 1
    col_end = int(np.ceil(window.col_off + window.width)) + 1
    row_end = int(np.ceil(window.row_off + window.height)) + 1

    col_off = max(col_off, 0)
    row_off = max(row_off, 0)
    col_end = min(col_end, source.width)
    row_end = min(row_end, source.height)
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def zonal_mean(
    paths: list[Path],
    polygons: gpd.GeoDataFrame,
    keys,
    *,
    column: str,
) -> pd.DataFrame:
    """Mean raster value inside each polygon, in the raster's own grid.

    Polygons are reprojected into the raster and rasterised there, so nothing resamples
    the product itself: a 100 m cell counts for the polygon whose interior holds its
    centre. Sums and counts accumulate across tiles, which is what lets a city that
    straddles a tile seam come out with one mean.

    A polygon smaller than a cell captures no centre at all. Dropping it would quietly
    restrict the comparison to large tracts and bias it, so the value is sampled at a
    representative interior point and the pixel count is recorded as zero, which leaves the
    fallback auditable and, if a referee asks, removable. The point comes from
    `representative_point` because the centroid of a concave tract can land outside it, and
    a coastal or L-shaped AGEB would then take a neighbour's cell.
    """
    if len(polygons) != len(keys):
        raise ValueError(f"{len(polygons)} polygons against {len(keys)} keys")

    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    interior_value: dict[int, float] = {}

    for path in paths:
        with rasterio.open(path) as source:
            local = polygons.to_crs(source.crs)
            window = _window_of(source, tuple(local.total_bounds))
            if window is None:
                continue
            data = source.read(1, window=window).astype("float64")
            transform = source.window_transform(window)
            if source.nodata is not None:
                data[data == source.nodata] = np.nan

            shapes = [
                (geometry, index + 1)
                for index, geometry in enumerate(local.geometry)
                if geometry is not None and not geometry.is_empty
            ]
            labels = features.rasterize(
                shapes,
                out_shape=data.shape,
                transform=transform,
                fill=0,
                dtype="int32",
                all_touched=False,
            )
            valid = (labels > 0) & np.isfinite(data)
            if valid.any():
                flat = labels[valid]
                values = data[valid]
                summed = np.bincount(flat, weights=values, minlength=len(local) + 1)
                seen = np.bincount(flat, minlength=len(local) + 1)
                for index in np.nonzero(seen)[0]:
                    totals[int(index) - 1] += float(summed[index])
                    counts[int(index) - 1] += int(seen[index])

            missing = [i for i in range(len(local)) if counts.get(i, 0) == 0]
            if missing:
                inside = local.geometry.iloc[missing].representative_point()
                sampled = list(source.sample(np.column_stack([inside.x, inside.y]).tolist()))
                for i, value in zip(missing, sampled, strict=True):
                    number = float(value[0])
                    if source.nodata is not None and number == source.nodata:
                        continue
                    if np.isfinite(number):
                        interior_value.setdefault(i, number)

    means = []
    pixels = []
    for index in range(len(polygons)):
        if counts.get(index, 0):
            means.append(totals[index] / counts[index])
        else:
            means.append(interior_value.get(index, np.nan))
        pixels.append(counts.get(index, 0))

    frame = pd.DataFrame({"cvegeo": list(keys), column: means, f"{column}_n_px": pixels})
    filled = int(frame[column].notna().sum())
    log.info(
        "%s: %d of %d polygons valued, %d of them from an interior point",
        column,
        filled,
        len(frame),
        int((frame[f"{column}_n_px"] == 0).sum()),
    )
    return frame
