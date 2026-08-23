"""Land cover fractions per AGEB, as a built-density baseline.

The project's open risk is the rurality shortcut: that the model gets it right by reading
how much is built rather than the morphology of deprivation. Ruling it out needs a
baseline that answers "here is how much is built according to another product, does the
model add anything on top?".

First-order statistics of the composites themselves do not answer that question. They come
from the same image the model sees, so a better texture model beats them by exploiting that
image better, without that ruling out the shortcut.

ESA WorldCover 2020 does: it is a published classification, at 10 m, from the census year,
and it is not tuned to the CONEVAL labels. How independent it is deserves a precise
statement: WorldCover is derived from 2020 Sentinel-1 and Sentinel-2, that is, the same
sensors that feed this work. The independence is one of product: another classification
chain over the same family of observations. The paper has to declare it with that
precision.

It was chosen over GHSL, which is what the plan mentions, for two reasons. GHSL is not
published in the Planetary Computer STAC, and its 100 m resolution would leave a fifth of
the pilot AGEB with fewer than ten pixels, too few for the baseline itself to mean
anything. WorldCover shares the 10 m grid of the composites.
"""

import logging

import numpy as np
import pandas as pd

from satinsight.aoi import AOI
from satinsight.catalog import open_catalogue
from satinsight.grid import Grid, polygon_window
from satinsight.raster import read_window

log = logging.getLogger(__name__)

COLLECTION = "esa-worldcover"
VERSION = "2020_v100"
"""The edition matching the census. The catalogue also publishes 2021_v200."""

CLASSES = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}

NO_DATA = 0
"""WorldCover uses zero for absent data, and `read_window` fills with zero outside the
tile. The two coincide, which is what allows mosaicking by overlay."""


def mosaic(area: AOI, grid: Grid, catalogue=None) -> np.ndarray:
    """Land cover over the city's grid, joining as many tiles as it takes.

    WorldCover is published in three-degree tiles and a city can touch several. Each tile
    is read over the whole grid —whatever falls outside arrives as zero— and laid over
    wherever the previous one had no data.
    """
    catalogue = catalogue or open_catalogue()
    items = [
        item
        for item in catalogue.search(collections=[COLLECTION], bbox=area.bbox).items()
        if VERSION in item.id
    ]
    if not items:
        raise RuntimeError(f"WorldCover {VERSION} does not cover the box of {area.key}")

    log.info("%s: %d WorldCover %s tiles", area.key, len(items), VERSION)
    output = np.zeros(grid.shape, dtype="uint8")
    for item in items:
        tile = read_window(item.assets["map"].href, area.bbox, grid.shape)
        missing = output == NO_DATA
        output[missing] = tile.astype("uint8")[missing]

    covered = float((output != NO_DATA).mean())
    log.info("%s: %.1f%% of the grid classified", area.key, 100 * covered)
    if covered < 0.99:
        log.warning(
            "%s: %.1f%% of the grid was left unclassified; the fractions of those AGEB "
            "are computed over fewer pixels than they hold",
            area.key,
            100 * (1 - covered),
        )
    return output


def fractions_per_ageb(
    classes: np.ndarray,
    transform,
    geometries,
    keys,
    *,
    prefix: str = "wc",
) -> pd.DataFrame:
    """Fraction of each cover class inside each AGEB.

    The denominator is the classified pixels of the polygon. That way a hole with no data
    stays out of the calculation, which avoids reading it as absence of construction. The
    number of classified pixels is reported so it can be audited.
    """
    if len(geometries) != len(keys):
        raise ValueError(f"{len(geometries)} geometries against {len(keys)} keys")

    columns = [f"{prefix}_{name}" for name in CLASSES.values()]
    rows = []
    for key, geometry in zip(keys, geometries, strict=True):
        base = {"cvegeo": key, f"{prefix}_n_px": 0}
        base.update(dict.fromkeys(columns, np.nan))

        window = polygon_window(transform, geometry, classes.shape)
        if window is not None:
            row_slice, col_slice, inside = window
            values = classes[row_slice, col_slice][inside]
            values = values[values != NO_DATA]
            base[f"{prefix}_n_px"] = int(values.size)
            if values.size:
                for code, name in CLASSES.items():
                    base[f"{prefix}_{name}"] = float((values == code).mean())
        rows.append(base)

    return pd.DataFrame(rows)
