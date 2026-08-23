"""Persistence of the annual composites as GeoTIFF.

Compositing a city costs about an hour per sensor, and the cost is dominated by the
overhead of opening each remote COG, far above the bytes transferred. Saving the result to
disk turns that expense into something paid once.

The Sentinel-1 composite is saved in linear power, which is how `composite_s1` delivers it.
Conversion to decibels happens on reading, never before writing: averaging in decibels and
averaging in power give different results, and the decibel average is biased towards the
low values.

The file carries the grid with it —affine transform, reference system and the name of each
band— so that reloading it recovers everything needed to cross it with polygons without
querying the catalogue again.
"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from satinsight.grid import Grid

log = logging.getLogger(__name__)

COMPOSITE_ROOT = Path("data") / "composites"


def composite_path(city: str, sensor: str, root: Path = COMPOSITE_ROOT) -> Path:
    """Canonical location of the composite of one city and one sensor."""
    return root / f"{city}_{sensor}.tif"


def save(bands: dict[str, np.ndarray], grid: Grid, destination: Path, **tags) -> Path:
    """Writes the bands of a composite into a GeoTIFF with its georeferencing.

    Band names are kept in each band's description, and any extra metadata —scenes used,
    orbit chosen— travels as a file tag so what it was built from can be audited later.
    """
    if not bands:
        raise ValueError("there are no bands to save")

    names = list(bands)
    shapes = {b.shape for b in bands.values()}
    if len(shapes) > 1:
        raise ValueError(f"the bands do not share a shape: {shapes}")
    shape = shapes.pop()
    if shape != grid.shape:
        raise ValueError(f"band shape {shape} does not match the grid {grid.shape}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": len(names),
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(destination, "w", **profile) as output:
        for index, name in enumerate(names, start=1):
            output.write(bands[name].astype("float32"), index)
            output.set_band_description(index, name)
        if tags:
            output.update_tags(**{k: json.dumps(v, ensure_ascii=False) for k, v in tags.items()})

    log.info("saved %s (%s, %.1f MP)", destination.name, ", ".join(names), grid.megapixels)
    return destination


def load(source: Path) -> tuple[dict[str, np.ndarray], Grid, dict]:
    """Recovers a composite and the grid it was written on."""
    with rasterio.open(source) as origin:
        names = [
            description or f"band_{i}" for i, description in enumerate(origin.descriptions, start=1)
        ]
        bands = {name: origin.read(i) for i, name in enumerate(names, start=1)}
        grid = Grid(
            transform=origin.transform,
            shape=(origin.height, origin.width),
            crs=str(origin.crs),
            bounds=tuple(origin.bounds),
        )
        tags = {}
        for key, value in origin.tags().items():
            try:
                tags[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                tags[key] = value
    return bands, grid, tags


def exists(city: str, sensor: str, root: Path = COMPOSITE_ROOT) -> bool:
    """Says whether the composite is already on disk."""
    return composite_path(city, sensor, root).exists()
