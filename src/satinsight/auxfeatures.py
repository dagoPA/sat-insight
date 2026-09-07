"""Per-token features from the free global products, as an input the head can fuse.

The global-product comparison measured GHSL and nighttime lights as competitors of the
map. On the test cities WorldCover alone orders tracts about as well as the map does, and
the products carry information the frozen encoder was never asked to hold: how much
surface is built, how tall it is, how many people the census grid puts there, and how
much light the place emits at night. This module turns each product into one number per
160 m token on the city grid, so the same head can read the products beside the
foundation-model vector.

Each product is reprojected from its own grid onto the 10 m grid of the city composite,
and the token value is the mean over the 16 by 16 window. Cells are far coarser than the
grid, so the reprojection only replicates them; averaging over the window then blends the
cells a token straddles in proportion to the area each covers. Population and built
surface are heavy-tailed counts and enter as log(1 + x); height and lights enter as they
are. Standardisation happens in the head, as for every other input.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from satinsight.download import DATA_ROOT
from satinsight.grid import Grid
from satinsight.incumbents import GHSL_PRODUCTS, NIGHTLIGHTS_NAME
from satinsight.tiling import TOKEN_SIZE

log = logging.getLogger(__name__)

FEATURES = ("built_surface", "built_height", "population", "nightlights")
"""Column order of the auxiliary matrix. Fixed here so a file and the head agree."""

LOG_SCALED = frozenset({"built_surface", "population"})
"""Products that enter as log(1 + x). Both are counts per cell with a long right tail."""


def product_files(product: str, root: Path = DATA_ROOT) -> list[Path]:
    """Every tile of one product already on disk, or the single nightlights layer."""
    if product == "nightlights":
        path = root / "externos" / "nightlights" / NIGHTLIGHTS_NAME
        return [path] if path.exists() else []
    spec = GHSL_PRODUCTS[product]
    return sorted((root / "externos" / "ghsl" / product).glob(f"{spec.dataset}_*.tif"))


def _overlaps(source, grid: Grid) -> bool:
    """Whether a raster's extent touches the city grid, judged in the raster's own system."""
    west, south, east, north = transform_bounds(grid.crs, source.crs, *grid.bounds)
    b = source.bounds
    return not (east < b.left or west > b.right or north < b.bottom or south > b.top)


def onto_grid(paths: list[Path], grid: Grid) -> np.ndarray:
    """One product resampled onto the city grid, tiles laid over one another.

    Every tile is reprojected into the full city grid with NaN outside its own extent,
    and each fills only what the previous ones left empty. Tiles of one product do not
    overlap, so the order does not matter; the loop exists because a city can straddle a
    tile seam.
    """
    output = np.full(grid.shape, np.nan, dtype="float32")
    for path in paths:
        with rasterio.open(path) as source:
            if not _overlaps(source, grid):
                continue
            piece = np.full(grid.shape, np.nan, dtype="float32")
            reproject(
                source=rasterio.band(source, 1),
                destination=piece,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            missing = np.isnan(output)
            output[missing] = piece[missing]
    return output


def token_matrix(
    layers: dict[str, np.ndarray], y0: np.ndarray, x0: np.ndarray, token_px: int = TOKEN_SIZE
) -> np.ndarray:
    """Mean of each layer inside every token window, in `FEATURES` order.

    A token with no valid cell in a layer gets zero after the log or raw scaling, the
    same convention the WorldCover token features use for unclassified ground.
    """
    matrix = np.empty((len(y0), len(FEATURES)), dtype="float32")
    for j, name in enumerate(FEATURES):
        layer = layers[name]
        if name in LOG_SCALED:
            layer = np.log1p(np.clip(layer, 0, None))
        for i, (r, c) in enumerate(zip(y0, x0, strict=True)):
            window = layer[r : r + token_px, c : c + token_px]
            matrix[i, j] = np.nanmean(window) if np.isfinite(window).any() else np.nan
    return np.nan_to_num(matrix, nan=0.0)


def city_features(grid: Grid, y0: np.ndarray, x0: np.ndarray, root: Path = DATA_ROOT) -> np.ndarray:
    """Auxiliary matrix of one city: every product on the grid, then one row per token."""
    layers = {}
    for name in FEATURES:
        paths = product_files(name, root)
        if not paths:
            raise FileNotFoundError(f"no {name} raster on disk under {root / 'externos'}")
        layers[name] = onto_grid(paths, grid)
        covered = float(np.isfinite(layers[name]).mean())
        if covered < 0.99:
            log.warning("%s covers %.1f%% of the grid", name, 100 * covered)
    return token_matrix(layers, y0, x0)
