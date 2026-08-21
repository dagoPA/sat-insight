"""Splits a city composite into the patches that become MIL instances.

A bag is a municipality and its instances are image patches. This module turns the
composite of a city into the list of patches that will be embedded, and records where
each one sits on the ground. That bookkeeping is what later lets an attention score be
traced back to an AGEB, which is the whole validation of the project.

Patches are laid out on a fixed grid rather than around anything found in the image, so
that the layout is reproducible and independent of the model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from satinsight.malla import Malla

log = logging.getLogger(__name__)

TILE_SIZE = 64
"""Patch side in pixels. At 10 m this is 640 m, a few city blocks across.

Measured on the national set, this leaves a median of about 1,250 patches per city,
which is the order of magnitude attention-based MIL is known to work with. Going up to
the 224 px that vision transformers take natively would leave around 100 patches per
city, and the smallest cities would drop to 32.
"""

MIN_VALID_FRACTION = 0.9
"""Share of observed pixels a patch needs to be kept.

Patches on the edge of a radar swath, or under a permanent cloud in the optical
composite, carry a median computed from very few scenes. Letting them into a bag adds
instances whose values say more about the acquisition than about the ground.
"""


class Tile(NamedTuple):
    """One patch, placed both on the pixel grid and on the ground."""

    row: int
    """Index down the patch grid, not the pixel grid."""

    col: int
    """Index across the patch grid."""

    y0: int
    """Topmost pixel row the patch covers."""

    x0: int
    """Leftmost pixel column the patch covers."""

    size: int

    @property
    def window(self) -> tuple[slice, slice]:
        """Slices that cut this patch out of a full-city array."""
        return slice(self.y0, self.y0 + self.size), slice(self.x0, self.x0 + self.size)

    @property
    def center_px(self) -> tuple[float, float]:
        """Centre of the patch in pixel coordinates, as (row, column)."""
        return self.y0 + self.size / 2, self.x0 + self.size / 2


def grid(shape: tuple[int, int], size: int = TILE_SIZE) -> list[Tile]:
    """Lays a grid of whole patches over an array of this shape.

    Partial patches along the right and bottom edges are dropped. A ragged instance would
    have to be padded, and padding invents texture in a project whose entire signal is
    texture. The loss is one patch width at two edges, under 4% of a typical city.
    """
    if size <= 0:
        raise ValueError(f"patch side must be positive, got {size}")
    height, width = shape
    rows, cols = height // size, width // size
    if rows == 0 or cols == 0:
        raise ValueError(f"a {height}x{width} array holds no whole {size}x{size} patch")

    dropped = 1 - (rows * size * cols * size) / (height * width)
    log.info("%dx%d patch grid, %.1f%% of the array left over", rows, cols, 100 * dropped)
    return [
        Tile(row=r, col=c, y0=r * size, x0=c * size, size=size)
        for r in range(rows)
        for c in range(cols)
    ]


def valid_fraction(bands: dict[str, np.ndarray], tile: Tile) -> float:
    """Share of pixels observed in every band of a patch.

    A pixel counts as observed only when all bands hold a finite value there, because a
    patch is embedded as a stack and one missing band ruins the instance.
    """
    window = tile.window
    observed = None
    for array in bands.values():
        finite = np.isfinite(array[window])
        observed = finite if observed is None else (observed & finite)
    return float(observed.mean()) if observed is not None else 0.0


def select(
    bands: dict[str, np.ndarray],
    size: int = TILE_SIZE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> list[Tile]:
    """Patches of a city that carry enough observed pixels to be worth embedding."""
    if not bands:
        raise ValueError("no bands to tile")
    shape = next(iter(bands.values())).shape
    mismatched = {name: a.shape for name, a in bands.items() if a.shape != shape}
    if mismatched:
        raise ValueError(f"bands disagree on shape: {shape} against {mismatched}")

    todos = grid(shape, size)
    kept = [t for t in todos if valid_fraction(bands, t) >= min_valid_fraction]
    log.info(
        "%d of %d patches kept above %.0f%% observed",
        len(kept),
        len(todos),
        100 * min_valid_fraction,
    )
    return kept


def centers(tiles: list[Tile], malla: Malla) -> np.ndarray:
    """Ground coordinates of each patch centre, in the grid's own reference system.

    Returns an array of (x, y) rows, ready to be handed to shapely or geopandas.
    """
    if not tiles:
        return np.empty((0, 2))
    filas = np.array([t.center_px[0] for t in tiles])
    columnas = np.array([t.center_px[1] for t in tiles])
    x, y = malla.transform * (columnas, filas)
    return np.column_stack([x, y])


def stack(bands: dict[str, np.ndarray], tile: Tile, order: list[str] | None = None) -> np.ndarray:
    """Cuts one patch out of every band and stacks it as (channel, row, column).

    Channel order is explicit because the foundation model is told the wavelength of each
    channel it receives, and a silent reordering would pair every channel with the wrong
    wavelength.
    """
    order = order or sorted(bands)
    window = tile.window
    return np.stack([bands[name][window] for name in order]).astype("float32")
