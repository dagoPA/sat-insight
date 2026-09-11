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
    from satinsight.grid import Grid

log = logging.getLogger(__name__)

WINDOW_SIZE = 224
"""Side of the window handed to the foundation model, in pixels.

Fixed by the model rather than chosen: DOFA builds its positional embedding for a 224 px
input and refuses anything else. At 10 m that is 2.24 km, far too coarse to be an
instance on its own, which is why a window is cut into tokens instead of being embedded
whole.
"""

TOKEN_SIZE = 16
"""Side of one token, in pixels. Also fixed by the model: it is the ViT patch size.

At 10 m a token covers 160 m, or 0.026 km². The median urban AGEB measures 0.255 km²,
so about ten tokens fall inside it. That ratio is what makes the attention map worth
comparing against the AGEB grade at all; embedding the whole 224 px window instead would
put one instance across twenty AGEB.
"""

TILE_SIZE = TOKEN_SIZE
"""What an instance measures. Kept as a name of its own because it is the unit the bags,
the attention scores and the validation all speak in."""

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


def grid(
    shape: tuple[int, int],
    size: int = WINDOW_SIZE,
    stride: int | None = None,
    *,
    flush: bool = False,
) -> list[Tile]:
    """Lays a grid of whole patches over an array of this shape.

    Partial patches along the right and bottom edges are dropped. A ragged instance would
    have to be padded, and padding invents texture in a project whose entire signal is
    texture. The loss is one patch width at two edges, under 4% of a typical city.

    With a `stride` smaller than `size` the patches overlap, so that a token is covered
    by several windows and its vectors can be averaged across them: the tokens near a
    window's edge then see context on both sides, and the seams between windows, which
    otherwise appear as blocks in the map, are averaged out. The stride must divide the
    token grid, so that overlapping windows share token positions exactly.

    With `flush` a last window is laid against the right and the bottom edge wherever the
    regular ones stop short, so that no ground goes unencoded. It costs a city under 4%
    of its area and a municipality of a few tracts most of it, which is why the small
    boxes of the national maps ask for it. Its offset is floored to a whole token so that
    every window still lands on the same token lattice.
    """
    if size <= 0:
        raise ValueError(f"patch side must be positive, got {size}")
    stride = size if stride is None else stride
    if stride <= 0 or stride > size or stride % TOKEN_SIZE:
        raise ValueError(f"stride must be a multiple of {TOKEN_SIZE} in (0, {size}], got {stride}")
    height, width = shape
    if height < size or width < size:
        raise ValueError(f"a {height}x{width} array holds no whole {size}x{size} patch")

    def offsets(extent: int) -> list[int]:
        steps = [i * stride for i in range((extent - size) // stride + 1)]
        if flush and steps[-1] + size < extent:
            steps.append((extent - size) // TOKEN_SIZE * TOKEN_SIZE)
        return sorted(set(steps))

    ys, xs = offsets(height), offsets(width)
    covered_rows, covered_cols = max(ys) + size, max(xs) + size
    dropped = 1 - (covered_rows * covered_cols) / (height * width)
    log.info("%dx%d patch grid, %.1f%% of the array left over", len(ys), len(xs), 100 * dropped)
    return [
        Tile(row=r, col=c, y0=y, x0=x, size=size)
        for r, y in enumerate(ys)
        for c, x in enumerate(xs)
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
    size: int = WINDOW_SIZE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    stride: int | None = None,
    *,
    flush: bool = False,
) -> list[Tile]:
    """Patches of a city that carry enough observed pixels to be worth embedding."""
    if not bands:
        raise ValueError("no bands to tile")
    shape = next(iter(bands.values())).shape
    mismatched = {name: a.shape for name, a in bands.items() if a.shape != shape}
    if mismatched:
        raise ValueError(f"bands disagree on shape: {shape} against {mismatched}")

    todos = grid(shape, size, stride, flush=flush)
    kept = [t for t in todos if valid_fraction(bands, t) >= min_valid_fraction]
    log.info(
        "%d of %d patches kept above %.0f%% observed",
        len(kept),
        len(todos),
        100 * min_valid_fraction,
    )
    return kept


def centers(tiles: list[Tile], malla: Grid) -> np.ndarray:
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


def tokens(window: Tile, token_size: int = TOKEN_SIZE) -> list[Tile]:
    """Cuts a window into the cells the model will return one vector for.

    Coordinates come back in the full image, not relative to the window, because every
    instance has to be locatable on the ground for the attention map to be scored against
    an AGEB. The row and column indices are the token's place inside the window, which is
    the order the model returns its vectors in.
    """
    if window.size % token_size:
        raise ValueError(f"a {window.size} px window does not divide into {token_size} px tokens")
    lado = window.size // token_size
    return [
        Tile(
            row=r,
            col=c,
            y0=window.y0 + r * token_size,
            x0=window.x0 + c * token_size,
            size=token_size,
        )
        for r in range(lado)
        for c in range(lado)
    ]


def instances(
    windows: list[Tile],
    bands: dict[str, np.ndarray],
    token_size: int = TOKEN_SIZE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> tuple[list[Tile], np.ndarray]:
    """Every token of every window, with the index it holds among the model's outputs.

    A window keeps its tokens even where a few are short of observed pixels, because the
    model is fed the window whole and returns all of them anyway. The ones below the
    threshold are dropped afterwards, and the returned index says which output row each
    surviving token came from.
    """
    kept: list[Tile] = []
    indices: list[int] = []
    seen: set[tuple[int, int]] = set()
    por_ventana = (windows[0].size // token_size) ** 2 if windows else 0
    for i, ventana in enumerate(windows):
        for j, token in enumerate(tokens(ventana, token_size)):
            # overlapping windows share tokens; a position enters the table once, from the
            # first window that holds it, because a repeated instance would weigh twice in
            # its bag and appear twice on the map
            if (token.y0, token.x0) in seen:
                continue
            if valid_fraction(bands, token) >= min_valid_fraction:
                seen.add((token.y0, token.x0))
                kept.append(token)
                indices.append(i * por_ventana + j)
    log.info(
        "%d tokens kept of %d, from %d windows",
        len(kept),
        len(windows) * por_ventana,
        len(windows),
    )
    return kept, np.array(indices, dtype="int64")
