"""Spatial context between instances of the same bag.

Every instance is classified in isolation today, and a 160 m patch of roofs is ambiguous on
its own while its surroundings are not. Deprivation is spatially autocorrelated, so the
neighbours of a token carry signal about it for free, and the measurement that motivated
this says the sampling is there to use: the median AGEB holds ten tokens.

The averaging happens on the hidden representation and not on the input vectors. Widening
the input would double its dimension for every instance of the training set at once, which
is gigabytes for no reason, while the hidden layer is a few hundred units and the
neighbourhood of a token does not change between epochs.

What travels with the bag is the adjacency: for each instance, which rows of the same bag
sit within a given radius of it on the token grid. It is built once and reused.
"""

from __future__ import annotations

import numpy as np

STRIDE = 16
"""Pixels between consecutive tokens, so one step of the grid is 160 m on the ground."""


def adjacency(y0: np.ndarray, x0: np.ndarray, radius: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour pairs of one bag, as the row and column of a sparse matrix.

    Chebyshev radius on the token grid: radius 1 is the eight surrounding tokens, radius 2
    the twenty-four. The instance itself is excluded, so the neighbourhood mean and the
    instance's own representation stay two distinct pieces of information.

    Positions come in pixels, which is how they are stored beside the instances. A bag does
    not tile a full rectangle, tokens with too few observed pixels were dropped, and a
    municipality is not a box, so the grid is sparse and the lookup goes through a
    dictionary of occupied cells.
    """
    row = (np.asarray(y0) // STRIDE).astype(np.int64)
    col = (np.asarray(x0) // STRIDE).astype(np.int64)
    where = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col, strict=True))}

    offsets = [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if (dr, dc) != (0, 0)
    ]
    src, dst = [], []
    for i, (r, c) in enumerate(zip(row, col, strict=True)):
        for dr, dc in offsets:
            j = where.get((int(r) + dr, int(c) + dc))
            if j is not None:
                src.append(i)
                dst.append(j)
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


def build_layer(hidden: int):
    """A layer that appends the neighbourhood mean to every instance representation.

    Torch is imported here and not at module level, the way the rest of the models do it.
    """
    import torch
    from torch import nn

    class Neighbourhood(nn.Module):
        """Concatenates each row with the mean of the rows adjacent to it.

        An instance with no neighbour on the grid, an isolated token, and nine per cent of
        AGEB hold a single one, takes its own representation as its context. Zero would
        put those rows in a region of the space no trained instance occupies, and they are
        exactly the small AGEB whose scores the map is most likely to get wrong already.
        """

        def __init__(self) -> None:
            super().__init__()
            self.out_features = 2 * hidden

        def forward(self, h, src, dst):
            pooled = torch.zeros_like(h)
            counts = torch.zeros(len(h), 1, device=h.device, dtype=h.dtype)
            if len(src):
                pooled.index_add_(0, src, h[dst])
                counts.index_add_(0, src, torch.ones(len(src), 1, device=h.device, dtype=h.dtype))
            alone = counts.squeeze(1) == 0
            pooled = pooled / counts.clamp(min=1)
            pooled[alone] = h[alone]
            return torch.cat([h, pooled], dim=1)

    return Neighbourhood()
