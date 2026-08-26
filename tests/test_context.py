"""The adjacency is pure geometry, so it is checked without torch or network."""

import numpy as np

from satinsight.context import STRIDE, adjacency


def grid(cells):
    """Pixel positions of a list of (row, column) cells of the token grid."""
    y0 = np.array([r * STRIDE for r, _ in cells])
    x0 = np.array([c * STRIDE for _, c in cells])
    return y0, x0


def neighbours_of(cells, radius=1):
    src, dst = adjacency(*grid(cells), radius=radius)
    out = {i: set() for i in range(len(cells))}
    for i, j in zip(src, dst, strict=True):
        out[int(i)].add(int(j))
    return out


def test_full_block_gives_every_surrounding_cell():
    cells = [(r, c) for r in range(3) for c in range(3)]
    found = neighbours_of(cells)
    centre = cells.index((1, 1))
    assert len(found[centre]) == 8
    assert centre not in found[centre], "an instance is not its own neighbour"
    corner = cells.index((0, 0))
    assert len(found[corner]) == 3


def test_gaps_in_the_grid_are_skipped():
    # a bag does not tile a rectangle: the token between the two is missing
    found = neighbours_of([(0, 0), (0, 2)])
    assert found[0] == set() and found[1] == set()


def test_radius_two_reaches_further():
    cells = [(0, 0), (0, 2)]
    assert neighbours_of(cells, radius=2)[0] == {1}


def test_isolated_instance_has_no_neighbours():
    found = neighbours_of([(0, 0), (5, 5), (5, 6)])
    assert found[0] == set()
    assert found[1] == {2}
