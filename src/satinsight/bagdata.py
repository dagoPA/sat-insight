"""Assembles the bags on disk into what the MIL training loop consumes.

The vectors live one file per city and sensor, the instances one parquet per city, and the
bag labels one parquet per city. This module joins the three and hands back a list of bags,
each one a matrix of instances plus the AGEB key of every row.

The AGEB keys travel with the bag but never reach the model. They are what the attention
map is scored against once training is done, and keeping them attached is what stops that
join from having to be reconstructed by position later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from satinsight.dataset import paths
from satinsight.download import DATA_ROOT
from satinsight.encoders import load

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bag:
    """One municipality: its instances, its label, and where each instance fell."""

    city: str
    municipality: str
    instances: np.ndarray
    cvegeo: np.ndarray
    ordinal: int
    shares: np.ndarray
    """Share of the municipality's population in AGEB of grade k or above, for k in 1..4.

    Carried beside the rounded grade because which of the two supervises is an open
    ablation, and recomputing them per experiment would mean rebuilding every bag."""

    y0: np.ndarray
    x0: np.ndarray
    """Pixel position of every instance on the city grid.

    It travels with the bag because a fused bag drops the instances with no counterpart in
    the other modality, so the rows no longer line up with the parquet they came from.
    Rebuilding the positions later by index would silently pair attention with the wrong
    ground."""

    def __len__(self) -> int:
        return len(self.instances)


def load_city(city: str, sensor: str, root: Path = DATA_ROOT, *, fuse: bool = False) -> list[Bag]:
    """Bags of one city, joining vectors, instances and labels.

    With `fuse` the two modalities are concatenated per instance. That needs both to have
    tiled identically, which they do because the window grid is derived from the composite
    shape and both composites share the grid of the city.
    """
    where = paths(root)
    instances = pd.read_parquet(where["instances"] / f"{city}_{sensor}.parquet")
    labels = pd.read_parquet(where["bags"] / f"{city}.parquet")

    vectors, tags = load(where["vectors"] / f"{city}_{sensor}.npz")
    if len(vectors) != len(instances):
        raise ValueError(
            f"{city}/{sensor}: {len(vectors)} vectors against {len(instances)} instances"
        )
    if "cvegeo" in tags and not (tags["cvegeo"] == instances.cvegeo.to_numpy()).all():
        raise ValueError(f"{city}/{sensor}: the vectors do not line up with the instances")

    if fuse:
        other = "s1" if sensor == "s2" else "s2"
        pair = pd.read_parquet(where["instances"] / f"{city}_{other}.parquet")
        other_vectors, _ = load(where["vectors"] / f"{city}_{other}.npz")
        # the two modalities tile the same grid, so an instance is identified by where it
        # sits; matching by position would silently pair different ground when one of the
        # two dropped a token for want of observed pixels
        index = {(y, x): i for i, (y, x) in enumerate(zip(pair.y0, pair.x0, strict=True))}
        rows = [index.get((y, x), -1) for y, x in zip(instances.y0, instances.x0, strict=True)]
        keep = np.array([r >= 0 for r in rows])
        if not keep.all():
            log.info("%s: %d instances without a counterpart dropped", city, (~keep).sum())
        instances = instances[keep].reset_index(drop=True)
        vectors = np.hstack([vectors[keep], other_vectors[np.array(rows)[keep]]])

    grades = dict(zip(labels.municipio, labels.ordinal, strict=True))
    columns = [f"p{k}" for k in range(1, 5)]
    shares = (
        {
            row.municipio: np.array([getattr(row, c) for c in columns], dtype="float32")
            for row in labels.itertuples()
        }
        if all(c in labels.columns for c in columns)
        else {}
    )
    bags = []
    for municipality, group in instances.groupby("municipio", observed=True):
        if municipality not in grades:
            continue
        bags.append(
            Bag(
                city=city,
                municipality=municipality,
                instances=vectors[group.index.to_numpy()],
                cvegeo=group.cvegeo.to_numpy(),
                ordinal=int(grades[municipality]),
                shares=shares.get(municipality, np.zeros(4, dtype="float32")),
                y0=group.y0.to_numpy(),
                x0=group.x0.to_numpy(),
            )
        )
    return bags


def load_split(
    cities: list[str], sensor: str, root: Path = DATA_ROOT, *, fuse: bool = False
) -> list[Bag]:
    """Bags of every city of one split. A city that fails does not stop the rest."""
    bags: list[Bag] = []
    for city in cities:
        try:
            bags.extend(load_city(city, sensor, root, fuse=fuse))
        except Exception:
            log.warning("no bags for %s", city, exc_info=True)
    if not bags:
        raise RuntimeError("no city yielded bags")
    sizes = np.array([len(b) for b in bags])
    log.info(
        "%d bags of %d cities · %d instances · median %d per bag",
        len(bags),
        len({b.city for b in bags}),
        sizes.sum(),
        np.median(sizes),
    )
    return bags
