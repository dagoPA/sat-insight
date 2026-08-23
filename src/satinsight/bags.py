"""Assembles MIL bags out of the patches of a city.

A bag is a municipality and its label is a single ordinal scalar. The instances are the
patches whose centre falls inside the municipality. The model is never told which patch
explains the label, which is exactly the weak supervision the project sets out to test.

Every patch also carries the AGEB it fell in. That column is held out of training
entirely and exists so the attention map can be scored against the AGEB grade later.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd

from satinsight.agebs import GRADES, ORDINAL
from satinsight.tiling import Tile, centers

if TYPE_CHECKING:
    from satinsight.grid import Grid

log = logging.getLogger(__name__)

MUNICIPALITY_KEY_LENGTH = 5
"""Characters of an AGEB key that name its municipality: two of state, three of county."""


def locate(tiles: list[Tile], grid: Grid, agebs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Says which AGEB and which municipality each patch fell into.

    A patch is placed by its centre. Patches straddling a boundary therefore land in one
    AGEB whole rather than being split, which keeps every instance a square of the same
    size; the alternative would weight instances by how much of them fell inside, and MIL
    attention has no place to put that weight.

    Patches whose centre lands outside every AGEB are dropped. The analysis window is the
    conurbated urban mass, so those fall on the countryside, on water, or in the gaps
    between detached settlements.
    """
    if not tiles:
        return pd.DataFrame(columns=["tile", "row", "col", "y0", "x0", "cvegeo", "municipio"])

    points = gpd.GeoDataFrame(
        {"tile": np.arange(len(tiles))},
        geometry=gpd.points_from_xy(*centers(tiles, grid).T),
        crs=grid.crs,
    )
    joined = points.sjoin(
        agebs.to_crs(grid.crs)[["cvegeo", "geometry"]], how="left", predicate="within"
    )
    # a patch centre sitting exactly on a shared edge matches both neighbours; the first
    # match keeps the assignment single-valued and the choice between two adjacent AGEB
    # is arbitrary either way
    joined = joined[~joined.index.duplicated(keep="first")]

    outside = int(joined.cvegeo.isna().sum())
    if outside:
        log.info("%d of %d patches fell outside every AGEB and were dropped", outside, len(tiles))
    joined = joined.dropna(subset=["cvegeo"])

    table = pd.DataFrame(
        {
            "tile": joined.tile.to_numpy(),
            "row": [tiles[i].row for i in joined.tile],
            "col": [tiles[i].col for i in joined.tile],
            "y0": [tiles[i].y0 for i in joined.tile],
            "x0": [tiles[i].x0 for i in joined.tile],
            "cvegeo": joined.cvegeo.to_numpy(),
        }
    )
    table["municipio"] = table.cvegeo.str[:MUNICIPALITY_KEY_LENGTH]
    return table.reset_index(drop=True)


def municipal_labels(agebs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregates the AGEB grades of a municipality into the single scalar a bag carries.

    CONEVAL publishes a Grado de Rezago Social for municipalities too, computed from its
    own principal component analysis over the whole municipality. Aggregating the urban
    AGEB grades instead keeps the bag label on the same scale and the same population as
    the labels the attention map is scored against, so a gap between the two cannot be
    blamed on two different indices disagreeing.

    The aggregate is the population-weighted mean of the ordinal grade, rounded. Weighting
    by population rather than by area stops a large empty AGEB from outvoting a dense one.
    """
    missing = {"cvegeo", "ordinal", "poblacion"} - set(agebs.columns)
    if missing:
        raise KeyError(f"the AGEB table is missing {sorted(missing)}")

    table = pd.DataFrame(
        {
            "municipio": agebs.cvegeo.str[:MUNICIPALITY_KEY_LENGTH],
            "ordinal": pd.to_numeric(agebs.ordinal, errors="coerce"),
            "poblacion": pd.to_numeric(agebs.poblacion, errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["ordinal"])

    def summarise(group: pd.DataFrame) -> pd.Series:
        weight = group.poblacion.to_numpy(dtype="float64")
        if weight.sum() <= 0:
            weight = np.ones(len(group))
        middle = float(np.average(group.ordinal.to_numpy(dtype="float64"), weights=weight))
        return pd.Series(
            {
                "ordinal": int(np.clip(round(middle), 0, len(GRADES) - 1)),
                "ordinal_continuo": middle,
                "poblacion": float(weight.sum()),
                "agebs": len(group),
            }
        )

    output = table.groupby("municipio", observed=True).apply(summarise, include_groups=False)
    output["grado"] = output.ordinal.map({v: k for k, v in ORDINAL.items()})
    return output.reset_index()


def build(
    tiles: list[Tile],
    grid: Grid,
    agebs: gpd.GeoDataFrame,
    city: str,
    *,
    min_instances: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the instance table and the bag table of one city.

    Returns the patches with their AGEB and municipality, and one row per bag with its
    label. `min_instances` drops bags too small to be worth an attention mechanism; a bag
    of three patches teaches the model nothing about where to look.
    """
    instances = locate(tiles, grid, agebs)
    if instances.empty:
        raise ValueError(f"{city}: no patch landed inside an AGEB")

    bag_table = municipal_labels(agebs)
    counts = instances.groupby("municipio", observed=True).size().rename("instances")
    bag_table = bag_table.merge(counts, on="municipio", how="inner")

    small = bag_table[bag_table.instances < min_instances]
    if not small.empty:
        log.info("%s: %d bags below %d instances dropped", city, len(small), min_instances)
        bag_table = bag_table[bag_table.instances >= min_instances]
        instances = instances[instances.municipio.isin(bag_table.municipio)]

    instances.insert(0, "ciudad", city)
    bag_table.insert(0, "ciudad", city)
    log.info(
        "%s: %d bags, %d instances, %.0f per bag on median",
        city,
        len(bag_table),
        len(instances),
        bag_table.instances.median(),
    )
    return instances.reset_index(drop=True), bag_table.reset_index(drop=True)
