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

LARGO_CLAVE_MUNICIPIO = 5
"""Characters of an AGEB key that name its municipality: two of state, three of county."""


def locate(tiles: list[Tile], malla: Grid, agebs: gpd.GeoDataFrame) -> pd.DataFrame:
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

    puntos = gpd.GeoDataFrame(
        {"tile": np.arange(len(tiles))},
        geometry=gpd.points_from_xy(*centers(tiles, malla).T),
        crs=malla.crs,
    )
    unido = puntos.sjoin(
        agebs.to_crs(malla.crs)[["cvegeo", "geometry"]], how="left", predicate="within"
    )
    # a patch centre sitting exactly on a shared edge matches both neighbours; the first
    # match keeps the assignment single-valued and the choice between two adjacent AGEB
    # is arbitrary either way
    unido = unido[~unido.index.duplicated(keep="first")]

    fuera = int(unido.cvegeo.isna().sum())
    if fuera:
        log.info("%d of %d patches fell outside every AGEB and were dropped", fuera, len(tiles))
    unido = unido.dropna(subset=["cvegeo"])

    tabla = pd.DataFrame(
        {
            "tile": unido.tile.to_numpy(),
            "row": [tiles[i].row for i in unido.tile],
            "col": [tiles[i].col for i in unido.tile],
            "y0": [tiles[i].y0 for i in unido.tile],
            "x0": [tiles[i].x0 for i in unido.tile],
            "cvegeo": unido.cvegeo.to_numpy(),
        }
    )
    tabla["municipio"] = tabla.cvegeo.str[:LARGO_CLAVE_MUNICIPIO]
    return tabla.reset_index(drop=True)


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
    faltantes = {"cvegeo", "ordinal", "poblacion"} - set(agebs.columns)
    if faltantes:
        raise KeyError(f"the AGEB table is missing {sorted(faltantes)}")

    tabla = pd.DataFrame(
        {
            "municipio": agebs.cvegeo.str[:LARGO_CLAVE_MUNICIPIO],
            "ordinal": pd.to_numeric(agebs.ordinal, errors="coerce"),
            "poblacion": pd.to_numeric(agebs.poblacion, errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["ordinal"])

    def resumir(grupo: pd.DataFrame) -> pd.Series:
        peso = grupo.poblacion.to_numpy(dtype="float64")
        if peso.sum() <= 0:
            peso = np.ones(len(grupo))
        medio = float(np.average(grupo.ordinal.to_numpy(dtype="float64"), weights=peso))
        return pd.Series(
            {
                "ordinal": int(np.clip(round(medio), 0, len(GRADES) - 1)),
                "ordinal_continuo": medio,
                "poblacion": float(peso.sum()),
                "agebs": len(grupo),
            }
        )

    salida = tabla.groupby("municipio", observed=True).apply(resumir, include_groups=False)
    salida["grado"] = salida.ordinal.map({v: k for k, v in ORDINAL.items()})
    return salida.reset_index()


def build(
    tiles: list[Tile],
    malla: Grid,
    agebs: gpd.GeoDataFrame,
    ciudad: str,
    *,
    minimo_instancias: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the instance table and the bag table of one city.

    Returns the patches with their AGEB and municipality, and one row per bag with its
    label. `minimo_instancias` drops bags too small to be worth an attention mechanism;
    a bag of three patches teaches the model nothing about where to look.
    """
    instancias = locate(tiles, malla, agebs)
    if instancias.empty:
        raise ValueError(f"{ciudad}: no patch landed inside an AGEB")

    bolsas = municipal_labels(agebs)
    cuenta = instancias.groupby("municipio", observed=True).size().rename("instancias")
    bolsas = bolsas.merge(cuenta, on="municipio", how="inner")

    chicas = bolsas[bolsas.instancias < minimo_instancias]
    if not chicas.empty:
        log.info("%s: %d bags below %d instances dropped", ciudad, len(chicas), minimo_instancias)
        bolsas = bolsas[bolsas.instancias >= minimo_instancias]
        instancias = instancias[instancias.municipio.isin(bolsas.municipio)]

    instancias.insert(0, "ciudad", ciudad)
    bolsas.insert(0, "ciudad", ciudad)
    log.info(
        "%s: %d bags, %d instances, %.0f per bag on median",
        ciudad,
        len(bolsas),
        len(instancias),
        bolsas.instancias.median(),
    )
    return instancias.reset_index(drop=True), bolsas.reset_index(drop=True)
