"""Partitions cities into training and test, with no city split between the two.

The unit of partition is the city, never the AGEB. Neighbouring AGEB share streets,
building stock and the same acquisition geometry, so dealing them at random puts halves
of the same neighbourhood on both sides and the score comes out inflated. Holding out
whole cities asks the question the project actually cares about: does this transfer to a
city the model has never seen.

A test set of whole cities is set aside once and left alone. The rest is cut into folds,
and each fold serves as validation in turn. Leaving a single city out at a time made
sense with five of them; with 138 it would mean 138 folds, each training on 99.3% of the
data, and the spread between folds would be mostly noise.
"""

from __future__ import annotations

import logging
from itertools import zip_longest

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

N_TEST = 20
"""Cities held out until the very end."""

N_FOLDS = 5
"""Folds the remaining cities are cut into."""

SEED = 20200101
"""Fixed so the partition can be rebuilt from scratch; it is the census reference date."""

ESTRATOS = 3
"""Bins per stratifying variable. Three by three leaves nine strata over 138 cities."""


def _bins(valores: pd.Series, n: int) -> pd.Series:
    """Rank-based bins that survive ties and skew.

    Quantile cuts on a heavily tied column collapse into fewer bins than asked for, and
    city size is skewed enough that fixed-width cuts would leave one bin holding almost
    everything.
    """
    rangos = valores.rank(method="first")
    return pd.cut(rangos, bins=n, labels=False).astype(int)


def assign(
    ciudades: pd.DataFrame,
    *,
    n_test: int = N_TEST,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    columna_ciudad: str = "clave",
    columna_tamano: str = "agebs",
    columna_estrato: str = "altos",
) -> pd.DataFrame:
    """Deals cities into a test set and into folds, balanced on size and on deprivation.

    Both variables are stratified because both bias the result on their own. Sorting only
    by size leaves the test set with the wrong mix of deprived AGEB, and the high grades
    are rare enough that an unlucky draw could leave a fold with almost none.

    Dealing round-robin inside each stratum, rather than sampling, guarantees the strata
    are spread evenly even when a stratum holds fewer cities than there are folds.
    """
    faltantes = {columna_ciudad, columna_tamano, columna_estrato} - set(ciudades.columns)
    if faltantes:
        raise KeyError(f"the city table is missing {sorted(faltantes)}")
    if n_test + n_folds > len(ciudades):
        raise ValueError(
            f"{len(ciudades)} cities cannot fill a test set of {n_test} plus {n_folds} folds"
        )

    tabla = ciudades[[columna_ciudad, columna_tamano, columna_estrato]].copy()
    tabla.columns = ["ciudad", "tamano", "estrato_valor"]
    tabla["estrato"] = (
        _bins(tabla.tamano, ESTRATOS).astype(str)
        + "-"
        + _bins(tabla.estrato_valor, ESTRATOS).astype(str)
    )

    azar = np.random.default_rng(seed)
    # las ciudades se ordenan alternando estratos, de modo que dos posiciones seguidas
    # vengan de estratos distintos; cualquier reparto posterior que camine ese orden
    # queda balanceado sin tener que contar cuotas por estrato
    barajados = []
    for _estrato, grupo in tabla.groupby("estrato", observed=True, sort=True):
        claves = grupo.ciudad.to_numpy().copy()
        azar.shuffle(claves)
        barajados.append(list(claves))
    orden = [c for ronda in zip_longest(*barajados) for c in ronda if c is not None]

    # el conjunto de prueba se toma en posiciones repartidas a lo largo del orden, no en
    # ronda contra los pliegues: repartir en ronda sobre n_test + n_folds destinos manda
    # a prueba la fracción n_test/(n_test+n_folds) de todo el catálogo, que con veinte
    # plazas y cinco pliegues son cuatro de cada cinco ciudades
    posiciones_prueba = {round(j * len(orden) / n_test) for j in range(n_test)}
    asignado: dict[str, str] = {}
    siguiente = 0
    for i, ciudad in enumerate(orden):
        if i in posiciones_prueba and sum(v == "test" for v in asignado.values()) < n_test:
            asignado[ciudad] = "test"
        else:
            asignado[ciudad] = f"fold{siguiente % n_folds}"
            siguiente += 1

    tabla["destino"] = tabla.ciudad.map(asignado)
    tabla["conjunto"] = np.where(tabla.destino == "test", "test", "train")
    tabla["pliegue"] = (
        tabla.destino.str.removeprefix("fold").where(tabla.conjunto == "train").astype("Int64")
    )

    resumen = tabla.groupby("conjunto", observed=True).agg(
        ciudades=("ciudad", "size"), rezago_medio=("estrato_valor", "mean")
    )
    log.info("partition:\n%s", resumen)
    return tabla[["ciudad", "conjunto", "pliegue", "tamano", "estrato_valor", "estrato"]]


def folds(particion: pd.DataFrame) -> list[tuple[list[str], list[str]]]:
    """Turns the partition into the (train, validation) city lists of each fold."""
    entrena = particion[particion.conjunto == "train"]
    salida = []
    for pliegue in sorted(entrena.pliegue.dropna().unique()):
        validacion = entrena[entrena.pliegue == pliegue].ciudad.tolist()
        resto = entrena[entrena.pliegue != pliegue].ciudad.tolist()
        salida.append((resto, validacion))
    return salida


def check(particion: pd.DataFrame, instancias: pd.DataFrame | None = None) -> None:
    """Fails loudly if a city, an AGEB or a bag ended up on both sides of the partition.

    Worth running even though `assign` cannot produce a leak by construction: the tables
    get rebuilt, filtered and merged by hand along the way, and a leak found by the
    reviewer instead of by us costs the paper.
    """
    repetidas = particion.ciudad[particion.ciudad.duplicated()].unique()
    if len(repetidas):
        raise ValueError(f"cities assigned more than once: {sorted(repetidas)}")

    if instancias is None:
        return
    por_ciudad = particion.set_index("ciudad").conjunto
    marcadas = instancias.assign(conjunto=instancias.ciudad.map(por_ciudad))
    sin_asignar = marcadas.conjunto.isna().sum()
    if sin_asignar:
        raise ValueError(f"{sin_asignar} instances belong to a city outside the partition")
    for columna in ("cvegeo", "municipio"):
        if columna not in marcadas.columns:
            continue
        cruzados = marcadas.groupby(columna, observed=True).conjunto.nunique()
        if (cruzados > 1).any():
            culpables = cruzados[cruzados > 1].index.tolist()[:5]
            raise ValueError(f"{columna} spanning both sides of the partition: {culpables}")
    log.info("partition clean over %d instances", len(instancias))
