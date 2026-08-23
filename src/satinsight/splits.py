"""Partitions cities into training and test, with no city split between the two.

The unit of partition is the city, never the AGEB. Neighbouring AGEB share streets,
building stock and the same acquisition geometry, so dealing them at random puts halves
of the same neighbourhood on both sides and the score comes out inflated. Holding out
whole cities asks the question the project actually cares about: does this transfer to a
city the model has never seen.

Cities are dealt into training, validation and test in an 80/10/10 split. Validation
tunes, test is opened once at the end. Leaving a single city out at a time made sense
with five of them; with 138 it would mean 138 folds, each training on 99.3% of the data,
and the spread between folds would be mostly noise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROPORTIONS = (0.8, 0.1, 0.1)
"""How cities are dealt between training, validation and test."""

SETS = ("train", "val", "test")

SEED = 20200101
"""Fixed so the partition can be rebuilt from scratch; it is the census reference date."""

STRATA = 3
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
    cities: pd.DataFrame,
    *,
    proportions: tuple[float, float, float] = PROPORTIONS,
    seed: int = SEED,
    columna_ciudad: str = "clave",
    columna_tamano: str = "agebs",
    columna_estrato: str = "altos",
) -> pd.DataFrame:
    """Deals whole cities into training, validation and test, balanced on two variables.

    Both are stratified because both bias the result on their own. Sorting only by size
    leaves the test set with the wrong mix of deprived AGEB, and the high grades are rare
    enough that an unlucky draw could leave a split with almost none.

    The deal walks an order that alternates strata and hands out places on a repeating
    pattern, so the proportions come out exact and every stratum is spread across the
    three sets rather than sampled into them.
    """
    missing = {columna_ciudad, columna_tamano, columna_estrato} - set(cities.columns)
    if missing:
        raise KeyError(f"the city table is missing {sorted(missing)}")
    if abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError(f"proportions must add up to one, got {proportions}")
    if len(cities) < len(SETS):
        raise ValueError(f"{len(cities)} cities cannot fill {len(SETS)} sets")

    table = cities[[columna_ciudad, columna_tamano, columna_estrato]].copy()
    table.columns = ["ciudad", "n_agebs", "stratum_value"]
    table["stratum"] = (
        _bins(table.n_agebs, STRATA).astype(str)
        + "-"
        + _bins(table.stratum_value, STRATA).astype(str)
    )

    rng = np.random.default_rng(seed)
    pattern = _patron(proportions)

    # the deal runs inside each stratum and not over a global order: walking a
    # single list, a whole stratum can land on positions the cycle always sends to the same
    # set, and validation ends up with half the deprivation of training even though the
    # global split adds up. Each stratum now hands over its own share.
    assigned: dict[str, str] = {}
    offset = 0
    for _estrato, grupo in table.groupby("stratum", observed=True, sort=True):
        keys = grupo.ciudad.to_numpy().copy()
        rng.shuffle(keys)
        for i, ciudad in enumerate(keys):
            assigned[ciudad] = pattern[(i + offset) % len(pattern)]
        # el offset evita que todos los estratos entreguen su primera ciudad al mismo
        # set, which with small strata would bias the whole deal
        offset = (offset + len(keys)) % len(pattern)
    table["split"] = table.ciudad.map(assigned)

    summary = table.groupby("split", observed=True).agg(
        cities=("ciudad", "size"),
        agebs=("n_agebs", "sum"),
        rezago_medio=("stratum_value", "mean"),
    )
    log.info("partition:\n%s", summary)
    return table[["ciudad", "split", "n_agebs", "stratum_value", "stratum"]]


def _patron(proportions: tuple[float, float, float], steps: int = 10) -> list[str]:
    """Ciclo de destinos que reproduce las proportions pedidas.

    Repartir por ciclo en vez de por muestreo hace que las proportions salgan exactas y
    that no set takes a run of the same stratum.
    """
    quotas = [round(p * steps) for p in proportions]
    quotas[0] += steps - sum(quotas)
    # test and validation sit at the start of the cycle, apart from each other, so that
    # queden repartidas a lo largo del order y no amontonadas en un extremo
    pattern = ["test", "val"] * min(quotas[2], quotas[1])
    pattern += ["test"] * (quotas[2] - min(quotas[2], quotas[1]))
    pattern += ["val"] * (quotas[1] - min(quotas[2], quotas[1]))
    pattern += ["train"] * quotas[0]
    return pattern


def cities_of(particion: pd.DataFrame, split: str) -> list[str]:
    """Claves de ciudad de uno de los tres conjuntos."""
    if split not in SETS:
        raise KeyError(f"unknown set {split!r}, expected one of {SETS}")
    return particion.loc[particion.split == split, "ciudad"].tolist()


def check(particion: pd.DataFrame, instancias: pd.DataFrame | None = None) -> None:
    """Fails loudly if a city, an AGEB or a bag ended up on both sides of the partition.

    Worth running even though `assign` cannot produce a leak by construction: the tables
    get rebuilt, filtered and merged by hand along the way, and a leak found by the
    reviewer instead of by us costs the paper.
    """
    repeated = particion.ciudad[particion.ciudad.duplicated()].unique()
    if len(repeated):
        raise ValueError(f"cities assigned more than once: {sorted(repeated)}")

    if instancias is None:
        return
    por_ciudad = particion.set_index("ciudad").split
    marcadas = instancias.assign(split=instancias.ciudad.map(por_ciudad))
    sin_asignar = marcadas.split.isna().sum()
    if sin_asignar:
        raise ValueError(f"{sin_asignar} instances belong to a city outside the partition")
    for columna in ("cvegeo", "municipio"):
        if columna not in marcadas.columns:
            continue
        cruzados = marcadas.groupby(columna, observed=True).split.nunique()
        if (cruzados > 1).any():
            culpables = cruzados[cruzados > 1].index.tolist()[:5]
            raise ValueError(f"{columna} spanning both sides of the partition: {culpables}")
    log.info("partition clean over %d instances", len(instancias))


def municipality_owner(table: pd.DataFrame, catalogue: dict) -> dict[str, str]:
    """Asigna cada municipality_key a una sola ciudad, para que ninguna AGEB pertenezca a dos.

    El recuadro de una ciudad envuelve su mancha urbana conurbada, y las manchas de dos
    neighbouring cities overlap: Guadalajara and Zapopan are separate catalogue entries and
    share 302 AGEB. Extracted under both, those AGEB enter the table twice, and if the two
    cities fall on different sides of the partition the model sees in training rows that
    are later measured on it. Over the national partition there were 1,880 such AGEB.

    The city whose municipality key is the AGEB's own decides, which is the one the
    catalogue names. A municipality no city claims —it enters by proximity, not by being
    the core— goes to whichever holds most of its AGEB, and ties break by name so the
    assignment does not depend on the order the files were read in.
    """
    owner = {c.municipality: clave for clave, c in catalogue.items()}
    output: dict[str, str] = {}
    for municipality_key, grupo in table.groupby(table.cvegeo.str[:5], observed=True):
        if municipality_key in owner and owner[municipality_key] in set(grupo.ciudad):
            output[municipality_key] = owner[municipality_key]
            continue
        cuenta = grupo.ciudad.value_counts()
        output[municipality_key] = sorted(cuenta[cuenta == cuenta.max()].index)[0]
    return output


def deduplicate(table: pd.DataFrame, catalogue: dict) -> pd.DataFrame:
    """Deja una sola fila por AGEB, bajo la ciudad que se queda con su municipality_key."""
    owner = municipality_owner(table, catalogue)
    municipality_key = table.cvegeo.str[:5]
    output = table[table.ciudad == municipality_key.map(owner)].copy()
    surplus = len(table) - len(output)
    if surplus:
        log.info("%d filas duplicadas entre cities conurbadas descartadas", surplus)
    repeated = output.cvegeo.duplicated().sum()
    if repeated:
        raise ValueError(f"quedan {repeated} AGEB repeated tras deduplicate")
    return output.reset_index(drop=True)
