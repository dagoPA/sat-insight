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

PROPORCIONES = (0.8, 0.1, 0.1)
"""Reparto de cities entre entrenamiento, validación y prueba."""

CONJUNTOS = ("train", "val", "test")

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
    cities: pd.DataFrame,
    *,
    proporciones: tuple[float, float, float] = PROPORCIONES,
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
    faltantes = {columna_ciudad, columna_tamano, columna_estrato} - set(cities.columns)
    if faltantes:
        raise KeyError(f"the city table is missing {sorted(faltantes)}")
    if abs(sum(proporciones) - 1.0) > 1e-9:
        raise ValueError(f"proportions must add up to one, got {proporciones}")
    if len(cities) < len(CONJUNTOS):
        raise ValueError(f"{len(cities)} cities cannot fill {len(CONJUNTOS)} sets")

    tabla = cities[[columna_ciudad, columna_tamano, columna_estrato]].copy()
    tabla.columns = ["ciudad", "tamano", "estrato_valor"]
    tabla["estrato"] = (
        _bins(tabla.tamano, ESTRATOS).astype(str)
        + "-"
        + _bins(tabla.estrato_valor, ESTRATOS).astype(str)
    )

    azar = np.random.default_rng(seed)
    patron = _patron(proporciones)

    # el reparto corre dentro de cada estrato y no sobre un orden global: recorriendo una
    # lista única, un estrato entero puede caer en posiciones que el ciclo manda siempre al
    # mismo conjunto, y la validación termina con la mitad del rezago del entrenamiento
    # aunque el reparto global cuadre. Cada estrato entrega ahora su propia proporción.
    asignado: dict[str, str] = {}
    desfase = 0
    for _estrato, grupo in tabla.groupby("estrato", observed=True, sort=True):
        claves = grupo.ciudad.to_numpy().copy()
        azar.shuffle(claves)
        for i, ciudad in enumerate(claves):
            asignado[ciudad] = patron[(i + desfase) % len(patron)]
        # el desfase evita que todos los estratos entreguen su primera ciudad al mismo
        # conjunto, que con estratos chicos sesgaría el reparto entero
        desfase = (desfase + len(claves)) % len(patron)
    tabla["conjunto"] = tabla.ciudad.map(asignado)

    resumen = tabla.groupby("conjunto", observed=True).agg(
        cities=("ciudad", "size"),
        agebs=("tamano", "sum"),
        rezago_medio=("estrato_valor", "mean"),
    )
    log.info("partition:\n%s", resumen)
    return tabla[["ciudad", "conjunto", "tamano", "estrato_valor", "estrato"]]


def _patron(proporciones: tuple[float, float, float], pasos: int = 10) -> list[str]:
    """Ciclo de destinos que reproduce las proporciones pedidas.

    Repartir por ciclo en vez de por muestreo hace que las proporciones salgan exactas y
    que ningún conjunto se lleve una racha del mismo estrato.
    """
    cupos = [round(p * pasos) for p in proporciones]
    cupos[0] += pasos - sum(cupos)
    # prueba y validación se colocan al principio del ciclo, separadas entre sí, para que
    # queden repartidas a lo largo del orden y no amontonadas en un extremo
    patron = ["test", "val"] * min(cupos[2], cupos[1])
    patron += ["test"] * (cupos[2] - min(cupos[2], cupos[1]))
    patron += ["val"] * (cupos[1] - min(cupos[2], cupos[1]))
    patron += ["train"] * cupos[0]
    return patron


def ciudades_de(particion: pd.DataFrame, conjunto: str) -> list[str]:
    """Claves de ciudad de uno de los tres conjuntos."""
    if conjunto not in CONJUNTOS:
        raise KeyError(f"unknown set {conjunto!r}, expected one of {CONJUNTOS}")
    return particion.loc[particion.conjunto == conjunto, "ciudad"].tolist()


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


def dueno_de_municipio(tabla: pd.DataFrame, catalogue: dict) -> dict[str, str]:
    """Asigna cada municipio a una sola ciudad, para que ninguna AGEB pertenezca a dos.

    El recuadro de una ciudad envuelve su mancha urbana conurbada, y las manchas de dos
    cities vecinas se solapan: Guadalajara y Zapopan son entradas distintas del catálogo
    y comparten 302 AGEB. Extraídas bajo ambas, esas AGEB entran dos veces a la tabla, y si
    las dos cities caen a lados distintos de la partición el modelo ve en entrenamiento
    filas que después se le miden. Sobre la partición nacional eran 1,880 AGEB.

    Manda la ciudad cuya clave de municipio es la del AGEB, que es la que el catálogo
    nombra. Un municipio que ninguna ciudad reclama —entra por vecindad, no por ser el
    centro— va a la que más AGEB suyas tenga, y el empate se rompe por nombre para que la
    asignación no dependa del orden en que se leyeron los archivos.
    """
    propietario = {c.municipality: clave for clave, c in catalogue.items()}
    salida: dict[str, str] = {}
    for municipio, grupo in tabla.groupby(tabla.cvegeo.str[:5], observed=True):
        if municipio in propietario and propietario[municipio] in set(grupo.ciudad):
            salida[municipio] = propietario[municipio]
            continue
        cuenta = grupo.ciudad.value_counts()
        salida[municipio] = sorted(cuenta[cuenta == cuenta.max()].index)[0]
    return salida


def desduplicar(tabla: pd.DataFrame, catalogue: dict) -> pd.DataFrame:
    """Deja una sola fila por AGEB, bajo la ciudad que se queda con su municipio."""
    dueno = dueno_de_municipio(tabla, catalogue)
    municipio = tabla.cvegeo.str[:5]
    salida = tabla[tabla.ciudad == municipio.map(dueno)].copy()
    sobrantes = len(tabla) - len(salida)
    if sobrantes:
        log.info("%d filas duplicadas entre cities conurbadas descartadas", sobrantes)
    repetidas = salida.cvegeo.duplicated().sum()
    if repetidas:
        raise ValueError(f"quedan {repetidas} AGEB repetidas tras desduplicar")
    return salida.reset_index(drop=True)
