"""Builds the MIL dataset of the national set, one city at a time.

Three artefacts come out of every city: the instance table saying where each patch sits
and which AGEB it fell in, the bag table with one row and one label per municipality,
and the matrix of feature vectors. They are written separately because the first two are
cheap and stable while the third is expensive and gets rebuilt whenever the foundation
model changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from satinsight import bags, encoders, tiling
from satinsight.agebs import ciudades_por_tamano
from satinsight.cache import cargar
from satinsight.ingesta import RAIZ_DATOS
from satinsight.pipeline import aoi_de_ciudad

log = logging.getLogger(__name__)

CANALES = {
    "s2": ["B02", "B03", "B04", "B08", "B11"],
    "s1": ["vh", "vv"],
}
"""Channels fed to the encoder per sensor, in the order their wavelengths are declared."""


def rutas(raiz: Path = RAIZ_DATOS) -> dict[str, Path]:
    """Where each artefact of the stage lives."""
    return {
        "instancias": raiz / "instancias",
        "bolsas": raiz / "bolsas",
        "vectores": raiz / "vectores",
        "particion": raiz / "particion.csv",
        "ciudades": raiz / "ciudades_nacional.csv",
    }


def city_table(raiz: Path = RAIZ_DATOS, *, forzar: bool = False) -> pd.DataFrame:
    """Size and deprivation of every city in the national set, which is what the split needs.

    Cached to disk because it walks the AGEB layer of all 32 states, and the partition has
    to be reproducible from the same numbers every time it is rebuilt.
    """
    destino = rutas(raiz)["ciudades"]
    if destino.exists() and not forzar:
        return pd.read_csv(destino, dtype={"clave": str})

    catalogo = ciudades_por_tamano(raiz=raiz, estratificar=True)
    filas = []
    for clave in catalogo:
        try:
            _, agebs = aoi_de_ciudad(clave, raiz, catalogo=catalogo)
        except Exception:
            log.warning("no geometry for %s", clave, exc_info=True)
            continue
        filas.append(
            {
                "clave": clave,
                "nombre": catalogo[clave].nombre,
                "entidad": catalogo[clave].entidad,
                "agebs": len(agebs),
                "altos": float(agebs.grado.isin(("Alto", "Muy alto")).mean()),
            }
        )
    tabla = pd.DataFrame(filas)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False)
    log.info("%d cities catalogued into %s", len(tabla), destino)
    return tabla


def build_city(
    clave: str,
    sensor: str,
    *,
    raiz: Path = RAIZ_DATOS,
    encoder: encoders.PatchEncoder | None = None,
    size: int = tiling.TILE_SIZE,
    min_valid_fraction: float = tiling.MIN_VALID_FRACTION,
    minimo_instancias: int = 32,
    catalogo: dict | None = None,
    forzar: bool = False,
) -> dict[str, Path]:
    """Tiles a city, assembles its bags, and encodes its patches if an encoder is given.

    Without an encoder the geometry half still runs, which is what lets the layout be
    checked and the partition be built before any deep learning stack is installed.
    """
    if sensor not in CANALES:
        raise KeyError(f"unknown sensor {sensor!r}, expected one of {sorted(CANALES)}")
    destino = rutas(raiz)
    salidas = {
        "instancias": destino["instancias"] / f"{clave}_{sensor}.parquet",
        "bolsas": destino["bolsas"] / f"{clave}.parquet",
    }

    compuesto = raiz / "compuestos" / f"{clave}_{sensor}.tif"
    if not compuesto.exists():
        raise FileNotFoundError(f"{clave} has no {sensor} composite yet: {compuesto}")

    bandas, malla, _ = cargar(compuesto)
    faltantes = [c for c in CANALES[sensor] if c not in bandas]
    if faltantes:
        raise KeyError(f"{compuesto.name} is missing {faltantes}")
    bandas = {c: bandas[c] for c in CANALES[sensor]}

    catalogo = catalogo or ciudades_por_tamano(raiz=raiz, estratificar=True)
    _, agebs = aoi_de_ciudad(clave, raiz, catalogo=catalogo)

    tiles = tiling.select(bandas, size=size, min_valid_fraction=min_valid_fraction)
    instancias, bolsas = bags.build(tiles, malla, agebs, clave, minimo_instancias=minimo_instancias)

    for nombre in ("instancias", "bolsas"):
        salidas[nombre].parent.mkdir(parents=True, exist_ok=True)
    instancias.to_parquet(salidas["instancias"], index=False)
    bolsas.to_parquet(salidas["bolsas"], index=False)

    if encoder is None:
        return salidas

    vectores = destino["vectores"] / f"{clave}_{sensor}.npz"
    if vectores.exists() and not forzar:
        log.info("%s already encoded", vectores.name)
        salidas["vectores"] = vectores
        return salidas

    # solo se codifican los parches que sobrevivieron al armado de bolsas: los que
    # cayeron fuera de toda AGEB o en una bolsa demasiado chica ya no son instancias
    conservados = [tiles[i] for i in instancias.tile]
    matriz = encoders.extract(bandas, conservados, encoder, order=CANALES[sensor])
    salidas["vectores"] = encoders.save(
        matriz, vectores, tile=instancias.tile.to_numpy(), cvegeo=instancias.cvegeo.to_numpy()
    )
    return salidas


def build_split(raiz: Path = RAIZ_DATOS, *, forzar: bool = False, **kwargs) -> pd.DataFrame:
    """Writes the train and test partition of the national set.

    Built once and read from disk afterwards, because a partition that quietly changes
    between runs turns every comparison of results into a comparison of partitions.
    """
    from satinsight import splits

    destino = rutas(raiz)["particion"]
    if destino.exists() and not forzar:
        return pd.read_csv(destino)
    particion = splits.assign(city_table(raiz), **kwargs)
    particion.to_csv(destino, index=False)
    log.info("partition written to %s", destino)
    return particion


def collect(sensor: str, raiz: Path = RAIZ_DATOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gathers the per-city tables of every city already built into two national ones."""
    destino = rutas(raiz)
    instancias = sorted(destino["instancias"].glob(f"*_{sensor}.parquet"))
    if not instancias:
        raise FileNotFoundError(f"no city has been tiled for {sensor} yet")
    i = pd.concat([pd.read_parquet(p) for p in instancias], ignore_index=True)
    claves = set(i.ciudad)
    b = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted(destino["bolsas"].glob("*.parquet"))
            if p.stem in claves
        ],
        ignore_index=True,
    )
    log.info("%d cities, %d bags, %d instances", len(claves), len(b), len(i))
    return i, b
