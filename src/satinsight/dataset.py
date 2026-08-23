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
from satinsight.agebs import cities_by_size
from satinsight.cache import load
from satinsight.download import DATA_ROOT
from satinsight.pipeline import city_aoi

log = logging.getLogger(__name__)

CHANNELS = {
    "s2": ["B02", "B03", "B04", "B08", "B11"],
    "s1": ["vh", "vv"],
}
"""Channels fed to the encoder per sensor, in the order their wavelengths are declared."""


def rutas(root: Path = DATA_ROOT) -> dict[str, Path]:
    """Where each artefact of the stage lives."""
    return {
        "instancias": root / "instancias",
        "bolsas": root / "bolsas",
        "vectores": root / "vectores",
        "particion": root / "particion.csv",
        "cities": root / "ciudades_nacional.csv",
    }


def city_table(root: Path = DATA_ROOT, *, force: bool = False) -> pd.DataFrame:
    """Size and deprivation of every city in the national set, which is what the split needs.

    Cached to disk because it walks the AGEB layer of all 32 states, and the partition has
    to be reproducible from the same numbers every time it is rebuilt.
    """
    destino = rutas(root)["cities"]
    if destino.exists() and not force:
        return pd.read_csv(destino, dtype={"clave": str})

    catalogue = cities_by_size(root=root, stratify=True)
    filas = []
    for clave in catalogue:
        try:
            _, agebs = city_aoi(clave, root, catalogue=catalogue)
        except Exception:
            log.warning("no geometry for %s", clave, exc_info=True)
            continue
        filas.append(
            {
                "clave": clave,
                "nombre": catalogue[clave].name,
                "entidad": catalogue[clave].state,
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
    root: Path = DATA_ROOT,
    encoder: encoders.PatchEncoder | None = None,
    size: int = tiling.WINDOW_SIZE,
    min_valid_fraction: float = tiling.MIN_VALID_FRACTION,
    minimo_instancias: int = 32,
    catalogue: dict | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Tiles a city, assembles its bags, and encodes its patches if an encoder is given.

    Without an encoder the geometry half still runs, which is what lets the layout be
    checked and the partition be built before any deep learning stack is installed.
    """
    if sensor not in CHANNELS:
        raise KeyError(f"unknown sensor {sensor!r}, expected one of {sorted(CHANNELS)}")
    destino = rutas(root)
    salidas = {
        "instancias": destino["instancias"] / f"{clave}_{sensor}.parquet",
        "bolsas": destino["bolsas"] / f"{clave}.parquet",
    }

    compuesto = root / "compuestos" / f"{clave}_{sensor}.tif"
    if not compuesto.exists():
        raise FileNotFoundError(f"{clave} has no {sensor} composite yet: {compuesto}")

    bandas, malla, _ = load(compuesto)
    faltantes = [c for c in CHANNELS[sensor] if c not in bandas]
    if faltantes:
        raise KeyError(f"{compuesto.name} is missing {faltantes}")
    bandas = {c: bandas[c] for c in CHANNELS[sensor]}

    catalogue = catalogue or cities_by_size(root=root, stratify=True)
    _, agebs = city_aoi(clave, root, catalogue=catalogue)

    ventanas = tiling.select(bandas, size=size, min_valid_fraction=min_valid_fraction)
    tokens, _ = tiling.instances(ventanas, bandas, min_valid_fraction=min_valid_fraction)
    instancias, bolsas = bags.build(
        tokens, malla, agebs, clave, minimo_instancias=minimo_instancias
    )

    for nombre in ("instancias", "bolsas"):
        salidas[nombre].parent.mkdir(parents=True, exist_ok=True)
    instancias.to_parquet(salidas["instancias"], index=False)
    bolsas.to_parquet(salidas["bolsas"], index=False)

    if encoder is None:
        return salidas

    vectores = destino["vectores"] / f"{clave}_{sensor}.npz"
    if vectores.exists() and not force:
        log.info("%s already encoded", vectores.name)
        salidas["vectores"] = vectores
        return salidas

    # el modelo recibe ventanas enteras y devuelve todos sus tokens, así que se codifica
    # una vez y después se conservan las filas que siguieron siendo instancias
    matriz, codificados = encoders.extract(
        bandas, ventanas, encoder, order=CHANNELS[sensor], min_valid_fraction=min_valid_fraction
    )
    posicion = {(t.y0, t.x0): i for i, t in enumerate(codificados)}
    filas = [posicion[(y, x)] for y, x in zip(instancias.y0, instancias.x0, strict=True)]
    salidas["vectores"] = encoders.save(
        matriz[filas],
        vectores,
        y0=instancias.y0.to_numpy(),
        x0=instancias.x0.to_numpy(),
        cvegeo=instancias.cvegeo.to_numpy(),
    )
    return salidas


def build_split(root: Path = DATA_ROOT, *, force: bool = False, **kwargs) -> pd.DataFrame:
    """Writes the train and test partition of the national set.

    Built once and read from disk afterwards, because a partition that quietly changes
    between runs turns every comparison of results into a comparison of partitions.
    """
    from satinsight import splits

    destino = rutas(root)["particion"]
    if destino.exists() and not force:
        return pd.read_csv(destino)
    particion = splits.assign(city_table(root), **kwargs)
    particion.to_csv(destino, index=False)
    log.info("partition written to %s", destino)
    return particion


def collect(sensor: str, root: Path = DATA_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gathers the per-city tables of every city already built into two national ones."""
    destino = rutas(root)
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
