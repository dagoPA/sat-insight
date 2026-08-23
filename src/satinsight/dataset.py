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


def paths(root: Path = DATA_ROOT) -> dict[str, Path]:
    """Where each artefact of the stage lives."""
    return {
        "instances": root / "instances",
        "bags": root / "bags",
        "vectors": root / "vectors",
        "partition": root / "partition.csv",
        "cities": root / "ciudades_nacional.csv",
    }


def city_table(root: Path = DATA_ROOT, *, force: bool = False) -> pd.DataFrame:
    """Size and deprivation of every city in the national set, which is what the split needs.

    Cached to disk because it walks the AGEB layer of all 32 states, and the partition has
    to be reproducible from the same numbers every time it is rebuilt.
    """
    destination = paths(root)["cities"]
    if destination.exists() and not force:
        return pd.read_csv(destination, dtype={"clave": str})

    catalogue = cities_by_size(root=root, stratify=True)
    rows = []
    for key in catalogue:
        try:
            _, agebs = city_aoi(key, root, catalogue=catalogue)
        except Exception:
            log.warning("no geometry for %s", key, exc_info=True)
            continue
        rows.append(
            {
                "clave": key,
                "nombre": catalogue[key].name,
                "entidad": catalogue[key].state,
                "agebs": len(agebs),
                "altos": float(agebs.grado.isin(("Alto", "Muy alto")).mean()),
            }
        )
    table = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    log.info("%d cities catalogued into %s", len(table), destination)
    return table


def build_city(
    key: str,
    sensor: str,
    *,
    root: Path = DATA_ROOT,
    encoder: encoders.PatchEncoder | None = None,
    size: int = tiling.WINDOW_SIZE,
    min_valid_fraction: float = tiling.MIN_VALID_FRACTION,
    min_instances: int = 32,
    catalogue: dict | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Tiles a city, assembles its bags, and encodes its patches if an encoder is given.

    Without an encoder the geometry half still runs, which is what lets the layout be
    checked and the partition be built before any deep learning stack is installed.
    """
    if sensor not in CHANNELS:
        raise KeyError(f"unknown sensor {sensor!r}, expected one of {sorted(CHANNELS)}")
    where = paths(root)
    outputs = {
        "instances": where["instances"] / f"{key}_{sensor}.parquet",
        "bags": where["bags"] / f"{key}.parquet",
    }

    composite = root / "composites" / f"{key}_{sensor}.tif"
    if not composite.exists():
        raise FileNotFoundError(f"{key} has no {sensor} composite yet: {composite}")

    bands, grid, _ = load(composite)
    missing = [c for c in CHANNELS[sensor] if c not in bands]
    if missing:
        raise KeyError(f"{composite.name} is missing {missing}")
    bands = {c: bands[c] for c in CHANNELS[sensor]}

    catalogue = catalogue or cities_by_size(root=root, stratify=True)
    _, agebs = city_aoi(key, root, catalogue=catalogue)

    windows = tiling.select(bands, size=size, min_valid_fraction=min_valid_fraction)
    tokens, _ = tiling.instances(windows, bands, min_valid_fraction=min_valid_fraction)
    instances, bag_table = bags.build(tokens, grid, agebs, key, min_instances=min_instances)

    for name in ("instances", "bags"):
        outputs[name].parent.mkdir(parents=True, exist_ok=True)
    instances.to_parquet(outputs["instances"], index=False)
    bag_table.to_parquet(outputs["bags"], index=False)

    if encoder is None:
        return outputs

    vectors = where["vectors"] / f"{key}_{sensor}.npz"
    if vectors.exists() and not force:
        log.info("%s already encoded", vectors.name)
        outputs["vectors"] = vectors
        return outputs

    # the model receives whole windows and returns all of their tokens, so it encodes once
    # and afterwards the rows that stayed instances are the ones kept
    matrix, encoded = encoders.extract(
        bands, windows, encoder, order=CHANNELS[sensor], min_valid_fraction=min_valid_fraction
    )
    position = {(t.y0, t.x0): i for i, t in enumerate(encoded)}
    rows_kept = [position[(y, x)] for y, x in zip(instances.y0, instances.x0, strict=True)]
    outputs["vectors"] = encoders.save(
        matrix[rows_kept],
        vectors,
        y0=instances.y0.to_numpy(),
        x0=instances.x0.to_numpy(),
        cvegeo=instances.cvegeo.to_numpy(),
    )
    return outputs


def build_split(root: Path = DATA_ROOT, *, force: bool = False, **kwargs) -> pd.DataFrame:
    """Writes the train, validation and test partition of the national set.

    Built once and read from disk afterwards, because a partition that quietly changes
    between runs turns every comparison of results into a comparison of partitions.
    """
    from satinsight import splits

    destination = paths(root)["partition"]
    if destination.exists() and not force:
        return pd.read_csv(destination)
    partition = splits.assign(city_table(root), **kwargs)
    partition.to_csv(destination, index=False)
    log.info("partition written to %s", destination)
    return partition


def collect(sensor: str, root: Path = DATA_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gathers the per-city tables of every city already built into two national ones."""
    where = paths(root)
    instance_files = sorted(where["instances"].glob(f"*_{sensor}.parquet"))
    if not instance_files:
        raise FileNotFoundError(f"no city has been tiled for {sensor} yet")
    instances = pd.concat([pd.read_parquet(p) for p in instance_files], ignore_index=True)
    keys = set(instances.ciudad)
    bag_table = pd.concat(
        [pd.read_parquet(p) for p in sorted(where["bags"].glob("*.parquet")) if p.stem in keys],
        ignore_index=True,
    )
    log.info("%d cities, %d bags, %d instances", len(keys), len(bag_table), len(instances))
    return instances, bag_table
