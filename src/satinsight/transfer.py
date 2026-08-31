"""Labels for the zero-shot transfer: Brazil's AGSN mask and Bogota's strata.

Neither country publishes anything like the GRS, and the evaluation never needs them to.
Zero-shot scoring only requires ground to correlate the instance predictions against, each
country against its own within-city chance level.

Brazil: the IBGE aglomerados subnormais, 2019 preliminary delimitation, a national polygon
layer of substandard settlements. Binary ground: a token either falls in one or it does
not. The map metric is AUROC of the instance score for recovering AGSN membership.

Colombia: Bogota's socioeconomic stratification, an ordinal 1-6 published per block by the
district and mandated by national law. Ordinal ground: Spearman between the instance score
and the stratum, inverted so higher means more deprived, evaluated within the city.

Both files are downloaded once by `herramientas/` and read from `data/transfer/`; the
paths and the column names are pinned here, next to the code that interprets them, so a
silent schema change in a re-download fails loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from satinsight.download import DATA_ROOT

log = logging.getLogger(__name__)

AGSN_PATH = Path("transfer/AGSN_2019/AGSN_2019.shp")
AGSN_CITY_COLUMN = "AGSN_NM_MU"
BOGOTA_PATH = Path("transfer/estratificacion_bogota.gpkg")
STRATUM_COLUMN = "ESTRATO"


def agsn_of_city(city_name: str, root: Path = DATA_ROOT) -> gpd.GeoDataFrame:
    """AGSN polygons of one Brazilian municipality, in WGS84.

    The layer names municipalities in Portuguese with accents; matching is done on the
    exact value the column carries, and an empty result raises so a typo in the name never
    reads as a city without settlements.
    """
    table = gpd.read_file(root / AGSN_PATH)
    chosen = table[table[AGSN_CITY_COLUMN] == city_name]
    if chosen.empty:
        known = sorted(table[AGSN_CITY_COLUMN].unique())
        raise KeyError(f"no AGSN named {city_name!r}; closest columns hold e.g. {known[:5]}")
    log.info("%s: %d AGSN polygons", city_name, len(chosen))
    return chosen.to_crs("EPSG:4326")[[AGSN_CITY_COLUMN, "geometry"]]


def bogota_strata(root: Path = DATA_ROOT) -> gpd.GeoDataFrame:
    """Blocks of Bogota with their stratum, in WGS84, stratum zero dropped.

    Zero marks unstratified ground, institutional, industrial, unbuilt, and carries no
    ordinal meaning, so keeping it would poison the correlation.
    """
    table = gpd.read_file(root / BOGOTA_PATH)
    table = table[table[STRATUM_COLUMN].between(1, 6)]
    log.info("Bogota: %d stratified blocks", len(table))
    return table.to_crs("EPSG:4326")[[STRATUM_COLUMN, "geometry"]]
