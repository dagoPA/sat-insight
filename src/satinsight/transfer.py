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

Both files are downloaded once by `scripts/` and read from `data/transfer/`; the
paths and the column names are pinned here, next to the code that interprets them, so a
silent schema change in a re-download fails loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

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


# ----------------------------------------------------------------------------------------
# Training in the transfer countries on their own municipal aggregates.
#
# Colombia publishes a municipal multidimensional poverty rate (IPM, DANE, census 2018), a
# share of people counted poor, which is a proportion and enters the label-proportion head
# unchanged. Brazil publishes per municipality the mean and median monthly income of the
# household head (IBGE, census 2022), a continuous aggregate. Both are what a statistics
# office hands out at municipal level, and neither carries any spatial detail.
#
# Fine ground for validation stays as in the zero-shot: Bogota's strata by block, and for
# Brazil the median income of the household head by census tract, the same variable one
# level down, which is exactly the Mexican design (municipal GRS against tract GRS).

MGN_PATH = Path("transfer/MGN2023_MPIO_POLITICO.zip")
MGN_LAYER = "MGN_ADM_MPIO_GRAFICO.shp"
IPM_PATH = Path("transfer/colombia_ipm_municipal.csv")
IBGE_MUNICIPAL_INCOME = Path("transfer/ibge_municipios_renda_responsavel.zip")
IBGE_SECTOR_INCOME = Path("transfer/ibge_setores_renda_responsavel.zip")
IBGE_SECTORS = {
    "RJ": "RJ_setores_CD2022.gpkg",
    "SP": "SP_setores_CD2022.gpkg",
    "MG": "MG_setores_CD2022.gpkg",
}
"""Census tract geometries of the three states the Brazilian boxes fall in."""

BOX_SHARE_FLOOR = 0.05
"""A municipality enters a box's bag set when at least this share of its area is inside.

A box clips its edge municipalities; one that only grazes the box would become a bag of a
few tokens carrying a label for ground mostly outside the image.
"""


def colombia_municipalities(root: Path = DATA_ROOT) -> gpd.GeoDataFrame:
    """Municipal polygons of Colombia (DANE MGN 2023) with their five-digit DIVIPOLA key."""
    table = gpd.read_file(f"zip://{root / MGN_PATH}!{MGN_LAYER}").to_crs("EPSG:4326")
    # mpio_cdpmp is the concatenated department and municipality code, the DIVIPOLA key
    return table.rename(columns={"mpio_cdpmp": "municipality", "mpio_cnmbr": "name"})[
        ["municipality", "name", "geometry"]
    ]


def colombia_ipm(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Municipal multidimensional poverty rate as a share in [0, 1], keyed by DIVIPOLA."""
    table = pd.read_csv(root / IPM_PATH, dtype=str)
    return pd.DataFrame(
        {
            "municipality": table["MPIO_CCDGO"].str.strip(),
            "name": table["MPIO_CNMBR"].str.strip(),
            "poor_share": table["IPM"].astype(float) / 100.0,
        }
    )


def _ibge_decimal(series: pd.Series) -> pd.Series:
    """IBGE writes decimals with a comma in some files and a point in others."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def brazil_municipal_income(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Mean and median monthly income of the household head per municipality, 2022."""
    import zipfile

    with zipfile.ZipFile(root / IBGE_MUNICIPAL_INCOME) as archive:
        table = pd.read_csv(
            archive.open(archive.namelist()[0]), sep=";", dtype=str, encoding="latin-1"
        )
    return pd.DataFrame(
        {
            "municipality": table["CD_MUN"].str.strip(),
            "name": table["NM_MUN"].str.strip(),
            "mean_income": _ibge_decimal(table["V06004"]),
            "median_income": _ibge_decimal(table["V06006"]),
            "heads": _ibge_decimal(table["V06001"]),
        }
    )


def brazil_sector_income(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Median income of the household head per census tract, 2022, keyed by CD_SETOR."""
    import zipfile

    with zipfile.ZipFile(root / IBGE_SECTOR_INCOME) as archive:
        table = pd.read_csv(
            archive.open(archive.namelist()[0]), sep=";", dtype=str, encoding="latin-1"
        )
    return pd.DataFrame(
        {
            "sector": table["CD_SETOR"].str.strip(),
            "median_income": _ibge_decimal(table["V06006"]),
            "heads": _ibge_decimal(table["V06001"]),
        }
    )


def brazil_sectors(state: str, bbox, root: Path = DATA_ROOT) -> gpd.GeoDataFrame:
    """Census tracts of one state inside a box, with municipality key and median income."""
    path = root / "transfer" / IBGE_SECTORS[state]
    tracts = gpd.read_file(path, bbox=tuple(bbox)).to_crs("EPSG:4326")
    tracts = tracts.rename(
        columns={"CD_SETOR": "sector", "CD_MUN": "municipality", "NM_MUN": "name"}
    )
    income = brazil_sector_income(root)
    return tracts[["sector", "municipality", "name", "SITUACAO", "geometry"]].merge(
        income, on="sector", how="left"
    )


def brazil_municipalities(state: str, bbox, root: Path = DATA_ROOT) -> gpd.GeoDataFrame:
    """Municipal polygons of one state inside a box, dissolved from its census tracts."""
    tracts = brazil_sectors(state, bbox, root)
    merged = tracts.dissolve(by="municipality")[["name", "geometry"]].reset_index()
    return merged


def municipalities_in_box(polygons: gpd.GeoDataFrame, bbox, floor: float = BOX_SHARE_FLOOR):
    """Keeps the municipalities with at least `floor` of their area inside the box."""
    from shapely.geometry import box as make_box

    window = make_box(*bbox)
    inside = polygons[polygons.intersects(window)].copy()
    share = inside.geometry.intersection(window).area / inside.geometry.area
    kept = inside[share >= floor]
    log.info(
        "%d of %d municipalities keep at least %.0f%% of their area in the box",
        len(kept),
        len(inside),
        100 * floor,
    )
    return kept.reset_index(drop=True)
