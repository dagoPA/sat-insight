"""Catalogue of municipalities to composite in Colombia and Brazil, one box per seat.

The first pass abroad used six hand-drawn metropolitan boxes and reached 35 and 48 bags,
the small-supply end of the Mexican curve. Walking each country's curve needs a supply
of the Mexican order, several hundred municipalities, and that means one box per
municipality derived from the official geography rather than by hand.

Colombia: the largest urban zone of class 1 (the municipal seat) in DANE's MGN 2023,
for every municipality with a poverty rate, kept when it covers at least the area floor.
Brazil: the urban census tracts of the seat district (the district named after the
municipality) in the 2022 tract geometries, for every municipality with an income
figure, ranked by urban population. Each box is the seat's bounding box with a margin.

Writes `data/transfer/boxes.csv` with key, country, municipality key and name, state,
bounding box and label, which the compositing, encoding and bag tools then consume.

Usage: transfer_catalogue.py [colombia_min_km2] [brazil_min_urban_population]
"""

import logging
import re
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

from math import cos, radians  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.transfer import IBGE_SECTORS, brazil_municipal_income, colombia_ipm  # noqa: E402

COLOMBIA_MIN_KM2 = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
BRAZIL_MIN_URBAN_POP = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
MARGIN_M = 1000
METRES_PER_DEGREE = 111_320
UF_CODES = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "11": "RO",
    "14": "RR", "43": "RS", "42": "SC", "28": "SE", "35": "SP", "17": "TO",
}  # fmt: skip
HAND_KEYS = {
    "colombia": {"bogota", "medellin", "cali"},
    "brazil": {"riodejaneiro", "saopaulo", "belohorizonte"},
}


def slug(name: str, code: str) -> str:
    """ASCII key from a name, suffixed with the code so homonyms never collide."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", plain.lower()) + code


def padded(bounds, margin_m: float = MARGIN_M):
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bounds)
    d_lat = margin_m / METRES_PER_DEGREE
    d_lon = d_lat / max(cos(radians((lat_min + lat_max) / 2)), 1e-6)
    return (
        round(lon_min - d_lon, 5),
        round(lat_min - d_lat, 5),
        round(lon_max + d_lon, 5),
        round(lat_max + d_lat, 5),
    )


def colombia() -> pd.DataFrame:
    zones = gpd.read_file(
        f"zip://{DATA_ROOT / 'transfer' / 'MGN2023_URB_ZONA_URBANA.zip'}!MGN_URB_ZONA_URBANA.shp"
    ).to_crs("EPSG:4326")
    seats = zones[zones.clas_ccdgo == "1"].copy()
    seats["area_km2"] = seats.to_crs("EPSG:3116").geometry.area / 1e6
    seats = seats.sort_values("area_km2", ascending=False).drop_duplicates("mpio_cdpmp")
    labels = colombia_ipm()
    table = seats.merge(labels, left_on="mpio_cdpmp", right_on="municipality")
    table = table[table.area_km2 >= COLOMBIA_MIN_KM2]
    rows = []
    for row in table.itertuples():
        rows.append(
            {
                "key": slug(row.name, row.municipality),
                "country": "colombia",
                "municipality": row.municipality,
                "name": row.name.title(),
                "state": row.municipality[:2],
                "bbox": " ".join(str(v) for v in padded(row.geometry.bounds)),
                "label": round(float(row.poor_share), 4),
                "size": round(float(row.area_km2), 2),
            }
        )
    logging.info("colombia: %d seats above %.1f km2", len(rows), COLOMBIA_MIN_KM2)
    return pd.DataFrame(rows)


def brazil() -> pd.DataFrame:
    urban = pd.read_csv(DATA_ROOT / "transfer" / "brazil_urban_population.csv", dtype=str)
    urban["pop"] = urban["pop"].astype(float)
    urban = urban[urban["pop"] >= BRAZIL_MIN_URBAN_POP]
    income = brazil_municipal_income().set_index("municipality")
    rows = []
    for state_code, group in urban.groupby("CD_UF"):
        state = UF_CODES[state_code]
        path = DATA_ROOT / "transfer" / "setores" / f"{state}_setores_CD2022.gpkg"
        if state in IBGE_SECTORS:
            path = DATA_ROOT / "transfer" / IBGE_SECTORS[state]
        if not path.exists():
            logging.warning(
                "%s: tract geometries missing, %d municipalities skipped", state, len(group)
            )
            continue
        try:
            tracts = gpd.read_file(
                path, columns=["CD_SETOR", "CD_MUN", "NM_MUN", "NM_DIST", "SITUACAO"]
            )
        except Exception as error:  # a file still downloading reads as malformed
            logging.warning("%s: tract geometries unreadable (%s); skipped", state, error)
            continue
        tracts = tracts[tracts.SITUACAO == "Urbana"].to_crs("EPSG:4326")
        for row in group.itertuples():
            own = tracts[tracts.CD_MUN == row.CD_MUN]
            seat = own[own.NM_DIST == own.NM_MUN]
            chosen = seat if len(seat) else own
            if chosen.empty or row.CD_MUN not in income.index:
                continue
            rows.append(
                {
                    "key": slug(row.NM_MUN, row.CD_MUN),
                    "country": "brazil",
                    "municipality": row.CD_MUN,
                    "name": row.NM_MUN,
                    "state": state,
                    "bbox": " ".join(str(v) for v in padded(chosen.total_bounds)),
                    "label": round(float(income.loc[row.CD_MUN, "mean_income"]), 2),
                    "size": int(row.pop),
                }
            )
        logging.info("%s: %d municipalities boxed", state, len(group))
    logging.info("brazil: %d seats above %d urban residents", len(rows), BRAZIL_MIN_URBAN_POP)
    return pd.DataFrame(rows)


def main() -> int:
    table = pd.concat([colombia(), brazil()], ignore_index=True)
    out = DATA_ROOT / "transfer" / "boxes.csv"
    table.to_csv(out, index=False)
    print(table.groupby("country").size().to_string(), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
