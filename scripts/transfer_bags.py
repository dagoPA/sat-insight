"""Bags of the transfer boxes: each token gets its municipality, its label, and its truth.

The Mexican pipeline builds bags from AGEB polygons that carry both the fine grade and
the municipality key. Colombia and Brazil have no such layer, so the join is done here in
three steps from the encoded tokens: the municipality that contains the token centre,
from each country's official boundaries; the municipal aggregate that country publishes,
which becomes the bag label; and the fine ground the evaluation runs on, which never
reaches training. Only built tokens are kept, judged by WorldCover, so a municipal share
of people is not diluted over countryside the aggregate does not describe.

Colombia: label is the municipal multidimensional poverty rate (share of people poor);
truth is Bogota's stratum by block, nearest block within 120 m, inverted so higher means
more deprived. Brazil: label is the municipal mean monthly income of the household head;
truth is the median income of the household head in the census tract the token falls in.

Usage: transfer_bags.py [key ...]   (default: the six boxes)
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from satinsight import landcover  # noqa: E402
from satinsight.cache import load  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.tiling import TOKEN_SIZE  # noqa: E402
from satinsight.transfer import (  # noqa: E402
    STRATUM_COLUMN,
    bogota_strata,
    brazil_municipal_income,
    brazil_municipalities,
    brazil_sectors,
    colombia_ipm,
    colombia_municipalities,
    municipalities_in_box,
)

sys.path.insert(0, "scripts")
from transfer_composites import transfer_aoi  # noqa: E402

COUNTRY = {
    "bogota": "colombia",
    "medellin": "colombia",
    "cali": "colombia",
    "riodejaneiro": "brazil",
    "saopaulo": "brazil",
    "belohorizonte": "brazil",
}
STATE = {"riodejaneiro": "RJ", "saopaulo": "SP", "belohorizonte": "MG"}
BUILT_CODE = 50
BUILT_FLOOR = 0.10
JOIN_M = 120
METRIC_CRS = {"colombia": "EPSG:3116", "brazil": "EPSG:5880"}


def built_fraction(key: str, tokens: pd.DataFrame) -> np.ndarray:
    _, grid, _ = load(DATA_ROOT / "composites" / f"{key}_s2.tif")
    classes = landcover.mosaic(transfer_aoi(key), grid)
    out = np.empty(len(tokens))
    for i, (y0, x0) in enumerate(zip(tokens.y0, tokens.x0, strict=True)):
        window = classes[y0 : y0 + TOKEN_SIZE, x0 : x0 + TOKEN_SIZE]
        valid = window[window != landcover.NO_DATA]
        out[i] = float((valid == BUILT_CODE).mean()) if valid.size else 0.0
    return out


def as_points(tokens: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        tokens, geometry=gpd.points_from_xy(tokens.lon, tokens.lat), crs="EPSG:4326"
    )


def assign_municipality(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> pd.Series:
    joined = gpd.sjoin(
        points, polygons[["municipality", "geometry"]], how="left", predicate="within"
    )
    return joined[~joined.index.duplicated()]["municipality"]


def colombia_truth(key: str, points: gpd.GeoDataFrame) -> pd.Series:
    """Deprivation from the stratum, Bogota only; elsewhere nothing to validate against."""
    truth = pd.Series(np.nan, index=points.index, dtype="float64")
    if key != "bogota":
        return truth
    crs = METRIC_CRS["colombia"]
    blocks = bogota_strata().to_crs(crs)[[STRATUM_COLUMN, "geometry"]]
    joined = gpd.sjoin_nearest(points.to_crs(crs), blocks, how="inner", max_distance=JOIN_M)
    joined = joined[~joined.index.duplicated()]
    truth.loc[joined.index] = -joined[STRATUM_COLUMN].astype(float)
    return truth


def brazil_truth(key: str, points: gpd.GeoDataFrame, bbox) -> tuple[pd.Series, pd.Series]:
    """Deprivation as minus the log median income of the tract, and the tract key."""
    tracts = brazil_sectors(STATE[key], bbox)
    joined = gpd.sjoin(
        points, tracts[["sector", "median_income", "geometry"]], how="left", predicate="within"
    )
    joined = joined[~joined.index.duplicated()]
    truth = -np.log(joined["median_income"].astype(float))
    return truth, joined["sector"]


def build(key: str) -> pd.DataFrame:
    country = COUNTRY[key]
    tokens = pd.read_parquet(DATA_ROOT / "transfer" / f"tokens_{key}.parquet")
    # row indexes the vector matrix, which the filtering below must not disturb
    tokens["row"] = np.arange(len(tokens))
    tokens["built"] = built_fraction(key, tokens)
    points = as_points(tokens)
    bbox = transfer_aoi(key).bbox

    if country == "colombia":
        polygons = municipalities_in_box(colombia_municipalities(), bbox)
        labels = colombia_ipm()[["municipality", "poor_share"]]
        tokens["municipality"] = assign_municipality(points, polygons).to_numpy()
        tokens = tokens.merge(labels, on="municipality", how="left")
        tokens["truth"] = colombia_truth(key, points).to_numpy()
        tokens["unit"] = None
    else:
        polygons = municipalities_in_box(brazil_municipalities(STATE[key], bbox), bbox)
        labels = brazil_municipal_income()[["municipality", "mean_income"]]
        tokens["municipality"] = assign_municipality(points, polygons).to_numpy()
        tokens = tokens.merge(labels, on="municipality", how="left")
        truth, unit = brazil_truth(key, points, bbox)
        tokens["truth"] = truth.to_numpy()
        tokens["unit"] = unit.to_numpy()

    tokens["country"] = country
    kept = tokens[(tokens.built >= BUILT_FLOOR) & tokens.municipality.notna()].reset_index(
        drop=True
    )
    label = "poor_share" if country == "colombia" else "mean_income"
    logging.info(
        "%s: %d tokens, %d built with a municipality, %d municipalities, %d labelled, "
        "%d with truth",
        key,
        len(tokens),
        len(kept),
        kept.municipality.nunique(),
        int(kept[label].notna().sum()),
        int(kept.truth.notna().sum()),
    )
    return kept


def main() -> int:
    keys = sys.argv[1:] or list(COUNTRY)
    failed = []
    for key in keys:
        try:
            table = build(key)
            table.to_parquet(DATA_ROOT / "transfer" / f"bags_{key}.parquet", index=False)
            print(
                f"OK {key} · {len(table)} built tokens · {table.municipality.nunique()} bags",
                flush=True,
            )
        except Exception as e:
            failed.append(key)
            logging.exception("FAIL %s", key)
            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
