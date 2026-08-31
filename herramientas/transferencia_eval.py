"""Scores the zero-shot maps of Bogota and Rio against their own ground.

Rio: does the score find the aglomerados subnormais among the built city? The metric is
AUROC of AGSN membership, and the evaluation is restricted to tokens with built ground
under them (WorldCover built fraction above a floor). Without that restriction the Tijuca
forest and the bay would pad the negatives with easy ground and flatter the number.

Bogota: does the score order the stratification? Tokens join their nearest block within
120 m, blocks only exist on urban ground, so the scope restriction is automatic, and the
metric is Spearman against the stratum, inverted so higher means more deprived, plus AUROC
for the deprived strata (1 and 2).

Both are within-city measurements against each country's own chance level; nothing here is
comparable to the Mexican +0.213 and the tables must never pretend otherwise.

Usage: transferencia_eval.py
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from satinsight import landcover  # noqa: E402
from satinsight.cache import load  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.tiling import TOKEN_SIZE  # noqa: E402
from satinsight.transfer import agsn_of_city, bogota_strata  # noqa: E402

sys.path.insert(0, "herramientas")
from transferencia_compuestos import transfer_aoi  # noqa: E402

BUILT_CODE = 50
BUILT_FLOOR = 0.10
JOIN_M = 120


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


def rio() -> dict:
    tokens = pd.read_parquet("data/zeroshot_riodejaneiro.parquet")
    tokens["built"] = built_fraction("riodejaneiro", tokens)
    urban = as_points(tokens[tokens.built >= BUILT_FLOOR].copy())
    agsn = agsn_of_city("Rio de Janeiro")
    joined = gpd.sjoin(urban, agsn[["geometry"]], how="left", predicate="within")
    urban["in_agsn"] = (~joined.index_right.isna()).groupby(level=0).any().astype(int)
    auroc = float(roc_auc_score(urban.in_agsn, urban.score))
    print(
        f"rio: {len(tokens)} tokens · {len(urban)} built · "
        f"{urban.in_agsn.mean():.1%} in AGSN · AUROC {auroc:.3f}",
        flush=True,
    )
    return {
        "city": "riodejaneiro",
        "tokens": len(urban),
        "positive_share": float(urban.in_agsn.mean()),
        "auroc": auroc,
        "spearman": float("nan"),
    }


def bogota() -> dict:
    tokens = as_points(pd.read_parquet("data/zeroshot_bogota.parquet"))
    blocks = bogota_strata()
    metric_crs = "EPSG:3116"
    joined = gpd.sjoin_nearest(
        tokens.to_crs(metric_crs),
        blocks.to_crs(metric_crs)[["ESTRATO", "geometry"]],
        how="inner",
        max_distance=JOIN_M,
    )
    joined = joined[~joined.index.duplicated()]
    deprivation = -joined.ESTRATO
    rho = float(spearmanr(joined.score, deprivation).statistic)
    auroc = float(roc_auc_score((joined.ESTRATO <= 2).astype(int), joined.score))
    print(
        f"bogota: {len(tokens)} tokens · {len(joined)} joined a block · "
        f"Spearman vs stratum {rho:+.3f} · AUROC(estrato 1-2) {auroc:.3f}",
        flush=True,
    )
    return {
        "city": "bogota",
        "tokens": len(joined),
        "positive_share": float((joined.ESTRATO <= 2).mean()),
        "auroc": auroc,
        "spearman": rho,
    }


def main() -> None:
    rows = [rio(), bogota()]
    pd.DataFrame(rows).to_csv("data/transferencia_zeroshot.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
