"""Independent same-year replication: score the map against CONAPO's urban marginacion.

The GRS is one institution's reading of the 2020 census. CONAPO published a different
index over the same census, different indicator set, different aggregation method, a
continuous score IM_2020 and a five-level grade GM_2020 per urban AGEB, and its 50,790
AGEB match this project's keys exactly. A map trained on CONEVAL aggregates that also
orders CONAPO's index within municipalities cannot be an artifact of one office's
methodology, and the continuous index gives the correlation more resolution than five
levels do.

Consumes the persisted validation scores; touches no training.

Usage: replicacion_imu.py
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from satinsight.llp import bootstrap_within  # noqa: E402

HIGH_GRADES = ("Alto", "Muy alto")


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/predicciones_val.parquet"
    split = "test" if "test" in source else "val"
    scores = pd.read_parquet(source)
    imu = pd.ExcelFile("data/externos/IMU_2020.xls").parse("IMU_2020")
    imu = imu.rename(columns={"CVE_AGEB": "cvegeo"})[["cvegeo", "IM_2020", "GM_2020"]]
    imu["cvegeo"] = imu.cvegeo.astype(str)
    # IM_2020 is a DP2 distance to the ideal: it grows toward LESS marginacion, and the
    # grade means confirm it (Muy alto 110.6 < Muy bajo 124.9). The comparable direction
    # is deprivation, so the sign flips here, once, at the door.
    imu["deprivation"] = -imu.IM_2020

    # promedio del ensamble por token, luego promedio por AGEB
    per_token = scores.groupby(
        ["city", "municipality", "cvegeo", "y0", "x0"], observed=True
    ).score.mean()
    per_ageb = (
        per_token.groupby(["city", "municipality", "cvegeo"], observed=True).mean().reset_index()
    )
    joined = per_ageb.merge(imu, on="cvegeo", how="inner")
    print(f"{len(per_ageb)} AGEB scored · {len(joined)} matched to IMU", flush=True)

    within, per_bag = [], []
    for municipality, group in joined.groupby("municipality", observed=True):
        if len(group) < 20 or group.deprivation.nunique() < 2:
            continue
        rho = float(spearmanr(group.score, group.deprivation).statistic)
        within.append(rho)
        per_bag.append((municipality, rho))
    city_of = dict(zip(joined.municipality, joined.city, strict=False))
    mean, half = bootstrap_within(per_bag, city_of)

    high = joined.GM_2020.isin(HIGH_GRADES).astype(int)
    auroc = float(roc_auc_score(high, joined.score)) if high.nunique() == 2 else float("nan")
    pooled = float(spearmanr(joined.score, joined.deprivation).statistic)

    result = {
        "agebs": len(joined),
        "municipalities_scored": len(within),
        "spearman_within": mean,
        "ci95_half": half,
        "spearman_pooled": pooled,
        "auroc_high": auroc,
    }
    pd.DataFrame([result]).to_csv(
        "data/replicacion_imu" + ("_test.csv" if split == "test" else ".csv"), index=False
    )
    print("\n===== CONAPO REPLICATION =====", flush=True)
    print(
        f"within-municipality Spearman vs CONAPO deprivation: {mean:+.3f} ± {half:.3f} "
        f"over {len(within)} municipalities · pooled {pooled:+.3f} · "
        f"AUROC(GM high) {auroc:.3f}",
        flush=True,
    )
    print("reference: +0.182 within against the GRS grade it was trained toward", flush=True)


if __name__ == "__main__":
    main()
