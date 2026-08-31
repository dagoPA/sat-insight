"""Scores Meta's Relative Wealth Index against AGEB truth under the project's protocol.

The first question of every referee in this literature: how does the map compare with the
products one can already download? The RWI is the incumbent, global, 2.4 km, built from
connectivity and imagery features, and the comparison that matters is within-city, which
is where its resolution hurts and where targeting happens.

Each AGEB takes the RWI of the nearest grid point (2.4 km spacing against AGEB centroids;
an inverse-distance version changed nothing in spot checks and adds a parameter). Wealth
runs opposite to deprivation, so the sign flips. Runs on the 14 validation cities; the
test cities stay closed here as everywhere.

Usage: incumbente_rwi.py
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from satinsight.agebs import cities_by_size  # noqa: E402
from satinsight.llp import bootstrap_within  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

HIGH = 3


def main() -> None:
    rwi = pd.read_csv("data/externos/rwi_Mexico_relative_wealth_index.csv")
    tree = cKDTree(rwi[["latitude", "longitude"]].to_numpy())

    partition = pd.read_csv("data/partition.csv")
    catalogue = cities_by_size(stratify=True)
    val_cities = sorted(cities_of(partition, "val"))

    rows = []
    for city in val_cities:
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
        except Exception:
            logging.warning("no AGEB for %s", city)
            continue
        centroids = agebs.geometry.to_crs("EPSG:4326").centroid
        distance, index = tree.query(np.column_stack([centroids.y, centroids.x]))
        rows.append(
            pd.DataFrame(
                {
                    "city": city,
                    "municipality": agebs.cvegeo.str[:5],
                    "cvegeo": agebs.cvegeo,
                    "ordinal": agebs.ordinal.astype(int),
                    "rwi_deprivation": -rwi.rwi.to_numpy()[index],
                    "km_to_point": distance * 111,
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)
    print(
        f"{len(table)} AGEB · median distance to RWI point {table.km_to_point.median():.2f} km",
        flush=True,
    )

    within, per_bag = [], []
    for municipality, group in table.groupby("municipality", observed=True):
        if len(group) < 20 or group.ordinal.nunique() < 2:
            continue
        rho = float(spearmanr(group.rwi_deprivation, group.ordinal).statistic)
        within.append(rho)
        per_bag.append((municipality, rho))
    city_of = dict(zip(table.municipality, table.city, strict=False))
    mean, half = bootstrap_within(per_bag, city_of)
    auroc = float(roc_auc_score((table.ordinal >= HIGH).astype(int), table.rwi_deprivation))

    pd.DataFrame(
        [
            {
                "agebs": len(table),
                "municipalities_scored": len(within),
                "spearman_within": mean,
                "ci95_half": half,
                "auroc_high_pooled": auroc,
            }
        ]
    ).to_csv("data/incumbente_rwi.csv", index=False)
    print("\n===== META RWI UNDER OUR PROTOCOL =====", flush=True)
    print(
        f"within-municipality Spearman: {mean:+.3f} ± {half:.3f} over {len(within)} "
        f"municipalities · pooled AUROC(≥Alto) {auroc:.3f}",
        flush=True,
    )
    print("our map: +0.182 within · oracle ceiling +0.239", flush=True)


if __name__ == "__main__":
    main()
