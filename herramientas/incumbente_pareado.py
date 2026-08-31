"""Pairs the RWI comparison on the exact same AGEB universe, as the referees require.

The first pass scored RWI on the 3,243 AGEB its grid reaches and the model on the full
validation set: different denominators, so the comparison proved nothing. Here both are
scored per AGEB on the intersection, the within-municipality correlations are computed on
identical ground, and the interval is a city-clustered bootstrap of the DIFFERENCE, which
is the quantity the claim is about.

Consumes the persisted validation scores; touches no training.

Usage: incumbente_pareado.py
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

from satinsight.agebs import cities_by_size  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

RESAMPLES = 2000


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/predicciones_val.parquet"
    split = "test" if "test" in source else "val"
    scores = pd.read_parquet(source)
    per_ageb = (
        scores.groupby(["city", "municipality", "cvegeo"], observed=True).score.mean().reset_index()
    )

    rwi = pd.read_csv("data/externos/rwi_Mexico_relative_wealth_index.csv")
    tree = cKDTree(rwi[["latitude", "longitude"]].to_numpy())
    partition = pd.read_csv("data/partition.csv")
    catalogue = cities_by_size(stratify=True)
    rows = []
    for city in sorted(cities_of(partition, split)):
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
        except Exception:
            continue
        centroids = agebs.geometry.to_crs("EPSG:4326").centroid
        _, index = tree.query(np.column_stack([centroids.y, centroids.x]))
        rows.append(
            pd.DataFrame(
                {
                    "cvegeo": agebs.cvegeo,
                    "ordinal": agebs.ordinal.astype(int),
                    "rwi_deprivation": -rwi.rwi.to_numpy()[index],
                }
            )
        )
    truth = pd.concat(rows, ignore_index=True)
    both = per_ageb.merge(truth, on="cvegeo", how="inner")
    print(f"shared universe: {len(both)} AGEB", flush=True)

    ours, theirs, cities_of_mun = {}, {}, {}
    for municipality, group in both.groupby("municipality", observed=True):
        if len(group) < 20 or group.ordinal.nunique() < 2:
            continue
        ours[municipality] = float(spearmanr(group.score, group.ordinal).statistic)
        theirs[municipality] = float(spearmanr(group.rwi_deprivation, group.ordinal).statistic)
        cities_of_mun[municipality] = group.city.iloc[0]

    muns = sorted(ours)
    diff = {m: ours[m] - theirs[m] for m in muns}
    by_city: dict = {}
    for m in muns:
        by_city.setdefault(cities_of_mun[m], []).append(diff[m])
    groups = [np.array(v) for v in by_city.values()]
    rng = np.random.default_rng(0)
    means = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        chosen = rng.integers(0, len(groups), len(groups))
        means[k] = float(np.concatenate([groups[j] for j in chosen]).mean())
    low, high = np.percentile(means, [2.5, 97.5])

    result = {
        "agebs": len(both),
        "municipalities": len(muns),
        "ours_within": float(np.mean(list(ours.values()))),
        "rwi_within": float(np.mean(list(theirs.values()))),
        "difference": float(np.mean(list(diff.values()))),
        "ci_low": float(low),
        "ci_high": float(high),
        "wins": int(sum(d > 0 for d in diff.values())),
    }
    pd.DataFrame([result]).to_csv(
        "data/incumbente_pareado" + ("_test.csv" if split == "test" else ".csv"), index=False
    )
    print(
        f"ours {result['ours_within']:+.3f} · RWI {result['rwi_within']:+.3f} · "
        f"paired difference {result['difference']:+.3f} [{low:+.3f}, {high:+.3f}] · "
        f"wins {result['wins']}/{len(muns)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
