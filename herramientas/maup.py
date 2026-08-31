"""B5: how the map's within-municipality agreement moves with the size of the AGEB.

The ecological caveat of every aggregate-label study needs a number, and this puts one on
it: per-AGEB agreement between the mean token score and the grade, sliced by how many
tokens the AGEB holds and by its population. Small AGEB are measured by fewer tokens and
carry noisier attention; if agreement collapses below some size, that size is the
resolution limit of the product, stated in units a statistics office understands.

Consumes the persisted validation scores; touches no training.

Usage: maup.py
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
from scipy.stats import spearmanr  # noqa: E402

from satinsight.agebs import cities_by_size, load_grs  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

BINS = ((1, 3), (4, 9), (10, 19), (20, 49), (50, 10_000))


def main() -> None:
    scores = pd.read_parquet("data/predicciones_val.parquet")
    per_ageb = (
        scores.groupby(["city", "municipality", "cvegeo"], observed=True)
        .agg(score=("score", "mean"), tokens=("score", "size"))
        .reset_index()
    )
    per_ageb["tokens"] //= scores.seed.nunique()

    partition = pd.read_csv("data/partition.csv")
    catalogue = cities_by_size(stratify=True)
    grades = {}
    for city in sorted(cities_of(partition, "val")):
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
            grades.update(dict(zip(agebs.cvegeo, agebs.ordinal.astype(int), strict=True)))
        except Exception:
            logging.warning("no grades for %s", city)
    per_ageb["ordinal"] = per_ageb.cvegeo.map(grades)
    census = load_grs()[["cvegeo", "poblacion"]]
    table = per_ageb.merge(census, on="cvegeo").dropna(subset=["ordinal"])

    rows = []
    for low, high in BINS:
        chosen = table[(table.tokens >= low) & (table.tokens <= high)]
        within = []
        for _, group in chosen.groupby("municipality", observed=True):
            if len(group) >= 10 and group.ordinal.nunique() > 1:
                within.append(float(spearmanr(group.score, group.ordinal).statistic))
        rows.append(
            {
                "tokens_low": low,
                "tokens_high": high,
                "agebs": len(chosen),
                "median_population": float(chosen.poblacion.median()),
                "municipalities": len(within),
                "spearman_within": float(np.mean(within)) if within else float("nan"),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv("data/maup.csv", index=False)
    print(result.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
