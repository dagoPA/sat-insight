"""E4: does the map jump exactly at municipal borders inside a metropolis?

The shortcut this test hunts: a model that memorises the bag label paints every token of a
municipality with its grade, and its map then jumps precisely where the administrative
border runs — ground that looks the same on both sides gets different scores because the
paperwork differs. A model that reads territory crosses the border smoothly.

The design compares adjacent token pairs, 160 m apart on the grid, inside validation
cities that span several municipalities. The comparison is conditioned on truth: only
pairs whose two AGEB carry the SAME grade enter, so any score jump in excess of the
within-municipality jump cannot be justified by deprivation actually changing. The
city-clustered bootstrap puts the interval on the excess.

A placebo runs beside it: pairs that cross an AGEB border of equal grade inside one
municipality. AGEB borders never reached the training loss, so the map should treat them
like open ground; if it jumps there too, the excess at municipal borders means texture
seams, and the test is measuring the composite, which would be its own finding.

Consumes the persisted validation scores; touches no training.

Usage: discontinuidad.py
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

from satinsight.agebs import cities_by_size  # noqa: E402
from satinsight.context import STRIDE  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

RESAMPLES = 2000


def pairs_of(city: pd.DataFrame) -> pd.DataFrame:
    """Adjacent token pairs (rook neighbours on the grid) with both sides' context."""
    where = {
        (r, c): i
        for i, (r, c) in enumerate(
            zip(city.y0.to_numpy() // STRIDE, city.x0.to_numpy() // STRIDE, strict=True)
        )
    }
    rows = []
    for (r, c), i in where.items():
        for dr, dc in ((0, 1), (1, 0)):
            j = where.get((r + dr, c + dc))
            if j is not None:
                rows.append((i, j))
    left = city.iloc[[a for a, _ in rows]].reset_index(drop=True)
    right = city.iloc[[b for _, b in rows]].reset_index(drop=True)
    return pd.DataFrame(
        {
            "city": left.city,
            "jump": (left.score - right.score).abs(),
            "same_municipality": (left.municipality == right.municipality).to_numpy(),
            "same_ageb": (left.cvegeo == right.cvegeo).to_numpy(),
            "same_grade": (left.ordinal == right.ordinal).to_numpy(),
        }
    )


def excess(pairs: pd.DataFrame, cross: str) -> tuple[float, float, float]:
    """Mean jump of the crossing pairs against the smooth reference, city-bootstrapped."""
    reference = pairs[pairs.same_ageb]
    crossing = {
        "municipal": pairs[~pairs.same_municipality & pairs.same_grade],
        "ageb": pairs[~pairs.same_ageb & pairs.same_municipality & pairs.same_grade],
    }[cross]
    cities = sorted(set(crossing.city))
    rng = np.random.default_rng(0)
    deltas = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        chosen = rng.choice(cities, size=len(cities))
        a = np.mean([crossing[crossing.city == c].jump.mean() for c in chosen])
        b = np.mean([reference[reference.city == c].jump.mean() for c in chosen])
        deltas[k] = a - b
    point = crossing.jump.mean() - reference.jump.mean()
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(point), float(low), float(high)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/predicciones_val.parquet"
    scores = pd.read_parquet(source)
    if "score_city" in scores.columns:
        scores = scores.rename(columns={"score_city": "score"})
    per_token = (
        scores.groupby(["city", "municipality", "cvegeo", "y0", "x0"], observed=True)
        .score.mean()
        .reset_index()
    )
    print(f"scores from {source}", flush=True)
    partition = pd.read_csv("data/partition.csv")
    catalogue = cities_by_size(stratify=True)
    grades = {}
    for city in sorted(cities_of(partition, "val")):
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
            grades.update(dict(zip(agebs.cvegeo, agebs.ordinal.astype(int), strict=True)))
        except Exception:
            logging.warning("no grades for %s", city)
    per_token["ordinal"] = per_token.cvegeo.map(grades)
    per_token = per_token.dropna(subset=["ordinal"])

    pairs = pd.concat(
        [pairs_of(g.reset_index(drop=True)) for _, g in per_token.groupby("city", observed=True)],
        ignore_index=True,
    )
    multi = pairs[~pairs.same_municipality]
    print(
        f"{len(pairs)} adjacent pairs · {len(multi)} cross a municipal border · "
        f"{int((~pairs.same_ageb).sum())} cross an AGEB border",
        flush=True,
    )

    rows = []
    for cross, label in (
        ("municipal", "municipal border, same grade"),
        ("ageb", "AGEB border, same grade, same municipality"),
    ):
        point, low, high = excess(pairs, cross)
        rows.append({"crossing": cross, "excess": point, "ci_low": low, "ci_high": high})
        print(f"excess jump at {label}: {point:+.4f} [{low:+.4f}, {high:+.4f}]", flush=True)
    reference = pairs[pairs.same_ageb].jump.mean()
    print(f"reference jump inside one AGEB: {reference:.4f}", flush=True)
    out = "data/discontinuidad_ciudad.csv" if "ciudad" in source else "data/discontinuidad.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
