"""Targeting at national scale, on every municipality the heads never trained on.

The validation experiment asks, city by city, how many people in high-deprivation tracts
a budget reaches when the allocation follows the municipal aggregate, the map, or the
tract census. This repeats it over the national out-of-sample ground: the validation and
test cities and every municipality beyond the catalogue, scored by the saved heads.
Two views. Per municipality, the gap closed by the map as in the paper, now over
thousands of municipalities. Nationally, one budget for the whole country and four
allocators: the aggregate alone (municipalities in order of their grade, tracts inside
them in random order), the aggregate refined by the map inside each municipality (the
way a statistics office would use it), the map alone, and the tract census as the upper
bound; and the same budget dealt to every municipality in proportion to its population
and spent inside it, where the within-municipality ordering is the whole decision. People
reached are those living in tracts of high or very high deprivation.

Usage: national_targeting.py [splits]   (default: val,test,beyond)
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

from satinsight import backbone  # noqa: E402
from satinsight.agebs import load_grs  # noqa: E402

sys.path.insert(0, "scripts")
from targeting import BUDGETS, reached  # noqa: E402

SPLITS = tuple((sys.argv[1] if len(sys.argv) > 1 else "val,test,beyond").split(","))
DRAWS = 200


def tract_table() -> pd.DataFrame:
    scores = pd.read_parquet(backbone.suffixed("data/national_scores.parquet"))
    scores = scores[scores.split.isin(SPLITS)]
    per_ageb = (
        scores.groupby(["key", "split", "municipality", "cvegeo"], observed=True)
        .score.mean()
        .reset_index()
    )
    census = load_grs()[["cvegeo", "population", "ordinal"]]
    table = per_ageb.merge(census, on="cvegeo", how="inner")
    table["aggregate_score"] = table.groupby("municipality", observed=True).ordinal.transform(
        "mean"
    )
    return table


def by_municipality(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for (key, municipality), group in table.groupby(["key", "municipality"], observed=True):
        if group.cvegeo.nunique() < 5 or group.ordinal.nunique() < 2:
            continue
        for budget in BUDGETS:
            oracle = reached(group.sort_values("ordinal", ascending=False), budget)
            map_reached = reached(group.sort_values("score", ascending=False), budget)
            draws = [
                reached(group.sample(frac=1, random_state=rng.integers(1 << 31)), budget)
                for _ in range(DRAWS)
            ]
            aggregate = float(np.mean(draws))
            gap = oracle - aggregate
            rows.append(
                {
                    "key": key,
                    "municipality": municipality,
                    "split": group.split.iloc[0],
                    "budget": budget,
                    "tracts": group.cvegeo.nunique(),
                    "aggregate": aggregate,
                    "map": map_reached,
                    "oracle": oracle,
                    "gap_closed": (map_reached - aggregate) / gap if gap > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def national(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for budget in BUDGETS:
        draws = []
        for _ in range(DRAWS):
            shuffled = table.sample(frac=1, random_state=rng.integers(1 << 31))
            draws.append(
                reached(
                    shuffled.sort_values("aggregate_score", ascending=False, kind="stable"), budget
                )
            )
        rows.append(
            {
                "budget": budget,
                "municipalities": table.municipality.nunique(),
                "tracts": len(table),
                "population": float(table.population.sum()),
                "aggregate": float(np.mean(draws)),
                "aggregate_then_map": reached(
                    table.sort_values(["aggregate_score", "score"], ascending=False), budget
                ),
                "map": reached(table.sort_values("score", ascending=False), budget),
                "census": reached(table.sort_values("ordinal", ascending=False), budget),
            }
        )
    out = pd.DataFrame(rows)
    gap = out["census"] - out["aggregate"]
    out["gap_closed_aggregate_then_map"] = (out["aggregate_then_map"] - out["aggregate"]) / gap
    out["gap_closed_map"] = (out["map"] - out["aggregate"]) / gap
    return out


def proportional(per: pd.DataFrame) -> pd.DataFrame:
    """One budget per municipality, in proportion to its population, spent inside it."""
    out = per.groupby("budget")[["aggregate", "map", "oracle"]].sum()
    out.columns = ["proportional_aggregate", "proportional_map", "proportional_census"]
    gap = out.proportional_census - out.proportional_aggregate
    out["gap_closed_proportional"] = (out.proportional_map - out.proportional_aggregate) / gap
    return out.reset_index()


def main() -> None:
    table = tract_table()
    logging.info(
        "%d tracts of %d municipalities in %s", len(table), table.municipality.nunique(), SPLITS
    )
    per = by_municipality(table)
    per.to_csv(backbone.suffixed("data/national_targeting_by_municipality.csv"), index=False)
    summary = national(table).merge(proportional(per), on="budget")
    summary.to_csv(backbone.suffixed("data/national_targeting.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False), flush=True)
    print(
        "gap closed per municipality, median by budget:\n"
        + per.groupby("budget").gap_closed.median().round(3).to_string(),
        flush=True,
    )
    losses = per[per["map"] < per["aggregate"]]
    print(f"municipality-budget cells where the map loses: {len(losses)}/{len(per)}", flush=True)


if __name__ == "__main__":
    main()
