"""Translates the map into targeting efficiency, the currency a general reader knows.

A correlation says the map orders ground; it does not say what that buys. The experiment a
policy reader understands: with a budget that reaches the q% most deprived people of a
city, how many people living in high-deprivation AGEB does each allocation actually reach?

Three allocators, same budget: the municipal aggregate alone (every AGEB of a municipality
ties, ordered by the municipal grade), the weakly supervised map, and the AGEB census
truth, which is the ceiling of geographic targeting. The map's value is the fraction of
the aggregate-to-census gap it closes. Population weights come from the census table.

Consumes the persisted validation scores; touches no training.

Usage: focalizacion.py
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

from satinsight.agebs import load_grs  # noqa: E402

BUDGETS = (0.05, 0.10, 0.20, 0.30)
HIGH = 3  # ordinal del rezago alto


def reached(order: pd.DataFrame, budget: float) -> float:
    """People in high-deprivation AGEB reached by funding the top of `order`.

    The budget is a fraction of the city's population; AGEB are funded whole, in order,
    until it runs out. Ties inside an allocator (every AGEB of a municipality under the
    aggregate) are broken at random, and the caller averages over draws.
    """
    cap = order.poblacion.sum() * budget
    cum = order.poblacion.cumsum()
    chosen = cum <= cap
    return float(order.loc[chosen & (order.ordinal >= HIGH), "poblacion"].sum())


def main() -> None:
    scores = pd.read_parquet("data/predicciones_val.parquet")
    per_ageb = (
        scores.groupby(["city", "municipality", "cvegeo"], observed=True).score.mean().reset_index()
    )
    census = load_grs()[["cvegeo", "poblacion", "ordinal"]]
    table = per_ageb.merge(census, on="cvegeo", how="inner")
    bag_grade = table.groupby("municipality", observed=True).ordinal.transform(lambda g: g.mean())
    table["aggregate_score"] = bag_grade

    rng = np.random.default_rng(0)
    rows = []
    for city, group in table.groupby("city", observed=True):
        deprived_population = float(group.loc[group.ordinal >= HIGH, "poblacion"].sum())
        if deprived_population == 0:
            continue
        for budget in BUDGETS:
            oracle = reached(group.sort_values("ordinal", ascending=False), budget)
            map_reached = reached(group.sort_values("score", ascending=False), budget)
            draws = []
            for _ in range(200):
                shuffled = group.sample(frac=1, random_state=rng.integers(1 << 31))
                draws.append(
                    reached(
                        shuffled.sort_values("aggregate_score", ascending=False, kind="stable"),
                        budget,
                    )
                )
            aggregate = float(np.mean(draws))
            gap = oracle - aggregate
            rows.append(
                {
                    "city": city,
                    "budget": budget,
                    "aggregate": aggregate,
                    "map": map_reached,
                    "oracle": oracle,
                    "gap_closed": (map_reached - aggregate) / gap if gap > 0 else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv("data/focalizacion.csv", index=False)
    print("===== TARGETING =====", flush=True)
    summary = result.groupby("budget")[["gap_closed"]].agg(["mean", "median", "count"])
    print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
