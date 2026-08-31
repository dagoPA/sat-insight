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

    The budget is a fraction of the city's population. The marginal AGEB is funded
    fractionally — the whole-AGEB greedy stopped at the first overflow, which let a
    lucky coarse allocator beat the census ceiling and broke the gap's denominator.
    With fractional funding the census ordering is a true upper bound by construction.
    Ties inside an allocator are broken at random by the caller, which averages draws.
    """
    cap = order.poblacion.sum() * budget
    population = order.poblacion.to_numpy(dtype="float64")
    deprived = (order.ordinal.to_numpy() >= HIGH).astype(float)
    cum = population.cumsum()
    full = cum <= cap
    total = float((population * deprived)[full].sum())
    edge = int(full.sum())
    if edge < len(order) and cap > (cum[edge - 1] if edge else 0.0):
        remainder = cap - (cum[edge - 1] if edge else 0.0)
        total += float(remainder * deprived[edge])
    return total


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/predicciones_val.parquet"
    split = "test" if "test" in source else "val"
    scores = pd.read_parquet(source)
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
    result.to_csv("data/focalizacion" + ("_test.csv" if split == "test" else ".csv"), index=False)
    print("===== TARGETING =====", flush=True)
    # per-city ratios explode when a city's gap is near zero; the honest figure pools
    # people first — reached people are additive, ratios of averages are not
    pooled = result.groupby("budget")[["aggregate", "map", "oracle"]].sum()
    pooled["gap_closed"] = (
        (pooled["map"] - pooled.aggregate_)
        if False
        else ((pooled["map"] - pooled["aggregate"]) / (pooled["oracle"] - pooled["aggregate"]))
    )
    print(pooled.round(3).to_string(), flush=True)
    print("\nper-city median of the ratio:", flush=True)
    print(result.groupby("budget").gap_closed.median().round(3).to_string(), flush=True)
    # la heterogeneidad es parte del resultado: el promedio no puede esconder a las
    # ciudades donde el mapa pierde contra el agregado
    losses = result[result["map"] < result["aggregate"]]
    print(
        f"\ncity-budget cells where the map loses to the aggregate: {len(losses)}/{len(result)}",
        flush=True,
    )
    for row in losses.itertuples():
        print(f"  {row.city} @ {row.budget:.0%}", flush=True)
    multi = pd.read_parquet("data/predicciones_val.parquet").groupby("city").municipality.nunique()
    several = set(multi[multi > 1].index)
    sub = result[result.city.isin(several)]
    pooled_multi = sub.groupby("budget")[["aggregate", "map", "oracle"]].sum()
    closed = (pooled_multi["map"] - pooled_multi["aggregate"]) / (
        pooled_multi["oracle"] - pooled_multi["aggregate"]
    )
    print(
        f"\npooled over the {len(several)} multi-municipality cities only "
        f"(where the aggregate is genuinely informative):",
        flush=True,
    )
    print(closed.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
