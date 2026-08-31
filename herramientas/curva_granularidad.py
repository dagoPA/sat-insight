"""Second axis of the curve: what a statistics office loses by publishing coarser.

The label of every training bag is replaced by the aggregate its municipality would get if
the office only published at state level, or only one national figure. The bags keep their
instances and their geometry; what changes is how much the number attached to each one
distinguishes it from its neighbours. At national granularity every bag carries the same
label and the only gradient left is the shared mean.

Aggregates are computed from the full census table, population-weighted, because that is
what the office would publish — the pool the model happens to train on does not shrink the
state it sits in. Evaluation never moves: the map is scored on the 14 held-out cities
against AGEB grades, identically across granularities.

Usage: curva_granularidad.py [epochs]
"""

import dataclasses
import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from satinsight.agebs import catalogue_with_extra, cities_extra, load_grs  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

sys.path.insert(0, "herramientas")
from curva_supervision import grades_of, links_of, train_once  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEEDS = (0, 1, 2)
LEVELS = ("municipality", "state", "national")
OUT = "data/curva_granularidad.csv"

log = logging.getLogger("granularity")


def shares_by(table: pd.DataFrame, keys: list[str] | None) -> dict | np.ndarray:
    """Population-weighted share of people in AGEB at grade k or above, for k in 1..4."""

    def of(group):
        weight = group.poblacion.to_numpy(dtype="float64")
        grade = group.ordinal.to_numpy()
        return np.array(
            [(weight * (grade >= k)).sum() / weight.sum() for k in range(1, 5)], dtype="float32"
        )

    if keys is None:
        return of(table)
    # pandas hands back tuple keys when grouping by a list, even a list of one column
    return {
        (name[0] if isinstance(name, tuple) else name): of(group)
        for name, group in table.groupby(keys, observed=True)
    }


def relabel(bags, level, table):
    if level == "municipality":
        return bags
    if level == "national":
        national = shares_by(table, None)
        return [dataclasses.replace(b, shares=national) for b in bags]
    state_of = dict(zip(table.cve_mun, table.cve_ent, strict=False))
    by_state = shares_by(table, ["cve_ent"])
    out = []
    for b in bags:
        state = state_of.get(b.municipality)
        if state is None:
            raise KeyError(f"no state for municipality {b.municipality}")
        out.append(dataclasses.replace(b, shares=by_state[state]))
    return out


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    train_cities = sorted(cities_of(partition, "train")) + sorted(cities_extra())
    val_cities = sorted(cities_of(partition, "val"))
    table = load_grs()

    pool = load_split(train_cities, "s2", fuse=True)
    val_bags = load_split(val_cities, "s2", fuse=True)
    grades = grades_of(val_cities, catalogue)
    print(f"pool of {len(pool)} bags · levels {LEVELS}", flush=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    val_links = links_of(val_bags, torch, device)

    results = []
    for level in LEVELS:
        bags = relabel(pool, level, table)
        spread = float(np.std([b.shares[2] for b in bags]))
        for seed in SEEDS:
            scored = train_once(bags, val_bags, val_links, grades, seed, torch, device)
            results.append({"level": level, "seed": seed, "label_spread": spread, **scored})
            log.info(
                "%-12s seed %d · map AUROC %.3f · within %+.3f · bag MAE %.4f",
                level,
                seed,
                scored["auroc_high"],
                scored["spearman_within"],
                scored["bag_mae"],
            )
            pd.DataFrame(results).to_csv(OUT, index=False)

    r = pd.DataFrame(results).groupby("level").mean(numeric_only=True)
    print("\n===== GRANULARITY =====", flush=True)
    print(r[["auroc_high", "spearman_within", "label_spread"]].round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
