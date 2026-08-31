"""Paired test-set comparison of the headline against the no-representation baseline.

On validation the land-cover head trailed the headline clearly (0.149 vs 0.227). On the
test cities its seed-mean within-municipality correlation came out higher (0.219 vs
0.195), and whether that is sample noise or a real reversal decides a sentence the
manuscript cannot fudge. The comparison that answers it is per-municipality, paired, on
identical bags, with the city-clustered bootstrap of the difference.

Selection stays anchored to validation: both heads train exactly as they always did, and
the test bags are only scored. This is analysis of the single opening, not a new one.

Usage: wc_pareado_test.py
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

from satinsight.agebs import catalogue_with_extra, cities_extra  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.llp import evaluate_map  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

sys.path.insert(0, "herramientas")
from curva_supervision import grades_of, links_of, train_once  # noqa: E402

SEEDS = (0, 1, 2)
RESAMPLES = 2000


def per_bag_of(arm, train_cities, val_pack, test_pack, torch, device):
    sensor, fuse = arm
    pool = load_split(train_cities, sensor, fuse=fuse)
    val_bags, val_links, val_grades = val_pack[(sensor, fuse)]
    test_bags, test_links, test_grades = test_pack[(sensor, fuse)]
    rhos: dict = {}
    for seed in SEEDS:
        train_once(pool, val_bags, val_links, val_grades, seed, torch, device)
        scored = evaluate_map(
            train_once.last_model, test_bags, test_links, test_grades, torch, device
        )
        for municipality, rho in scored["per_bag"]:
            rhos.setdefault(municipality, []).append(rho)
    return {m: float(np.mean(v)) for m, v in rhos.items()}


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    train_cities = sorted(cities_of(partition, "train")) + sorted(cities_extra())
    val_cities = sorted(cities_of(partition, "val"))
    test_cities = sorted(cities_of(partition, "test"))
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    val_grades = grades_of(val_cities, catalogue)
    test_grades = grades_of(test_cities, catalogue)

    arms = {("s2", True): None, ("wc", False): None}
    val_pack, test_pack = {}, {}
    city_of = {}
    for key in arms:
        sensor, fuse = key
        vb = load_split(val_cities, sensor, fuse=fuse)
        tb = load_split(test_cities, sensor, fuse=fuse)
        val_pack[key] = (vb, links_of(vb, torch, device), val_grades)
        test_pack[key] = (tb, links_of(tb, torch, device), test_grades)
        for b in tb:
            city_of[b.municipality] = b.city

    ours = per_bag_of(("s2", True), train_cities, val_pack, test_pack, torch, device)
    wc = per_bag_of(("wc", False), train_cities, val_pack, test_pack, torch, device)

    shared = sorted(set(ours) & set(wc))
    diff = {m: ours[m] - wc[m] for m in shared}
    by_city: dict = {}
    for m in shared:
        by_city.setdefault(city_of[m], []).append(diff[m])
    groups = [np.array(v) for v in by_city.values()]
    rng = np.random.default_rng(0)
    means = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        chosen = rng.integers(0, len(groups), len(groups))
        means[k] = float(np.concatenate([groups[j] for j in chosen]).mean())
    low, high = np.percentile(means, [2.5, 97.5])

    result = {
        "municipalities": len(shared),
        "ours_within": float(np.mean([ours[m] for m in shared])),
        "wc_within": float(np.mean([wc[m] for m in shared])),
        "difference": float(np.mean(list(diff.values()))),
        "ci_low": float(low),
        "ci_high": float(high),
        "wins": int(sum(d > 0 for d in diff.values())),
    }
    pd.DataFrame([result]).to_csv("data/wc_pareado_test.csv", index=False)
    print(
        f"ours {result['ours_within']:+.3f} · worldcover {result['wc_within']:+.3f} · "
        f"paired {result['difference']:+.3f} [{low:+.3f}, {high:+.3f}] · "
        f"wins {result['wins']}/{len(shared)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
