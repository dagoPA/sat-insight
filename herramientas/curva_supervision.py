"""First axis of the supervision-efficiency curve: how much map N municipal aggregates buy.

The programme's headline quantity is the exchange rate between aggregate statistics and
maps: the fraction of a fully supervised ceiling that weak supervision recovers, as a
function of how much aggregate supervision exists. This tool sweeps the first axis, the
number of bags, with the granularity of the label and the sensor resolution held fixed.

Training bags are sampled from the 110 national training cities plus the expansion beyond
the national set. Evaluation never moves: the map is scored on the 14 cities held for
validation, so every point of the curve is comparable and the test cities stay closed.

Bags are sampled without replacement, stratified by the municipal grade so a small draw
does not come out all comfortable by chance, and nested: the bags of a smaller N are
contained in every larger N of the same seed. Nesting removes sampling noise from the
shape of the curve, which is the object of interest.

Usage: curva_supervision.py [epochs] [radius] [sizes]

Radius sweeps the context sensitivity on the expanded pool. Sizes "full" runs only the
whole-pool point, which is what the radius sensitivity needs.
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
from satinsight.context import adjacency  # noqa: E402
from satinsight.llp import build, evaluate_map  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RADIUS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SIZES = (None,) if len(sys.argv) > 3 and sys.argv[3] == "full" else (50, 100, 200, 400, None)
SEEDS = (0, 1, 2)
PATIENCE = 8
OUT = f"data/curva_supervision_r{RADIUS}.csv" if RADIUS != 1 else "data/curva_supervision.csv"

log = logging.getLogger("curve")


def grades_of(cities, catalogue):
    out = {}
    for city in cities:
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
            out.update(dict(zip(agebs.cvegeo, agebs.ordinal.astype(int), strict=True)))
        except Exception:
            logging.warning("no grades for %s", city)
    return out


def links_of(bags, torch, device):
    out = []
    for bag in bags:
        src, dst = adjacency(bag.y0, bag.x0, radius=RADIUS)
        out.append((torch.from_numpy(src).to(device), torch.from_numpy(dst).to(device)))
    return out


def nested_sample(bags, seed):
    """One shuffled order per seed, stratified by grade; every N is a prefix of it.

    Interleaving the per-grade shuffles keeps each prefix close to the pool's grade
    balance, and a prefix structure is what makes the points of one seed nested.
    """
    rng = np.random.default_rng(seed)
    by_grade = {}
    for index, bag in enumerate(bags):
        by_grade.setdefault(bag.ordinal, []).append(index)
    for indices in by_grade.values():
        rng.shuffle(indices)
    order = []
    pools = sorted(by_grade.values(), key=len, reverse=True)
    longest = max(len(p) for p in pools)
    for step in range(longest):
        for pool in pools:
            if step < len(pool):
                order.append(pool[step])
    return order


def train_once(train_bags, val_bags, val_links, grades, seed, torch, device):
    from torch import nn

    torch.manual_seed(seed)
    model = build(train_bags[0].instances.shape[1], radius=RADIUS, standardize=True).to(device)
    rng_stats = np.random.default_rng(seed)
    sample = np.vstack([b.instances[rng_stats.permutation(len(b))[:200]] for b in train_bags])
    model.fit_scaler(sample.mean(axis=0), sample.std(axis=0))
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()
    train_links = links_of(train_bags, torch, device)
    rng = np.random.default_rng(seed)
    best, waited, best_state = np.inf, 0, None

    for _ in range(EPOCHS):
        model.train()
        for index in rng.permutation(len(train_bags)):
            bag, (src, dst) = train_bags[index], train_links[index]
            optimiser.zero_grad()
            x = torch.from_numpy(bag.instances).float().to(device)
            shares, _ = model(x, src, dst)
            loss = criterion(shares, torch.from_numpy(bag.shares).float().to(device))
            loss.backward()
            optimiser.step()
        scored = evaluate_map(model, val_bags, val_links, grades, torch, device)
        if scored["bag_mae"] < best:
            best, waited = scored["bag_mae"], 0
            best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    model.load_state_dict(best_state)
    final = evaluate_map(model, val_bags, val_links, grades, torch, device)
    final.pop("per_bag")
    return final


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    train_cities = sorted(cities_of(partition, "train")) + sorted(cities_extra())
    val_cities = sorted(cities_of(partition, "val"))
    assert not set(val_cities) & set(cities_extra()), "an expansion key collides with validation"
    assert not set(cities_of(partition, "test")) & set(cities_extra()), "collides with test"

    pool = load_split(train_cities, "s2", fuse=True)
    val_bags = load_split(val_cities, "s2", fuse=True)
    grades = grades_of(val_cities, catalogue)
    print(f"pool of {len(pool)} bags · sizes {SIZES} · seeds {SEEDS}", flush=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    val_links = links_of(val_bags, torch, device)

    results = []
    for seed in SEEDS:
        order = nested_sample(pool, seed)
        for size in SIZES:
            chosen = order if size is None else order[:size]
            bags = [pool[i] for i in chosen]
            scored = train_once(bags, val_bags, val_links, grades, seed, torch, device)
            results.append({"seed": seed, "bags": len(bags), "radius": RADIUS, **scored})
            log.info(
                "seed %d · %4d bags · map AUROC %.3f · within %+.3f · bag MAE %.4f",
                seed,
                len(bags),
                scored["auroc_high"],
                scored["spearman_within"],
                scored["bag_mae"],
            )
            pd.DataFrame(results).to_csv(OUT, index=False)

    r = pd.DataFrame(results).groupby("bags").mean(numeric_only=True)
    print("\n===== THE CURVE =====", flush=True)
    print(r[["auroc_high", "spearman_within", "bag_mae"]].round(4).to_string(), flush=True)
    print("ceiling with instance supervision and context: within +0.239", flush=True)


if __name__ == "__main__":
    main()
