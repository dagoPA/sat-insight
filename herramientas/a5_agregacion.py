"""A5: makes the bag prediction follow the process that produced the bag label.

The GRS label of a municipality is a population-weighted aggregate of its AGEB, and every
label-proportion method in the literature averages instances uniformly. The mismatch was
measured at 0.029 of mean discrepancy on the held-out cities, real, and small against a
bag error of 0.110. This is the experiment that says whether modelling the aggregation
buys anything.

The model changes in one place: the bag prediction becomes a weighted mean of the instance
predictions, with each token carrying the population of its AGEB split evenly among the
tokens of that AGEB in the bag. Population at AGEB level is public census data and is not
the label being disaggregated, using it is the setting where instance weights are known,
which the label-proportions literature allows but never instantiates with real weights.

Evaluation is identical to every other run: the map on the 14 held-out cities, paired
against the uniform-mean model, three seeds.

Usage: a5_agregacion.py [epochs]
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

from satinsight.agebs import catalogue_with_extra, cities_extra, load_grs  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.llp import build, evaluate_map  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

sys.path.insert(0, "herramientas")
from curva_supervision import grades_of  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEEDS = (0, 1, 2)
RADIUS, PATIENCE = 1, 8
OUT = "data/a5_agregacion.csv"

log = logging.getLogger("a5")


def population_weights(bags):
    """One weight per instance: its AGEB's population split among that AGEB's tokens.

    Normalised to mean one inside the bag so the loss scale stays comparable with the
    uniform model. An AGEB missing from the census table gets the bag's median AGEB
    population, which keeps its tokens counted without letting them dominate.
    """
    table = load_grs()
    population = dict(zip(table.cvegeo, table.poblacion.astype("float64"), strict=False))
    out = []
    for bag in bags:
        counts: dict = {}
        for c in bag.cvegeo:
            counts[c] = counts.get(c, 0) + 1
        known = [population[c] for c in counts if c in population]
        fallback = float(np.median(known)) if known else 1.0
        w = np.array([population.get(c, fallback) / counts[c] for c in bag.cvegeo], dtype="float32")
        out.append(w * len(w) / w.sum())
    return out


def train_weighted(train_bags, weights, val_bags, val_links, grades, seed, torch, device):
    from torch import nn

    torch.manual_seed(seed)
    model = build(train_bags[0].instances.shape[1], radius=RADIUS, standardize=True).to(device)
    rng_stats = np.random.default_rng(seed)
    sample = np.vstack([b.instances[rng_stats.permutation(len(b))[:200]] for b in train_bags])
    model.fit_scaler(sample.mean(axis=0), sample.std(axis=0))
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()
    links = [
        tuple(torch.from_numpy(a).to(device) for a in adjacency(b.y0, b.x0, radius=RADIUS))
        for b in train_bags
    ]
    tensors = [torch.from_numpy(w).float().to(device).unsqueeze(1) for w in weights]
    rng = np.random.default_rng(seed)
    best, waited, best_state = np.inf, 0, None

    for _ in range(EPOCHS):
        model.train()
        for index in rng.permutation(len(train_bags)):
            bag, (src, dst), w = train_bags[index], links[index], tensors[index]
            optimiser.zero_grad()
            x = torch.from_numpy(bag.instances).float().to(device)
            _, per_instance = model(x, src, dst)
            shares = (per_instance * w).sum(dim=0) / w.sum()
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
    train_weighted.last_model = model
    final = evaluate_map(model, val_bags, val_links, grades, torch, device)
    per_bag = final.pop("per_bag")
    return final, per_bag


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    train_cities = sorted(cities_of(partition, "train")) + sorted(cities_extra())
    val_cities = sorted(cities_of(partition, "val"))

    pool = load_split(train_cities, "s2", fuse=True)
    val_bags = load_split(val_cities, "s2", fuse=True)
    grades = grades_of(val_cities, catalogue)
    weights = population_weights(pool)
    print(f"pool of {len(pool)} bags · weighted aggregation", flush=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    val_links = [
        tuple(torch.from_numpy(a).to(device) for a in adjacency(b.y0, b.x0, radius=RADIUS))
        for b in val_bags
    ]

    results, bags_rows = [], []
    for seed in SEEDS:
        scored, per_bag = train_weighted(
            pool, weights, val_bags, val_links, grades, seed, torch, device
        )
        results.append({"seed": seed, **scored})
        bags_rows.extend({"seed": seed, "municipality": m, "rho": r} for m, r in per_bag)
        log.info(
            "seed %d · map AUROC %.3f · within %+.3f · bag MAE %.4f",
            seed,
            scored["auroc_high"],
            scored["spearman_within"],
            scored["bag_mae"],
        )
        pd.DataFrame(results).to_csv(OUT, index=False)
        pd.DataFrame(bags_rows).to_csv("data/a5_agregacion_bags.csv", index=False)

    r = pd.DataFrame(results)
    print("\n===== WEIGHTED AGGREGATION =====", flush=True)
    print(
        f"map AUROC {r.auroc_high.mean():.3f} · within {r.spearman_within.mean():+.3f} · "
        f"bag MAE {r.bag_mae.mean():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
