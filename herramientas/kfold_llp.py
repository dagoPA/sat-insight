"""Grouped k-fold of the label-proportion model.

Usage: kfold_llp.py [folds] [epochs] [radius]

A radius above zero widens every instance with the mean of the tokens around it before
scoring. Radius 1 is the eight neighbours, 480 m of ground; radius 2 is 800 m.
"""

import json
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
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.llp import build, evaluate_map  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
RADIUS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SEED, PATIENCE = 0, 8
OUT = f"data/llp_kfold_r{RADIUS}.csv" if RADIUS else "data/llp_kfold.csv"


def grades_of(cities, catalogue):
    out = {}
    for city in cities:
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
            out.update(dict(zip(agebs.cvegeo, agebs.ordinal.astype(int), strict=True)))
        except Exception:
            logging.warning("no grades for %s", city)
    return out


def neighbours(bags, torch, device):
    """Adjacency of every bag, built once and kept on the device.

    The neighbourhood of a token does not change between epochs, and rebuilding it inside
    the loop would cost more than the forward pass it feeds.
    """
    if not RADIUS:
        return [(None, None)] * len(bags)
    out = []
    for bag in bags:
        src, dst = adjacency(bag.y0, bag.x0, radius=RADIUS)
        out.append(
            (
                torch.from_numpy(src).to(device),
                torch.from_numpy(dst).to(device),
            )
        )
    return out


def main() -> None:
    import torch
    from torch import nn

    partition = pd.read_csv("data/partition.csv")
    cities = sorted(cities_of(partition, "train"))
    catalogue = cities_by_size(stratify=True)
    print(f"{len(cities)} training cities · {FOLDS} folds · radius {RADIUS}", flush=True)

    rng = np.random.default_rng(SEED)
    shuffled = list(cities)
    rng.shuffle(shuffled)
    folds = [shuffled[i::FOLDS] for i in range(FOLDS)]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    results = []
    for k, held in enumerate(folds):
        rest = [c for c in cities if c not in held]
        train_bags = load_split(rest, "s2", fuse=True)
        val_bags = load_split(held, "s2", fuse=True)
        grades = grades_of(held, catalogue)
        train_links = neighbours(train_bags, torch, device)
        val_links = neighbours(val_bags, torch, device)

        torch.manual_seed(SEED + k)
        model = build(train_bags[0].instances.shape[1], radius=RADIUS).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()
        best, waited, best_state = np.inf, 0, None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total = 0.0
            for index in rng.permutation(len(train_bags)):
                bag = train_bags[index]
                src, dst = train_links[index]
                optimiser.zero_grad()
                x = torch.from_numpy(bag.instances).float().to(device)
                shares, _ = model(x, src, dst)
                loss = criterion(shares, torch.from_numpy(bag.shares).float().to(device))
                loss.backward()
                optimiser.step()
                total += float(loss)
            scored = evaluate_map(model, val_bags, val_links, grades, torch, device)
            log.info(
                "fold %d epoch %d · loss %.5f · bag MAE %.4f · map AUROC %.3f · rho %+.3f",
                k,
                epoch,
                total / len(train_bags),
                scored["bag_mae"],
                scored["auroc_high"],
                scored["spearman_within"],
            )
            if scored["bag_mae"] < best:
                best, waited = scored["bag_mae"], 0
                best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
            else:
                waited += 1
                if waited >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        results.append(
            {
                "fold": k,
                "bags": len(val_bags),
                "radius": RADIUS,
                **{
                    key: value
                    for key, value in evaluate_map(
                        model, val_bags, val_links, grades, torch, device
                    ).items()
                    if key != "per_bag"
                },
            }
        )
        print(json.dumps(results[-1], default=float), flush=True)
        pd.DataFrame(results).to_csv(OUT, index=False)

    r = pd.DataFrame(results)
    print("\n===== SUMMARY =====", flush=True)
    print(r.round(4).to_string(index=False), flush=True)
    print(
        f"\nmap AUROC {r.auroc_high.mean():.3f} ± {r.auroc_high.std():.3f} · "
        f"Spearman dentro {r.spearman_within.mean():+.3f} "
        f"± {r.spearman_within.std():.3f} · "
        f"agrupado {r.spearman_pooled.mean():+.3f} · bag MAE {r.bag_mae.mean():.4f}",
        flush=True,
    )
    print("attention MIL reached 0.463 · the instance-supervised ceiling is 0.826", flush=True)


log = logging.getLogger("llp")

if __name__ == "__main__":
    main()
