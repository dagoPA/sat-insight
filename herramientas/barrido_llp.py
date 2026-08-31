"""Sweeps the hyperparameters the label-proportion head never had swept.

Every number in that head was the first one written down. The sweeps in the history (entropy
weight, clustering constraint, cumulative label) were run on the attention MIL,
the family that was abandoned. When the composition was inverted the new model inherited
untouched defaults and got a single architectural ablation, the context radius.

That matters for what comes next. The argument for collecting more bags is that the model
falls short of the instance-supervised ceiling, and part of that gap could be a learning
rate nobody compared against anything.

Two decisions about the protocol.

The sweep runs on grouped folds of the 110 training cities. The 14 held for validation are
opened once, with the winner, and not twenty times to choose it: a gain measured on
thirty-odd bags is fragile enough that picking a winner on them would be picking noise.

Configurations are ranked by bag error, which is the only signal weak supervision has a
right to look at. Ranking them by the quality of the map would use the AGEB labels the
project claims to hold out. The map metrics are recorded anyway, for one specific purpose:
to check whether the criterion that has to be used tracks the one that is cared about.

Folds are the outer loop and configurations the inner one, so each fold's bags are read
from disk once and reused across the grid.

Usage: barrido_llp.py [folds] [epochs] [round]

Round 1 varies one ingredient at a time. Round 2 combines the ones that survived it.
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
from satinsight.llp import build, evaluate_map  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ROUND = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SEED, PATIENCE = 0, 8
OUT = f"data/barrido_llp_r{ROUND}.csv"

BASE = {"lr": 2e-4, "decay": 1e-4, "hidden": 256, "dropout": 0.25, "loss": "mse", "norm": False}

GRID = [
    ("base", {}),
    ("lr 5e-5", {"lr": 5e-5}),
    ("lr 1e-4", {"lr": 1e-4}),
    ("lr 5e-4", {"lr": 5e-4}),
    ("lr 1e-3", {"lr": 1e-3}),
    ("bce", {"loss": "bce"}),
    ("hidden 64", {"hidden": 64}),
    ("hidden 128", {"hidden": 128}),
    ("hidden 512", {"hidden": 512}),
    ("dropout 0.0", {"dropout": 0.0}),
    ("dropout 0.5", {"dropout": 0.5}),
    ("standardize", {"norm": True}),
    ("decay 1e-3", {"decay": 1e-3}),
    ("decay 1e-5", {"decay": 1e-5}),
]

# standardising the input won round 1 on both criteria at once; these ask whether the
# ingredients that came next add anything on top of it, and whether centring the features
# makes the larger steps usable that were marginal without it
COMBINED = [
    ("standardize", {"norm": True}),
    ("std + bce", {"norm": True, "loss": "bce"}),
    ("std + lr 5e-4", {"norm": True, "lr": 5e-4}),
    ("std + bce + lr 5e-4", {"norm": True, "loss": "bce", "lr": 5e-4}),
    ("std + lr 1e-3", {"norm": True, "lr": 1e-3}),
]

log = logging.getLogger("sweep")


def grades_of(cities, catalogue):
    out = {}
    for city in cities:
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
            out.update(dict(zip(agebs.cvegeo, agebs.ordinal.astype(int), strict=True)))
        except Exception:
            logging.warning("no grades for %s", city)
    return out


def statistics(bags):
    """Per-feature centre and spread over a sample of the training instances."""
    rng = np.random.default_rng(SEED)
    sample = np.vstack([b.instances[rng.permutation(len(b))[:200]] for b in bags])
    return sample.mean(axis=0), sample.std(axis=0)


def run(config, train_bags, val_bags, grades, centre, spread, torch, device, fold):
    from torch import nn

    setting = {**BASE, **config}
    torch.manual_seed(SEED + fold)
    model = build(
        train_bags[0].instances.shape[1],
        hidden=setting["hidden"],
        dropout=setting["dropout"],
        standardize=setting["norm"],
    ).to(device)
    if setting["norm"]:
        model.fit_scaler(centre, spread)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setting["lr"], weight_decay=setting["decay"]
    )
    criterion = nn.BCELoss() if setting["loss"] == "bce" else nn.MSELoss()
    links = [(None, None)] * max(len(train_bags), len(val_bags))
    rng = np.random.default_rng(SEED + fold)
    best, waited, best_state = np.inf, 0, None

    for _ in range(EPOCHS):
        model.train()
        for index in rng.permutation(len(train_bags)):
            bag = train_bags[index]
            optimiser.zero_grad()
            shares, _ = model(torch.from_numpy(bag.instances).float().to(device))
            loss = criterion(shares, torch.from_numpy(bag.shares).float().to(device))
            loss.backward()
            optimiser.step()
        scored = evaluate_map(model, val_bags, links[: len(val_bags)], grades, torch, device)
        if scored["bag_mae"] < best:
            best, waited = scored["bag_mae"], 0
            best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    model.load_state_dict(best_state)
    final = evaluate_map(model, val_bags, links[: len(val_bags)], grades, torch, device)
    final.pop("per_bag")
    return final


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    cities = sorted(cities_of(partition, "train"))
    catalogue = cities_by_size(stratify=True)
    grid = GRID if ROUND == 1 else COMBINED
    print(
        f"{len(cities)} training cities · {FOLDS} folds · {len(grid)} configurations",
        flush=True,
    )

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
        centre, spread = statistics(train_bags)

        for name, config in grid:
            scored = run(config, train_bags, val_bags, grades, centre, spread, torch, device, k)
            results.append({"fold": k, "config": name, **scored})
            log.info(
                "fold %d · %-12s · bag MAE %.4f · map AUROC %.3f · within %+.3f",
                k,
                name,
                scored["bag_mae"],
                scored["auroc_high"],
                scored["spearman_within"],
            )
            pd.DataFrame(results).to_csv(OUT, index=False)

    r = pd.DataFrame(results).groupby("config").mean(numeric_only=True)
    r = r.sort_values("bag_mae")
    print("\n===== SUMMARY, ranked by the criterion weak supervision may use =====", flush=True)
    print(r[["bag_mae", "auroc_high", "spearman_within"]].round(4).to_string(), flush=True)
    tracks = r.bag_mae.corr(r.spearman_within, method="spearman")
    print(f"\nrank correlation between bag error and map quality: {tracks:+.3f}", flush=True)
    print(json.dumps({"best_by_bag_mae": r.index[0]}), flush=True)


if __name__ == "__main__":
    main()
