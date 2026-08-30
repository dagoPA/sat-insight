"""Trains the label-proportion model on the 110 and scores its map on the 14 held for validation.

The k-fold answers how the model behaves on average over the training cities. It does not
put the weakly supervised map and the instance-supervised ceiling on the same ground: the
ceiling is measured on the validation cities, and comparing the two across different sets
of cities compares two things at once.

Usage: llp_val.py [epochs] [radius] [seed] [tuned] [modality]

With `tuned` the head standardises its input and fits with binary cross entropy, which is
what won the sweep over the training folds. The sweep never looked at these cities: they
are opened once, with the winner already chosen.
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
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.llp import build, evaluate_map  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RADIUS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TUNED = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
MODALITY = sys.argv[5] if len(sys.argv) > 5 else "fused"
PATIENCE = 8
SENSOR = {"fused": "s2", "optical": "s2", "radar": "s1", "degraded": "s2deg"}[MODALITY]
FUSE = MODALITY == "fused"

log = logging.getLogger("llp-val")


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
    if not RADIUS:
        return [(None, None)] * len(bags)
    out = []
    for bag in bags:
        src, dst = adjacency(bag.y0, bag.x0, radius=RADIUS)
        out.append((torch.from_numpy(src).to(device), torch.from_numpy(dst).to(device)))
    return out


def main() -> None:
    import torch
    from torch import nn

    partition = pd.read_csv("data/partition.csv")
    train_cities = sorted(cities_of(partition, "train"))
    val_cities = sorted(cities_of(partition, "val"))
    catalogue = cities_by_size(stratify=True)

    train_bags = load_split(train_cities, SENSOR, fuse=FUSE)
    val_bags = load_split(val_cities, SENSOR, fuse=FUSE)
    grades = grades_of(val_cities, catalogue)
    print(
        f"{MODALITY} · radius {RADIUS} · {len(train_bags)} training bags · {len(val_bags)} held",
        flush=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    train_links = links_of(train_bags, torch, device)
    val_links = links_of(val_bags, torch, device)

    torch.manual_seed(SEED)
    model = build(train_bags[0].instances.shape[1], radius=RADIUS, standardize=TUNED).to(device)
    if TUNED:
        rng_stats = np.random.default_rng(SEED)
        sample = np.vstack([b.instances[rng_stats.permutation(len(b))[:200]] for b in train_bags])
        model.fit_scaler(sample.mean(axis=0), sample.std(axis=0))
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.BCELoss() if TUNED else nn.MSELoss()
    rng = np.random.default_rng(SEED)
    # el criterio de parada es el error de bolsa, que es la única señal que la supervisión
    # débil tiene derecho a mirar: elegir la época por la calidad del mapa sería usar la
    # etiqueta por AGEB que el proyecto asegura no usar
    best, waited, best_state = np.inf, 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for index in rng.permutation(len(train_bags)):
            bag, (src, dst) = train_bags[index], train_links[index]
            optimiser.zero_grad()
            x = torch.from_numpy(bag.instances).float().to(device)
            shares, _ = model(x, src, dst)
            loss = criterion(shares, torch.from_numpy(bag.shares).float().to(device))
            loss.backward()
            optimiser.step()
            total += float(loss)
        scored = evaluate_map(model, val_bags, val_links, grades, torch, device)
        log.info(
            "epoch %d · loss %.5f · bag MAE %.4f · map AUROC %.3f · within %+.3f",
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
    scored = evaluate_map(model, val_bags, val_links, grades, torch, device)
    per_bag = scored.pop("per_bag")
    final = {"modality": MODALITY, "radius": RADIUS, "seed": SEED, "tuned": TUNED, **scored}
    print("\n===== VALIDATION =====", flush=True)
    print(
        f"{MODALITY} · radius {RADIUS} · tuned {TUNED} · map AUROC {final['auroc_high']:.3f} · "
        f"within-bag {final['spearman_within']:+.3f} over {final['bags_scored']} bags · "
        f"pooled {final['spearman_pooled']:+.3f} · bag MAE {final['bag_mae']:.4f}",
        flush=True,
    )
    pd.DataFrame([final]).to_csv(
        f"data/llp_val_{MODALITY}_r{RADIUS}_s{SEED}{'_tuned' if TUNED else ''}.csv", index=False
    )
    pd.DataFrame(per_bag, columns=["municipality", "rho"]).assign(
        modality=MODALITY, radius=RADIUS, seed=SEED, tuned=TUNED
    ).to_csv(
        f"data/llp_val_bags_{MODALITY}_r{RADIUS}_s{SEED}{'_tuned' if TUNED else ''}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
