"""Upper bound on what the attention map could achieve with these instance vectors.

Trains the same projection on the same instances, but supervised directly with the grade
of the AGEB each one fell in. That is not the project: it is the ceiling. If a classifier
that is told the answer per instance cannot separate deprived from comfortable ground, the
160 m vectors do not carry the signal and no amount of ablating the MIL will find it.

The comparison that matters is against the attention map of the weakly supervised model,
which reached 0.463 of area under the curve. The gap between that and this ceiling is what
the weak supervision is costing.
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from satinsight.agebs import cities_by_size  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BATCH = 4096
SEED = 0


def grades_of(cities, catalogue):
    out = {}
    for city in cities:
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
            out.update(dict(zip(agebs.cvegeo, agebs.ordinal.astype(int), strict=True)))
        except Exception:
            logging.warning("no grades for %s", city)
    return out


def flatten(bags, grades):
    """Every instance as its own row, labelled by the AGEB it fell in."""
    x, y, city = [], [], []
    for bag in bags:
        g = np.array([grades.get(c, -1) for c in bag.cvegeo])
        keep = g >= 0
        x.append(bag.instances[keep])
        y.append(g[keep])
        city.extend([bag.city] * int(keep.sum()))
    return np.vstack(x), np.concatenate(y), np.array(city)


def main() -> None:
    import torch
    from torch import nn

    partition = pd.read_csv("data/partition.csv")
    train_cities = sorted(cities_of(partition, "train"))
    val_cities = sorted(cities_of(partition, "val"))
    catalogue = cities_by_size(stratify=True)

    train_bags = load_split(train_cities, "s2", fuse=True)
    val_bags = load_split(val_cities, "s2", fuse=True)
    x_tr, y_tr, _ = flatten(train_bags, grades_of(train_cities, catalogue))
    x_va, y_va, _ = flatten(val_bags, grades_of(val_cities, catalogue))
    print(f"train {x_tr.shape} · val {x_va.shape}", flush=True)
    print("grade distribution:", np.bincount(y_tr, minlength=5).tolist(), flush=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(x_tr.shape[1], 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 5)
    ).to(device)
    counts = np.bincount(y_tr, minlength=5).astype("float64")
    weights = torch.tensor(
        len(y_tr) / (5 * np.maximum(counts, 1)), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    xt = torch.from_numpy(x_tr).float()
    yt = torch.from_numpy(y_tr).long()
    xv = torch.from_numpy(x_va).float().to(device)
    rng = np.random.default_rng(SEED)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(xt))
        total = 0.0
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            optimiser.zero_grad()
            loss = criterion(model(xt[idx].to(device)), yt[idx].to(device))
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)

        model.eval()
        with torch.inference_mode():
            logits = model(xv)
            probability = torch.softmax(logits, dim=1).cpu().numpy()
        # la puntuación comparable con la atención es la esperanza del grado: un solo
        # número por instancia que ordena de menos a más rezagada
        score = probability @ np.arange(5)
        auroc = roc_auc_score((y_va >= 3).astype(int), score)
        rho = spearmanr(score, y_va).statistic
        print(
            f"epoch {epoch} · loss {total / len(order):.4f} · "
            f"AUROC(≥High) {auroc:.3f} · Spearman {rho:+.3f}",
            flush=True,
        )

    print("\n===== CEILING =====", flush=True)
    print(f"instance level, fully supervised: AUROC {auroc:.3f} · Spearman {rho:+.3f}", flush=True)
    print("weakly supervised attention map:  AUROC 0.463 · Spearman -0.107", flush=True)


if __name__ == "__main__":
    main()
