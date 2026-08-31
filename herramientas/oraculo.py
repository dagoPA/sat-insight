"""Upper bound on what the attention map could achieve with these instance vectors.

Trains the same projection on the same instances, but supervised directly with the grade
of the AGEB each one fell in. That is not the project: it is the ceiling. If a classifier
that is told the answer per instance cannot separate deprived from comfortable ground, the
160 m vectors do not carry the signal and no amount of ablating the MIL will find it.

The comparison that matters is against the attention map of the weakly supervised model,
which reached 0.463 of area under the curve. The gap between that and this ceiling is what
the weak supervision is costing.

Usage: oraculo.py [epochs] [radius] [seed] [pool] [eval_split]

Pool "expanded" trains on the 110 national training cities plus the expansion beyond the
national set. The ceiling is the denominator of the supervision-efficiency curve, and a
curve trained on the expanded pool needs a ceiling measured on the same pool.

With a radius the ceiling is measured on instances widened with their neighbourhood, which
answers a different question than the weakly supervised run does. If context lifts the
ceiling too, what a lone 160 m token encodes was the limit. If it lifts the weakly
supervised map while leaving the ceiling flat, context was only helping the supervision
find what the vectors already held, and raising the ceiling needs the foundation model to
stop being frozen.
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

from satinsight.agebs import catalogue_with_extra, cities_by_size, cities_extra  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.context import adjacency, build_layer  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
RADIUS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
POOL = sys.argv[4] if len(sys.argv) > 4 else "base"
EVAL_SPLIT = sys.argv[5] if len(sys.argv) > 5 else "val"
BATCH = 4096


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


def labelled(bags, grades):
    """Grade of every instance of every bag, with -1 where the AGEB is unknown.

    Unknown instances stay in the bag. They never contribute to the loss, and they are
    still ground the neighbourhood of their neighbours is made of, so dropping them would
    punch holes in the grid the context is read from.
    """
    return [np.array([grades.get(c, -1) for c in bag.cvegeo]) for bag in bags]


def links_of(bags, torch, device):
    """Adjacency of every bag, built once, empty when no context is asked for."""
    if not RADIUS:
        return [(None, None)] * len(bags)
    out = []
    for bag in bags:
        src, dst = adjacency(bag.y0, bag.x0, radius=RADIUS)
        out.append((torch.from_numpy(src).to(device), torch.from_numpy(dst).to(device)))
    return out


def build_oracle(dim_in, torch):
    """The same shallow head the weakly supervised model uses, told the answer per instance."""
    from torch import nn

    class Oracle(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(dim_in, 256), nn.ReLU(), nn.Dropout(0.25))
            self.neighbourhood = build_layer(256) if RADIUS else None
            self.score = nn.Linear(512 if RADIUS else 256, 5)

        def forward(self, x, src=None, dst=None):
            h = self.project(x)
            if self.neighbourhood is not None:
                h = self.neighbourhood(h, src, dst)
            return self.score(h)

    return Oracle()


def score_bags(model, bags, links, truth, torch, device):
    """Expected grade of every labelled instance, which is what the map is ranked by."""
    model.eval()
    scores, truths = [], []
    with torch.inference_mode():
        for bag, (src, dst), g in zip(bags, links, truth, strict=True):
            keep = g >= 0
            if not keep.any():
                continue
            x = torch.from_numpy(bag.instances).float().to(device)
            probability = torch.softmax(model(x, src, dst), dim=1).cpu().numpy()
            scores.extend((probability @ np.arange(5))[keep])
            truths.extend(g[keep])
    return np.array(scores), np.array(truths)


def main() -> None:
    import torch
    from torch import nn

    partition = pd.read_csv("data/partition.csv")
    train_cities = sorted(cities_of(partition, "train"))
    val_cities = sorted(cities_of(partition, EVAL_SPLIT))
    if POOL == "expanded":
        train_cities += sorted(cities_extra())
        catalogue = catalogue_with_extra()
    else:
        catalogue = cities_by_size(stratify=True)

    train_bags = load_split(train_cities, "s2", fuse=True)
    val_bags = load_split(val_cities, "s2", fuse=True)
    train_truth = labelled(train_bags, grades_of(train_cities, catalogue))
    val_truth = labelled(val_bags, grades_of(val_cities, catalogue))
    counts = np.bincount(np.concatenate(train_truth).clip(min=0), minlength=5).astype("float64")
    total_labelled = int(sum(int((g >= 0).sum()) for g in train_truth))
    print(
        f"{POOL} pool · radius {RADIUS} · {total_labelled} labelled training instances", flush=True
    )
    print("grade distribution:", counts.astype(int).tolist(), flush=True)

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
    model = build_oracle(train_bags[0].instances.shape[1], torch).to(device)
    weights = torch.tensor(
        total_labelled / (5 * np.maximum(counts, 1)), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, seen = 0.0, 0
        for index in rng.permutation(len(train_bags)):
            bag, (src, dst), g = train_bags[index], train_links[index], train_truth[index]
            keep = g >= 0
            if not keep.any():
                continue
            optimiser.zero_grad()
            x = torch.from_numpy(bag.instances).float().to(device)
            logits = model(x, src, dst)[torch.from_numpy(keep).to(device)]
            target = torch.from_numpy(g[keep]).long().to(device)
            loss = criterion(logits, target)
            loss.backward()
            optimiser.step()
            total += float(loss) * int(keep.sum())
            seen += int(keep.sum())

        score, truth = score_bags(model, val_bags, val_links, val_truth, torch, device)
        auroc = roc_auc_score((truth >= 3).astype(int), score)
        rho = spearmanr(score, truth).statistic
        print(
            f"epoch {epoch} · loss {total / seen:.4f} · "
            f"AUROC(>=High) {auroc:.3f} · Spearman {rho:+.3f}",
            flush=True,
        )

    # apareado por municipio: el techo con y sin contexto se compara sobre las mismas bolsas
    within = []
    with torch.inference_mode():
        for bag, (src, dst), g in zip(val_bags, val_links, val_truth, strict=True):
            keep = g >= 0
            if keep.sum() < 20 or len(set(g[keep])) < 2:
                continue
            x = torch.from_numpy(bag.instances).float().to(device)
            probability = torch.softmax(model(x, src, dst), dim=1).cpu().numpy()
            within.append(
                (
                    bag.municipality,
                    float(spearmanr((probability @ np.arange(5))[keep], g[keep]).statistic),
                )
            )

    print("\n===== CEILING =====", flush=True)
    print(
        f"radius {RADIUS} · instance level, fully supervised: AUROC {auroc:.3f} · "
        f"pooled Spearman {rho:+.3f} · "
        f"within-bag {np.mean([r for _, r in within]):+.3f} over {len(within)} bags",
        flush=True,
    )
    pd.DataFrame(
        [
            {
                "radius": RADIUS,
                "seed": SEED,
                "pool": POOL,
                "auroc": auroc,
                "pooled": rho,
                "within": float(np.mean([r for _, r in within])),
            }
        ]
    ).to_csv(f"data/oraculo_{POOL}_r{RADIUS}_s{SEED}_{EVAL_SPLIT}.csv", index=False)
    pd.DataFrame(within, columns=["municipality", "rho"]).assign(radius=RADIUS, seed=SEED).to_csv(
        f"data/oraculo_bags_{POOL}_r{RADIUS}_s{SEED}_{EVAL_SPLIT}.csv", index=False
    )


if __name__ == "__main__":
    main()
