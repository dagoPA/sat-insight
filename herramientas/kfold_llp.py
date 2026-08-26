"""Grouped k-fold of the label-proportion model. Usage: kfold_llp.py [folds] [epochs]"""

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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from satinsight.agebs import cities_by_size  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.llp import build, instance_scores  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SEED, PATIENCE = 0, 8


def grades_of(cities, catalogue):
    out = {}
    for city in cities:
        try:
            _, agebs = city_aoi(city, catalogue=catalogue)
            out.update(dict(zip(agebs.cvegeo, agebs.ordinal.astype(int), strict=True)))
        except Exception:
            logging.warning("no grades for %s", city)
    return out


def evaluate(model, bags, grades, torch, device):
    """Bag error and whether the instance scores find the deprived AGEB.

    The map is scored twice on purpose, and the two answer different questions.

    Pooling every instance of every bag measures whether the model orders AGEB across the
    whole country. Much of that is easy: knowing a municipality is deprived on average
    already ranks its AGEB above those of a comfortable one, and no disaggregation is
    involved.

    Averaging the correlation computed inside each bag measures what the project claims:
    telling apart the deprived parts of one municipality from its comfortable parts. That
    is the honest figure for the contribution, and it is the lower of the two.
    """
    model.eval()
    bag_true, bag_pred, scores, truths, within = [], [], [], [], []
    with torch.inference_mode():
        for bag in bags:
            shares, per_instance = model(torch.from_numpy(bag.instances).float().to(device))
            bag_true.append(bag.shares)
            bag_pred.append(shares.cpu().numpy())
            g = np.array([grades.get(c, -1) for c in bag.cvegeo])
            keep = g >= 0
            if not keep.any():
                continue
            s = instance_scores(per_instance.cpu().numpy())[keep]
            scores.extend(s)
            truths.extend(g[keep])
            # una bolsa cuyas AGEB comparten grado no tiene orden interno que recuperar
            if len(set(g[keep])) > 1 and len(s) >= 20:
                within.append(float(spearmanr(s, g[keep]).statistic))
    bag_true, bag_pred = np.vstack(bag_true), np.vstack(bag_pred)
    scores, truths = np.array(scores), np.array(truths)
    return {
        "bag_mae": float(np.abs(bag_true - bag_pred).mean()),
        "auroc_high": float(roc_auc_score((truths >= 3).astype(int), scores)),
        "spearman_pooled": float(spearmanr(scores, truths).statistic),
        "spearman_within": float(np.mean(within)) if within else float("nan"),
        "bags_scored": len(within),
        "instances": len(truths),
    }


def main() -> None:
    import torch
    from torch import nn

    partition = pd.read_csv("data/partition.csv")
    cities = sorted(cities_of(partition, "train"))
    catalogue = cities_by_size(stratify=True)
    print(f"{len(cities)} training cities · {FOLDS} folds", flush=True)

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

        torch.manual_seed(SEED + k)
        model = build(train_bags[0].instances.shape[1]).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()
        best, waited, best_state = np.inf, 0, None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total = 0.0
            for index in rng.permutation(len(train_bags)):
                bag = train_bags[index]
                optimiser.zero_grad()
                shares, _ = model(torch.from_numpy(bag.instances).float().to(device))
                loss = criterion(shares, torch.from_numpy(bag.shares).float().to(device))
                loss.backward()
                optimiser.step()
                total += float(loss)
            scored = evaluate(model, val_bags, grades, torch, device)
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
            {"fold": k, "bags": len(val_bags), **evaluate(model, val_bags, grades, torch, device)}
        )
        print(json.dumps(results[-1], default=float), flush=True)
        pd.DataFrame(results).to_csv("data/llp_kfold.csv", index=False)

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
