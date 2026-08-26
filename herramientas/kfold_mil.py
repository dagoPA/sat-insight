"""Grouped k-fold of the attention MIL over the training cities.

Usage: kfold_mil.py [folds] [epochs] [objective] [entropy] [cluster]

Folds are cut by city, never by bag: two bags of the same conurbation share urban fabric
and acquisition geometry, so splitting them would let the model recognise the place rather
than the deprivation. The test cities are never touched here.
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
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402
from satinsight.train_mil import SEED, predict, score_heatmap, train  # noqa: E402

FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OBJECTIVE = sys.argv[3] if len(sys.argv) > 3 else "classes"
ENTROPY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
CLUSTER = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
SENSOR = "s2"
FUSE = True

partition = pd.read_csv("data/partition.csv")
cities = sorted(cities_of(partition, "train"))
print(
    f"{len(cities)} training cities · {FOLDS} folds · {EPOCHS} epochs max · objective {OBJECTIVE}",
    flush=True,
)

catalogue = cities_by_size(stratify=True)
grades: dict[str, int] = {}
for city in cities:
    try:
        _, agebs = city_aoi(city, catalogue=catalogue)
        grades.update(dict(zip(agebs.cvegeo, agebs.ordinal.astype(int), strict=True)))
    except Exception:
        logging.warning("no grades for %s", city)
print(f"{len(grades):,} AGEB with a grade held out of training", flush=True)

rng = np.random.default_rng(SEED)
shuffled = list(cities)
rng.shuffle(shuffled)
folds = [shuffled[i::FOLDS] for i in range(FOLDS)]

histories, results = [], []
for k, held_out in enumerate(folds):
    rest = [c for c in cities if c not in held_out]
    print(f"\n===== fold {k}: {len(rest)} train · {len(held_out)} validate =====", flush=True)
    train_bags = load_split(rest, SENSOR, fuse=FUSE)
    val_bags = load_split(held_out, SENSOR, fuse=FUSE)

    model, history = train(
        train_bags,
        val_bags,
        epochs=EPOCHS,
        seed=SEED + k,
        objective=OBJECTIVE,
        entropy_weight=ENTROPY,
        cluster_weight=CLUSTER,
    )
    history["fold"] = k
    histories.append(history)

    scored = predict(model, val_bags, objective=OBJECTIVE)
    heat = score_heatmap(val_bags, scored["attention"], grades)
    results.append(
        {
            "fold": k,
            "cities": len(held_out),
            "bags": len(val_bags),
            "kappa": float(history.val_kappa.max()),
            "auroc": float(history.loc[history.val_kappa.idxmax(), "val_auroc"]),
            "spearman_mean": heat["spearman_mean"],
            "spearman_pooled": heat["spearman_pooled"],
            "auroc_high": heat["auroc_high"],
            "bags_scored": heat["bags_scored"],
        }
    )
    print(json.dumps(results[-1], default=float), flush=True)
    pd.concat(histories, ignore_index=True).to_csv(
        f"data/mil_history_{OBJECTIVE}_{ENTROPY}_{CLUSTER}.csv", index=False
    )
    pd.DataFrame(results).to_csv(f"data/mil_kfold_{OBJECTIVE}_{ENTROPY}_{CLUSTER}.csv", index=False)

r = pd.DataFrame(results)
print("\n===== SUMMARY =====", flush=True)
print(r.round(4).to_string(index=False), flush=True)
print(
    f"\nkappa {r.kappa.mean():.3f} ± {r.kappa.std():.3f} · "
    f"auroc {r.auroc.mean():.3f} · "
    f"spearman del mapa {r.spearman_mean.mean():.3f} · "
    f"auroc del mapa {r.auroc_high.mean():.3f}",
    flush=True,
)
