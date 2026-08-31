"""The single opening of the test set: every reported row, scored once, frozen.

The fourteen test cities have never been touched. This scores them under every
configuration the manuscript reports, with one discipline throughout: model selection —
epochs by early stopping, hyperparameters, the choice of head — stays anchored to the
validation cities exactly as it was; the test cities are only ever scored. Nothing here
may be re-run with different settings afterwards: a second opening would turn the
confirmatory column back into a development set.

Rows, in order: the supervision-efficiency curve (five sizes), label granularity (three
levels), single-sensor configurations, the population-weighted variant, the
instance-supervised oracle, and the headline model from its saved validation-selected
weights, whose per-token scores are persisted for the downstream analyses (RWI pairing,
CONAPO replication, targeting, MAUP, border test).

Usage: apertura_test.py [epochs]
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
from satinsight.llp import build, evaluate_map, instance_scores  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

sys.path.insert(0, "herramientas")
from curva_granularidad import relabel  # noqa: E402
from curva_supervision import grades_of, links_of, nested_sample, train_once  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEEDS = (0, 1, 2)
SIZES = (50, 100, 200, 400, None)
OUT = "data/apertura_test.csv"


def score_test(model, test_bags, test_links, grades, torch, device):
    scored = evaluate_map(model, test_bags, test_links, grades, torch, device)
    scored.pop("per_bag")
    return scored


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

    pool = load_split(train_cities, "s2", fuse=True)
    val_bags = load_split(val_cities, "s2", fuse=True)
    test_bags = load_split(test_cities, "s2", fuse=True)
    val_grades = grades_of(val_cities, catalogue)
    test_grades = grades_of(test_cities, catalogue)
    val_links = links_of(val_bags, torch, device)
    test_links = links_of(test_bags, torch, device)
    table = load_grs()
    print(f"test opens: {len(test_bags)} bags of {len(test_cities)} cities", flush=True)

    rows = []

    def record(row):
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        logging.info("%s", {k: v for k, v in row.items() if k != "seed"})

    # 1 · the curve
    for seed in SEEDS:
        order = nested_sample(pool, seed)
        for size in SIZES:
            bags = [pool[i] for i in (order if size is None else order[:size])]
            train_once(bags, val_bags, val_links, val_grades, seed, torch, device)
            scored = score_test(
                train_once.last_model, test_bags, test_links, test_grades, torch, device
            )
            record({"row": "curve", "detail": len(bags), "seed": seed, **scored})

    # 2 · granularity
    for level in ("municipality", "state", "national"):
        bags = relabel(pool, level, table)
        for seed in SEEDS:
            train_once(bags, val_bags, val_links, val_grades, seed, torch, device)
            scored = score_test(
                train_once.last_model, test_bags, test_links, test_grades, torch, device
            )
            record({"row": "granularity", "detail": level, "seed": seed, **scored})

    # 3 · single sensors
    for modality, (sensor, fuse) in {
        "optical": ("s2", False),
        "degraded": ("s2deg", False),
        "radar": ("s1", False),
        "worldcover": ("wc", False),
    }.items():
        m_pool = load_split(train_cities, sensor, fuse=fuse)
        m_val = load_split(val_cities, sensor, fuse=fuse)
        m_test = load_split(test_cities, sensor, fuse=fuse)
        m_val_links = links_of(m_val, torch, device)
        m_test_links = links_of(m_test, torch, device)
        for seed in SEEDS:
            train_once(m_pool, m_val, m_val_links, val_grades, seed, torch, device)
            scored = score_test(
                train_once.last_model, m_test, m_test_links, test_grades, torch, device
            )
            record({"row": "modality", "detail": modality, "seed": seed, **scored})

    # 4 · population-weighted variant
    from a5_agregacion import population_weights, train_weighted

    weights = population_weights(pool)
    for seed in SEEDS:
        _, _ = train_weighted(pool, weights, val_bags, val_links, val_grades, seed, torch, device)
        # train_weighted keeps no handle; rebuild scoring through its returned map on val
        # is not what we need — so it must expose the model the same way train_once does
        model = train_weighted.last_model
        scored = score_test(model, test_bags, test_links, test_grades, torch, device)
        record({"row": "weighted", "detail": "population", "seed": seed, **scored})

    # the oracle runs standalone in the queue: its schedule is fixed, so scoring the
    # test split directly is safe and needs no selection anchor

    # 5 · headline from the saved validation-selected weights, city-wide adjacency
    by_city: dict = {}
    for bag in test_bags:
        by_city.setdefault(bag.city, []).append(bag)
    dim = test_bags[0].instances.shape[1]
    frames = []
    for seed in SEEDS:
        model = build(dim, radius=1, standardize=True).to(device)
        model.load_state_dict(torch.load(f"data/weights/llp_final_s{seed}.pt", map_location=device))
        model.eval()
        scored = score_test(model, test_bags, test_links, test_grades, torch, device)
        record({"row": "headline_saved", "detail": "bag_adjacency", "seed": seed, **scored})
        for city, bags in sorted(by_city.items()):
            x = np.vstack([b.instances for b in bags])
            y0 = np.concatenate([b.y0 for b in bags])
            x0 = np.concatenate([b.x0 for b in bags])
            src, dst = adjacency(y0, x0, radius=1)
            with torch.inference_mode():
                _, per_instance = model(
                    torch.from_numpy(x).float().to(device),
                    torch.from_numpy(src).to(device),
                    torch.from_numpy(dst).to(device),
                )
            frames.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "city": city,
                        "municipality": np.concatenate([[b.municipality] * len(b) for b in bags]),
                        "cvegeo": np.concatenate([b.cvegeo for b in bags]),
                        "y0": y0,
                        "x0": x0,
                        "score": instance_scores(per_instance.cpu().numpy()),
                    }
                )
            )
    pd.concat(frames, ignore_index=True).to_parquet("data/predicciones_test.parquet", index=False)
    print("DONE: test scored once; per-token scores persisted", flush=True)


if __name__ == "__main__":
    main()
