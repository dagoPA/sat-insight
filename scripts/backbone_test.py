"""Scores the held-out test cities with a backbone's saved validation-selected heads.

Mirrors the headline row of the test column for one alternative backbone: the three
seeds saved by predictions_val.py are loaded, every test bag is scored with city-wide
adjacency, the per-token scores are persisted with the backbone's suffix, and one row per
seed records the map metrics on test, per token and per AGEB.

Usage: backbone_test.py <sensor>   (for example s2_dofal)
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

from satinsight.agebs import catalogue_with_extra  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.llp import build, evaluate_map, instance_scores  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

sys.path.insert(0, "scripts")
from supervision_curve import grades_of, links_of  # noqa: E402

SENSOR = sys.argv[1]
SUFFIX = "" if SENSOR == "s2" else f"_{SENSOR[3:]}"
SEEDS = (0, 1, 2)


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    test_cities = sorted(cities_of(partition, "test"))
    test_bags = load_split(test_cities, SENSOR, fuse=True)
    grades = grades_of(test_cities, catalogue)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    links = links_of(test_bags, torch, device)
    by_city: dict = {}
    for bag in test_bags:
        by_city.setdefault(bag.city, []).append(bag)

    rows, frames = [], []
    for seed in SEEDS:
        model = build(test_bags[0].instances.shape[1], radius=1, standardize=True).to(device)
        state = torch.load(f"data/weights/llp_final{SUFFIX}_s{seed}.pt", map_location=device)
        model.load_state_dict(state)
        scored = evaluate_map(model, test_bags, links, grades, torch, device)
        per_bag = scored.pop("per_bag")
        rows.append({"seed": seed, **scored})
        logging.info(
            "seed %d · test within %+.3f (token) %+.3f (AGEB) · AUROC %.3f",
            seed,
            scored["spearman_within"],
            scored["spearman_within_ageb"],
            scored["auroc_high"],
        )
        pd.DataFrame(per_bag, columns=["municipality", "rho"]).assign(seed=seed).to_csv(
            f"data/backbone_test_bags{SUFFIX}_s{seed}.csv", index=False
        )
        model.eval()
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
    pd.DataFrame(rows).to_csv(f"data/backbone_test{SUFFIX}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_parquet(
        f"data/predictions_test{SUFFIX}.parquet", index=False
    )
    print(f"DONE {SENSOR}: {len(rows)} seeds scored on test", flush=True)


if __name__ == "__main__":
    main()
