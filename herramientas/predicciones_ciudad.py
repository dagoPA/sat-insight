"""Re-scores the validation tokens with city-wide adjacency, from the saved weights.

The border test found the map jumping at municipal borders where deprivation does not
change, and there is a mechanical suspect before any claim about label memorisation: the
480 m neighbourhood is built inside the bag, so a token beside the border never sees its
neighbours across it, at training and at scoring alike. Administrative paperwork has no
business shaping who counts as a neighbour at inference.

This loads the saved heads — no retraining — and scores every validation city as one
block, with the adjacency drawn over the whole city grid. The border test rerun on both
columns separates the hypotheses: a jump that dies was our seam; a jump that survives is
the model carrying municipal information it was never given.

Usage: predicciones_ciudad.py
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

from satinsight.bagdata import load_split  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.llp import build, instance_scores  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

SEEDS = (0, 1, 2)
RADIUS = 1


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    val_cities = sorted(cities_of(partition, "val"))
    val_bags = load_split(val_cities, "s2", fuse=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    by_city: dict = {}
    for bag in val_bags:
        by_city.setdefault(bag.city, []).append(bag)

    models = []
    dim = val_bags[0].instances.shape[1]
    for seed in SEEDS:
        model = build(dim, radius=RADIUS, standardize=True).to(device)
        model.load_state_dict(torch.load(f"data/weights/llp_final_s{seed}.pt", map_location=device))
        model.eval()
        models.append(model)

    rows = []
    for city, bags in sorted(by_city.items()):
        x = np.vstack([b.instances for b in bags])
        y0 = np.concatenate([b.y0 for b in bags])
        x0 = np.concatenate([b.x0 for b in bags])
        cvegeo = np.concatenate([b.cvegeo for b in bags])
        municipality = np.concatenate([[b.municipality] * len(b) for b in bags])
        src, dst = adjacency(y0, x0, radius=RADIUS)
        src_t = torch.from_numpy(src).to(device)
        dst_t = torch.from_numpy(dst).to(device)
        xt = torch.from_numpy(x).float().to(device)
        scores = []
        with torch.inference_mode():
            for model in models:
                _, per_instance = model(xt, src_t, dst_t)
                scores.append(instance_scores(per_instance.cpu().numpy()))
        mean_score = np.mean(scores, axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "city": city,
                    "municipality": municipality,
                    "cvegeo": cvegeo,
                    "y0": y0,
                    "x0": x0,
                    "score_city": mean_score,
                }
            )
        )
        logging.info("%s: %d tokens rescored city-wide", city, len(mean_score))

    pd.DataFrame(pd.concat(rows, ignore_index=True)).to_parquet(
        "data/predicciones_val_ciudad.parquet", index=False
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
