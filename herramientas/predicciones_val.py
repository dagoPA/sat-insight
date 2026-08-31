"""Persists per-instance scores on the validation cities, which every new analysis needs.

Every training run so far evaluated and discarded its predictions. The adopted slate
(targeting efficiency, the CONAPO replication, the border-discontinuity test, the
uncertainty layer) all consume the same object: the score of every token of the 14
held-out cities under the final configuration. This trains that configuration (tuned head,
480 m context, expanded pool), saves the model weights, and writes one row per token.

Three seeds, saved separately: the analyses that follow decide how to combine them, and an
ensemble mean is itself one of the planned analyses.

Usage: predicciones_val.py [epochs]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

from satinsight.agebs import catalogue_with_extra, cities_extra  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.llp import instance_scores  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

sys.path.insert(0, "herramientas")
from curva_supervision import grades_of, links_of, train_once  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEEDS = (0, 1, 2)
RADIUS = 1

log = logging.getLogger("scores")


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    train_cities = sorted(cities_of(partition, "train")) + sorted(cities_extra())
    val_cities = sorted(cities_of(partition, "val"))

    pool = load_split(train_cities, "s2", fuse=True)
    val_bags = load_split(val_cities, "s2", fuse=True)
    grades = grades_of(val_cities, catalogue)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    val_links = links_of(val_bags, torch, device)

    rows = []
    for seed in SEEDS:
        # train_once ya devuelve el modelo cargado con su mejor estado vía closure; aquí
        # se necesita el modelo mismo, así que se re-implementa el remate de la función
        scored = train_once(pool, val_bags, val_links, grades, seed, torch, device)
        log.info("seed %d trained · within %+.3f", seed, scored["spearman_within"])
        model = train_once.last_model
        torch.save(model.state_dict(), f"data/weights/llp_final_s{seed}.pt")
        model.eval()
        with torch.inference_mode():
            for bag, (src, dst) in zip(val_bags, val_links, strict=True):
                x = torch.from_numpy(bag.instances).float().to(device)
                _, per_instance = model(x, src, dst)
                score = instance_scores(per_instance.cpu().numpy())
                for i in range(len(bag)):
                    rows.append(
                        {
                            "seed": seed,
                            "city": bag.city,
                            "municipality": bag.municipality,
                            "cvegeo": bag.cvegeo[i],
                            "y0": int(bag.y0[i]),
                            "x0": int(bag.x0[i]),
                            "score": float(score[i]),
                        }
                    )
        pd.DataFrame(rows).to_parquet("data/predicciones_val.parquet", index=False)
        log.info("seed %d · %d rows persisted", seed, len(rows))

    print(f"DONE · {len(rows)} token scores over {len(SEEDS)} seeds", flush=True)


if __name__ == "__main__":
    main()
