"""Grouped cross-validation of the weakly supervised predictor over all 138 cities.

The validation and test columns rest on 14 cities each, too few to separate backbones.
This scores every city of the partition once as a held-out fold: cities are dealt into
k folds stratified as the partition was (by size and by share of high deprivation), and
for each fold the head trains on the other cities plus the expansion municipalities,
with every bag sharing a municipality key with the held-out fold removed from training.
Early stopping uses bag-level error on an inner set of cities drawn from the training
side, so the held-out fold plays no part in selection. Three seeds per fold; the
per-token scores of every held-out city are persisted so the pooled metrics, per token
and per AGEB with city-clustered bootstrap intervals, are computed once for every
backbone by backbone_cv_compare.py.

Usage: backbone_cv.py <sensor> [folds] [epochs]   (sensor s2, s2_dofal or s2_cfm)
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

from satinsight import backbone  # noqa: E402
from satinsight.agebs import catalogue_with_extra, cities_extra  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.llp import evaluate_map, instance_scores  # noqa: E402

sys.path.insert(0, "scripts")
from supervision_curve import grades_of, links_of, train_once  # noqa: E402

SENSOR = sys.argv[1] if len(sys.argv) > 1 else "s2"
SUFFIX = (
    backbone.SUFFIX
    if backbone.sensor("s2") == SENSOR
    else ""
    if SENSOR == "s2"
    else f"_{SENSOR[3:]}"
)
"""With the sensor of the environment the suffix also carries the head of the ablation."""
FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEEDS = (0, 1, 2)
INNER = 14
"""Cities held from each fold's training side for early stopping, as in the partition."""
FOLD_SEED = 0
RADIUS = 1

log = logging.getLogger("cv")


def stratified_folds(partition: pd.DataFrame, folds: int, seed: int) -> list[list[str]]:
    """Deals cities into folds round-robin within each stratum, after a seeded shuffle."""
    rng = np.random.default_rng(seed)
    out: list[list[str]] = [[] for _ in range(folds)]
    start = 0
    for _, group in partition.groupby("stratum"):
        cities = list(group.city)
        rng.shuffle(cities)
        for i, city in enumerate(cities):
            out[(start + i) % folds].append(city)
        start += len(cities)
    return [sorted(f) for f in out]


def inner_cities(training: list[str], partition: pd.DataFrame, seed: int) -> list[str]:
    """A proportionally stratified draw of INNER cities from the fold's training side.

    Cities are shuffled within each stratum and laid out stratum by stratum; taking every
    (n / INNER)-th city of that sequence allocates the draw to strata in proportion to
    their size, as the partition itself was dealt.
    """
    rng = np.random.default_rng(seed)
    strata = partition.set_index("city").stratum
    ordered: list[str] = []
    for _, group in sorted(pd.Series(training).groupby(strata[training].to_numpy())):
        cities = list(group)
        rng.shuffle(cities)
        ordered.extend(cities)
    picks = np.round(np.linspace(0, len(ordered) - 1, INNER)).astype(int)
    return sorted(ordered[i] for i in picks)


def main() -> None:
    import torch

    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    cities = sorted(partition.city)
    folds = stratified_folds(partition, FOLDS, FOLD_SEED)
    city_bags = load_split(cities, SENSOR, fuse=True)
    expansion = load_split(sorted(cities_extra()), SENSOR, fuse=True)
    grades = grades_of(cities, catalogue)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info(
        "%d cities in %d folds · %d city bags · %d expansion bags · %d dims",
        len(cities),
        FOLDS,
        len(city_bags),
        len(expansion),
        city_bags[0].instances.shape[1],
    )

    rows, frames = [], []
    for k, held in enumerate(folds):
        held_set = set(held)
        held_bags = [b for b in city_bags if b.city in held_set]
        held_keys = {b.municipality for b in held_bags}
        training = [c for c in cities if c not in held_set]
        inner = set(inner_cities(training, partition, FOLD_SEED + k))
        inner_bags = [b for b in city_bags if b.city in inner]
        inner_keys = {b.municipality for b in inner_bags}
        banned = held_keys | inner_keys
        train_bags = [
            b for b in city_bags if b.city not in held_set | inner and b.municipality not in banned
        ] + [b for b in expansion if b.municipality not in banned]
        inner_links = links_of(inner_bags, torch, device)
        held_links = links_of(held_bags, torch, device)
        by_city: dict = {}
        for bag in held_bags:
            by_city.setdefault(bag.city, []).append(bag)
        log.info(
            "fold %d · %d held cities (%d bags) · %d training bags · %d inner bags",
            k,
            len(held),
            len(held_bags),
            len(train_bags),
            len(inner_bags),
        )
        for seed in SEEDS:
            train_once(train_bags, inner_bags, inner_links, grades, seed, torch, device)
            model = train_once.last_model
            scored = evaluate_map(model, held_bags, held_links, grades, torch, device)
            scored.pop("per_bag")
            rows.append({"fold": k, "seed": seed, **scored})
            log.info(
                "fold %d seed %d · held within %+.3f (token) %+.3f (AGEB) · AUROC %.3f",
                k,
                seed,
                scored["spearman_within"],
                scored["spearman_within_ageb"],
                scored["auroc_high"],
            )
            model.eval()
            for city, bags in sorted(by_city.items()):
                x = np.vstack([b.instances for b in bags])
                y0 = np.concatenate([b.y0 for b in bags])
                x0 = np.concatenate([b.x0 for b in bags])
                src, dst = adjacency(y0, x0, radius=RADIUS)
                with torch.inference_mode():
                    _, per_instance = model(
                        torch.from_numpy(x).float().to(device),
                        torch.from_numpy(src).to(device),
                        torch.from_numpy(dst).to(device),
                    )
                frames.append(
                    pd.DataFrame(
                        {
                            "fold": k,
                            "seed": seed,
                            "city": city,
                            "municipality": np.concatenate(
                                [[b.municipality] * len(b) for b in bags]
                            ),
                            "cvegeo": np.concatenate([b.cvegeo for b in bags]),
                            "y0": y0,
                            "x0": x0,
                            "score": instance_scores(per_instance.cpu().numpy()),
                        }
                    )
                )
            pd.DataFrame(rows).to_csv(f"data/cv_folds{SUFFIX}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_parquet(
        f"data/cv_predictions{SUFFIX}.parquet", index=False
    )
    print(f"DONE {SENSOR}: {FOLDS} folds · {len(rows)} runs", flush=True)


if __name__ == "__main__":
    main()
