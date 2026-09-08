"""Fine-tunes the last blocks of the base backbone with the head, on cached states.

The ablation the frozen protocol invites: the encoder's suffix (blocks from `split_at` on
and the final norm, shared by both sensors) trains together with the label-proportion
head on the 771 municipal bags, under the same loss, seeds, early stopping on validation
bag error and evaluation as the frozen runs. The suffix starts from the pretrained
weights, so step zero is the frozen model; the head's scaler is fit on the frozen
vectors, as in every other run. Per-token scores of the validation and test cities are
persisted with the suffix `_ft`, so every downstream tool can consume them.

Usage: finetune_train.py [epochs] [seed] [split_at] [backbone_lr]
"""

import logging
import sys
import warnings
from dataclasses import replace

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from satinsight import encoders, finetune  # noqa: E402
from satinsight.agebs import catalogue_with_extra, cities_extra  # noqa: E402
from satinsight.bagdata import load_split  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.llp import build, evaluate_map, instance_scores  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

sys.path.insert(0, "scripts")
from supervision_curve import grades_of  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SPLIT_AT = int(sys.argv[3]) if len(sys.argv) > 3 else 10
BACKBONE_LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
RADIUS = 1
TAG = f"ft_b{SPLIT_AT}"

log = logging.getLogger("finetune")


class Cities:
    """Cached states opened on demand, one pair of sensors per city."""

    def __init__(self) -> None:
        self.open: dict = {}

    def __getitem__(self, city: str) -> dict:
        if city not in self.open:
            self.open[city] = {s: finetune.CityStates.open(city, s, SPLIT_AT) for s in ("s2", "s1")}
        return self.open[city]


def links_of(bags, torch, device):
    out = []
    for bag in bags:
        src, dst = adjacency(bag.y0, bag.x0, radius=RADIUS)
        out.append((torch.from_numpy(src).to(device), torch.from_numpy(dst).to(device)))
    return out


def rebuilt(suffix, bags, addresses, cities, torch, device):
    """Bags with their instances replaced by the suffix's current vectors, for scoring."""
    out = []
    with torch.inference_mode():
        for bag, address in zip(bags, addresses, strict=True):
            vectors = finetune.fused_vectors(suffix, bag, address, cities[bag.city], torch, device)
            out.append(replace(bag, instances=vectors.float().cpu().numpy()))
    return out


def main() -> None:
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    train_cities = sorted(cities_of(partition, "train")) + sorted(cities_extra())
    val_cities = sorted(cities_of(partition, "val"))
    test_cities = sorted(cities_of(partition, "test"))
    pool = load_split(train_cities, "s2", fuse=True)
    val_bags = load_split(val_cities, "s2", fuse=True)
    test_bags = load_split(test_cities, "s2", fuse=True)
    val_grades = grades_of(val_cities, catalogue)
    test_grades = grades_of(test_cities, catalogue)

    encoder = encoders.DofaEncoder()
    device = encoder.device
    cities = Cities()
    addresses = {
        id(bag): finetune.address_bag(bag, cities[bag.city]) for bag in pool + val_bags + test_bags
    }
    suffix = finetune.build_suffix(encoder, SPLIT_AT).to(device)
    head = build(pool[0].instances.shape[1], radius=RADIUS, standardize=True).to(device)
    rng_stats = np.random.default_rng(SEED)
    sample = np.vstack([b.instances[rng_stats.permutation(len(b))[:200]] for b in pool])
    head.fit_scaler(sample.mean(axis=0), sample.std(axis=0))
    optimiser = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": HEAD_LR},
            {"params": suffix.parameters(), "lr": BACKBONE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCELoss()
    train_links = links_of(pool, torch, device)
    val_links = links_of(val_bags, torch, device)
    test_links = links_of(test_bags, torch, device)
    log.info(
        "seed %d · %d training bags · suffix of %d blocks, %d parameters · lr %.0e",
        SEED,
        len(pool),
        12 - SPLIT_AT,
        sum(p.numel() for p in suffix.parameters()),
        BACKBONE_LR,
    )

    rng = np.random.default_rng(SEED)
    best, waited, best_state = np.inf, 0, None
    for epoch in range(EPOCHS):
        suffix.train()
        head.train()
        total = 0.0
        for index in rng.permutation(len(pool)):
            bag, (src, dst) = pool[index], train_links[index]
            optimiser.zero_grad()
            vectors = finetune.fused_vectors(
                suffix, bag, addresses[id(bag)], cities[bag.city], torch, device
            )
            shares, _ = head(vectors, src, dst)
            loss = criterion(shares, torch.from_numpy(bag.shares).float().to(device))
            loss.backward()
            optimiser.step()
            total += float(loss)
        suffix.eval()
        head.eval()
        scored = evaluate_map(
            head,
            rebuilt(suffix, val_bags, [addresses[id(b)] for b in val_bags], cities, torch, device),
            val_links,
            val_grades,
            torch,
            device,
        )
        log.info(
            "epoch %d · loss %.4f · val bag MAE %.4f · within %+.3f · AUROC %.3f",
            epoch,
            total / len(pool),
            scored["bag_mae"],
            scored["spearman_within"],
            scored["auroc_high"],
        )
        if scored["bag_mae"] < best:
            best, waited = scored["bag_mae"], 0
            best_state = (
                {n: v.detach().clone() for n, v in suffix.state_dict().items()},
                {n: v.detach().clone() for n, v in head.state_dict().items()},
            )
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    suffix.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    suffix.eval()
    head.eval()
    torch.save(
        {"suffix": suffix.state_dict(), "head": head.state_dict()},
        f"data/weights/llp_final_{TAG}_s{SEED}.pt",
    )

    rows, frames = [], []
    for split, bags, links, grades in (
        ("val", val_bags, val_links, val_grades),
        ("test", test_bags, test_links, test_grades),
    ):
        current = rebuilt(suffix, bags, [addresses[id(b)] for b in bags], cities, torch, device)
        scored = evaluate_map(head, current, links, grades, torch, device)
        scored.pop("per_bag")
        rows.append({"seed": SEED, "split": split, "split_at": SPLIT_AT, **scored})
        log.info(
            "%s · within %+.3f (token) %+.3f (AGEB) · AUROC %.3f · bag MAE %.4f",
            split,
            scored["spearman_within"],
            scored["spearman_within_ageb"],
            scored["auroc_high"],
            scored["bag_mae"],
        )
        with torch.inference_mode():
            for bag, (src, dst) in zip(current, links, strict=True):
                x = torch.from_numpy(bag.instances).float().to(device)
                _, per_instance = head(x, src, dst)
                frames.append(
                    pd.DataFrame(
                        {
                            "split": split,
                            "seed": SEED,
                            "city": bag.city,
                            "municipality": bag.municipality,
                            "cvegeo": bag.cvegeo,
                            "y0": bag.y0,
                            "x0": bag.x0,
                            "score": instance_scores(per_instance.cpu().numpy()),
                        }
                    )
                )
    pd.DataFrame(rows).to_csv(f"data/finetune_{TAG}_s{SEED}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_parquet(
        f"data/predictions_{TAG}_s{SEED}.parquet", index=False
    )
    print(f"DONE seed {SEED}", flush=True)


if __name__ == "__main__":
    main()
