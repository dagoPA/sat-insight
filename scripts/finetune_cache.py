"""Caches the frozen-prefix states of every window, for the partial fine-tuning ablation.

The base backbone is split after `split_at` of its twelve blocks; the prefix runs once
over every window of every city of the partition and the expansion, both sensors, and
its output is stored under data/activations/. With `check <city>` the script instead
verifies on one city that the pretrained suffix applied to the cache reproduces the
frozen vectors on disk, which is what makes the fine-tuned and frozen protocols
comparable at step zero.

Usage: finetune_cache.py [split_at] [index total]
       finetune_cache.py check <city> [split_at]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

from satinsight import encoders, finetune  # noqa: E402
from satinsight.agebs import cities_extra  # noqa: E402
from satinsight.bagdata import load_city  # noqa: E402
from satinsight.dataset import paths  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402


def check(city: str, split_at: int) -> None:
    import torch

    encoder = encoders.DofaEncoder()
    for sensor in ("s2", "s1"):
        print(f"{sensor}: {finetune.cache_city(city, sensor, encoder, split_at)}", flush=True)
    states = {s: finetune.CityStates.open(city, s, split_at) for s in ("s2", "s1")}
    suffix = finetune.build_suffix(encoder, split_at).to(encoder.device).eval()
    bags = load_city(city, "s2", fuse=True)
    worst = 0.0
    with torch.inference_mode():
        for bag in bags:
            address = finetune.address_bag(bag, states)
            rebuilt = finetune.fused_vectors(suffix, bag, address, states, torch, encoder.device)
            frozen = torch.from_numpy(bag.instances).float().to(encoder.device)
            gap = (rebuilt - frozen).abs().max().item()
            scale = frozen.abs().mean().item()
            worst = max(worst, gap)
            print(f"  bag {bag.municipality}: {len(bag)} tokens, gap {gap:.4f} (|v| {scale:.3f})")
    print(f"CHECK {'OK' if worst < 0.05 else 'FAILED'}: worst gap {worst:.4f}", flush=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
        return 0
    split_at = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    partition = pd.read_csv("data/partition.csv")
    keys = sorted(partition.city) + sorted(cities_extra())
    if len(sys.argv) >= 4:
        keys = keys[int(sys.argv[2]) :: int(sys.argv[3])]
    encoder = encoders.DofaEncoder()
    where = paths(DATA_ROOT)
    failed = []
    for n, city in enumerate(keys, start=1):
        for sensor in ("s2", "s1"):
            if not (where["instances"] / f"{city}_{sensor}.parquet").exists():
                continue
            try:
                result = finetune.cache_city(city, sensor, encoder, split_at)
                print(f"{result} {city}/{sensor} ({n}/{len(keys)})", flush=True)
            except Exception as e:
                failed.append(f"{city}/{sensor}")
                print(f"FAIL {city}/{sensor}: {type(e).__name__}: {e}", flush=True)
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
