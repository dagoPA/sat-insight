"""Auxiliary per-token features from the free global products, one file per city.

Writes `{city}_aux.npz` beside the optical vectors with the four columns of
`satinsight.auxfeatures.FEATURES`, and an instances table identical to the optical one,
so the loaders fuse them by position with `extras=("aux",)` and every training and
evaluation tool runs on the fused input unchanged.

Usage: aux_tokens.py [index total]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

from satinsight.auxfeatures import city_features  # noqa: E402
from satinsight.cache import load  # noqa: E402
from satinsight.dataset import paths  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.encoders import save  # noqa: E402


def token_features(city: str) -> None:
    where = paths(DATA_ROOT)
    out_vectors = where["vectors"] / f"{city}_aux.npz"
    out_instances = where["instances"] / f"{city}_aux.parquet"
    if out_vectors.exists() and out_instances.exists():
        print(f"SKIP {city}", flush=True)
        return

    instances = pd.read_parquet(where["instances"] / f"{city}_s2.parquet")
    _, grid, _ = load(DATA_ROOT / "composites" / f"{city}_s2.tif")
    matrix = city_features(grid, instances.y0.to_numpy(), instances.x0.to_numpy())

    save(
        matrix,
        out_vectors,
        y0=instances.y0.to_numpy(),
        x0=instances.x0.to_numpy(),
        cvegeo=instances.cvegeo.to_numpy(),
    )
    instances.to_parquet(out_instances, index=False)
    print(f"OK {city} · {len(instances)} tokens · {matrix.shape[1]} features", flush=True)


def main() -> int:
    where = paths(DATA_ROOT)
    keys = sorted(p.stem[:-3] for p in where["instances"].glob("*_s2.parquet"))
    arguments = sys.argv[1:]
    if len(arguments) >= 2 and arguments[0].isdigit():
        keys = keys[int(arguments[0]) :: int(arguments[1])]

    failed = []
    for n, city in enumerate(keys, start=1):
        try:
            token_features(city)
        except Exception as e:
            failed.append(city)
            print(f"FAIL {city} ({n}/{len(keys)}): {type(e).__name__}: {e}", flush=True)
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
