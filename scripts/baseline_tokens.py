"""B2: per-token features with no learned representation, for the obvious objection.

The first thing a reviewer will ask of the curve is whether a foundation model is doing any
work, or whether published land cover plus a vegetation index already buys the same map.
Stage one hinted that it might at the municipal level; the curve claims the instance level,
so the baseline has to live there too.

Each surviving token of a city gets the WorldCover class fractions inside its 160 m window
plus the mean and dispersion of NDVI from the same composite the encoder saw. The features
are saved under the sensor name "wc" with an instances table identical to the optical one,
so every training and evaluation tool of the project runs on them unchanged, same bags,
same protocol, same head. Whatever difference appears is the representation.

Usage: baseline_tokens.py [index total]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from satinsight import landcover  # noqa: E402
from satinsight.agebs import catalogue_with_extra  # noqa: E402
from satinsight.dataset import load, paths  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.encoders import save  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.tiling import TOKEN_SIZE  # noqa: E402

TOKEN_PX = TOKEN_SIZE


def token_features(city: str, catalogue) -> None:
    where = paths(DATA_ROOT)
    out_vectors = where["vectors"] / f"{city}_wc.npz"
    out_instances = where["instances"] / f"{city}_wc.parquet"
    if out_vectors.exists() and out_instances.exists():
        print(f"SKIP {city}", flush=True)
        return

    instances = pd.read_parquet(where["instances"] / f"{city}_s2.parquet")
    bands, grid, _ = load(DATA_ROOT / "composites" / f"{city}_s2.tif")
    area, _ = city_aoi(city, catalogue=catalogue)
    classes = landcover.mosaic(area, grid)

    red, nir = bands["B04"], bands["B08"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(nir + red > 0, (nir - red) / (nir + red), np.nan)

    codes = sorted(landcover.CLASSES)
    rows = np.empty((len(instances), len(codes) + 2), dtype="float32")
    for i, (y0, x0) in enumerate(zip(instances.y0, instances.x0, strict=True)):
        window_classes = classes[y0 : y0 + TOKEN_PX, x0 : x0 + TOKEN_PX]
        valid = window_classes[window_classes != landcover.NO_DATA]
        fractions = (
            [float((valid == c).mean()) for c in codes] if valid.size else [np.nan] * len(codes)
        )
        window_ndvi = ndvi[y0 : y0 + TOKEN_PX, x0 : x0 + TOKEN_PX]
        rows[i] = [*fractions, np.nanmean(window_ndvi), np.nanstd(window_ndvi)]
    matrix = np.nan_to_num(rows, nan=0.0)

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
    catalogue = catalogue_with_extra()
    where = paths(DATA_ROOT)
    keys = sorted(p.stem[:-3] for p in where["instances"].glob("*_s2.parquet"))
    argumentos = sys.argv[1:]
    if len(argumentos) >= 2 and argumentos[0].isdigit():
        keys = keys[int(argumentos[0]) :: int(argumentos[1])]

    failed = []
    for n, city in enumerate(keys, start=1):
        try:
            token_features(city, catalogue)
        except Exception as e:
            failed.append(city)
            print(f"FAIL {city} ({n}/{len(keys)}): {type(e).__name__}: {e}", flush=True)
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
