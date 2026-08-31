"""A3: re-encodes the optical arm at the radar's effective resolution.

The modality ablation left a causal claim untested. Radar fails to localize and its real
resolution is 20 by 22 metres against the optical 10; the claim is that resolution, and
never the sensing physics, is what kills the map. The test is to hand the encoder the same
optical composite with its resolution destroyed to the radar's level and nothing else
changed: same windows, same tokens, same protocol.

Degradation is a 2x2 area average replicated back to the native grid, which resamples the
image to 20 metres while keeping every token where it was. A blur kernel would need an
argument about point-spread functions; block averaging is the statement "you only get one
number per 20 metres" made literally.

Vectors are saved under the sensor name "s2deg" beside an instances table copied from the
optical one, so the training tools run on them unchanged.

Usage: degradar_optico.py [index total]
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

from satinsight import encoders, tiling  # noqa: E402
from satinsight.dataset import CHANNELS, load, paths  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402

FACTOR = 2


def degrade(band: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    """Area-average blocks of `factor` pixels, replicated back to the native grid.

    NaN stays NaN through a nan-mean, so nodata does not leak into its neighbours as a
    lowered value.
    """
    rows = band.shape[0] - band.shape[0] % factor
    cols = band.shape[1] - band.shape[1] % factor
    blocks = band[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coarse = np.nanmean(blocks, axis=(1, 3))
    out = band.copy()
    out[:rows, :cols] = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)
    return out


def one_city(city: str) -> None:
    where = paths(DATA_ROOT)
    out_vectors = where["vectors"] / f"{city}_s2deg.npz"
    out_instances = where["instances"] / f"{city}_s2deg.parquet"
    if out_vectors.exists() and out_instances.exists():
        print(f"SKIP {city}", flush=True)
        return

    source = where["instances"] / f"{city}_s2.parquet"
    instances = pd.read_parquet(source)
    bands, _, _ = load(DATA_ROOT / "composites" / f"{city}_s2.tif")
    bands = {c: degrade(bands[c]) for c in CHANNELS["s2"]}

    encoder = one_city.encoder
    windows = tiling.select(bands, min_valid_fraction=tiling.MIN_VALID_FRACTION)
    matrix, tokens = encoders.extract(bands, windows, encoder, order=CHANNELS["s2"])
    position = {(t.y0, t.x0): i for i, t in enumerate(tokens)}
    rows = [position.get((y, x), -1) for y, x in zip(instances.y0, instances.x0, strict=True)]
    keep = np.array([r >= 0 for r in rows])
    if not keep.all():
        print(f"{city}: {(~keep).sum()} optical tokens without degraded twin", flush=True)
    kept = instances[keep].reset_index(drop=True)

    encoders.save(
        matrix[np.array(rows)[keep]],
        out_vectors,
        y0=kept.y0.to_numpy(),
        x0=kept.x0.to_numpy(),
        cvegeo=kept.cvegeo.to_numpy(),
    )
    kept.to_parquet(out_instances, index=False)
    print(f"OK {city} · {len(kept)} tokens", flush=True)


def main() -> int:
    from satinsight.encoders import DofaEncoder

    partition = pd.read_csv("data/partition.csv")
    keys = sorted(partition.ciudad)
    argumentos = sys.argv[1:]
    if len(argumentos) >= 2 and argumentos[0].isdigit():
        keys = keys[int(argumentos[0]) :: int(argumentos[1])]

    one_city.encoder = DofaEncoder()
    failed = []
    for n, city in enumerate(keys, start=1):
        try:
            one_city(city)
        except Exception as e:
            failed.append(city)
            print(f"FAIL {city} ({n}/{len(keys)}): {type(e).__name__}: {e}", flush=True)
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
