"""Re-encodes every tiled city with another frozen backbone, for the backbone ablation.

The weakly supervised head recovers most of what the DOFA base vectors support, so the
remaining room is in the vectors. This extracts the same tokens of the same cities with a
different foundation model and writes them under a suffixed sensor name, `s2_dofal` and
`s1_dofal` for DOFA large, `s2_cfm` and `s1_cfm` for Copernicus-FM, beside an instance
table identical to the base one, so every training and evaluation tool runs on the new
vectors with the sensor name as its only change.

Copernicus-FM also takes the location, date and footprint of each window; they come from
the composite grid, with the date fixed at the middle of the composited year.

Usage: backbone_extract.py <dofa_large|copernicusfm> [index total]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

from satinsight import encoders, tiling  # noqa: E402
from satinsight.cache import load  # noqa: E402
from satinsight.dataset import CHANNELS, paths  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402

BACKBONES = {
    "dofa_large": ("dofal", lambda: encoders.DofaEncoder("dofa_large_patch16_224")),
    "copernicusfm": ("cfm", lambda: encoders.CopernicusFmEncoder()),
}


def encode_city(city: str, sensor: str, tag: str, encoder) -> str:
    where = paths(DATA_ROOT)
    out_vectors = where["vectors"] / f"{city}_{sensor}_{tag}.npz"
    out_instances = where["instances"] / f"{city}_{sensor}_{tag}.parquet"
    if out_vectors.exists() and out_instances.exists():
        return "SKIP"
    instances = pd.read_parquet(where["instances"] / f"{city}_{sensor}.parquet")
    bands, grid, _ = load(DATA_ROOT / "composites" / f"{city}_{sensor}.tif")
    bands = {c: bands[c] for c in CHANNELS[sensor]}
    windows = tiling.select(bands, min_valid_fraction=tiling.MIN_VALID_FRACTION)
    metadata = (
        encoders.window_metadata(windows, grid)
        if getattr(encoder, "needs_metadata", False)
        else None
    )
    matrix, tokens = encoders.extract(
        bands, windows, encoder, order=CHANNELS[sensor], metadata=metadata
    )
    position = {(t.y0, t.x0): i for i, t in enumerate(tokens)}
    rows = [position[(y, x)] for y, x in zip(instances.y0, instances.x0, strict=True)]
    encoders.save(
        matrix[rows],
        out_vectors,
        y0=instances.y0.to_numpy(),
        x0=instances.x0.to_numpy(),
        cvegeo=instances.cvegeo.to_numpy(),
    )
    instances.to_parquet(out_instances, index=False)
    return f"OK {len(rows)} tokens · {matrix.shape[1]} dims"


def main() -> int:
    name = sys.argv[1]
    tag, make = BACKBONES[name]
    encoder = make()
    where = paths(DATA_ROOT)
    keys = sorted(p.stem[:-3] for p in where["instances"].glob("*_s2.parquet"))
    if len(sys.argv) >= 4 and sys.argv[2].isdigit():
        keys = keys[int(sys.argv[2]) :: int(sys.argv[3])]
    failed = []
    for n, city in enumerate(keys, start=1):
        for sensor in ("s2", "s1"):
            if not (where["instances"] / f"{city}_{sensor}.parquet").exists():
                continue
            try:
                result = encode_city(city, sensor, tag, encoder)
                print(f"{result} {city}/{sensor} ({n}/{len(keys)})", flush=True)
            except Exception as e:
                failed.append(f"{city}/{sensor}")
                print(
                    f"FAIL {city}/{sensor} ({n}/{len(keys)}): {type(e).__name__}: {e}", flush=True
                )
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
