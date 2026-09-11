"""Tiles every Mexican municipality beyond the catalogue whose composites are on disk.

The geometry half of build_city only: 16 px tokens, bag tables and instance tables under
the base sensor names, so that backbone_extract.py can encode them with each feature
extractor exactly as it encodes the catalogue. Bags here are never trained on, so the
32-token floor of the training bags does not apply: every token that falls in a tract is
kept, the window grid lays a flush window against the right and bottom edges so that no
tract is left outside it, and a municipality of a single small tract still gets its map.
Resumable: a
municipality with both instance tables filled is skipped; an empty table left by an
earlier floor is rebuilt and its stale vectors and scores removed.

Usage: national_tiles.py
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

from satinsight.agebs import catalogue_with_extra, cities_beyond  # noqa: E402
from satinsight.dataset import build_city, paths  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402

MIN_INSTANCES = 1


def filled(path) -> bool:
    return path.exists() and len(pd.read_parquet(path, columns=["y0"])) > 0


def forget(key: str, where: dict) -> None:
    """Drops the vectors and scores derived from an instance table about to be rebuilt."""
    for old in [
        *where["vectors"].glob(f"{key}_s[12]_*.npz"),
        *(DATA_ROOT / "national").glob(f"{key}_*.parquet"),
    ]:
        old.unlink()


def main() -> int:
    beyond = cities_beyond()
    catalogue = {**catalogue_with_extra(), **beyond}
    where = paths(DATA_ROOT)
    composites = DATA_ROOT / "composites"
    failed, done, skipped = [], 0, 0
    for key in beyond:
        if not all((composites / f"{key}_{s}.tif").exists() for s in ("s2", "s1")):
            continue
        if all(filled(where["instances"] / f"{key}_{s}.parquet") for s in ("s2", "s1")):
            skipped += 1
            continue
        try:
            forget(key, where)
            for sensor in ("s2", "s1"):
                build_city(
                    key, sensor, catalogue=catalogue, min_instances=MIN_INSTANCES, flush=True
                )
            done += 1
            print(f"OK {key}", flush=True)
        except Exception as e:
            failed.append(key)
            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
    print(f"END: {done} tiled, {skipped} already tiled, {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
