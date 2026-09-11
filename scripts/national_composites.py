"""Composites for every Mexican municipality with urban tracts beyond the catalogue.

The national maps score all of urban Mexico with the heads trained on the 771 bags. The
138 cities and the expansion cover 430 municipalities; the 2,469 with urban tracts are
imaged here, both sensors, boxes drawn from their tracts as for the expansion. Nothing is
encoded or bagged: the vectors follow with backbone_extract.py for each feature
extractor, and none of these municipalities ever enters training. Resumable: a
composite already on disk is skipped. Splitting into processes takes index and total.

Usage: national_composites.py [index total [reverse]]

With `reverse` the process walks its share of the list from the end, so a forward and a
reverse process can share one index; each stops when it meets a municipality the other
has already attempted, which is read from the partner's progress table.
"""

import logging
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

from satinsight.agebs import catalogue_with_extra, cities_beyond  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.pipeline import city_aoi, ensure_composite  # noqa: E402

FREE_FLOOR_GB = 20


def main() -> int:
    beyond = cities_beyond()
    catalogue = {**catalogue_with_extra(), **beyond}
    keys = list(beyond)
    label = "run"
    reverse = len(sys.argv) == 4 and sys.argv[3] == "reverse"
    if len(sys.argv) >= 3 and sys.argv[1].isdigit():
        index, total = int(sys.argv[1]), int(sys.argv[2])
        keys, label = keys[index::total], f"process {index}"
    if reverse:
        keys, label = keys[::-1], f"{label} reverse"
    progress = Path(f"data/national_composites_{label.replace(' ', '_')}.csv")
    partner = Path(
        str(progress).replace("_reverse", "") if reverse else str(progress)[:-4] + "_reverse.csv"
    )
    print(f"{label}: {len(keys)} municipalities", flush=True)
    failed, rows = [], []
    for n, key in enumerate(keys, start=1):
        if partner.exists() and key in set(pd.read_csv(partner, dtype=str).key):
            print(f"MET the partner process at {key}; stopping", flush=True)
            break
        free = shutil.disk_usage(DATA_ROOT).free / 1024**3
        if free <= FREE_FLOOR_GB:
            print(f"STOP: {free:.1f} GB free is under the {FREE_FLOOR_GB} GB floor", flush=True)
            break
        start = time.time()
        try:
            area, agebs = city_aoi(key, catalogue=catalogue)
            for sensor in ("s2", "s1"):
                ensure_composite(key, sensor, area=area)
            rows.append(
                {
                    "key": key,
                    "status": "ok",
                    "municipality": beyond[key].municipality,
                    "agebs": len(agebs),
                }
            )
            minutes = (time.time() - start) / 60
            print(f"OK {key} ({n}/{len(keys)}) {len(agebs)} AGEB in {minutes:.1f} min", flush=True)
        except Exception as e:
            failed.append(key)
            rows.append({"key": key, "status": "fail", "municipality": beyond[key].municipality})
            print(f"FAIL {key} ({n}/{len(keys)}): {type(e).__name__}: {e}", flush=True)
        pd.DataFrame(rows).to_csv(progress, index=False)
    print(f"END {label}: {len(keys) - len(failed)} done, {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
