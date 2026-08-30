"""Downloads and processes the municipalities beyond the 138 of the national set.

Usage: zonas_extra.py [index total]

The selected 138 are the well-off half of urban Mexico: 14.4% of their AGEB sit at high
grade against 30.5% of everything left out. That biases every number the project has
produced, and it also makes the expansion the cleanest domain shift available — the label
definition, the census and the sensor all stay fixed, and only the urban form changes.

Each municipality runs the whole chain: the analysis box from its AGEB, the annual median
composites of both sensors, the tiling into bags, and the frozen encoder over the surviving
tokens. A municipality that fails leaves the rest running and is named at the end.

The run takes days. It lives in the repository and not in a scratchpad, because a scratch
directory that gets cleaned mid-run leaves the work failing in silence against a file that
no longer exists. Splitting it into processes takes the index and total, as the national
composite run did.
"""

import logging
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

from satinsight.agebs import catalogue_with_extra, cities_extra  # noqa: E402
from satinsight.dataset import build_city  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.pipeline import city_aoi, ensure_composite  # noqa: E402

FREE_FLOOR_GB = 20
"""What the run leaves untouched on the volume.

The guard used to be a 30 GB budget for the data directory. The budget was the constraint
while it was the binding one; with a terabyte free, stopping a multi-day run because the
project passed an arbitrary line would throw away work for nothing. What matters is the
volume filling up, so the guard watches free space instead. It stops and says so, and
deletes nothing: which city to drop is not a decision this script gets to make.
"""


def disk_used_gb() -> float:
    return sum(f.stat().st_size for f in DATA_ROOT.rglob("*") if f.is_file()) / 1024**3


def disk_free_gb() -> float:
    usage = shutil.disk_usage(DATA_ROOT)
    return usage.free / 1024**3


def main() -> int:
    from satinsight.encoders import DofaEncoder

    catalogue = catalogue_with_extra()
    extra = cities_extra()
    keys = list(extra)
    argumentos = sys.argv[1:]
    if len(argumentos) >= 2 and argumentos[0].isdigit():
        index, total = int(argumentos[0]), int(argumentos[1])
        keys, label = keys[index::total], f"process {index}"
    else:
        label = "run"

    encoder = DofaEncoder()
    print(
        f"{label}: {len(keys)} municipalities · "
        f"{disk_used_gb():.1f} GB used · {disk_free_gb():.1f} GB free",
        flush=True,
    )

    failed, done, rows = [], 0, []
    for n, key in enumerate(keys, start=1):
        free = disk_free_gb()
        if free <= FREE_FLOOR_GB:
            print(f"STOP: {free:.1f} GB free is under the {FREE_FLOOR_GB} GB floor", flush=True)
            break
        start = time.time()
        try:
            area, agebs = city_aoi(key, catalogue=catalogue)
            for sensor in ("s2", "s1"):
                ensure_composite(key, sensor, area=area)
            for sensor in ("s2", "s1"):
                build_city(key, sensor, encoder=encoder, catalogue=catalogue)
            done += 1
            rows.append({"key": key, "municipality": extra[key].municipality, "agebs": len(agebs)})
            pd.DataFrame(rows).to_csv(
                f"data/zonas_extra_{label.replace(' ', '_')}.csv", index=False
            )
            print(
                f"OK {key} ({n}/{len(keys)}) {len(agebs)} AGEB "
                f"in {(time.time() - start) / 60:.1f} min · {disk_used_gb():.1f} GB",
                flush=True,
            )
        except Exception as e:
            failed.append(key)
            print(f"FAIL {key} ({n}/{len(keys)}): {type(e).__name__}: {e}", flush=True)

    print(f"END {label}: {done} done, {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
