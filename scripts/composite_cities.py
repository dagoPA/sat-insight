"""Composites one slice of the national set. Usage: composite_cities.py [index total] [city ...]

It lives in the repository rather than in a temporary directory: the run lasts days, and a
scratchpad that gets cleaned leaves the work failing silently against a file that no
longer exists.
"""

import logging
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

from satinsight.agebs import cities_by_size  # noqa: E402
from satinsight.pipeline import city_aoi, ensure_composite  # noqa: E402

catalogue = cities_by_size(stratify=True)
arguments = sys.argv[1:]
if arguments and arguments[0].isdigit():
    index, total = int(arguments[0]), int(arguments[1])
    mine = list(catalogue)[index::total]
    label = f"process {index}"
else:
    mine = arguments or list(catalogue)
    label = "run"

print(f"{label}: {len(mine)} cities", flush=True)
failed = []
for n, key in enumerate(mine, start=1):
    start = time.time()
    try:
        area, agebs = city_aoi(key, catalogue=catalogue)
        for sensor in ("s2", "s1"):
            ensure_composite(key, sensor, area=area)
        minutes = (time.time() - start) / 60
        print(f"OK {key} ({n}/{len(mine)}) {len(agebs)} AGEB in {minutes:.1f} min", flush=True)
    except Exception as e:
        failed.append(key)
        print(f"FAIL {key} ({n}/{len(mine)}): {type(e).__name__}: {e}", flush=True)

print(f"END {label}: {len(failed)} failed {failed}", flush=True)
# the exit code tells a complete run from one that left cities behind, so a retry loop
# knows whether to call it again
sys.exit(1 if failed else 0)
