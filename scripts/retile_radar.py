"""Retiles the radar of catalogue cities whose composite was rebuilt, base vectors included.

After the radar composite of a city is laid again on the optical grid, its radar instance
table and every radar vector are stale. This rebuilds the table and the base DOFA vectors
exactly as the national pipeline built them, with the same window grid and the same
32-token floor, so the catalogue keeps one convention; the other feature extractors
follow through backbone_extract.py, which re-encodes a city whose instances changed.

Usage: retile_radar.py <key ...>
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

from satinsight.agebs import catalogue_with_extra  # noqa: E402
from satinsight.dataset import build_city  # noqa: E402


def main() -> int:
    from satinsight.encoders import DofaEncoder

    catalogue = catalogue_with_extra()
    encoder = DofaEncoder()
    failed, done = [], 0
    for n, key in enumerate(sys.argv[1:], start=1):
        if key not in catalogue:
            print(f"SKIP {key}: not in the catalogue", flush=True)
            continue
        try:
            build_city(key, "s1", encoder=encoder, catalogue=catalogue, force=True)
            done += 1
            print(f"OK {key} ({n}/{len(sys.argv) - 1})", flush=True)
        except Exception as e:
            failed.append(key)
            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
    print(f"END: {done} retiled, {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
