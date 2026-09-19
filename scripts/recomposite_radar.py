"""Rebuilds the radar composite of Mexican cities on the grid of their optical composite.

The optical and the radar composites of a city on the edge of a UTM zone used to be laid
each in its own zone, a few pixels apart in shape and a projection's skew apart in
position, so the fused vector of a token paired ground that was not the same. The
compositing now lays the radar on the optical grid; this tool applies that to cities
already on disk: it drops the radar composite and builds it again inside the box the
optical composite was built for, read from that file's own tags so that both composites
declare one box. Resumable: a city whose two composites already share a grid is skipped.

Usage: recomposite_radar.py <key ...>
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import rasterio  # noqa: E402

from satinsight import cache  # noqa: E402
from satinsight.aoi import AOI  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.pipeline import ensure_composite  # noqa: E402


def grids_agree(key: str) -> bool:
    shapes = []
    for sensor in ("s2", "s1"):
        path = DATA_ROOT / "composites" / f"{key}_{sensor}.tif"
        if not path.exists():
            return False
        with rasterio.open(path) as source:
            shapes.append((source.height, source.width, str(source.crs)))
    return shapes[0] == shapes[1]


def main() -> int:
    keys = sys.argv[1:]
    failed, done = [], 0
    for n, key in enumerate(keys, start=1):
        if grids_agree(key):
            print(f"SKIP {key}: one grid already", flush=True)
            continue
        try:
            optical = DATA_ROOT / "composites" / f"{key}_s2.tif"
            _, _, tags = cache.load(optical)
            area = AOI(key=key, name=key, state="", bbox=tuple(float(v) for v in tags["bbox"]))
            radar = DATA_ROOT / "composites" / f"{key}_s1.tif"
            radar.unlink(missing_ok=True)
            ensure_composite(key, "s1", area=area)
            if not grids_agree(key):
                raise RuntimeError("the rebuilt radar still sits on another grid")
            done += 1
            print(f"OK {key} ({n}/{len(keys)})", flush=True)
        except Exception as e:
            failed.append(key)
            print(f"FAIL {key} ({n}/{len(keys)}): {type(e).__name__}: {e}", flush=True)
    print(f"END: {done} rebuilt, {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
