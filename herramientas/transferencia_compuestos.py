"""Builds the composites for the transfer cities, with boxes derived from their labels.

The imaging half of the pipeline never assumed Mexico: it works from a WGS84 box and the
Planetary Computer STAC. What each foreign city needs is a box, and the honest source for
it is the label layer itself, the ground the evaluation will run on.

Bogota's box wraps the stratified blocks. Rio's wraps its AGSN polygons: they spread
across the whole municipality, so the box covers favelas and formal city alike, which is
exactly what a binary detection metric needs.

Usage: transferencia_compuestos.py [key ...]   (default: both)
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

from satinsight.aoi import AOI  # noqa: E402
from satinsight.pipeline import ensure_composite  # noqa: E402
from satinsight.transfer import agsn_of_city, bogota_strata  # noqa: E402

MARGIN_M = 2000


HAND_BOXES = {
    # metro training boxes for the tri-country atlas; the box just has to cover the
    # conurbation, municipal assignment comes later from each country's boundaries
    "medellin": ("Valle de Aburr\u00e1", "Colombia", (-75.70, 6.05, -75.20, 6.55)),
    "cali": ("Cali y alrededores", "Colombia", (-76.60, 3.30, -76.24, 3.56)),
    "saopaulo": ("S\u00e3o Paulo metro", "Brazil", (-46.85, -23.75, -46.30, -23.35)),
    "belohorizonte": ("Belo Horizonte metro", "Brazil", (-44.20, -20.05, -43.85, -19.75)),
}


def transfer_aoi(key: str) -> AOI:
    if key in HAND_BOXES:
        name, country, bbox = HAND_BOXES[key]
        return AOI(key=key, name=name, state=country, bbox=bbox)
    if key == "bogota":
        return AOI.from_polygons(
            "bogota", "Bogotá D.C.", "Colombia", bogota_strata(), margin_m=MARGIN_M
        )
    if key == "riodejaneiro":
        return AOI.from_polygons(
            "riodejaneiro",
            "Rio de Janeiro",
            "Brazil",
            agsn_of_city("Rio de Janeiro"),
            margin_m=MARGIN_M,
        )
    raise KeyError(f"unknown transfer key {key!r}")


def main() -> int:
    keys = sys.argv[1:] or ["bogota", "riodejaneiro"]
    failed = []
    for key in keys:
        area = transfer_aoi(key)
        shape = area.approximate_shape()
        print(f"{key}: bbox {tuple(round(v, 3) for v in area.bbox)} · ~{shape} px", flush=True)
        for sensor in ("s2", "s1"):
            try:
                ensure_composite(key, sensor, area=area)
                print(f"OK {key}/{sensor}", flush=True)
            except Exception as e:
                failed.append(f"{key}/{sensor}")
                print(f"FAIL {key}/{sensor}: {type(e).__name__}: {e}", flush=True)
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
