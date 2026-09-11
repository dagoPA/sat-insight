"""Builds the composites for the transfer cities, with boxes derived from their labels.

The imaging half of the pipeline never assumed Mexico: it works from a WGS84 box and the
Planetary Computer STAC. What each foreign city needs is a box, and the honest source for
it is the label layer itself, the ground the evaluation will run on.

Bogota's box wraps the stratified blocks. Rio's wraps its AGSN polygons: they spread
across the whole municipality, so the box covers favelas and formal city alike, which is
exactly what a binary detection metric needs.

Usage: transfer_composites.py [key ...]   (default: the hand boxes plus every catalogued seat)
       transfer_composites.py <index> <total> [reverse]   (one of `total` interleaved
       processes; `reverse` walks the same share from the end and stops on meeting the
       forward process, read from its progress table)
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

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


BOXES_PATH = Path("data/transfer/boxes.csv")


def catalogued_boxes() -> dict[str, tuple[str, str, tuple[float, float, float, float]]]:
    """Boxes written by transfer_catalogue.py, one per municipal seat, keyed like the hand boxes."""
    if not BOXES_PATH.exists():
        return {}
    table = pd.read_csv(BOXES_PATH, dtype={"municipality": str})
    return {
        row.key: (row.name, row.country, tuple(float(v) for v in row.bbox.split()))
        for row in table.itertuples()
    }


def transfer_aoi(key: str) -> AOI:
    boxes = HAND_BOXES if key in HAND_BOXES else catalogued_boxes()
    if key in boxes:
        name, country, bbox = boxes[key]
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
    keys = sys.argv[1:] or ["bogota", "riodejaneiro", *HAND_BOXES, *catalogued_boxes()]
    progress = partner = None
    if len(sys.argv) >= 3 and sys.argv[1].isdigit() and sys.argv[2].isdigit():
        index, total = int(sys.argv[1]), int(sys.argv[2])
        keys = ["bogota", "riodejaneiro", *HAND_BOXES, *catalogued_boxes()][index::total]
        reverse = len(sys.argv) == 4 and sys.argv[3] == "reverse"
        if reverse:
            keys = keys[::-1]
        stem = f"data/transfer_composites_process_{index}"
        progress = Path(f"{stem}_reverse.csv" if reverse else f"{stem}.csv")
        partner = Path(f"{stem}.csv" if reverse else f"{stem}_reverse.csv")
        print(
            f"process {index} of {total}{' reverse' if reverse else ''}: {len(keys)} boxes",
            flush=True,
        )
    failed, rows = [], []
    for key in keys:
        if (
            partner is not None
            and partner.exists()
            and key in set(pd.read_csv(partner, dtype=str).key)
        ):
            print(f"MET the partner process at {key}; stopping", flush=True)
            break
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
        if progress is not None:
            rows.append({"key": key})
            pd.DataFrame(rows).to_csv(progress, index=False)
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
