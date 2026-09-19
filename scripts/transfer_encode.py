"""Frozen DOFA vectors of every token of the transfer boxes, saved for reuse.

The zero-shot run encoded Bogota and Rio and kept only their scores. Training in each
country on its own aggregates needs the vectors themselves, for six boxes now, and the
oracle upper bound and the folds re-read them many times. So they are extracted once here
and written like the Mexican ones: a token table with grid position and lon/lat of the
centre, and a half-precision matrix beside it.

Usage: transfer_encode.py [key ...]   (default: every box with both composites on disk)
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import pandas as pd  # noqa: E402

from satinsight import backbone, encoders  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402

sys.path.insert(0, "scripts")
from transfer_zeroshot import encode_city  # noqa: E402

HAND_KEYS = ("bogota", "medellin", "cali", "riodejaneiro", "saopaulo", "belohorizonte")


def composited_keys() -> list[str]:
    """Hand boxes plus every catalogued seat whose two composites are already on disk."""
    from transfer_composites import catalogued_boxes

    keys = [*HAND_KEYS, *catalogued_boxes()]
    root = DATA_ROOT / "composites"
    return [k for k in keys if (root / f"{k}_s2.tif").exists() and (root / f"{k}_s1.tif").exists()]


def one_grid(key: str) -> None:
    """Refuses a box whose two composites do not share shape and reference system.

    The fused vector of a token pairs the optical and the radar rows at one grid
    position, which only names the same ground when both composites were laid on one
    grid. A radar composite built in the neighbouring UTM zone differs by a few pixels
    and a projection's skew, and the pairing would go quietly wrong.
    """
    import rasterio

    grids = {}
    for sensor in ("s2", "s1"):
        with rasterio.open(DATA_ROOT / "composites" / f"{key}_{sensor}.tif") as source:
            grids[sensor] = (source.height, source.width, str(source.crs))
    if grids["s2"] != grids["s1"]:
        raise RuntimeError(f"{key}: composites on different grids {grids}; recomposite the radar")


def stale(key: str, tokens_path) -> bool:
    """Whether a saved token table was cut from a composite other than the one on disk.

    A box recomposited on a tighter or wider footprint keeps its old token table, whose
    grid positions then point at the wrong ground: 21 Amazonian seats came out with no
    built token at all that way. A table is stale when its positions reach past the
    raster or stop more than one token short of its far edges.
    """
    import rasterio

    from satinsight.tiling import TOKEN_SIZE

    with rasterio.open(DATA_ROOT / "composites" / f"{key}_s2.tif") as source:
        height, width = source.height, source.width
    table = pd.read_parquet(tokens_path, columns=["y0", "x0"])
    if table.empty:
        return True
    far_y, far_x = int(table.y0.max()) + TOKEN_SIZE, int(table.x0.max()) + TOKEN_SIZE
    beyond = far_y > height or far_x > width
    # with the flush window the last token ends within one token of each far edge
    short = far_y < height - TOKEN_SIZE or far_x < width - TOKEN_SIZE
    if beyond or short:
        print(
            f"STALE {key}: tokens reach ({far_y}, {far_x}) on a {height}x{width} raster", flush=True
        )
    return beyond or short


def main() -> int:
    keys = sys.argv[1:] or composited_keys()
    encoder = backbone.encoder()
    out = DATA_ROOT / "transfer"
    failed = []
    for key in keys:
        tokens_path, vectors_path = (
            out / f"tokens_{key}{backbone.SUFFIX}.parquet",
            out / f"vectors_{key}{backbone.SUFFIX}.npz",
        )
        if tokens_path.exists() and vectors_path.exists() and not stale(key, tokens_path):
            print(f"SKIP {key}", flush=True)
            continue
        try:
            one_grid(key)
            table, fused = encode_city(key, encoder)
            encoders.save(fused, vectors_path, y0=table.y0.to_numpy(), x0=table.x0.to_numpy())
            table.to_parquet(tokens_path, index=False)
            print(f"OK {key} · {len(table)} tokens · {fused.shape[1]} dims", flush=True)
        except Exception as e:
            failed.append(key)
            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
    print(f"END: {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
