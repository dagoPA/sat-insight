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

from satinsight import encoders  # noqa: E402
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


def main() -> int:
    keys = sys.argv[1:] or composited_keys()
    encoder = encoders.DofaEncoder()
    out = DATA_ROOT / "transfer"
    failed = []
    for key in keys:
        tokens_path, vectors_path = out / f"tokens_{key}.parquet", out / f"vectors_{key}.npz"
        if tokens_path.exists() and vectors_path.exists():
            print(f"SKIP {key}", flush=True)
            continue
        try:
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
