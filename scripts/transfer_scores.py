"""The Mexican heads applied to every composited box of Colombia and Brazil.

The maps abroad that the paper shows for Bogota and Rio, drawn for the whole catalogue:
each box's built tokens are scored by the three saved heads of the feature extractor of
the environment, unadapted, and every token keeps its seed-mean score beside its
municipality, the local fine truth where there is one, and the longitude and latitude of
its centre. One table per box under data/transfer/scores, resumable, then one table per
country. This is the zero-shot arm; the heads trained on each country's own aggregates
come from transfer_train.py.

Usage: transfer_scores.py [key ...]   (default: every box with bags on disk)
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from satinsight import backbone  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.encoders import load  # noqa: E402
from satinsight.llp import build, instance_scores  # noqa: E402

SEEDS = (0, 1, 2)
RADIUS = 1
OUT = DATA_ROOT / "transfer" / "scores"
KEEP = ["key", "country", "municipality", "lon", "lat", "built", "truth", "unit"]


def bagged_keys() -> list[str]:
    root = DATA_ROOT / "transfer"
    tail = f"{backbone.SUFFIX}.parquet"
    return sorted(p.name[len("bags_") : -len(tail)] for p in root.glob(f"bags_*{tail}"))


def heads(dim: int, torch, device) -> list:
    made = []
    for seed in SEEDS:
        head = build(dim, radius=RADIUS, standardize=True).to(device)
        head.load_state_dict(
            torch.load(f"data/weights/llp_final{backbone.SUFFIX}_s{seed}.pt", map_location=device)
        )
        head.eval()
        made.append(head)
    return made


def main() -> int:
    import torch

    keys = sys.argv[1:] or bagged_keys()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    made, failed, done = [], [], 0
    for key in keys:
        out = OUT / f"{key}{backbone.SUFFIX}.parquet"
        if out.exists() and not sys.argv[1:]:
            continue
        try:
            bags = pd.read_parquet(DATA_ROOT / "transfer" / f"bags_{key}{backbone.SUFFIX}.parquet")
            if bags.empty:
                continue
            matrix, _ = load(DATA_ROOT / "transfer" / f"vectors_{key}{backbone.SUFFIX}.npz")
            if not made:
                made = heads(matrix.shape[1], torch, device)
            x = torch.from_numpy(matrix[bags.row.to_numpy()]).float().to(device)
            src, dst = adjacency(bags.y0.to_numpy(), bags.x0.to_numpy(), radius=RADIUS)
            src_t, dst_t = torch.from_numpy(src).to(device), torch.from_numpy(dst).to(device)
            scores = []
            with torch.inference_mode():
                for head in made:
                    _, per_instance = head(x, src_t, dst_t)
                    scores.append(instance_scores(per_instance.cpu().numpy()))
            table = bags[[c for c in KEEP if c in bags.columns]].copy()
            table["key"] = key
            table["score"] = np.mean(scores, axis=0)
            table.to_parquet(out, index=False)
            done += 1
            logging.info("%s: %d tokens scored", key, len(table))
        except Exception as e:
            failed.append(key)
            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
    tables = sorted(OUT.glob(f"*{backbone.SUFFIX}.parquet"))
    if tables:
        national = pd.concat([pd.read_parquet(t) for t in tables], ignore_index=True)
        national.to_parquet(backbone.suffixed("data/transfer_scores.parquet"), index=False)
        counts = national.groupby("country").key.nunique().to_dict()
        print(
            f"END: {done} scored now, {len(tables)} boxes in the country tables "
            f"({counts}), {len(national)} tokens, {len(failed)} failed {failed[:5]}",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
