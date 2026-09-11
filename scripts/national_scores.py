"""The national maps: every Mexican municipality scored with the saved heads.

The heads trained on the 771 bags are applied, unchanged, to every municipality with urban
tracts: the 138 cities of the partition, the expansion, and the municipalities beyond the
catalogue that were imaged for the maps and never entered training. Each municipality is
scored as one block with the 480 m neighbourhood drawn over its whole grid, by the three
seeds of the feature extractor of the environment, and every token keeps its seed-mean
score, the three seed scores, its tract, and the longitude and latitude of its centre.
One table per municipality under data/national, resumable, then one national table.

Usage: national_scores.py [key ...]   (default: every municipality with vectors on disk)
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
import rasterio  # noqa: E402
import rasterio.warp  # noqa: E402

from satinsight import backbone  # noqa: E402
from satinsight.agebs import cities_beyond, cities_extra  # noqa: E402
from satinsight.bagdata import load_city  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.dataset import paths  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.llp import build, instance_scores  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402
from satinsight.tiling import TOKEN_SIZE  # noqa: E402

SEEDS = (0, 1, 2)
RADIUS = 1
OUT = DATA_ROOT / "national"


def split_of() -> dict[str, str]:
    """Which part of the design each key belongs to: train, val, test, expansion or beyond."""
    partition = pd.read_csv("data/partition.csv")
    out = {}
    for split in ("train", "val", "test"):
        out.update(dict.fromkeys(cities_of(partition, split), split))
    out.update(dict.fromkeys(cities_extra(), "expansion"))
    out.update(dict.fromkeys(cities_beyond(), "beyond"))
    return out


def centres(key: str, y0: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of token centres from the optical composite's grid."""
    with rasterio.open(DATA_ROOT / "composites" / f"{key}_s2.tif") as origin:
        east, north = origin.transform * (x0 + TOKEN_SIZE / 2, y0 + TOKEN_SIZE / 2)
        lon, lat = rasterio.warp.transform(origin.crs, "EPSG:4326", east, north)
    return np.asarray(lon), np.asarray(lat)


def encoded_keys(sensor: str) -> list[str]:
    where = paths(DATA_ROOT)
    partner = "s1" + sensor[2:]
    keys = [p.stem[: -len(f"_{sensor}")] for p in where["vectors"].glob(f"*_{sensor}.npz")]
    return sorted(k for k in keys if (where["vectors"] / f"{k}_{partner}.npz").exists())


def main() -> int:
    import torch

    sensor = backbone.sensor("s2")
    keys = sys.argv[1:] or encoded_keys(sensor)
    splits = split_of()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    models, failed, done = [], [], 0
    for key in keys:
        out = OUT / f"{key}{backbone.SUFFIX}.parquet"
        if out.exists() and not sys.argv[1:]:
            continue
        if key not in splits:
            continue
        try:
            bags = load_city(key, sensor, fuse=True)
            if not bags:
                raise ValueError("no bags")
            if not models:
                dim = bags[0].instances.shape[1]
                for seed in SEEDS:
                    model = build(dim, radius=RADIUS, standardize=True).to(device)
                    model.load_state_dict(
                        torch.load(
                            f"data/weights/llp_final{backbone.SUFFIX}_s{seed}.pt",
                            map_location=device,
                        )
                    )
                    model.eval()
                    models.append(model)
            x = np.vstack([b.instances for b in bags])
            y0 = np.concatenate([b.y0 for b in bags])
            x0 = np.concatenate([b.x0 for b in bags])
            src, dst = adjacency(y0, x0, radius=RADIUS)
            xt = torch.from_numpy(x).float().to(device)
            src_t, dst_t = torch.from_numpy(src).to(device), torch.from_numpy(dst).to(device)
            scores = []
            with torch.inference_mode():
                for model in models:
                    _, per_instance = model(xt, src_t, dst_t)
                    scores.append(instance_scores(per_instance.cpu().numpy()))
            lon, lat = centres(key, y0, x0)
            table = pd.DataFrame(
                {
                    "key": key,
                    "split": splits[key],
                    "municipality": np.concatenate([[b.municipality] * len(b) for b in bags]),
                    "cvegeo": np.concatenate([b.cvegeo for b in bags]),
                    "y0": y0,
                    "x0": x0,
                    "lon": lon,
                    "lat": lat,
                    "score": np.mean(scores, axis=0),
                    **{f"score_s{seed}": s for seed, s in zip(SEEDS, scores, strict=True)},
                }
            )
            table.to_parquet(out, index=False)
            done += 1
            logging.info("%s: %d tokens, %d municipalities", key, len(table), len(bags))
        except Exception as e:
            failed.append(key)
            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
    tables = sorted(OUT.glob(f"*{backbone.SUFFIX}.parquet"))
    if backbone.SUFFIX == "":
        tables = [t for t in tables if "_" not in t.stem or t.stem in splits]
    national = pd.concat([pd.read_parquet(t) for t in tables], ignore_index=True)
    national.to_parquet(backbone.suffixed("data/national_scores.parquet"), index=False)
    print(
        f"END: {done} scored now, {national.key.nunique()} municipality boxes in the national "
        f"table, {len(national)} tokens, {len(failed)} failed {failed}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
