"""Zero-shot scoring of the transfer cities with the Mexican model, untouched.

The claim under test: what the model learned from Mexican aggregates is urban form, and
urban form travels. Nothing is retrained and nothing is recalibrated here, the three
final models score every token of Bogota and Rio exactly as they scored Guadalajara, and
whatever ordering comes out is compared against ground the model has never heard of.

The circuit mirrors the Mexican one without the label machinery: windows from the
composite, both sensors encoded, fusion by grid position, 480 m adjacency, the saved
heads, the seed-ensemble mean. Token centres are georeferenced through the composite's
affine transform so the evaluation can join them to label polygons.

Usage: transferencia_zeroshot.py [key ...]   (default: bogota riodejaneiro)
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
import rasterio.warp  # noqa: E402

from satinsight import encoders, tiling  # noqa: E402
from satinsight.cache import load  # noqa: E402
from satinsight.context import adjacency  # noqa: E402
from satinsight.dataset import CHANNELS  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.llp import build, instance_scores  # noqa: E402

SEEDS = (0, 1, 2)
RADIUS = 1
HALF_TOKEN_PX = tiling.TOKEN_SIZE // 2


def encode_city(key: str, encoder) -> tuple[pd.DataFrame, np.ndarray]:
    """Fused vectors of every token the two sensors share, with lon/lat of each centre."""
    vectors, positions, grid = {}, {}, None
    for sensor in ("s2", "s1"):
        bands, grid, _ = load(DATA_ROOT / "composites" / f"{key}_{sensor}.tif")
        bands = {c: bands[c] for c in CHANNELS[sensor]}
        windows = tiling.select(bands, min_valid_fraction=tiling.MIN_VALID_FRACTION)
        matrix, tokens = encoders.extract(bands, windows, encoder, order=CHANNELS[sensor])
        vectors[sensor] = matrix
        positions[sensor] = {(t.y0, t.x0): i for i, t in enumerate(tokens)}
        logging.info("%s/%s: %d tokens", key, sensor, len(tokens))

    shared = sorted(set(positions["s2"]) & set(positions["s1"]))
    fused = np.hstack(
        [
            vectors["s2"][[positions["s2"][p] for p in shared]],
            vectors["s1"][[positions["s1"][p] for p in shared]],
        ]
    )
    ys = np.array([p[0] + HALF_TOKEN_PX for p in shared])
    xs = np.array([p[1] + HALF_TOKEN_PX for p in shared])
    east, north = grid.transform * (xs + 0.5, ys + 0.5)
    lon, lat = rasterio.warp.transform(grid.crs, "EPSG:4326", east, north)
    table = pd.DataFrame(
        {
            "city": key,
            "y0": [p[0] for p in shared],
            "x0": [p[1] for p in shared],
            "lon": lon,
            "lat": lat,
        }
    )
    return table, fused


def main() -> None:
    import torch

    keys = sys.argv[1:] or ["bogota", "riodejaneiro"]
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    encoder = encoders.DofaEncoder()

    for key in keys:
        table, fused = encode_city(key, encoder)
        src, dst = adjacency(table.y0.to_numpy(), table.x0.to_numpy(), radius=RADIUS)
        src_t = torch.from_numpy(src).to(device)
        dst_t = torch.from_numpy(dst).to(device)
        x = torch.from_numpy(fused).float().to(device)

        scores = []
        for seed in SEEDS:
            model = build(fused.shape[1], radius=RADIUS, standardize=True).to(device)
            state = torch.load(f"data/weights/llp_final_s{seed}.pt", map_location=device)
            model.load_state_dict(state)
            model.eval()
            with torch.inference_mode():
                _, per_instance = model(x, src_t, dst_t)
            scores.append(instance_scores(per_instance.cpu().numpy()))
        table["score"] = np.mean(scores, axis=0)
        table.to_parquet(f"data/zeroshot_{key}.parquet", index=False)
        print(f"OK {key} · {len(table)} tokens scored", flush=True)


if __name__ == "__main__":
    main()
