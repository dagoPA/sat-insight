"""C5: packages the frozen benchmark, vectors, labels, splits, protocol, for release.

The citation magnet is a bundle any lab can train against in minutes with no satellite
pipeline: the frozen foundation-model vectors per city, the bag labels, the AGEB grades
the maps are scored against, the partition, and a manifest with hashes so a result can
name exactly what it ran on.

Usage: empaquetar_benchmark.py [destination]
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from satinsight.dataset import paths
from satinsight.download import DATA_ROOT

DESTINATION = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist/benchmark")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    where = paths(DATA_ROOT)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {"files": {}}

    for folder, pattern in (
        ("vectors", "*.npz"),
        ("instances", "*.parquet"),
        ("bags", "*.parquet"),
    ):
        out = DESTINATION / folder
        out.mkdir(exist_ok=True)
        for f in sorted(where[folder].glob(pattern)):
            shutil.copy2(f, out / f.name)
            manifest["files"][f"{folder}/{f.name}"] = digest(f)

    for name in ("partition.csv", "grs_ageb_2020.parquet"):
        source = DATA_ROOT / name
        shutil.copy2(source, DESTINATION / name)
        manifest["files"][name] = digest(source)

    partition = pd.read_csv(DATA_ROOT / "partition.csv")
    manifest["cities"] = {
        split: sorted(partition[partition.split == split].ciudad) for split in ("train", "val")
    }
    manifest["protocol"] = {
        "evaluation": "held-out cities only; the test split stays closed until final",
        "map_metrics": [
            "Spearman within municipality, averaged over bags with >=20 instances and >1 grade",
            "AUROC for grade >= 3, pooled per city then averaged",
        ],
        "ceiling": "instance-supervised oracle on the same vectors; report recovery fraction",
        "comparisons": "paired by municipality across configurations, 3 seeds minimum",
    }
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in DESTINATION.rglob("*") if f.is_file()) / 1024**3
    print(f"{len(manifest['files'])} files · {total:.1f} GB → {DESTINATION}")


if __name__ == "__main__":
    main()
