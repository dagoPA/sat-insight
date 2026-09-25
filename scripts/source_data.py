"""Exports per-figure source-data files, as Nature requires for every graph.

One CSV per figure panel and per feature extractor with the plotted values, named after
the figure and the extractor, gathered from the same artifacts the figures read: the
manuscript reports DOFA-B and DOFA-L side by side, so every file comes in both versions.
The manuscript's Data availability statement points here.

Usage: source_data.py
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from satinsight.manuscript import BACKBONES, block, suffixed

OUT = Path("docs/manuscript/source_data")
COPIED = (
    ("fig2b_validation", "data/granularity_curve.csv"),
    ("fig3b_sweep", "data/llp_sweep_r1.csv"),
    ("fig4a_rwi_val", "data/rwi_paired.csv"),
    ("fig4a_rwi_test", "data/rwi_paired_test.csv"),
    ("fig4b_conapo_val", "data/conapo_replication.csv"),
    ("fig4b_conapo_test", "data/conapo_replication_test.csv"),
    ("fig4c_targeting_val", "data/targeting.csv"),
    ("fig4c_targeting_test", "data/targeting_test.csv"),
    ("fig4d_transfer_training", "data/transfer_training.csv"),
    ("fig5c_conformal_coverage_by_city", "data/uncertainty_coverage_city.csv"),
    ("table8_maup", "data/maup.csv"),
    ("table3_products_val", "data/global_products.csv"),
    ("table3_products_test", "data/global_products_test.csv"),
    ("table6_conformal", "data/uncertainty_conformal.csv"),
    ("table7_city_audit", "data/city_audit_test.csv"),
    ("uncertainty_selective_curve", "data/uncertainty_selective.csv"),
    ("uncertainty_spread_vs_error", "data/uncertainty_spread.csv"),
)
"""Artifacts copied as they are, one per extractor, under the figure's name."""


def slug(name: str) -> str:
    return name.lower().replace("-", "_")


def curve_file(tag: str) -> str:
    return f"data/supervision_curve_s2_{tag}.csv" if tag else "data/supervision_curve.csv"


def export(tag: str, name: str, canon: dict) -> None:
    own = block(canon, tag)
    ext = slug(name)
    curve = pd.read_csv(curve_file(tag))
    curve.groupby("bags")[["spearman_within", "auroc_high", "bag_mae"]].agg(["mean", "std"]).to_csv(
        OUT / f"fig2a_validation_{ext}.csv"
    )
    pd.DataFrame(
        [{"bags": int(k), "spearman_within_test": v} for k, v in own["test"]["curve"].items()]
    ).to_csv(OUT / f"fig2a_test_{ext}.csv", index=False)
    pd.DataFrame([own["test"]["granularity"]]).to_csv(OUT / f"fig2b_test_{ext}.csv", index=False)
    pd.DataFrame([own["test"]["modality"]]).to_csv(OUT / f"fig2c_test_{ext}.csv", index=False)
    pd.DataFrame(
        [
            {"family": "attention MIL (ABMIL)", "map_auroc": own["abmil"]["map_auroc"]},
            {"family": "label proportions", "map_auroc": own["abmil"]["llp_kfold_auroc"]},
            {"family": "+ sweep + context", "map_auroc": own["curve"]["771"]["auroc_high"]},
            {"family": "oracle (instance labels)", "map_auroc": own["ceiling_auroc"]["1"]},
        ]
    ).to_csv(OUT / f"fig3a_families_{ext}.csv", index=False)
    for stem, src in COPIED:
        shutil.copy2(suffixed(src, tag), OUT / f"{stem}_{ext}.csv")
    bags = suffixed("data/transfer_training.csv", tag).replace(".csv", "_bags.csv")
    shutil.copy2(bags, OUT / f"fig4d_transfer_training_by_municipality_{ext}.csv")


def three_countries() -> None:
    """Figure 9: one row per municipality mapped, with its mean score, for DOFA-B."""
    sys.path.insert(0, "scripts")
    from fig9_three_countries import foreign_tokens, mexican_tokens, municipal

    _tag, name = BACKBONES[0]
    tokens = pd.concat([mexican_tokens(), foreign_tokens()], ignore_index=True)
    municipal(tokens).to_csv(OUT / f"fig9_municipalities_{slug(name)}.csv", index=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    canon = json.loads(Path("docs/manuscript/canonical_results.json").read_text())
    for tag, name in BACKBONES:
        export(tag, name, canon)
    # the zero-shot protocol table of the non-overlapping reference has no per-extractor
    # version; it is the one file that keeps the root artifact
    shutil.copy2("data/transfer_zeroshot.csv", OUT / "table5_zeroshot_protocol.csv")
    three_countries()
    print(f"{len(list(OUT.iterdir()))} source-data files in {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
