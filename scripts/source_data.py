"""Exports per-figure source-data files, as Nature requires for every graph.

One CSV per figure panel with the plotted values, named after the figure, gathered from
the same artifacts the figures read. The manuscript's Data availability statement points
here.

Usage: source_data.py
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

OUT = Path("docs/manuscript/source_data")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    canon = json.loads(Path("docs/manuscript/canonical_results.json").read_text())

    curve = pd.read_csv("data/supervision_curve.csv")
    curve.groupby("bags")[["spearman_within", "auroc_high", "bag_mae"]].agg(["mean", "std"]).to_csv(
        OUT / "fig2a_validation.csv"
    )
    pd.DataFrame(
        [{"bags": int(k), "spearman_within_test": v} for k, v in canon["test"]["curve"].items()]
    ).to_csv(OUT / "fig2a_test.csv", index=False)
    pd.read_csv("data/granularity_curve.csv").to_csv(OUT / "fig2b_validation.csv", index=False)
    pd.DataFrame([canon["test"]["granularity"]]).to_csv(OUT / "fig2b_test.csv", index=False)
    pd.DataFrame([canon["test"]["modality"]]).to_csv(OUT / "fig2c_test.csv", index=False)
    pd.read_csv("data/llp_sweep_r1.csv").to_csv(OUT / "fig3b_sweep.csv", index=False)
    pd.DataFrame(
        [
            {"family": "attention MIL (ABMIL)", "map_auroc": canon["abmil"]["map_auroc"]},
            {"family": "label proportions", "map_auroc": canon["abmil"]["llp_kfold_auroc"]},
            {"family": "+ sweep + context", "map_auroc": canon["curve"]["771"]["auroc_high"]},
            {"family": "oracle (instance labels)", "map_auroc": canon["ceiling_auroc"]["1"]},
        ]
    ).to_csv(OUT / "fig3a_families.csv", index=False)
    for name, src in (
        ("fig4a_rwi_val.csv", "data/rwi_paired.csv"),
        ("fig4a_rwi_test.csv", "data/rwi_paired_test.csv"),
        ("fig4b_conapo_val.csv", "data/conapo_replication.csv"),
        ("fig4b_conapo_test.csv", "data/conapo_replication_test.csv"),
        ("fig4c_targeting_val.csv", "data/targeting.csv"),
        ("fig4c_targeting_test.csv", "data/targeting_test.csv"),
        ("fig4d_transfer_training.csv", "data/transfer_training.csv"),
        ("fig4d_transfer_training_by_municipality.csv", "data/transfer_training_bags.csv"),
        ("table5_zeroshot_protocol.csv", "data/transfer_zeroshot.csv"),
        ("fig5c_conformal_coverage_by_city.csv", "data/uncertainty_coverage_city.csv"),
        ("table8_maup.csv", "data/maup.csv"),
        ("table3_products_val.csv", "data/global_products.csv"),
        ("table3_products_test.csv", "data/global_products_test.csv"),
        ("table6_conformal.csv", "data/uncertainty_conformal.csv"),
        ("table7_city_audit.csv", "data/city_audit_test.csv"),
        ("uncertainty_selective_curve.csv", "data/uncertainty_selective.csv"),
        ("uncertainty_spread_vs_error.csv", "data/uncertainty_spread.csv"),
    ):
        shutil.copy2(src, OUT / name)
    print(f"{len(list(OUT.iterdir()))} source-data files in {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
