"""Exports per-figure source-data files, as Nature requires for every graph.

One CSV per figure panel with the plotted values, named after the figure, gathered from
the same artifacts the figures read. The manuscript's Data availability statement points
here.

Usage: datos_fuente.py
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

OUT = Path("docs/manuscript/source_data")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    canon = json.loads(Path("data/canon_manuscrito.json").read_text())

    curve = pd.read_csv("data/curva_supervision.csv")
    curve.groupby("bags")[["spearman_within", "auroc_high", "bag_mae"]].agg(["mean", "std"]).to_csv(
        OUT / "fig2a_validation.csv"
    )
    pd.DataFrame(
        [{"bags": int(k), "spearman_within_test": v} for k, v in canon["test"]["curve"].items()]
    ).to_csv(OUT / "fig2a_test.csv", index=False)
    pd.read_csv("data/curva_granularidad.csv").to_csv(OUT / "fig2b_validation.csv", index=False)
    pd.DataFrame([canon["test"]["granularity"]]).to_csv(OUT / "fig2b_test.csv", index=False)
    pd.DataFrame([canon["test"]["modality"]]).to_csv(OUT / "fig2c_test.csv", index=False)
    pd.read_csv("data/barrido_llp_r1.csv").to_csv(OUT / "fig3b_sweep.csv", index=False)
    for name, src in (
        ("fig4a_rwi_val.csv", "data/incumbente_pareado.csv"),
        ("fig4a_rwi_test.csv", "data/incumbente_pareado_test.csv"),
        ("fig4b_conapo_val.csv", "data/replicacion_imu.csv"),
        ("fig4b_conapo_test.csv", "data/replicacion_imu_test.csv"),
        ("fig4c_targeting_val.csv", "data/focalizacion.csv"),
        ("fig4c_targeting_test.csv", "data/focalizacion_test.csv"),
        ("fig4d_zeroshot.csv", "data/transferencia_zeroshot.csv"),
        ("supp_table2_maup.csv", "data/maup.csv"),
    ):
        shutil.copy2(src, OUT / name)
    print(f"{len(list(OUT.iterdir()))} source-data files in {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
