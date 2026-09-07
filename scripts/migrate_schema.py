"""Renames the data schema on disk from the Spanish column names to the English ones.

The library now writes `city`, `municipality`, `grade`, `population` and the English grade
labels. Tables built before that change carry `ciudad`, `municipio`, `grado`, `poblacion`
and CONEVAL's Spanish labels, and every loader would fail on them. This walks the tables
under `data/` and `dist/benchmark/` once, renames what it finds, and leaves files that are
already in the new schema untouched, so it is safe to rerun.

Usage: migrate_schema.py
"""

import sys
from pathlib import Path

import pandas as pd

from satinsight.agebs import CONEVAL_GRADES

COLUMNS = {
    "ciudad": "city",
    "municipio": "municipality",
    "grado": "grade",
    "poblacion": "population",
    "viviendas": "dwellings",
    "ordinal_continuo": "ordinal_continuous",
    "altos": "high_share",
    "clave": "key",
    "nombre": "name",
    "entidad": "state",
}
CONEVAL_COLUMNS = {
    "cve_ent": "state_key",
    "entidad": "state_name",
    "cve_mun": "municipality_key",
    "municipio": "municipality_name",
    "cve_loc": "locality_key",
    "localidad": "locality_name",
    "poblacion": "population",
    "viviendas": "dwellings",
    "grado": "grade",
}
BASELINE_COLUMNS = {
    "modalidad": "modality",
    "conjunto": "set",
    "exactitud": "accuracy",
    "auroc_muy_bajo": "auroc_very_low",
    "auroc_bajo": "auroc_low",
    "auroc_medio": "auroc_medium",
    "auroc_alto": "auroc_high",
    "auroc_muy_alto": "auroc_very_high",
    "auroc_acumulada": "auroc_cumulative_mean",
    "n_entrena": "n_train",
    "n_mide": "n_measured",
    "ciudades_mide": "cities_measured",
}
BASELINE_VALUES = {
    "modality": {"óptico": "optical", "fusión": "fused", "radar": "radar"},
    "set": {"cobertura": "cover", "densidad": "intensity", "textura": "texture", "completo": "all"},
}


def rename(table: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, bool]:
    present = {k: v for k, v in mapping.items() if k in table.columns}
    return table.rename(columns=present), bool(present)


def migrate_grades(table: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if "grade" in table.columns and table["grade"].isin(CONEVAL_GRADES).any():
        table["grade"] = table["grade"].map(lambda g: CONEVAL_GRADES.get(g, g))
        return table, True
    return table, False


def migrate(path: Path, mapping: dict[str, str]) -> bool:
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    table, changed = rename(table, mapping)
    table, regraded = migrate_grades(table)
    if changed or regraded:
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
    return changed or regraded


def migrate_baseline(path: Path) -> bool:
    table = pd.read_csv(path)
    columns = {}
    for column in table.columns:
        new = column
        for old, replacement in BASELINE_COLUMNS.items():
            if new == old:
                new = replacement
        new = new.replace("_ic_bajo", "_ci_low").replace("_ic_alto", "_ci_high")
        if new != column:
            columns[column] = new
    changed = bool(columns)
    table = table.rename(columns=columns)
    for column, values in BASELINE_VALUES.items():
        if column in table.columns and table[column].isin(values).any():
            table[column] = table[column].map(lambda v, values=values: values.get(v, v))
            changed = True
    if changed:
        table.to_csv(path, index=False)
    return changed


def main() -> int:
    roots = [Path("data"), Path("dist/benchmark")]
    touched = 0
    for root in roots:
        if not root.exists():
            continue
        for folder in ("instances", "bags"):
            for path in sorted((root / folder).glob("*.parquet")):
                touched += migrate(path, COLUMNS)
        for name in ("partition.csv", "cities_national.csv", "extensions_national.csv"):
            if (root / name).exists():
                touched += migrate(root / name, COLUMNS)
        if (root / "grs_ageb_2020.parquet").exists():
            touched += migrate(root / "grs_ageb_2020.parquet", CONEVAL_COLUMNS)
        for path in sorted(root.glob("features_*.parquet")):
            touched += migrate(path, COLUMNS)
        for name in ("baseline_val.csv", "baseline_test.csv", "model_val.csv"):
            if (root / name).exists():
                touched += migrate_baseline(root / name)
    print(f"{touched} tables migrated", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
