"""Pooled cross-validation metrics per backbone, with city-clustered bootstrap intervals.

Reads the per-token scores that backbone_cv.py persisted for every held-out city, averages
the three seeds per token, and reports for each backbone the standard map metrics over the
pooled predictions of all 138 cities: within-municipality Spearman per token and per
AGEB (mean over municipalities), pooled Spearman and AUROC for high deprivation. Every
interval is a 95% percentile bootstrap that resamples cities, the unit of independence in
the design; the difference between two backbones is bootstrapped on the same city draws,
so its interval is the standard paired one.

Usage: backbone_cv_compare.py [suffix ...]   (the first suffix is the reference the
       differences are paired against; default: base, dofal and cfm where present)
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from satinsight.agebs import load_grs  # noqa: E402

sys.path.insert(0, "scripts")
from backbone_paired import within  # noqa: E402

NAMES = {
    "": "DOFA base",
    "dofal": "DOFA large",
    "cfm": "Copernicus-FM base",
    "ov": "DOFA base, seam-free",
    "dofalov": "DOFA large, seam-free",
}
RESAMPLES = 2000
HIGH = 3
"""Grades High and Very high count as high deprivation, as everywhere in the paper."""


def tokens_of(suffix: str, grades: pd.Series) -> pd.DataFrame:
    table = pd.read_parquet(f"data/cv_predictions{'_' + suffix if suffix else ''}.parquet")
    tokens = (
        table.groupby(["city", "municipality", "cvegeo", "y0", "x0"], observed=True)
        .score.mean()
        .reset_index()
    )
    tokens = tokens.drop_duplicates(["cvegeo", "y0", "x0"])
    tokens["grade"] = tokens.cvegeo.map(grades)
    return tokens.dropna(subset=["grade"])


def pooled(tokens: pd.DataFrame) -> dict:
    return {
        "spearman_pooled": float(spearmanr(tokens.score, tokens.grade).statistic),
        "auroc_high": float(roc_auc_score(tokens.grade >= HIGH, tokens.score)),
    }


def bootstrap(per_city: dict[str, np.ndarray], draws: np.ndarray) -> np.ndarray:
    """Mean over municipalities of the resampled cities, one value per draw."""
    keys = sorted(per_city)
    groups = [per_city[k] for k in keys]
    return np.array([np.concatenate([groups[j] for j in draw]).mean() for draw in draws])


def main() -> None:
    suffixes = sys.argv[1:] or [
        s
        for s in ("", "dofal", "cfm")
        if Path(f"data/cv_predictions{'_' + s if s else ''}.parquet").exists()
    ]
    suffixes = ["" if s == "base" else s for s in suffixes]
    grades = load_grs().set_index("cvegeo").ordinal.astype(float)
    per_backbone = {s: within(tokens_of(s, grades), grades) for s in suffixes}
    cities = sorted(set.intersection(*(set(t.city) for t in per_backbone.values())))
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(cities), size=(RESAMPLES, len(cities)))

    rows = []
    for suffix, table in per_backbone.items():
        table = table[table.city.isin(cities)]
        tokens = tokens_of(suffix, grades)
        row = {"backbone": NAMES[suffix], "cities": len(cities), "municipalities": len(table)}
        for unit in ("token", "ageb"):
            values = table.dropna(subset=[unit])
            per_city = {c: g[unit].to_numpy() for c, g in values.groupby("city")}
            means = bootstrap({c: per_city.get(c, np.array([])) for c in cities}, draws)
            row[f"within_{unit}"] = float(values[unit].mean())
            row[f"within_{unit}_low"], row[f"within_{unit}_high"] = np.percentile(
                means, [2.5, 97.5]
            )
        row.update(pooled(tokens[tokens.city.isin(cities)]))
        rows.append(row)
    summary = pd.DataFrame(rows)

    diffs = []
    reference = suffixes[0]
    base = per_backbone.get(reference)
    for suffix, table in per_backbone.items():
        if suffix == reference or base is None:
            continue
        merged = base.merge(table, on=["city", "municipality"], suffixes=("_base", "_other"))
        for unit in ("token", "ageb"):
            pair = merged.dropna(subset=[f"{unit}_base", f"{unit}_other"])
            delta = pair[f"{unit}_other"] - pair[f"{unit}_base"]
            per_city = {c: np.array([]) for c in cities}
            per_city.update({c: d.to_numpy() for c, d in delta.groupby(pair.city)})
            means = bootstrap(per_city, draws)
            low, high = np.percentile(means, [2.5, 97.5])
            diffs.append(
                {
                    "comparison": f"{NAMES[suffix]} minus {NAMES[reference]}",
                    "unit": unit,
                    "municipalities": len(pair),
                    "base": float(pair[f"{unit}_base"].mean()),
                    "other": float(pair[f"{unit}_other"].mean()),
                    "difference": float(delta.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    differences = pd.DataFrame(diffs)
    summary.to_csv("data/cv_backbones.csv", index=False)
    differences.to_csv("data/cv_backbone_differences.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print(differences.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
