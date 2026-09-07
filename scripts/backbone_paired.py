"""Pairs an alternative backbone against the canonical one, municipality by municipality.

Both backbones score the same tokens of the same municipalities, so the honest comparison
is the difference inside each municipality, averaged over seeds, with a city-clustered
bootstrap of the mean. It is computed at both units the paper reports, per token and per
AGEB, on validation and on test, from the persisted per-token scores.

Usage: backbone_paired.py <suffix>   (for example dofal)
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from satinsight.agebs import load_grs  # noqa: E402

SUFFIX = sys.argv[1]
MIN_TOKENS = 20
MIN_TRACTS = 5


def within(table: pd.DataFrame, grades: pd.Series) -> pd.DataFrame:
    """Per-municipality Spearman of the seed-mean score against the grade, both units."""
    tokens = (
        table.groupby(["city", "municipality", "cvegeo", "y0", "x0"], observed=True)
        .score.mean()
        .reset_index()
    )
    tokens["grade"] = tokens.cvegeo.map(grades)
    tokens = tokens.dropna(subset=["grade"])
    rows = []
    for (city, municipality), group in tokens.groupby(["city", "municipality"]):
        if len(group) < MIN_TOKENS or group.grade.nunique() < 2:
            continue
        token_rho = float(spearmanr(group.score, group.grade).statistic)
        tract = group.groupby("cvegeo").agg(score=("score", "mean"), grade=("grade", "first"))
        tract_rho = (
            float(spearmanr(tract.score, tract.grade).statistic)
            if len(tract) >= MIN_TRACTS and tract.grade.nunique() > 1
            else np.nan
        )
        rows.append(
            {"city": city, "municipality": municipality, "token": token_rho, "ageb": tract_rho}
        )
    return pd.DataFrame(rows)


def clustered_interval(delta: pd.Series, cities: pd.Series, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    groups = [delta[cities == c].to_numpy() for c in cities.unique()]
    means = [
        np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean()
        for _ in range(2000)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    grades = load_grs().set_index("cvegeo").ordinal.astype(float)
    rows = []
    for split in ("val", "test"):
        base = within(pd.read_parquet(f"data/predictions_{split}.parquet"), grades)
        other = within(pd.read_parquet(f"data/predictions_{split}_{SUFFIX}.parquet"), grades)
        merged = base.merge(other, on=["city", "municipality"], suffixes=("_base", "_other"))
        for unit in ("token", "ageb"):
            pair = merged.dropna(subset=[f"{unit}_base", f"{unit}_other"])
            delta = pair[f"{unit}_other"] - pair[f"{unit}_base"]
            low, high = clustered_interval(delta, pair.city)
            rows.append(
                {
                    "split": split,
                    "unit": unit,
                    "municipalities": len(pair),
                    "base_within": float(pair[f"{unit}_base"].mean()),
                    "other_within": float(pair[f"{unit}_other"].mean()),
                    "difference": float(delta.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int((delta > 0).sum()),
                }
            )
            print(
                f"{split} · {unit}: base {rows[-1]['base_within']:+.3f} · {SUFFIX} "
                f"{rows[-1]['other_within']:+.3f} · Δ {delta.mean():+.3f} [{low:+.3f}, {high:+.3f}] "
                f"· {rows[-1]['wins']}/{len(pair)} municipalities",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(f"data/backbone_paired_{SUFFIX}.csv", index=False)


if __name__ == "__main__":
    main()
