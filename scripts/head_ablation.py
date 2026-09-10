"""Head ablation: the three parameterisations of the instance head under one protocol.

The reference head predicts four cumulative shares with independent thresholds; the CORAL
head shares one weight vector across the thresholds, so its cumulative probabilities are
ordered; the softmax head predicts five class probabilities and is trained with
cross-entropy on the class shares of the bag. All three are scored exactly as the rest of
the paper: within-municipality Spearman of the seed-mean token score, per token and per
AGEB, on the validation and test cities with city-clustered bootstrap intervals; the
classification view (quadratic-weighted kappa of the isotonic grade read from the
validation tracts) on both splits; the bag error and AUROC on test; and grouped
cross-validation over the 138 cities with the difference against the reference head
bootstrapped on the same city draws. Written to data/head_ablation.csv.

Usage: head_ablation.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from satinsight.agebs import load_grs  # noqa: E402
from satinsight.manuscript import BACKBONES  # noqa: E402

sys.path.insert(0, "scripts")
from backbone_cv_compare import RESAMPLES, bootstrap, pooled, tokens_of  # noqa: E402
from backbone_paired import clustered_interval, within  # noqa: E402
from headline_intervals import per_seed_within  # noqa: E402
from uncertainty import per_ageb, with_truth  # noqa: E402

HEADS = (("", "cumulative"), ("_coral", "coral"), ("_softmax", "softmax"))
"""File suffix and name of each head; the empty suffix is the reference head."""


def split_rows(suffix: str, grades: pd.Series) -> list[dict]:
    """Validation and test rows: headline values, classification view, test bag error."""
    if not Path(f"data/predictions_test{suffix}.parquet").exists():
        return []
    rows = []
    frames = {
        split: with_truth(per_ageb(f"data/predictions_{split}{suffix}.parquet"))
        for split in ("val", "test")
    }
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        frames["val"].score, frames["val"].ordinal
    )
    test_runs = pd.read_csv(f"data/backbone_test{suffix}.csv")
    for split in ("val", "test"):
        scores = pd.read_parquet(f"data/predictions_{split}{suffix}.parquet")
        table = per_seed_within(scores, grades)
        row = {"split": split}
        for unit in ("token", "ageb"):
            values = table.dropna(subset=[unit])
            low, high = clustered_interval(values[unit], values.city)
            row[f"within_{unit}"] = float(values[unit].mean())
            row[f"within_{unit}_low"], row[f"within_{unit}_high"] = low, high
        frame = frames[split]
        predicted = np.rint(isotonic.predict(frame.score)).astype(int)
        row["kappa_quadratic"] = cohen_kappa_score(frame.ordinal, predicted, weights="quadratic")
        row["accuracy"] = float((predicted == frame.ordinal).mean())
        if split == "test":
            row["bag_mae"] = float(test_runs.bag_mae.mean())
            row["auroc_high"] = float(test_runs.auroc_high.mean())
        rows.append(row)
    return rows


def cv_row(suffix: str, reference: str, grades: pd.Series) -> dict | None:
    """Pooled cross-validation values and the paired difference against the reference head."""
    path = f"data/cv_predictions{suffix}.parquet"
    if not Path(path).exists():
        return None
    table = within(tokens_of(suffix.lstrip("_"), grades), grades)
    base = within(tokens_of(reference.lstrip("_"), grades), grades)
    cities = sorted(set(table.city) & set(base.city))
    draws = np.random.default_rng(0).integers(0, len(cities), size=(RESAMPLES, len(cities)))
    row = {"split": "cv", "municipalities": int(table.city.isin(cities).sum())}
    for unit in ("token", "ageb"):
        values = table[table.city.isin(cities)].dropna(subset=[unit])
        per_city = {c: np.array([]) for c in cities}
        per_city.update({c: g[unit].to_numpy() for c, g in values.groupby("city")})
        means = bootstrap(per_city, draws)
        row[f"within_{unit}"] = float(values[unit].mean())
        row[f"within_{unit}_low"], row[f"within_{unit}_high"] = np.percentile(means, [2.5, 97.5])
        if suffix != reference:
            merged = base.merge(table, on=["city", "municipality"], suffixes=("_base", "_other"))
            pair = merged.dropna(subset=[f"{unit}_base", f"{unit}_other"])
            delta = pair[f"{unit}_other"] - pair[f"{unit}_base"]
            per_city = {c: np.array([]) for c in cities}
            per_city.update({c: d.to_numpy() for c, d in delta.groupby(pair.city)})
            means = bootstrap(per_city, draws)
            row[f"diff_{unit}"] = float(delta.mean())
            row[f"diff_{unit}_low"], row[f"diff_{unit}_high"] = np.percentile(means, [2.5, 97.5])
    tokens = tokens_of(suffix.lstrip("_"), grades)
    row.update(pooled(tokens[tokens.city.isin(cities)]))
    return row


def main() -> None:
    grades = load_grs().set_index("cvegeo").ordinal.astype(float)
    rows = []
    for tag, name in BACKBONES:
        reference = f"_{tag}"
        for head_suffix, head in HEADS:
            suffix = f"_{tag}{head_suffix}"
            found = split_rows(suffix, grades)
            cv = cv_row(suffix, reference, grades)
            if cv is not None:
                found.append(cv)
            for row in found:
                rows.append({"features": name, "head": head, **row})
    out = pd.DataFrame(rows)
    out.to_csv("data/head_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
