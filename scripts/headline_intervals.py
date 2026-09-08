"""City-clustered bootstrap intervals of the headline map values, both units and backbones.

The per-token scores persisted for the validation and test cities give one
within-municipality Spearman per municipality (seed-mean score, per token and per AGEB);
the mean over municipalities is the headline value and its 95% interval resamples cities,
the unit of independence in the partition. Written to data/headline_intervals.csv and
quoted wherever the paper states a headline value.

Usage: headline_intervals.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from satinsight.agebs import load_grs  # noqa: E402
from satinsight.manuscript import BACKBONES, suffixed  # noqa: E402

sys.path.insert(0, "scripts")
from backbone_paired import clustered_interval, within  # noqa: E402


def per_seed_within(table: pd.DataFrame, grades: pd.Series) -> pd.DataFrame:
    """Within-municipality Spearman per seed, then averaged over seeds per municipality.

    This is the quantity the headline rows report (mean over seeds of per-seed values);
    the ensemble of seed-mean scores is a different and slightly higher number, reported
    beside it.
    """
    parts = []
    for seed, group in table.groupby("seed"):
        one = within(group, grades)
        one["seed"] = seed
        parts.append(one)
    stacked = pd.concat(parts)
    return stacked.groupby(["city", "municipality"])[["token", "ageb"]].mean().reset_index()


def main() -> None:
    grades = load_grs().set_index("cvegeo").ordinal.astype(float)
    rows = []
    for tag, name in BACKBONES:
        for split in ("val", "test"):
            scores = pd.read_parquet(suffixed(f"data/predictions_{split}.parquet", tag))
            for estimator, table in (
                ("seed mean", per_seed_within(scores, grades)),
                ("ensemble", within(scores, grades)),
            ):
                for unit in ("token", "ageb"):
                    values = table.dropna(subset=[unit])
                    low, high = clustered_interval(values[unit], values.city)
                    rows.append(
                        {
                            "backbone": name,
                            "split": split,
                            "estimator": estimator,
                            "unit": unit,
                            "municipalities": len(values),
                            "within": float(values[unit].mean()),
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv("data/headline_intervals.csv", index=False)
    print(out.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
