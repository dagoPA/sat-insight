"""The map read as a five-grade classifier, for readers of that literature.

The paper's metric is the within-municipality rank correlation, which cannot borrow the
between-city gradient. This computes the classification quantities such readers expect,
for both feature extractors and both splits: the score is turned into a grade by the
isotonic map fitted on the validation tracts (the same map the conformal intervals use),
and scored per tract with quadratic-weighted Cohen's kappa and accuracy over the five
grades; the score itself is scored with the AUROC of each cumulative threshold (grade at
least k), averaged over the four thresholds, pooled over the split and averaged over
municipalities with at least twenty tracts. Written to data/classification_view.csv.

Usage: classification_view.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score  # noqa: E402

from satinsight.manuscript import BACKBONES, suffixed  # noqa: E402

sys.path.insert(0, "scripts")
from uncertainty import per_ageb, with_truth  # noqa: E402

MIN_TRACTS = 20
THRESHOLDS = (1, 2, 3, 4)


def threshold_aurocs(frame: pd.DataFrame) -> list[float]:
    return [
        roc_auc_score(frame.ordinal >= k, frame.score)
        for k in THRESHOLDS
        if 0 < int((frame.ordinal >= k).sum()) < len(frame)
    ]


def main() -> None:
    rows = []
    for tag, name in BACKBONES:
        frames = {
            split: with_truth(per_ageb(suffixed(f"data/predictions_{split}.parquet", tag)))
            for split in ("val", "test")
        }
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(
            frames["val"].score, frames["val"].ordinal
        )
        for split, frame in frames.items():
            predicted = np.rint(isotonic.predict(frame.score)).astype(int)
            within = [
                np.mean(threshold_aurocs(group))
                for _, group in frame.groupby("municipality", observed=True)
                if len(group) >= MIN_TRACTS and threshold_aurocs(group)
            ]
            rows.append(
                {
                    "features": name,
                    "split": split,
                    "tracts": len(frame),
                    "kappa_quadratic": cohen_kappa_score(
                        frame.ordinal, predicted, weights="quadratic"
                    ),
                    "accuracy": float((predicted == frame.ordinal).mean()),
                    "auroc_thresholds_pooled": float(np.mean(threshold_aurocs(frame))),
                    "auroc_thresholds_within": float(np.mean(within)),
                    "municipalities": len(within),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv("data/classification_view.csv", index=False)
    print(table.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
