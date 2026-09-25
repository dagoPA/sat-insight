"""Writes the cross-validation block of the canonical results file from the comparison.

backbone_cv_compare.py leaves two tables under data/: one row per feature extractor with
the pooled cross-validation metrics and their city bootstrap intervals, and one row per
paired difference against the reference it was run with. The manuscript's Table 3 and
Figure 3 read those numbers from the canon, so this copies them there, rounded to four
decimals like every other block, under book["backbone_cv"]. The block used to be pasted
by hand and went stale when the data were rebuilt; regenerating it here keeps the paper's
claim that every number derives from the artifacts on disk.

Usage: canon_cv.py
"""

import json

import pandas as pd

from satinsight.manuscript import CANON_PATH

PROTOCOL = (
    "grouped 5-fold cross-validation over the 138 partition cities, 3 seeds, seed-mean "
    "token scores pooled, 95% bootstrap over cities; differences paired against the "
    "reference the comparison was run with"
)
COLUMNS = [
    "cities",
    "municipalities",
    "within_token",
    "within_token_low",
    "within_token_high",
    "within_ageb",
    "within_ageb_low",
    "within_ageb_high",
    "spearman_pooled",
    "auroc_high",
]


def block() -> dict:
    summary = pd.read_csv("data/cv_backbones.csv")
    differences = pd.read_csv("data/cv_backbone_differences.csv")
    backbones = {
        row.backbone: {c: round(float(row[c]), 4) for c in COLUMNS} for _, row in summary.iterrows()
    }
    paired = [
        {
            "comparison": row.comparison,
            "unit": row.unit,
            "municipalities": int(row.municipalities),
            **{
                c: round(float(row[c]), 4)
                for c in ("base", "other", "difference", "ci_low", "ci_high")
            },
        }
        for _, row in differences.iterrows()
    ]
    return {"protocol": PROTOCOL, "backbones": backbones, "differences": paired}


def main() -> None:
    book = json.loads(CANON_PATH.read_text())
    book["backbone_cv"] = block()
    CANON_PATH.write_text(json.dumps(book, indent=1, ensure_ascii=False))
    for name, values in book["backbone_cv"]["backbones"].items():
        print(
            f"{name}: within token {values['within_token']:.4f} "
            f"[{values['within_token_low']:.4f}, {values['within_token_high']:.4f}]"
        )
    print(f"END: {len(book['backbone_cv']['differences'])} paired differences written")


if __name__ == "__main__":
    main()
