"""Tests whether an alternative backbone differs from the canonical one, without advocacy.

The decision rule is fixed before looking: the canonical backbone is replaced only when
the weakly supervised predictor, the quantity the paper is about, improves significantly
on validation at the full pool of aggregates. Everything else is reported beside it with
absolute values. Three quantities are tested, each paired by municipality so that the
same tokens of the same ground are compared:

* the predictor, on validation and on test, per token and per AGEB, from the persisted
  per-token scores of the three validation-selected seeds (seed-mean score per token);
* the fully supervised oracle, on validation and on test, from its per-bag files;
* every point of the supervision curve, seed by seed, since the three seeds of each
  backbone share the same nested bag samples.

Paired tests: Wilcoxon signed-rank, exact two-sided sign test, and a sign-flip
permutation test that flips whole cities, the unit of independence in the partition.

Usage: backbone_significance.py <suffix>   (for example dofal)
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402

from satinsight.agebs import load_grs  # noqa: E402

sys.path.insert(0, "scripts")
from backbone_paired import within  # noqa: E402

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "dofal"
SEEDS = (0, 1, 2)
PERMUTATIONS = 20_000


def city_permutation_p(delta: pd.Series, cities: pd.Series, seed: int = 0) -> float:
    """Two-sided p-value of the mean difference under sign flips of whole cities."""
    rng = np.random.default_rng(seed)
    d, c = delta.to_numpy(), cities.to_numpy()
    uniq, index = np.unique(c, return_inverse=True)
    observed = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(PERMUTATIONS, len(uniq)))
    means = (flips[:, index] * d).mean(axis=1)
    return float(((np.abs(means) >= observed).sum() + 1) / (PERMUTATIONS + 1))


def paired_row(base: pd.Series, other: pd.Series, cities=None) -> dict:
    delta = other - base
    nonzero = int((delta != 0).sum())
    return {
        "n": len(delta),
        "base": float(base.mean()),
        "other": float(other.mean()),
        "difference": float(delta.mean()),
        "wins": int((delta > 0).sum()),
        "wilcoxon_p": float(wilcoxon(delta).pvalue),
        "sign_p": float(binomtest(int((delta > 0).sum()), nonzero).pvalue) if nonzero else 1.0,
        "city_permutation_p": (
            city_permutation_p(delta, cities) if cities is not None else float("nan")
        ),
    }


def predictor_rows(grades: pd.Series) -> list[dict]:
    rows = []
    for split in ("val", "test"):
        base = within(pd.read_parquet(f"data/predictions_{split}.parquet"), grades)
        other = within(pd.read_parquet(f"data/predictions_{split}_{SUFFIX}.parquet"), grades)
        merged = base.merge(other, on=["city", "municipality"], suffixes=("_base", "_other"))
        for unit in ("token", "ageb"):
            pair = merged.dropna(subset=[f"{unit}_base", f"{unit}_other"])
            rows.append(
                {
                    "quantity": "predictor",
                    "split": split,
                    "unit": unit,
                    **paired_row(pair[f"{unit}_base"], pair[f"{unit}_other"], pair.city),
                }
            )
    return rows


def oracle_rows() -> list[dict]:
    """The oracle's per-bag files carry no city, so the permutation test is not run."""
    rows = []
    for split, base_suffix, other_suffix in (("val", "", "_val"), ("test", "_test", "_test")):
        base = pd.concat(
            pd.read_csv(f"data/oracle_bags_expanded_r1_s{s}{base_suffix}.csv") for s in SEEDS
        )
        other = pd.concat(
            pd.read_csv(f"data/oracle_bags_expanded_r1_s{s}{other_suffix}_s2_{SUFFIX}.csv")
            for s in SEEDS
        )
        merged = pd.concat(
            [
                base.groupby("municipality").rho.mean().rename("base"),
                other.groupby("municipality").rho.mean().rename("other"),
            ],
            axis=1,
        ).dropna()
        rows.append(
            {
                "quantity": "oracle",
                "split": split,
                "unit": "token",
                **paired_row(merged.base, merged.other),
            }
        )
    return rows


def curve_rows() -> list[dict]:
    """Seed-wise comparison at every point of the supervision curve on validation."""
    base = pd.read_csv("data/supervision_curve.csv").query("radius == 1")
    other = pd.read_csv(f"data/supervision_curve_s2_{SUFFIX}.csv").query("radius == 1")
    rows = []
    for bags in sorted(other.bags.unique()):
        b = base[base.bags == bags].set_index("seed").spearman_within
        o = other[other.bags == bags].set_index("seed").spearman_within
        common = b.index.intersection(o.index)
        rows.append(
            {
                "quantity": "curve",
                "split": "val",
                "unit": f"token@{bags}",
                "n": len(common),
                "base": float(b[common].mean()),
                "base_sd": float(b[common].std()),
                "other": float(o[common].mean()),
                "other_sd": float(o[common].std()),
                "difference": float((o[common] - b[common]).mean()),
                "wins": int((o[common] > b[common]).sum()),
            }
        )
    return rows


def main() -> None:
    grades = load_grs().set_index("cvegeo").ordinal.astype(float)
    table = pd.DataFrame(predictor_rows(grades) + oracle_rows() + curve_rows())
    table.to_csv(f"data/backbone_significance_{SUFFIX}.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
