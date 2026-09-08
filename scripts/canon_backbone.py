"""Builds one backbone's block of the canonical results file from its artifacts.

The canon of DOFA base was assembled by hand as the results came in. A second backbone
needs the same quantities from the same artifacts with a suffix, so this computes every
block from the files under data/ and, run for DOFA base itself, checks that it
reproduces the published numbers: any block it cannot reproduce is printed and the run
fails, so the derivation written here is the one the paper actually used.

Usage: canon_backbone.py [tag]   (no tag verifies DOFA base; "dofal" writes book["dofal"])
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from satinsight.agebs import load_grs  # noqa: E402
from satinsight.manuscript import CANON_PATH  # noqa: E402

sys.path.insert(0, "scripts")
from backbone_paired import within  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SUFFIX = f"_{TAG}" if TAG else ""
SENSOR_SUFFIX = f"_s2_{TAG}" if TAG else ""
SEEDS = (0, 1, 2)
BUDGETS = (0.05, 0.1, 0.2, 0.3)
ARMS = ("optical", "radar", "degraded", "worldcover")
RESAMPLES = 2000


def r4(x) -> float:
    return round(float(x), 4)


def seeds_mean(pattern: str, column: str) -> float:
    files = sorted(Path().glob(pattern))
    assert files, pattern
    return r4(pd.concat(pd.read_csv(f) for f in files)[column].mean())


def curve_file() -> str:
    return f"data/supervision_curve_s2_{TAG}.csv" if TAG else "data/supervision_curve.csv"


def radius_file(radius: int) -> str:
    return (
        f"data/supervision_curve_s2_{TAG}_r{radius}.csv"
        if TAG
        else f"data/supervision_curve_r{radius}.csv"
    )


def oracle_files(radius: int, split: str) -> str:
    if TAG:
        return f"data/oracle_expanded_r{radius}_s[0-9]_{split}{SENSOR_SUFFIX}.csv"
    return f"data/oracle_expanded_r{radius}_s[0-9]{'_test' if split == 'test' else ''}.csv"


def oracle_bag_files(split: str) -> str:
    if TAG:
        return f"data/oracle_bags_expanded_r1_s[0-9]_{split}{SENSOR_SUFFIX}.csv"
    return f"data/oracle_bags_expanded_r1_s[0-9]{'_test' if split == 'test' else ''}.csv"


def pooled_targeting(path: str) -> dict:
    table = pd.read_csv(path).groupby("budget")[["aggregate", "map", "oracle"]].sum()
    pooled = (table["map"] - table["aggregate"]) / (table["oracle"] - table["aggregate"])
    return {str(b): round(float(pooled.loc[b]), 3) for b in BUDGETS}


def rows_of(path: str) -> list[dict]:
    return [
        {k: (r4(v) if isinstance(v, float) else v) for k, v in row.items()}
        for row in pd.read_csv(path).to_dict("records")
    ]


def modality(arm: str) -> dict:
    files = sorted(Path().glob(f"data/llp_val_{arm}_r1_s[0-9]_tuned{SUFFIX}.csv"))
    if not files:
        return {}
    table = pd.concat(pd.read_csv(f) for f in files)
    return {
        "within": r4(table.spearman_within.mean()),
        "auroc": r4(table.auroc_high.mean()),
        "bag_mae": r4(table.bag_mae.mean()),
        "seeds": len(files),
    }


def fraction_interval(split: str, grades: pd.Series) -> dict:
    """Model-to-oracle ratio of the mean per-municipality rho, city-clustered bootstrap."""
    model = (
        within(pd.read_parquet(f"data/predictions_{split}{SUFFIX}.parquet"), grades)
        .groupby("municipality")
        .agg(token=("token", "mean"), city=("city", "first"))
        .reset_index()
    )
    bags = pd.concat(
        pd.read_csv(f, dtype={"municipality": str})
        for f in sorted(Path().glob(oracle_bag_files(split)))
    )
    oracle = bags.groupby("municipality").rho.mean()
    pair = model.merge(oracle.rename("oracle"), left_on="municipality", right_index=True)
    groups = [g[["token", "oracle"]].to_numpy() for _, g in pair.groupby("city")]
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(RESAMPLES):
        chosen = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        ratios.append(chosen[:, 0].mean() / chosen[:, 1].mean())
    low, high = np.percentile(ratios, [2.5, 97.5])
    return {
        "fraction": round(float(pair.token.mean() / pair.oracle.mean()), 3),
        "ci_low": round(float(low), 3),
        "ci_high": round(float(high), 3),
        "municipalities": len(pair),
    }


def sweep_blindness() -> dict:
    path = f"data/llp_sweep_r1{SUFFIX}.csv"
    try:
        sweep = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    per_config = sweep.groupby("config")[["bag_mae", "spearman_within", "auroc_high"]].mean()
    return {
        "rank_corr_mae_vs_within": round(
            float(spearmanr(per_config.bag_mae, per_config.spearman_within).statistic), 3
        ),
        "rank_corr_mae_vs_auroc": round(
            float(spearmanr(per_config.bag_mae, per_config.auroc_high).statistic), 3
        ),
        "configs": len(per_config),
    }


def abmil() -> dict:
    """The canon's pooled figure is the mean over folds of the per-fold mean Spearman."""
    mil_path = "data/mil_kfold.csv" if not TAG else f"data/mil_kfold_classes_0.0_0.0{SUFFIX}.csv"
    mil = pd.read_csv(mil_path)
    llp = pd.read_csv(f"data/llp_kfold{SUFFIX}.csv")
    return {
        "map_auroc": round(float(mil.auroc_high.mean()), 3),
        "pooled_spearman": round(float(mil.spearman_mean.mean()), 3),
        "bag_kappa": round(float(mil.kappa.mean()), 3),
        "llp_kfold_auroc": round(float(llp.auroc_high.mean()), 3),
        "llp_kfold_within": round(float(llp.spearman_within.mean()), 3),
    }


def transfer_training() -> dict:
    path = f"data/transfer_training{SUFFIX}.csv"
    try:
        table = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    out = {}
    for country, group in table.groupby("country"):
        methods = {}
        for method, rows in group.groupby("method"):
            rows = rows.sort_values("seed")
            methods[method] = {
                "within_seeds": [r4(v) for v in rows.within],
                "within": r4(rows.within.mean()),
                "within_ci_low": r4(rows.within_ci_low.mean()),
                "within_ci_high": r4(rows.within_ci_high.mean()),
                "pooled": r4(rows.pooled.mean()),
                "auroc_top_quartile": r4(rows.auroc_top_quartile.mean()),
            }
        out[country] = {
            "bags": int(group.bags.iloc[0]),
            "municipalities_scored": int(group.municipalities_scored.iloc[0]),
            "tokens_scored": int(group.tokens_scored.iloc[0]),
            "methods": methods,
        }
    return out


def build() -> dict:
    grades = load_grs().set_index("cvegeo").ordinal.astype(float)
    curve = pd.read_csv(curve_file()).groupby("bags")
    columns = ["auroc_high", "spearman_within", "spearman_pooled", "bag_mae"]
    granularity = pd.read_csv(f"data/granularity_curve{SUFFIX}.csv").groupby("level")
    a5 = pd.read_csv(f"data/a5_aggregation{SUFFIX}.csv")
    test = pd.read_csv(f"data/test_column{SUFFIX}.csv")

    def test_rows(row: str) -> dict:
        chosen = test[test.row == row]
        return {str(d): r4(v) for d, v in chosen.groupby("detail").spearman_within.mean().items()}

    headline = test[test.row == "headline_saved"]
    ensemble = within(pd.read_parquet(f"data/predictions_test{SUFFIX}.parquet"), grades)
    ceiling_test_r1 = seeds_mean(oracle_files(1, "test"), "within")
    return {
        "curve": {str(b): {c: r4(v) for c, v in g[columns].mean().items()} for b, g in curve},
        "granularity": {
            str(level): {c: r4(v) for c, v in g[["auroc_high", "spearman_within"]].mean().items()}
            for level, g in granularity
        },
        "ceiling": {str(r): seeds_mean(oracle_files(r, "val"), "within") for r in (0, 1)},
        "ceiling_auroc": {str(r): seeds_mean(oracle_files(r, "val"), "auroc") for r in (0, 1)},
        "radius_sweep": {
            "0": r4(pd.read_csv(radius_file(0)).spearman_within.mean()),
            "2": r4(pd.read_csv(radius_file(2)).spearman_within.mean()),
            "1": r4(curve.get_group(771).spearman_within.mean()),
        },
        "a5_weighted": {c: r4(a5[c].mean()) for c in ("auroc_high", "spearman_within", "bag_mae")},
        "conapo": rows_of(f"data/conapo_replication{SUFFIX}.csv")[0],
        "rwi_paired": rows_of(f"data/rwi_paired{SUFFIX}.csv")[0],
        "targeting_pooled": pooled_targeting(f"data/targeting{SUFFIX}.csv"),
        "border": rows_of(f"data/border_discontinuity_city{SUFFIX}.csv"),
        "maup": rows_of(f"data/maup{SUFFIX}.csv"),
        **{f"modality_{arm}": modality(arm) for arm in ARMS},
        "abmil": abmil(),
        "sweep_blindness": sweep_blindness(),
        "test": {
            "headline_token_within": r4(headline.spearman_within.mean()),
            "headline_token_within_ensemble": r4(ensemble.token.mean()),
            "headline_auroc": r4(headline.auroc_high.mean()),
            "ceiling_r1": ceiling_test_r1,
            "ceiling_r0": seeds_mean(oracle_files(0, "test"), "within"),
            "fraction": round(float(headline.spearman_within.mean() / ceiling_test_r1), 2),
            "curve": test_rows("curve"),
            "granularity": test_rows("granularity"),
            "modality": test_rows("modality"),
            "weighted": test_rows("weighted")["population"],
            "wc_paired": rows_of(f"data/wc_paired_test{SUFFIX}.csv")[0],
            "rwi_paired": rows_of(f"data/rwi_paired_test{SUFFIX}.csv")[0],
            "conapo": rows_of(f"data/conapo_replication_test{SUFFIX}.csv")[0],
            "targeting": pooled_targeting(f"data/targeting_test{SUFFIX}.csv"),
            "maup": rows_of(f"data/maup_test{SUFFIX}.csv"),
        },
        "fraction_ci": {split: fraction_interval(split, grades) for split in ("val", "test")},
        "transfer_training": transfer_training(),
    }


def compare(computed, published, path="", tolerance=1e-3) -> list[str]:
    """Every leaf of `computed` that departs from `published`."""
    problems = []
    if isinstance(computed, dict):
        if not isinstance(published, dict):
            return [f"{path}: published is not a block"]
        for key, value in computed.items():
            if key not in published:
                problems.append(f"{path}/{key}: not in the canon")
            else:
                problems.extend(compare(value, published[key], f"{path}/{key}", tolerance))
    elif isinstance(computed, list):
        for i, (c, p) in enumerate(zip(computed, published, strict=False)):
            problems.extend(compare(c, p, f"{path}[{i}]", tolerance))
    elif isinstance(computed, (int, float)) and isinstance(published, (int, float)):
        if abs(float(computed) - float(published)) > tolerance:
            problems.append(f"{path}: computed {computed} vs canon {published}")
    elif computed != published:
        problems.append(f"{path}: computed {computed!r} vs canon {published!r}")
    return problems


def main() -> int:
    book = json.loads(CANON_PATH.read_text())
    block = build()
    if not TAG:
        problems = compare(block, book)
        print("\n".join(problems) if problems else "DOFA base canon reproduced", flush=True)
        return 1 if problems else 0
    book[TAG] = block
    CANON_PATH.write_text(json.dumps(book, indent=1, ensure_ascii=False))
    print(json.dumps(block, indent=1)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
