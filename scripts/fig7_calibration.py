"""Figure 7: what the tract-level score means in grades, and why an interval is wide.

(a) On the held-out test cities, the distribution of the tract-mean score at each true
    grade, for both backbones: the score orders the grades and the distributions overlap.
(b) The same per city: the median tract score by grade in each test city (base backbone),
    which is the ordering holding inside cities while the level shifts between them, the
    reason a marginal conformal interval undercovers.
(c) The isotonic map from score to grade fitted on the validation cities, with the
    conformal half-widths of Table 8 drawn as bands (base backbone).

Usage: fig7_calibration.py <destination.pdf>
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from satinsight.manuscript import (  # noqa: E402
    BACKBONES,
    INK,
    LARGE_HATCH,
    MUTED,
    STYLE,
    TEST,
    VALIDATION,
    suffixed,
)

sys.path.insert(0, "scripts")
from uncertainty import per_ageb, with_truth  # noqa: E402

GRADES = ["very low", "low", "medium", "high", "very high"]


def _distributions(ax, frames: dict) -> None:
    width = 0.36
    for index, (tag, _name) in enumerate(BACKBONES):
        frame = frames[("test", tag)]
        data = [frame[frame.ordinal == g].score.to_numpy() for g in range(5)]
        positions = np.arange(5) + (index - 0.5) * width
        box = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(TEST)
            patch.set_edgecolor("white")
            patch.set_alpha(0.85)
            if tag:
                patch.set_hatch(LARGE_HATCH)
        counts = [len(d) for d in data]
        if not tag:
            for g, n in enumerate(counts):
                ax.text(
                    g,
                    ax.get_ylim()[0] if False else -0.02,
                    f"n={n}",
                    ha="center",
                    size=7,
                    color=INK,
                )
    ax.set_xticks(range(5))
    ax.set_xticklabels(GRADES)
    ax.set_xlabel("census-tract grade (truth)")
    ax.set_ylabel("tract-mean score")
    ax.set_title("a  Score by true grade, test cities")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TEST, edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor=TEST, edgecolor="white", hatch=LARGE_HATCH),
    ]
    ax.legend(handles, [name for _, name in BACKBONES], frameon=False, fontsize=8, loc="upper left")


def _per_city(ax, frame: pd.DataFrame, names: dict) -> None:
    for city, group in frame.groupby("city", observed=True):
        medians = group.groupby("ordinal").score.median()
        ax.plot(medians.index, medians.to_numpy(), "-o", color=MUTED, lw=1, markersize=3, alpha=0.9)
        ax.text(
            medians.index[-1] + 0.05,
            medians.iloc[-1],
            names.get(city, city),
            size=5.5,
            color=INK,
            va="center",
        )
    pooled = frame.groupby("ordinal").score.median()
    ax.plot(
        pooled.index,
        pooled.to_numpy(),
        "-o",
        color=TEST,
        lw=2.2,
        markersize=5,
        label="all test tracts",
    )
    ax.set_xticks(range(5))
    ax.set_xticklabels(GRADES)
    ax.set_xlim(-0.3, 5.2)
    ax.set_xlabel("census-tract grade (truth)")
    ax.set_ylabel("median tract score in the city")
    ax.set_title("b  The ordering holds inside cities; the level shifts between them")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _isotonic(ax, frames: dict, tag: str = "") -> None:
    calibration = frames[("val", tag)]
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(calibration.score, calibration.ordinal)
    grid = np.linspace(calibration.score.min(), calibration.score.max(), 300)
    fitted = isotonic.predict(grid)
    widths = (
        pd.read_csv(suffixed("data/uncertainty_conformal.csv", tag))
        .set_index("rule")
        .half_width_grades
    )
    ax.fill_between(
        grid,
        fitted - widths["clustered"],
        fitted + widths["clustered"],
        color=INK,
        alpha=0.12,
        label=f"clustered 90% interval ($\\pm${widths['clustered']:.2f})",
    )
    ax.fill_between(
        grid,
        fitted - widths["marginal"],
        fitted + widths["marginal"],
        color=VALIDATION,
        alpha=0.2,
        label=f"marginal 90% interval ($\\pm${widths['marginal']:.2f})",
    )
    ax.plot(
        grid, fitted, color=VALIDATION, lw=2.2, label="isotonic score-to-grade map (validation)"
    )
    test = frames[("test", tag)]
    ax.scatter(
        test.score,
        test.ordinal + np.random.default_rng(0).uniform(-0.12, 0.12, len(test)),
        s=4,
        color=TEST,
        alpha=0.25,
        label="test tracts (jittered)",
    )
    ax.set_yticks(range(5))
    ax.set_yticklabels(GRADES)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("tract-mean score")
    ax.set_ylabel("grade")
    ax.set_title("c  Score to grade, with the conformal half-widths")
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def draw(destination: str) -> None:
    sns.set_theme(**STYLE)
    frames = {
        (split, tag): with_truth(per_ageb(suffixed(f"data/predictions_{split}.parquet", tag)))
        for split in ("val", "test")
        for tag, _ in BACKBONES
    }
    names = dict(pd.read_csv("data/cities_national.csv")[["key", "name"]].to_numpy())
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    _distributions(axes[0], frames)
    _per_city(axes[1], frames[("test", "")], names)
    _isotonic(axes[2], frames)
    figure.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "docs/manuscript/figures/fig7_calibration.pdf")
