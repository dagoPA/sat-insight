"""Figure 2: what the map recovers as a function of the aggregates that supervise it.

Three panels, all within-municipality Spearman against held-out tract grades, all means
over the three seeds, for both frozen backbones. Validation carries the seed spread
because it was scored repeatedly during selection; test carries none because it was
scored once with the validation-selected models.

(a) How many municipal aggregates supervise training, against what the map recovers, with
    the fully supervised oracle drawn on each split so the reader sees the fraction rather
    than an absolute the scale of which means nothing on its own.
(b) The same 771 bags relabeled at coarser levels of publication.
(c) Single-sensor arms, including optical block-averaged to the radar's effective grain,
    which is what separates sensor content from ground sampling distance.

Every plotted number is recomputed from the per-seed artifacts and checked against
`docs/manuscript/canonical_results.json`; a divergence raises rather than redrawing the page.

Usage: fig2_curve.py <destination.pdf>
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

from satinsight.manuscript import (  # noqa: E402
    BACKBONES,
    LARGE_HATCH,
    MUTED,
    ORACLE,
    STYLE,
    TEST,
    VALIDATION,
    agrees,
    block,
    canon,
    suffixed,
)

BAG_SIZES = (50, 100, 200, 400, 771)
LEVELS = ("municipality", "state", "national")
LEVEL_LABELS = {"municipality": "municipal", "state": "state", "national": "national"}
ARMS = ("optical", "degraded", "radar")
ARM_LABELS = {
    "optical": "optical\n10 m",
    "degraded": "optical\ndegraded",
    "radar": "radar\n20$\\times$22 m",
}


def _curve_file(tag: str) -> str:
    return f"data/supervision_curve_s2_{tag}.csv" if tag else "data/supervision_curve.csv"


def _validation_seeds(paths) -> np.ndarray:
    """Per-seed within-municipality correlation from one artifact per seed."""
    return np.array([pd.read_csv(path).spearman_within.iloc[0] for path in sorted(paths)])


def _test_row(column: pd.DataFrame, row: str, detail) -> float:
    """Mean over seeds of one row of the test column."""
    chosen = column[(column.row == row) & (column.detail.astype(str) == str(detail))]
    if chosen.empty:
        raise KeyError(f"the test column carries no row {row}/{detail}")
    return float(chosen.spearman_within.mean())


def _quantity(ax, book) -> None:
    """Panel a: map recovered against the number of aggregates that supervise it."""
    x = np.arange(len(BAG_SIZES))
    for tag, name in BACKBONES:
        own = block(book, tag)
        curve = pd.read_csv(_curve_file(tag))
        grouped = curve.groupby("bags").spearman_within.agg(["mean", "std"]).loc[list(BAG_SIZES)]
        for bags in BAG_SIZES:
            agrees(
                grouped.loc[bags, "mean"],
                own["curve"][str(bags)]["spearman_within"],
                name=f"{name} validation curve at {bags} bags",
            )
        column = pd.read_csv(suffixed("data/test_column.csv", tag))
        test = [
            agrees(
                _test_row(column, "curve", bags),
                own["test"]["curve"][str(bags)],
                name=f"{name} test curve at {bags} bags",
            )
            for bags in BAG_SIZES
        ]
        solid = not tag
        ax.axhline(
            own["ceiling"]["1"],
            color=ORACLE,
            ls="--",
            lw=1.3 if solid else 1.0,
            alpha=1.0 if solid else 0.45,
            label="oracle, validation" if solid else None,
        )
        ax.axhline(
            own["test"]["ceiling_r1"],
            color=ORACLE,
            ls=":",
            lw=1.3 if solid else 1.0,
            alpha=1.0 if solid else 0.45,
            label="oracle, test" if solid else None,
        )
        ax.errorbar(
            x,
            grouped["mean"],
            yerr=grouped["std"],
            fmt="o-" if solid else "o--",
            color=VALIDATION,
            markerfacecolor=VALIDATION if solid else "white",
            capsize=3,
            lw=2 if solid else 1.4,
            label=f"validation, {name}",
        )
        ax.plot(
            x,
            test,
            "s-" if solid else "s--",
            color=TEST,
            markerfacecolor=TEST if solid else "white",
            lw=1.8 if solid else 1.3,
            label=f"test, {name}",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in BAG_SIZES])
    ax.set_xlabel("municipal aggregates (bags)")
    ax.set_ylabel(r"within-municipality $\rho$")
    ax.set_ylim(0, 0.34)
    ax.set_title("a  Number of aggregates")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)


def _granularity(ax, book) -> None:
    """Panel b: the same bags relabeled at coarser levels of publication."""
    validation, test = {}, {}
    for tag, name in BACKBONES:
        own = block(book, tag)
        table = pd.read_csv(suffixed("data/granularity_curve.csv", tag))
        grouped = table.groupby("level").spearman_within.agg(["mean", "std"]).loc[list(LEVELS)]
        for level in LEVELS:
            agrees(
                grouped.loc[level, "mean"],
                own["granularity"][level]["spearman_within"],
                name=f"{name} validation granularity, {level}",
            )
        column = pd.read_csv(suffixed("data/test_column.csv", tag))
        validation[tag] = grouped
        test[tag] = [
            agrees(
                _test_row(column, "granularity", level),
                own["test"]["granularity"][level],
                name=f"{name} test granularity, {level}",
            )
            for level in LEVELS
        ]
    _paired_bars(ax, [LEVEL_LABELS[level] for level in LEVELS], validation, test)
    ax.set_title("b  Granularity of the label")


def _content(ax, book) -> None:
    """Panel c: single-sensor arms, with optical degraded to the radar's grain."""
    validation, test = {}, {}
    for tag, name in BACKBONES:
        own = block(book, tag)
        means, spreads = [], []
        for arm in ARMS:
            pattern = f"llp_val_{arm}_r1_s*_tuned{'_' + tag if tag else ''}.csv"
            seeds = _validation_seeds(Path("data").glob(pattern))
            means.append(
                agrees(seeds.mean(), own[f"modality_{arm}"]["within"], name=f"{name} arm, {arm}")
            )
            spreads.append(float(seeds.std(ddof=1)))
        validation[tag] = pd.DataFrame({"mean": means, "std": spreads}, index=list(ARMS))
        column = pd.read_csv(suffixed("data/test_column.csv", tag))
        test[tag] = [
            agrees(
                _test_row(column, "modality", arm),
                own["test"]["modality"][arm],
                name=f"{name} test arm, {arm}",
            )
            for arm in ARMS
        ]
    _paired_bars(ax, [ARM_LABELS[arm] for arm in ARMS], validation, test)
    ax.set_title("c  Sensor and resolution")


def _paired_bars(ax, labels, validation: dict, test: dict) -> None:
    """Validation and test side by side per backbone; the seed spread on validation alone.

    Color is the split; the second backbone is hatched.
    """
    x = np.arange(len(labels))
    width = 0.19
    for index, (tag, name) in enumerate(BACKBONES):
        shift = (index - 0.5) * 2 * width
        hatch = LARGE_HATCH if tag else None
        ax.bar(
            x + shift - width / 2,
            validation[tag]["mean"].to_numpy(),
            width,
            yerr=validation[tag]["std"].to_numpy(),
            capsize=2,
            color=VALIDATION,
            hatch=hatch,
            edgecolor="white",
            ecolor="#2c3e50",
            label=f"validation, {name}",
        )
        ax.bar(
            x + shift + width / 2,
            test[tag],
            width,
            color=TEST,
            hatch=hatch,
            edgecolor="white",
            label=f"test, {name}",
        )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.34)
    ax.set_ylabel("")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)


def draw(destination: str) -> None:
    book = canon()
    sns.set_theme(**STYLE)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    _quantity(axes[0], book)
    _granularity(axes[1], book)
    _content(axes[2], book)
    figure.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "docs/manuscript/figures/fig2_curve.pdf")
