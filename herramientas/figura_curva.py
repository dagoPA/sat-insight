"""Figure 2: the exchange rate between published aggregates and recovered map.

Three panels, all within-municipality Spearman against held-out tract grades, all means
over the three seeds. Validation carries error bars because it was scored repeatedly
during selection; test carries none because it was scored once with frozen models, and
drawing a spread there would suggest a freedom the protocol did not have.

(a) How many municipal aggregates supervise training, against what the map recovers, with
    the fully supervised oracle drawn on each split so the reader sees the fraction rather
    than an absolute the scale of which means nothing on its own.
(b) The same 771 bags relabeled at coarser levels of publication.
(c) Single-sensor arms, including optical block-averaged to the radar's effective grain,
    which is what separates sensor content from ground sampling distance.

Every plotted number is recomputed from the per-seed artifacts and checked against
`data/canon_manuscrito.json`; a divergence raises rather than redrawing the page.

Usage: figura_curva.py <destination.pdf>
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
    MUTED,
    ORACLE,
    STYLE,
    TEST,
    VALIDATION,
    agrees,
    canon,
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


def _validation_seeds(paths) -> np.ndarray:
    """Per-seed within-municipality correlation from one artifact per seed."""
    return np.array([pd.read_csv(path).spearman_within.iloc[0] for path in sorted(paths)])


def _test_row(opening: pd.DataFrame, row: str, detail) -> float:
    """Mean over seeds of one row of the single test opening."""
    chosen = opening[(opening.row == row) & (opening.detail.astype(str) == str(detail))]
    if chosen.empty:
        raise KeyError(f"the test opening carries no row {row}/{detail}")
    return float(chosen.spearman_within.mean())


def _quantity(ax, book) -> None:
    """Panel a: map recovered against the number of aggregates that supervise it."""
    curve = pd.read_csv("data/curva_supervision.csv")
    grouped = curve.groupby("bags").spearman_within.agg(["mean", "std"]).loc[list(BAG_SIZES)]
    for bags in BAG_SIZES:
        agrees(
            grouped.loc[bags, "mean"],
            book["curve"][str(bags)]["spearman_within"],
            name=f"validation curve at {bags} bags",
        )

    opening = pd.read_csv("data/apertura_test.csv")
    test = [
        agrees(
            _test_row(opening, "curve", bags),
            book["test"]["curve"][str(bags)],
            name=f"test curve at {bags} bags",
        )
        for bags in BAG_SIZES
    ]

    x = range(len(BAG_SIZES))
    ax.axhline(
        book["ceiling"]["1"], color=ORACLE, ls="--", lw=1.3, label="oracle bound, validation"
    )
    ax.axhline(book["test"]["ceiling_r1"], color=ORACLE, ls=":", lw=1.3, label="oracle bound, test")
    ax.errorbar(
        x,
        grouped["mean"],
        yerr=grouped["std"],
        fmt="o-",
        color=VALIDATION,
        capsize=3,
        lw=2,
        label="validation (selection)",
    )
    ax.plot(x, test, "s--", color=TEST, lw=1.8, label="test (single opening)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in BAG_SIZES])
    ax.set_xlabel("municipal aggregates (bags)")
    ax.set_ylabel(r"within-municipality $\rho$")
    ax.set_ylim(0, 0.30)
    ax.set_title("a  Number of aggregates")
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def _granularity(ax, book) -> None:
    """Panel b: the same bags relabeled at coarser levels of publication."""
    table = pd.read_csv("data/curva_granularidad.csv")
    grouped = table.groupby("level").spearman_within.agg(["mean", "std"]).loc[list(LEVELS)]
    for level in LEVELS:
        agrees(
            grouped.loc[level, "mean"],
            book["granularity"][level]["spearman_within"],
            name=f"validation granularity, {level}",
        )

    opening = pd.read_csv("data/apertura_test.csv")
    test = [
        agrees(
            _test_row(opening, "granularity", level),
            book["test"]["granularity"][level],
            name=f"test granularity, {level}",
        )
        for level in LEVELS
    ]
    _paired_bars(ax, [LEVEL_LABELS[level] for level in LEVELS], grouped, test)
    ax.set_title("b  Granularity of the label")


def _content(ax, book) -> None:
    """Panel c: single-sensor arms, with optical degraded to the radar's grain."""
    means, spreads = [], []
    for arm in ARMS:
        seeds = _validation_seeds(Path("data").glob(f"llp_val_{arm}_r1_s*_tuned.csv"))
        means.append(
            agrees(
                seeds.mean(),
                book[f"modality_{arm}"]["within"],
                name=f"validation arm, {arm}",
            )
        )
        spreads.append(float(seeds.std(ddof=1)))
    grouped = pd.DataFrame({"mean": means, "std": spreads}, index=list(ARMS))

    opening = pd.read_csv("data/apertura_test.csv")
    test = [
        agrees(
            _test_row(opening, "modality", arm),
            book["test"]["modality"][arm],
            name=f"test arm, {arm}",
        )
        for arm in ARMS
    ]
    _paired_bars(ax, [ARM_LABELS[arm] for arm in ARMS], grouped, test)
    ax.set_title("c  Sensor and resolution")


def _paired_bars(ax, labels, validation, test) -> None:
    """Validation and test side by side, with the seed spread on validation alone."""
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(
        x - width / 2,
        validation["mean"].to_numpy(),
        width,
        yerr=validation["std"].to_numpy(),
        capsize=3,
        color=VALIDATION,
        ecolor="#2c3e50",
        label="validation",
    )
    ax.bar(x + width / 2, test, width, color=TEST, label="test")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("")
    ax.legend(frameon=False, fontsize=9, loc="upper right")


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
