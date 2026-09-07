"""Figure 4: four independent checks that the map measures deprivation.

(a) The leading downloadable wealth product, scored on the identical tract universe its
    grid reaches, with the paired difference and its city-clustered interval annotated
    because the difference, and not either bar, is the quantity the claim is about.
(b) Replication against an index built by another institution from a different indicator
    set, beside the map's agreement with the construct it was trained toward.
(c) The share of the aggregate-to-census targeting gap the map closes at each budget,
    people pooled over cities, including the validation budget where it loses.
(d) Zero-shot transfer to two countries, in purple because neither belongs to a Mexican
    split.

Every plotted number is recomputed from the committed artifacts and checked against
`docs/manuscript/canonical_results.json`; a divergence raises rather than redrawing the page.

Usage: fig4_validation.py <destination.pdf>
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
    INK,
    MUTED,
    STYLE,
    TEST,
    TRANSFER,
    VALIDATION,
    agrees,
    canon,
)

BUDGETS = (0.05, 0.10, 0.20, 0.30)
CHANCE = 0.5


def _paired_bars(ax, labels, validation, test, *, errors=None) -> None:
    """One group per quantity, validation and test side by side."""
    x = np.arange(len(labels))
    width = 0.38
    val_error = None if errors is None else errors[0]
    test_error = None if errors is None else errors[1]
    ax.bar(
        x - width / 2,
        validation,
        width,
        yerr=val_error,
        capsize=3,
        color=VALIDATION,
        ecolor=INK,
        label="validation",
    )
    ax.bar(
        x + width / 2,
        test,
        width,
        yerr=test_error,
        capsize=3,
        color=TEST,
        ecolor=INK,
        label="test",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color=MUTED, lw=0.8)


def _incumbent(ax, book) -> None:
    """Panel a: the map and Meta's RWI on the AGEB its grid reaches."""
    validation = pd.read_csv("data/rwi_paired.csv").iloc[0]
    test = pd.read_csv("data/rwi_paired_test.csv").iloc[0]
    for row, key, split in (
        (validation, book["rwi_paired"], "validation"),
        (test, book["test"]["rwi_paired"], "test"),
    ):
        agrees(row.ours_within, key["ours_within"], name=f"map within on the RWI universe, {split}")
        agrees(row.rwi_within, key["rwi_within"], name=f"RWI within, {split}")
        agrees(row.difference, key["difference"], name=f"paired difference, {split}")

    _paired_bars(
        ax,
        ["Meta RWI", "this work"],
        [validation.rwi_within, validation.ours_within],
        [test.rwi_within, test.ours_within],
    )
    ax.set_ylabel(r"within-municipality $\rho$")
    ax.set_ylim(0, 0.45)
    ax.set_title("a  Incumbent, same AGEB")
    ax.text(
        0.03,
        0.97,
        f"val $\\Delta$ {validation.difference:+.2f} "
        f"[{validation.ci_low:+.2f}, {validation.ci_high:+.2f}]\n"
        f"test $\\Delta$ {test.difference:+.2f} "
        f"[{test.ci_low:+.2f}, {test.ci_high:+.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        size=8.5,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")


def _replication(ax, book) -> None:
    """Panel b: an independent institution's index beside the training construct."""
    own_validation = pd.read_csv("data/supervision_curve.csv").query("bags == 771").spearman_within
    opening = pd.read_csv("data/test_column.csv")
    own_test = opening[opening.row == "headline_saved"].spearman_within
    agrees(
        own_validation.mean(),
        book["curve"]["771"]["spearman_within"],
        name="map against its own construct, validation",
    )
    agrees(
        own_test.mean(),
        book["test"]["headline_token_within"],
        name="map against its own construct, test",
    )

    conapo_validation = pd.read_csv("data/conapo_replication.csv").iloc[0]
    conapo_test = pd.read_csv("data/conapo_replication_test.csv").iloc[0]
    agrees(
        conapo_validation.spearman_within,
        book["conapo"]["spearman_within"],
        name="CONAPO replication, validation",
    )
    agrees(
        conapo_test.spearman_within,
        book["test"]["conapo"]["spearman_within"],
        name="CONAPO replication, test",
    )

    _paired_bars(
        ax,
        ["CONEVAL\ngrade", "CONAPO\nindex"],
        [own_validation.mean(), conapo_validation.spearman_within],
        [own_test.mean(), conapo_test.spearman_within],
        errors=(
            [own_validation.std(), conapo_validation.ci95_half],
            [own_test.std(), conapo_test.ci95_half],
        ),
    )
    ax.set_ylim(0, 0.45)
    ax.set_title("b  Replication")


def _targeting(ax, book) -> None:
    """Panel c: share of the aggregate-to-census gap closed, people pooled per split."""
    lines = {}
    for split, path, expected in (
        ("validation", "data/targeting.csv", book["targeting_pooled"]),
        ("test", "data/targeting_test.csv", book["test"]["targeting"]),
    ):
        table = pd.read_csv(path).groupby("budget")[["aggregate", "map", "oracle"]].sum()
        pooled = (table["map"] - table["aggregate"]) / (table["oracle"] - table["aggregate"])
        lines[split] = [
            agrees(pooled.loc[budget], expected[str(budget)], name=f"targeting {split} at {budget}")
            for budget in BUDGETS
        ]

    x = [100 * budget for budget in BUDGETS]
    ax.plot(
        x, [100 * v for v in lines["validation"]], "o-", color=VALIDATION, lw=2, label="validation"
    )
    ax.plot(x, [100 * v for v in lines["test"]], "s--", color=TEST, lw=2, label="test")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x)
    ax.set_xlabel("budget (% population)")
    ax.set_ylabel("gap closed (%)")
    ax.set_title("c  Targeting")
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def _transfer(ax, book) -> None:
    """Panel d: the Mexican model applied unchanged to two other countries."""
    table = pd.read_csv("data/transfer_zeroshot.csv").set_index("city")
    stored = {row["city"]: row for row in book["zeroshot"]}
    for city in ("bogota", "riodejaneiro"):
        agrees(table.loc[city, "auroc"], stored[city]["auroc"], name=f"zero-shot AUROC, {city}")

    labels = ["Bogotá\nstrata 1–2", "Rio\nAGSN"]
    values = [table.loc["bogota", "auroc"], table.loc["riodejaneiro", "auroc"]]
    ax.bar(labels, values, color=TRANSFER, width=0.55)
    ax.axhline(CHANCE, color=INK, ls=":", lw=1.1)
    ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("zero-shot AUROC")
    ax.set_title("d  Transfer")
    for index, value in enumerate(values):
        ax.text(index, value + 0.006, f"{value:.3f}", ha="center", size=9, color=INK)


def draw(destination: str) -> None:
    book = canon()
    sns.set_theme(**STYLE)
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    _incumbent(axes[0], book)
    _replication(axes[1], book)
    _targeting(axes[2], book)
    _transfer(axes[3], book)
    figure.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "docs/manuscript/figures/fig4_validation.pdf")
