"""Figure 5: what free products already do, and what confidence the map can carry.

Three panels, all from committed result tables so the figure regenerates without touching
the pipeline, for both frozen backbones:

(a) The paired difference between the map and each downloadable product, within
    municipalities, on both splits, with city-clustered intervals. Zero is drawn because
    the reader's question is which of these intervals clears it.
(b) Pooled detection of high-grade tracts on the test cities. Every product lands at or
    above the map here, which is the dissociation of the previous figure arriving on
    external ground, so all sit on one axis.
(c) Conformal coverage per test city under the two rules against the nominal level. The
    marginal rule's spread below the line is the argument for the clustered one.

Usage: fig5_incumbents.py <destination.pdf>
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

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

SHORT = {
    "built_surface": "GHSL built-up",
    "built_height": "GHSL height",
    "population": "GHSL population",
    "nightlights": "Night lights",
}
SPLIT_COLOR = {"val": VALIDATION, "test": TEST}
NOMINAL = 0.90


def _differences(ax) -> None:
    """Paired within-municipality difference against each product, both splits."""
    order = list(SHORT)
    offsets = {
        ("val", ""): -0.27,
        ("test", ""): -0.09,
        ("val", "dofal"): 0.09,
        ("test", "dofal"): 0.27,
    }
    for tag, name in BACKBONES:
        for split, path in (
            ("val", "data/global_products.csv"),
            ("test", "data/global_products_test.csv"),
        ):
            rows = pd.read_csv(suffixed(path, tag)).set_index("product").loc[order]
            y = [order.index(p) + offsets[(split, tag)] for p in order]
            ax.errorbar(
                rows.difference,
                y,
                xerr=[rows.difference - rows.ci_low, rows.ci_high - rows.difference],
                fmt="o",
                color=SPLIT_COLOR[split],
                ecolor=SPLIT_COLOR[split],
                markerfacecolor=SPLIT_COLOR[split] if not tag else "white",
                elinewidth=1.6 if not tag else 1.1,
                capsize=3,
                markersize=6,
                label=f"{'validation' if split == 'val' else 'test'}, {name}",
            )
    ax.axvline(0, color=INK, ls="--", lw=1.1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[p] for p in order])
    ax.invert_yaxis()
    ax.set_xlabel(r"paired $\Delta\rho$ within municipality (map minus product)")
    ax.set_title("a  Map against free global products")
    ax.legend(frameon=False, loc="lower right", fontsize=8)


def _detection(ax) -> None:
    """Pooled AUROC for high and very-high grade tracts on the test cities."""
    table = pd.read_csv("data/global_products_test.csv")
    names = [SHORT[p] for p in table["product"]]
    values = list(table.proxy_auc_high)
    colors = [MUTED] * len(table)
    hatches = [None] * len(table)
    for tag, name in BACKBONES:
        own = pd.read_csv(suffixed("data/global_products_test.csv", tag))
        names.append(f"map, {name}")
        values.append(float(own.ours_auc_high.iloc[0]))
        colors.append(TEST)
        hatches.append(LARGE_HATCH if tag else None)
    order = sorted(range(len(values)), key=lambda i: values[i])
    bars = ax.barh(
        [names[i] for i in order],
        [values[i] for i in order],
        color=[colors[i] for i in order],
        edgecolor="white",
    )
    for bar, i in zip(bars, order, strict=True):
        bar.set_hatch(hatches[i])
    ax.set_xlim(0.5, 0.95)
    ax.axvline(0.5, color=INK, ls="--", lw=1.1)
    ax.set_xlabel("pooled AUROC, high-grade tracts (test)")
    ax.set_title("b  Pooled detection: products at or above the map")
    for index, position in enumerate(order):
        ax.text(values[position] + 0.005, index, f"{values[position]:.3f}", va="center", size=9)


def _city_names() -> dict:
    """Keys to printable names, from the committed city catalogue."""
    catalogue = pd.read_csv("data/cities_national.csv")
    return dict(zip(catalogue.key, catalogue.name, strict=True))


def _coverage(ax) -> None:
    """Per-city conformal coverage under both rules, against the nominal level."""
    names = _city_names()
    base = pd.read_csv("data/uncertainty_coverage_city.csv")
    order = list(base[base.rule == "marginal"].sort_values("coverage").city)
    for tag, name in BACKBONES:
        table = pd.read_csv(suffixed("data/uncertainty_coverage_city.csv", tag))
        for rule, color, label in (
            ("marginal", MUTED, "marginal"),
            ("clustered", INK, "clustered by city"),
        ):
            rows = table[table.rule == rule].set_index("city").loc[order]
            width = float(rows.half_width_grades.iloc[0])
            ax.plot(
                range(len(order)),
                rows.coverage,
                "o-" if not tag else "o--",
                color=color,
                markerfacecolor=color if not tag else "white",
                lw=1.8 if not tag else 1.2,
                markersize=5,
                label=f"{label}, {name} ($\\pm${width:.2f} grades)",
            )
    ax.axhline(NOMINAL, color=TEST, ls="--", lw=1.2, label="nominal 90%")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names.get(key, key) for key in order], rotation=45, ha="right", size=8)
    ax.set_ylabel("empirical coverage")
    ax.set_title("c  Coverage holds only when cities are clustered")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)


def draw(destination: str) -> None:
    sns.set_theme(**STYLE)
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    _differences(axes[0])
    _detection(axes[1])
    _coverage(axes[2])
    figure.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "docs/manuscript/figures/fig5_incumbents.pdf")
