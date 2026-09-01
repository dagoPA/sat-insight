"""Figure 5: what free products already do, and what confidence the map can carry.

Three panels, all from committed result tables so the figure regenerates without touching
the pipeline:

(a) The paired difference between the map and each downloadable product, within
    municipalities, on both splits, with city-clustered intervals. Zero is drawn because
    the reader's question is which of these intervals clears it.
(b) Pooled detection of high-grade tracts on the test cities. Every product lands at or
    above the map here, which is the dissociation of the previous figure arriving on
    external ground, so all five sit on one axis.
(c) Conformal coverage per test city under the two rules against the nominal level. The
    marginal rule's spread below the line is the argument for the clustered one.

Usage: figura_incumbentes.py <destination.pdf>
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

SHORT = {
    "built_surface": "GHSL built-up",
    "built_height": "GHSL height",
    "population": "GHSL population",
    "nightlights": "Night lights",
}
SPLIT_COLOR = {"val": "#7f8c8d", "test": "#c0392b"}
NOMINAL = 0.90


def _differences(ax) -> None:
    """Paired within-municipality difference against each product, both splits."""
    frames = []
    for split, path in (
        ("val", "data/incumbentes_globales.csv"),
        ("test", "data/incumbentes_globales_test.csv"),
    ):
        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)

    order = list(SHORT)
    for offset, split in ((-0.16, "val"), (0.16, "test")):
        rows = table[table.split == split].set_index("product").loc[order]
        y = [order.index(p) + offset for p in order]
        ax.errorbar(
            rows.difference,
            y,
            xerr=[rows.difference - rows.ci_low, rows.ci_high - rows.difference],
            fmt="o",
            color=SPLIT_COLOR[split],
            ecolor=SPLIT_COLOR[split],
            elinewidth=1.8,
            capsize=3,
            markersize=6,
            label="Validation" if split == "val" else "Test",
        )
    ax.axvline(0, color="#2c3e50", ls="--", lw=1.1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[p] for p in order])
    ax.invert_yaxis()
    ax.set_xlabel(r"Paired $\Delta\rho$ within municipality (map minus product)")
    ax.set_title("a  Map against free global products")
    ax.legend(frameon=False, loc="lower right")


def _detection(ax) -> None:
    """Pooled AUROC for high and very-high grade tracts on the test cities."""
    table = pd.read_csv("data/incumbentes_globales_test.csv")
    names = [SHORT[p] for p in table["product"]] + ["Map"]
    values = [*table.proxy_auc_high, float(table.ours_auc_high.iloc[0])]
    colors = ["#95a5a6"] * len(table) + ["#c0392b"]
    order = sorted(range(len(values)), key=lambda i: values[i])
    ax.barh(
        [names[i] for i in order],
        [values[i] for i in order],
        color=[colors[i] for i in order],
    )
    ax.set_xlim(0.5, 0.95)
    ax.axvline(0.5, color="#2c3e50", ls="--", lw=1.1)
    ax.set_xlabel("Pooled AUROC, high-grade tracts (test)")
    ax.set_title("b  Pooled detection: products at or above the map")
    for index, position in enumerate(order):
        ax.text(values[position] + 0.005, index, f"{values[position]:.3f}", va="center", size=9)


def _city_names() -> dict:
    """Keys to printable names, from the committed city catalogue."""
    catalogue = pd.read_csv("data/ciudades_nacional.csv")
    return dict(zip(catalogue.clave, catalogue.nombre, strict=True))


def _coverage(ax) -> None:
    """Per-city conformal coverage under both rules, against the nominal level."""
    table = pd.read_csv("data/incertidumbre_cobertura_ciudad.csv")
    names = _city_names()
    marginal = table[table.rule == "marginal"].sort_values("coverage")
    order = list(marginal.city)
    for rule, color, label in (
        ("marginal", "#95a5a6", "Marginal"),
        ("clustered", "#2980b9", "Clustered by city"),
    ):
        rows = table[table.rule == rule].set_index("city").loc[order]
        width = float(rows.half_width_grades.iloc[0])
        ax.plot(
            range(len(order)),
            rows.coverage,
            "o-",
            color=color,
            lw=1.8,
            markersize=5,
            label=f"{label} ($\\pm${width:.2f} grades)",
        )
    ax.axhline(NOMINAL, color="#c0392b", ls="--", lw=1.2, label="Nominal 90%")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names.get(key, key) for key in order], rotation=45, ha="right", size=8)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("c  Coverage holds only when cities are clustered")
    ax.legend(frameon=False, loc="lower right", fontsize=9)


def draw(destination: str) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.66)
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
