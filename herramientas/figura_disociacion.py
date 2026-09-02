"""Figure 3: predicting the bag and localizing inside it are different abilities.

(a) Map quality by model family on identical features and labels. The ramp runs from the
    standard weakly supervised localizer, which lands at chance, through label proportions
    and the swept head, to the oracle trained on tract labels. The ramp is model families,
    so it carries no split colors: the first two bars are measured on the national-set
    pool, the last two on the expanded pool, which the caption states.
(b) The blindness of the only selection signal weak supervision may legitimately use.
    Across the fourteen sweep configurations, bag error barely ranks map quality, so a
    pipeline whose product is the map cannot be tuned on the error it can see.

Every plotted number is recomputed from the per-fold or per-seed artifacts and checked
against `data/canon_manuscrito.json`; a divergence raises rather than redrawing the page.

Usage: figura_disociacion.py <destination.pdf>
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
from scipy.stats import spearmanr  # noqa: E402

from satinsight.manuscript import (  # noqa: E402
    INK,
    MUTED,
    ORACLE,
    STYLE,
    VALIDATION,
    agrees,
    canon,
)

CHANCE = 0.5
STEPS = (
    ("attention MIL\n(ABMIL)", MUTED),
    ("label\nproportions", "#8fb8d4"),
    ("+ sweep +\ncontext", VALIDATION),
    ("oracle\n(instance labels)", ORACLE),
)


def _families(ax, book) -> None:
    """Panel a: the map each model family produces, on identical features and labels."""
    abmil = agrees(
        pd.read_csv("data/mil_kfold.csv").auroc_high.mean(),
        book["abmil"]["map_auroc"],
        name="ABMIL map AUROC",
    )
    proportions = agrees(
        pd.read_csv("data/llp_kfold.csv").auroc_high.mean(),
        book["abmil"]["llp_kfold_auroc"],
        name="label proportions map AUROC",
    )
    swept = agrees(
        pd.read_csv("data/curva_supervision.csv").query("bags == 771").auroc_high.mean(),
        book["curve"]["771"]["auroc_high"],
        name="swept head map AUROC",
    )
    oracle_seeds = [
        pd.read_csv(path).auroc.iloc[0]
        for path in sorted(Path("data").glob("oraculo_expanded_r1_s*.csv"))
        if "_test" not in path.name
    ]
    oracle = agrees(
        sum(oracle_seeds) / len(oracle_seeds),
        book["ceiling_auroc"]["1"],
        name="oracle map AUROC",
    )

    values = [abmil, proportions, swept, oracle]
    labels = [label for label, _ in STEPS]
    ax.bar(labels, values, color=[color for _, color in STEPS])
    ax.axhline(CHANCE, color=INK, ls=":", lw=1.1)
    ax.text(-0.45, CHANCE + 0.008, "chance", color=INK, size=9)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("map AUROC (high deprivation)")
    ax.set_title("a  The map, model by model")
    for index, value in enumerate(values):
        ax.text(index, value + 0.008, f"{value:.3f}", ha="center", size=9, color=INK)


def _blindness(ax, book) -> None:
    """Panel b: bag error against map quality over the fourteen sweep configurations."""
    sweep = pd.read_csv("data/barrido_llp_r1.csv")
    per_config = sweep.groupby("config")[["bag_mae", "spearman_within"]].mean()
    if len(per_config) != book["sweep_blindness"]["configs"]:
        raise ValueError(
            f"the sweep holds {len(per_config)} configurations against "
            f"{book['sweep_blindness']['configs']} in the canon"
        )
    correlation = agrees(
        spearmanr(per_config.bag_mae, per_config.spearman_within).statistic,
        book["sweep_blindness"]["rank_corr_mae_vs_within"],
        name="rank correlation of bag error against map quality",
    )

    ax.scatter(per_config.bag_mae, per_config.spearman_within, s=48, color=VALIDATION, zorder=3)
    ax.set_xlabel("bag error (the only signal weak supervision may use)")
    ax.set_ylabel(r"map quality (within-municipality $\rho$)")
    ax.set_title("b  Aggregate validation cannot see map quality")
    ax.text(
        0.97,
        0.95,
        f"rank correlation {correlation:.2f}\n({len(per_config)} configurations)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        size=10,
        color=INK,
    )


def draw(destination: str) -> None:
    book = canon()
    sns.set_theme(**STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    _families(axes[0], book)
    _blindness(axes[1], book)
    figure.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "docs/manuscript/figures/fig3_dissociation.pdf")
