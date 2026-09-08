"""Figure 3: predicting the bag and localizing inside it are different abilities.

(a) Map quality by model family on identical features and labels, for both frozen
    backbones. The ramp runs from the standard weakly supervised localizer, which lands
    at chance, through label proportions and the swept head, to the oracle trained on
    tract labels. The ramp is model families, so it carries no split colors: the first
    two bars are measured on the national-set pool, the last two on the expanded pool,
    which the caption states. The second backbone is hatched.
(b) The blindness of the only selection signal weak supervision may legitimately use.
    Across the fourteen sweep configurations, bag error barely ranks map quality, so a
    pipeline whose product is the map cannot be tuned on the error it can see. Drawn for
    every backbone whose sweep exists.

Every plotted number is recomputed from the per-fold or per-seed artifacts and checked
against `docs/manuscript/canonical_results.json`; a divergence raises rather than redrawing
the page.

Usage: fig3_dissociation.py <destination.pdf>
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
from scipy.stats import spearmanr  # noqa: E402

from satinsight.manuscript import (  # noqa: E402
    BACKBONES,
    INK,
    LARGE_HATCH,
    MUTED,
    ORACLE,
    STYLE,
    VALIDATION,
    agrees,
    block,
    canon,
    suffixed,
)

CHANCE = 0.5
STEPS = (
    ("attention MIL\n(ABMIL)", MUTED),
    ("label\nproportions", "#8fb8d4"),
    ("+ sweep +\ncontext", VALIDATION),
    ("oracle\n(instance labels)", ORACLE),
)


def _mil_file(tag: str) -> str:
    return "data/mil_kfold.csv" if not tag else f"data/mil_kfold_classes_0.0_0.0_{tag}.csv"


def _curve_file(tag: str) -> str:
    return f"data/supervision_curve_s2_{tag}.csv" if tag else "data/supervision_curve.csv"


def _oracle_files(tag: str):
    # the canonical oracle files carry no split or extras tag; the fusion ablations and
    # the test column share the prefix and must stay out of the mean
    pattern = (
        "oracle_expanded_r1_s[0-9].csv"
        if not tag
        else f"oracle_expanded_r1_s[0-9]_val_s2_{tag}.csv"
    )
    return sorted(Path("data").glob(pattern))


def _families(ax, book) -> None:
    """Panel a: the map each model family produces, on identical features and labels."""
    labels = [label for label, _ in STEPS]
    x = np.arange(len(STEPS))
    width = 0.38
    for index, (tag, name) in enumerate(BACKBONES):
        own = block(book, tag)
        abmil = agrees(
            pd.read_csv(_mil_file(tag)).auroc_high.mean(),
            own["abmil"]["map_auroc"],
            name=f"{name} ABMIL map AUROC",
        )
        proportions = agrees(
            pd.read_csv(suffixed("data/llp_kfold.csv", tag)).auroc_high.mean(),
            own["abmil"]["llp_kfold_auroc"],
            name=f"{name} label proportions map AUROC",
        )
        swept = agrees(
            pd.read_csv(_curve_file(tag)).query("bags == 771").auroc_high.mean(),
            own["curve"]["771"]["auroc_high"],
            name=f"{name} swept head map AUROC",
        )
        oracle_seeds = [pd.read_csv(path).auroc.iloc[0] for path in _oracle_files(tag)]
        oracle = agrees(
            sum(oracle_seeds) / len(oracle_seeds),
            own["ceiling_auroc"]["1"],
            name=f"{name} oracle map AUROC",
        )
        values = [abmil, proportions, swept, oracle]
        shift = (index - 0.5) * width
        ax.bar(
            x + shift,
            values,
            width,
            color=[color for _, color in STEPS],
            hatch=LARGE_HATCH if tag else None,
            edgecolor="white",
            label=name,
        )
        for position, value in zip(x + shift, values, strict=True):
            ax.text(position, value + 0.008, f"{value:.3f}", ha="center", size=7.5, color=INK)
    ax.axhline(CHANCE, color=INK, ls=":", lw=1.1)
    ax.text(-0.45, CHANCE + 0.008, "chance", color=INK, size=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 0.92)
    ax.set_ylabel("map AUROC (high deprivation)")
    ax.set_title("a  The map, model by model")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=VALIDATION, edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor=VALIDATION, edgecolor="white", hatch=LARGE_HATCH),
    ]
    ax.legend(handles, [name for _, name in BACKBONES], frameon=False, fontsize=8, loc="upper left")


def _blindness(ax, book) -> None:
    """Panel b: bag error against map quality over the sweep configurations."""
    for tag, name in BACKBONES:
        path = Path(suffixed("data/llp_sweep_r1.csv", tag))
        own = block(book, tag)
        if not path.exists() or not own.get("sweep_blindness"):
            continue
        sweep = pd.read_csv(path)
        per_config = sweep.groupby("config")[["bag_mae", "spearman_within"]].mean()
        if len(per_config) != own["sweep_blindness"]["configs"]:
            raise ValueError(
                f"the {name} sweep holds {len(per_config)} configurations against "
                f"{own['sweep_blindness']['configs']} in the canon"
            )
        correlation = agrees(
            spearmanr(per_config.bag_mae, per_config.spearman_within).statistic,
            own["sweep_blindness"]["rank_corr_mae_vs_within"],
            name=f"{name} rank correlation of bag error against map quality",
        )
        ax.scatter(
            per_config.bag_mae,
            per_config.spearman_within,
            s=48,
            facecolor=VALIDATION if not tag else "white",
            edgecolor=VALIDATION,
            linewidth=1.4,
            zorder=3,
            label=f"{name}: rank correlation {correlation:.2f} ({len(per_config)} configurations)",
        )
    ax.set_xlabel("bag error (the only signal weak supervision may use)")
    ax.set_ylabel(r"map quality (within-municipality $\rho$)")
    ax.set_title("b  Aggregate validation cannot see map quality")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


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
