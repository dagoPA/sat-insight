"""Figure 4: four independent checks that the map measures deprivation.

(a) The leading downloadable wealth product, scored on the identical tract universe its
    grid reaches, with the paired difference and its city-clustered interval annotated
    because the difference, and not either bar, is the quantity the claim is about.
(b) Replication against an index built by another institution from a different indicator
    set, beside the map's agreement with the construct it was trained toward.
(c) The share of the aggregate-to-census targeting gap the map closes at each budget,
    people pooled over cities, including the validation budget where it loses.
(d) Colombia and Brazil trained on their own municipal aggregates, beside the Mexican
    model unadapted and fine-tuned and the fully supervised oracle, in purple because
    neither country belongs to a Mexican split.

Every panel draws both frozen backbones, the second hatched or hollow. Every plotted
number is recomputed from the committed artifacts and checked against
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
    BACKBONES,
    INK,
    LARGE_HATCH,
    MUTED,
    STYLE,
    TEST,
    TRANSFER,
    VALIDATION,
    agrees,
    block,
    canon,
    suffixed,
)

BUDGETS = (0.05, 0.10, 0.20, 0.30)
CHANCE = 0.5


def _paired_bars(ax, labels, validation: dict, test: dict, *, errors=None) -> None:
    """One group per quantity, validation and test side by side, one pair per backbone.

    `validation` and `test` map a backbone tag to the values per label; `errors`, when
    given, maps a tag to a (validation, test) pair of error lists.
    """
    x = np.arange(len(labels))
    width = 0.19
    for index, (tag, name) in enumerate(BACKBONES):
        shift = (index - 0.5) * 2 * width
        hatch = LARGE_HATCH if tag else None
        val_error = None if errors is None else errors[tag][0]
        test_error = None if errors is None else errors[tag][1]
        ax.bar(
            x + shift - width / 2,
            validation[tag],
            width,
            yerr=val_error,
            capsize=2,
            color=VALIDATION,
            hatch=hatch,
            edgecolor="white",
            ecolor=INK,
            label=f"validation, {name}",
        )
        ax.bar(
            x + shift + width / 2,
            test[tag],
            width,
            yerr=test_error,
            capsize=2,
            color=TEST,
            hatch=hatch,
            edgecolor="white",
            ecolor=INK,
            label=f"test, {name}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color=MUTED, lw=0.8)


def _incumbent(ax, book) -> None:
    """Panel a: the map and Meta's RWI on the AGEB its grid reaches."""
    validation, test, notes = {}, {}, []
    for tag, name in BACKBONES:
        own = block(book, tag)
        val_row = pd.read_csv(suffixed("data/rwi_paired.csv", tag)).iloc[0]
        test_row = pd.read_csv(suffixed("data/rwi_paired_test.csv", tag)).iloc[0]
        for row, key, split in (
            (val_row, own["rwi_paired"], "validation"),
            (test_row, own["test"]["rwi_paired"], "test"),
        ):
            agrees(row.ours_within, key["ours_within"], name=f"{name} map within on RWI, {split}")
            agrees(row.rwi_within, key["rwi_within"], name=f"{name} RWI within, {split}")
            agrees(row.difference, key["difference"], name=f"{name} paired difference, {split}")
        validation[tag] = [val_row.rwi_within, val_row.ours_within]
        test[tag] = [test_row.rwi_within, test_row.ours_within]
        notes.append(
            f"{name}: val $\\Delta$ {val_row.difference:+.2f} "
            f"[{val_row.ci_low:+.2f}, {val_row.ci_high:+.2f}], "
            f"test $\\Delta$ {test_row.difference:+.2f} "
            f"[{test_row.ci_low:+.2f}, {test_row.ci_high:+.2f}]"
        )
    _paired_bars(ax, ["Meta RWI", "this work"], validation, test)
    ax.set_ylabel(r"within-municipality $\rho$")
    ax.set_ylim(0, 0.62)
    ax.set_title("a  Incumbent, same AGEB")
    ax.text(
        0.03,
        0.97,
        "\n".join(notes),
        transform=ax.transAxes,
        ha="left",
        va="top",
        size=6.5,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")


def _curve_file(tag: str) -> str:
    return f"data/supervision_curve_s2_{tag}.csv" if tag else "data/supervision_curve.csv"


def _replication(ax, book) -> None:
    """Panel b: an independent institution's index beside the training construct."""
    validation, test, errors = {}, {}, {}
    for tag, name in BACKBONES:
        own = block(book, tag)
        own_validation = pd.read_csv(_curve_file(tag)).query("bags == 771").spearman_within
        column = pd.read_csv(suffixed("data/test_column.csv", tag))
        own_test = column[column.row == "headline_saved"].spearman_within
        agrees(
            own_validation.mean(),
            own["curve"]["771"]["spearman_within"],
            name=f"{name} map against its own construct, validation",
        )
        agrees(
            own_test.mean(),
            own["test"]["headline_token_within"],
            name=f"{name} map against its own construct, test",
        )
        conapo_validation = pd.read_csv(suffixed("data/conapo_replication.csv", tag)).iloc[0]
        conapo_test = pd.read_csv(suffixed("data/conapo_replication_test.csv", tag)).iloc[0]
        agrees(
            conapo_validation.spearman_within,
            own["conapo"]["spearman_within"],
            name=f"{name} CONAPO replication, validation",
        )
        agrees(
            conapo_test.spearman_within,
            own["test"]["conapo"]["spearman_within"],
            name=f"{name} CONAPO replication, test",
        )
        validation[tag] = [own_validation.mean(), conapo_validation.spearman_within]
        test[tag] = [own_test.mean(), conapo_test.spearman_within]
        errors[tag] = (
            [own_validation.std(), conapo_validation.ci95_half],
            [own_test.std(), conapo_test.ci95_half],
        )
    _paired_bars(ax, ["CONEVAL\ngrade", "CONAPO\nindex"], validation, test, errors=errors)
    ax.set_ylim(0, 0.5)
    ax.set_title("b  Replication")


def _targeting(ax, book) -> None:
    """Panel c: share of the aggregate-to-census gap closed, people pooled per split."""
    x = [100 * budget for budget in BUDGETS]
    for tag, name in BACKBONES:
        own = block(book, tag)
        for split, path, expected, color, marker in (
            ("validation", "data/targeting.csv", own["targeting_pooled"], VALIDATION, "o"),
            ("test", "data/targeting_test.csv", own["test"]["targeting"], TEST, "s"),
        ):
            table = (
                pd.read_csv(suffixed(path, tag))
                .groupby("budget")[["aggregate", "map", "oracle"]]
                .sum()
            )
            pooled = (table["map"] - table["aggregate"]) / (table["oracle"] - table["aggregate"])
            line = [
                agrees(pooled.loc[b], expected[str(b)], name=f"{name} targeting {split} at {b}")
                for b in BUDGETS
            ]
            ax.plot(
                x,
                [100 * v for v in line],
                f"{marker}-" if not tag else f"{marker}--",
                color=color,
                markerfacecolor=color if not tag else "white",
                lw=2 if not tag else 1.4,
                label=f"{split}, {name}",
            )
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x)
    ax.set_xlabel("budget (% population)")
    ax.set_ylabel("gap closed (%)")
    ax.set_title("c  Targeting")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")


METHODS = (
    ("zero-shot", "Mexican\nunadapted"),
    ("aggregates", "local\naggregates"),
    ("mexico-init", "Mexican\nfine-tuned"),
    ("oracle", "oracle"),
)


def _transfer(ax, book) -> None:
    """Panel d: each country trained on its own municipal aggregates, with the references.

    Within-municipality Spearman against the fine truth under grouped folds; the oracle
    is drawn as the same bar family so the recovered fraction reads off the panel. One
    bar per method and backbone; a backbone whose transfer run is absent is skipped.
    """
    available = [
        (tag, name)
        for tag, name in BACKBONES
        if Path(suffixed("data/transfer_training.csv", tag)).exists()
        and block(book, tag).get("transfer_training")
    ]
    width = 0.8 / (len(METHODS) * len(available))
    countries = (("colombia", "Bogot\u00e1 · block strata"), ("brazil", "Brazil · tract income"))
    for index, (country, _label) in enumerate(countries):
        slot = 0
        for method, _ in METHODS:
            for tag, name in available:
                table = pd.read_csv(suffixed("data/transfer_training.csv", tag))
                stored = block(book, tag)["transfer_training"]
                rows = table[table.country == country]
                seeds = rows[rows.method == method].sort_values("seed").within
                expected = stored[country]["methods"][method]
                mean = agrees(
                    seeds.mean(), expected["within"], name=f"{name} transfer {country} {method}"
                )
                x = index - 0.4 + (slot + 0.5) * width
                slot += 1
                color = MUTED if method == "oracle" else TRANSFER
                alpha = 1.0 if method in ("aggregates", "oracle") else 0.55
                ax.bar(
                    x,
                    mean,
                    width * 0.92,
                    yerr=seeds.std() if len(seeds) > 1 else None,
                    capsize=2,
                    color=color,
                    alpha=alpha,
                    hatch=LARGE_HATCH if tag else None,
                    edgecolor="white",
                    ecolor=INK,
                )
                top = mean + (seeds.std() if len(seeds) > 1 else 0.0)
                ax.text(x, top + 0.012, f"{mean:.2f}", ha="center", size=6, color=INK)
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels([label for _, label in countries])
    ax.set_ylabel(r"within-municipality $\rho$")
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("d  Own aggregates abroad")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TRANSFER, alpha=0.55),
        plt.Rectangle((0, 0), 1, 1, color=TRANSFER),
        plt.Rectangle((0, 0), 1, 1, color=MUTED),
        plt.Rectangle((0, 0), 1, 1, facecolor=TRANSFER, edgecolor="white", hatch=LARGE_HATCH),
    ]
    ax.legend(
        handles,
        ["Mexican, unadapted / fine-tuned", "trained on local aggregates", "oracle", "DOFA large"],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )


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
