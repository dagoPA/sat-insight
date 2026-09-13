"""Figure 9: the design carried across Mexico, Colombia and Brazil.

One row per country. The left panel places every municipality the model mapped at its
centre, colored by the mean deprivation score of its tokens, which shows both the reach
of the catalogue and the gradient the model reads between municipalities. The right panel
zooms into one city of that country at the 160 m token lattice, where the quantity the
paper claims lives: the ordering inside a municipality.

Mexico is scored by the heads trained on its own 771 municipal aggregates, on
municipalities that never entered training. Colombia and Brazil are scored by those same
Mexican heads applied unadapted, which is the zero-shot arm of the transfer section.

Usage: fig9_three_countries.py [output stem]
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from satinsight.manuscript import BACKBONES, INK, STYLE  # noqa: E402

TAG, EXTRACTOR = BACKBONES[0]
STEM = sys.argv[1] if len(sys.argv) > 1 else "docs/manuscript/figures/fig9_three_countries"
ZOOM = {
    "mexico": ("tuxtla", "Tuxtla Guti\u00e9rrez"),
    "colombia": ("bogota", "Bogot\u00e1"),
    "brazil": ("riodejaneiro", "Rio de Janeiro"),
}
"""The city each row zooms into, one per country, and the name to print."""
TRAINED_ON = ("train", "expansion")
"""Splits whose municipalities supplied bag labels; the rest are out of sample."""
MIN_TOKENS = 20
BUILT_FLOOR = 0.10
"""Built-up share a transfer token needs, the floor of the transfer protocol."""


def mexican_tokens() -> pd.DataFrame:
    table = pd.read_parquet(f"data/national_scores_{TAG}.parquet")
    return table.assign(country="mexico", built=1.0)


def foreign_tokens() -> pd.DataFrame:
    """The transfer tokens the protocol keeps: those with built-up ground under them."""
    table = pd.read_parquet(f"data/transfer_scores_{TAG}.parquet")
    return table[table.built >= BUILT_FLOOR] if "built" in table.columns else table


def municipal(tokens: pd.DataFrame) -> pd.DataFrame:
    """One row per municipality: where it is, how big it is, and its mean score."""
    grouped = tokens.groupby(["country", "municipality"], observed=True)
    aggregation = {
        "lon": ("lon", "mean"),
        "lat": ("lat", "mean"),
        "tokens": ("score", "size"),
        "score": ("score", "mean"),
    }
    if "split" in tokens.columns:
        aggregation["split"] = ("split", "first")
    out = grouped.agg(**aggregation).reset_index()
    return out[out.tokens >= MIN_TOKENS]


def paint_country(ax, points: pd.DataFrame, title: str):
    low, high = points.score.quantile([0.02, 0.98])
    drawn = ax.scatter(
        points.lon,
        points.lat,
        c=points.score.clip(low, high),
        s=np.clip(points.tokens / 60, 3, 45),
        cmap="RdYlBu_r",
        linewidths=0,
        alpha=0.85,
    )
    ax.set_title(title, color=INK, loc="left")
    ax.set_aspect(1 / np.cos(np.radians(points.lat.mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return drawn


def paint_city(ax, tokens: pd.DataFrame, title: str) -> None:
    low, high = tokens.score.quantile([0.02, 0.98])
    ax.scatter(
        tokens.lon,
        tokens.lat,
        c=tokens.score.clip(low, high),
        s=1.4,
        cmap="RdYlBu_r",
        linewidths=0,
        marker="s",
    )
    ax.set_title(title, color=INK, loc="left")
    ax.set_aspect(1 / np.cos(np.radians(tokens.lat.mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def main() -> None:
    sns.set_theme(**STYLE)
    mexico, foreign = mexican_tokens(), foreign_tokens()
    tokens = pd.concat([mexico, foreign], ignore_index=True)
    places = municipal(tokens)

    figure, axes = plt.subplots(3, 2, figsize=(11, 13.5), gridspec_kw={"width_ratios": [2, 1]})
    names = {"mexico": "Mexico", "colombia": "Colombia", "brazil": "Brazil"}
    painted = None
    for row, (country, label) in enumerate(names.items()):
        points = places[places.country == country]
        drawn = tokens[tokens.country == country]
        reach = f"{len(points):,} municipalities mapped"
        if "split" in points.columns and points.split.notna().any():
            held = int((~points.split.isin(TRAINED_ON)).sum())
            reach += f", {held:,} of them outside the training bags"
        painted = paint_country(axes[row][0], points, f"{chr(97 + row * 2)}, {label}: {reach}")
        key, city_name = ZOOM[country]
        city = drawn[drawn.key == key] if "key" in drawn.columns else drawn.iloc[:0]
        if city.empty:
            axes[row][1].axis("off")
            continue
        paint_city(
            axes[row][1],
            city,
            f"{chr(98 + row * 2)}, {city_name}: {len(city):,} tokens of 160 m",
        )
    figure.suptitle(
        f"Deprivation maps trained on municipal aggregates alone, {EXTRACTOR} features",
        color=INK,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    bar = figure.colorbar(
        painted, ax=axes, orientation="horizontal", fraction=0.02, pad=0.02, location="bottom"
    )
    bar.set_label("predicted deprivation, expected grade over the five census levels", color=INK)
    for extension in ("pdf", "png"):
        figure.savefig(f"{STEM}.{extension}", dpi=200, bbox_inches="tight")
    print(
        f"{STEM}.pdf · "
        + " · ".join(
            f"{names[c]} {int((places.country == c).sum())} municipalities" for c in names
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
