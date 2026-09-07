"""Graphical abstract: what goes in, the three ways of learning from it, what comes out.

Built from the project's own artifacts rather than illustrations: the municipal
aggregates on the national map, the Sentinel-2 and Sentinel-1 composites of the display
city with the token lattice, and the token map beside the tract truth it is scored
against. The middle column names the three approaches compared in the paper with the map
quality each reaches, so the reader sees in one glance why prediction and aggregation are
ordered the way they are.

Follows the Elsevier proportions (about 2.5:1) and is meant to be read at column width.

Usage: graphical_abstract.py [city]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from satinsight import agebs  # noqa: E402
from satinsight.cache import load  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.manuscript import canon  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.raster import stretch, to_db  # noqa: E402
from satinsight.tiling import TOKEN_SIZE  # noqa: E402

sys.path.insert(0, "scripts")
from fig1_design import CMAP, municipal_points, rgb_of, token_raster  # noqa: E402

CITY = sys.argv[1] if len(sys.argv) > 1 else "acambaro"
OUT = "docs/manuscript/figures/graphical_abstract"
INK = "#222222"
BOX = "#f4f4f4"
plt.rcParams.update({"font.size": 8, "font.family": "Helvetica Neue", "figure.dpi": 300})


def municipal_grades() -> pd.DataFrame:
    """One point per study municipality with the aggregate grade the model trained on."""
    points = municipal_points()
    grades = []
    for path in sorted((DATA_ROOT / "bags").glob("*.parquet")):
        table = pd.read_parquet(path, columns=["municipality", "ordinal"])
        grades.append(table)
    labels = pd.concat(grades).drop_duplicates("municipality").set_index("municipality").ordinal
    points["grade"] = points.municipality.map(labels)
    return points.dropna(subset=["grade"])


def radar_rgb(city: str) -> np.ndarray:
    bands, _, _ = load(DATA_ROOT / "composites" / f"{city}_s1.tif")
    vv, vh = to_db(bands["vv"]), to_db(bands["vh"])
    return np.dstack([stretch(vv), stretch(vh), stretch(vv - vh)])


def label(ax, text, y=1.02, size=8.5):
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", pad=3)
    ax.set_axis_off()


def arrow(fig, x0, x1, y, text=None):
    fig.patches.append(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            color=INK,
            lw=1.2,
        )
    )
    if text:
        fig.text((x0 + x1) / 2, y + 0.03, text, ha="center", va="bottom", fontsize=7, color=INK)


def main() -> None:
    book = canon()
    fig = plt.figure(figsize=(13, 5.2))

    # 1. what the model is told: one aggregate grade per municipality
    ax = fig.add_axes([0.005, 0.04, 0.245, 0.9])
    states = gpd.read_file("data/naturalearth/ne_10m_admin_1_states_provinces.shp")
    states[states.admin == "Mexico"].boundary.plot(ax=ax, color="#d0d0d0", linewidth=0.3)
    points = municipal_grades()
    norm = colors.Normalize(vmin=0, vmax=4)
    ax.scatter(points.lon, points.lat, c=points.grade, cmap=CMAP, norm=norm, s=7, linewidths=0)
    ax.set_xlim(-118, -86)
    ax.set_ylim(14, 33)
    label(ax, "1  Published municipal aggregates")
    ax.text(
        0.02,
        0.02,
        f"{len(points)} study units, 771 municipal bags,\n"
        "one deprivation grade each,\nno spatial annotation of any kind",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )

    # 2. what it sees: the two free composites, tiled into 160 m tokens
    rgb, grid = rgb_of(CITY)
    catalogue = agebs.cities_by_size(stratify=True)
    _, layer = city_aoi(CITY, catalogue=catalogue)
    for index, (image, name) in enumerate(
        ((rgb, "Sentinel-2 optical"), (radar_rgb(CITY), "Sentinel-1 radar"))
    ):
        ax = fig.add_axes([0.285, 0.52 - 0.42 * index, 0.16, 0.36])
        ax.imshow(image)
        ax.add_patch(
            Rectangle(
                (24, 24),
                10 * TOKEN_SIZE,
                10 * TOKEN_SIZE,
                fill=False,
                edgecolor="#ffd92f",
                lw=0.9,
            )
        )
        ax.text(28, 18, "10×10 tokens, 1.6 km", color="#ffd92f", fontsize=5.5)
        label(ax, ("2  Free imagery, 160 m tokens" if index == 0 else "") or " ", size=8.5)
        ax.text(0.02, 0.03, name, transform=ax.transAxes, color="white", fontsize=7)
    arrow(fig, 0.255, 0.283, 0.5)

    # 3. the three ways of learning from one label per bag
    ax = fig.add_axes([0.475, 0.08, 0.24, 0.84])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    label(ax, "3  Three ways to learn from a bag label")
    rows = (
        (
            "Attention MIL (ABMIL, CLAM)",
            "pool tokens, then predict the bag",
            f"map AUROC {book['abmil']['map_auroc']:.2f}: chance",
            "#b2182b",
        ),
        (
            "Label proportions (this work)",
            "predict every token, average, match the share",
            f"map AUROC {book['curve']['771']['auroc_high']:.2f} · within-municipality ρ "
            f"{book['test']['headline_token_within']:.2f} on test",
            "#2166ac",
        ),
        (
            "Fully supervised oracle",
            "same head, tract labels: the upper bound",
            f"within-municipality ρ {book['test']['ceiling_r1']:.2f} on test",
            "#555555",
        ),
    )
    for index, (title, how, result, colour) in enumerate(rows):
        y = 0.86 - index * 0.31
        ax.add_patch(
            FancyBboxPatch(
                (0.02, y - 0.24),
                0.96,
                0.26,
                boxstyle="round,pad=0.01",
                facecolor=BOX,
                edgecolor=colour,
                lw=1.2,
            )
        )
        ax.text(0.06, y - 0.02, title, fontsize=8, fontweight="bold", color=colour, va="top")
        ax.text(0.06, y - 0.10, how, fontsize=7, va="top", color=INK)
        ax.text(0.06, y - 0.17, result, fontsize=7, va="top", color=INK)
    arrow(fig, 0.45, 0.473, 0.5)

    # 4. what comes out, beside the truth it never saw
    scores = pd.read_parquet("data/predictions_val.parquet")
    tokens = (
        scores[scores.city == CITY]
        .groupby(["cvegeo", "y0", "x0"], observed=True)
        .score.mean()
        .reset_index()
    )
    shape = rgb.shape[:2]
    bounds = layer.to_crs(grid.crs)
    inverse = ~grid.transform
    for index, (title, kind) in enumerate(
        (("4  Map, from aggregates alone", "map"), ("Tract truth, never trained on", "truth"))
    ):
        ax = fig.add_axes([0.745, 0.52 - 0.42 * index, 0.16, 0.36])
        ax.imshow(rgb * 0.35)
        if kind == "map":
            ax.imshow(
                token_raster(tokens, tokens.score.to_numpy(), shape),
                cmap=CMAP,
                norm=norm,
                interpolation="nearest",
            )
        else:
            for geometry, ordinal in zip(bounds.geometry, bounds.ordinal.astype(int), strict=True):
                for part in getattr(geometry, "geoms", [geometry]):
                    xs, ys = part.exterior.xy
                    pixels = [inverse * (x, y) for x, y in zip(xs, ys, strict=True)]
                    ax.add_patch(
                        Polygon(
                            pixels, closed=True, facecolor=CMAP(norm(ordinal)), edgecolor="none"
                        )
                    )
            ax.set_xlim(0, shape[1])
            ax.set_ylim(shape[0], 0)
        label(ax, title)
    arrow(fig, 0.72, 0.743, 0.5)
    cax = fig.add_axes([0.915, 0.14, 0.012, 0.72])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    bar.set_ticks([0, 4])
    bar.set_ticklabels(["very low", "very high"])
    bar.set_label("deprivation grade", fontsize=7)
    fig.text(
        0.745,
        0.035,
        f"{book['test']['fraction']:.0%} of the fully supervised bound recovered on held-out "
        "cities; replicated on Colombia's and Brazil's own aggregates",
        fontsize=7,
        color=INK,
    )

    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", bbox_inches="tight", dpi=300)
    print(f"{OUT} saved", flush=True)


if __name__ == "__main__":
    main()
