"""Figure 6: the map beside the truth in six cities, for both backbones.

One row per city, three panels: the tract truth as AGEB polygons, and the prediction of
each backbone averaged to the tract, the unit every evaluation scores. Two validation
cities and four held-out test cities: the best and the median of validation, and on test
the two highest and the two lowest within-municipality correlations of the audit table,
so the reader sees the map where it works and where it does not. The per-tract
correlation of each panel is annotated, recomputed from the persisted scores.

Usage: fig6_gallery.py [output stem]
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from satinsight import agebs  # noqa: E402
from satinsight.agebs import load_grs  # noqa: E402
from satinsight.manuscript import BACKBONES, suffixed  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402

sys.path.insert(0, "scripts")
from backbone_paired import within  # noqa: E402
from fig1_design import CMAP, rgb_of  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/manuscript/figures/fig6_gallery"
TEST_CITIES = ("igualadelaindependencia", "lapaz", "guerrero", "escarcega")

plt.rcParams.update({"font.size": 8, "font.family": "Helvetica Neue", "figure.dpi": 300})


def tract_scores(split: str, tag: str) -> pd.DataFrame:
    """Seed-mean score per token of one split and backbone."""
    scores = pd.read_parquet(suffixed(f"data/predictions_{split}.parquet", tag))
    return (
        scores.groupby(["city", "municipality", "cvegeo", "y0", "x0"], observed=True)
        .score.mean()
        .reset_index()
    )


MIN_TRACTS = 20
"""Municipalities with fewer graded tracts are not scored, as in the audit table."""


def city_rho(table: pd.DataFrame, grades: pd.Series) -> pd.Series:
    """Per-city mean of the per-municipality tract-level correlation, audit rule."""
    per = within(table, grades)
    graded = table.assign(grade=table.cvegeo.map(grades)).dropna(subset=["grade"])
    tracts = graded.groupby("municipality").cvegeo.nunique()
    per = per[per.municipality.map(tracts).fillna(0) >= MIN_TRACTS]
    return per.groupby("city").ageb.mean()


def validation_pick(rho: pd.Series) -> list[str]:
    """The best validation city and the one nearest the median."""
    ordered = rho.dropna().sort_values()
    median_city = ordered.index[len(ordered) // 2]
    best = ordered.index[-1]
    return [best, median_city] if best != median_city else [best, ordered.index[-2]]


def fill(ax, rgb, bounds, inverse, values, norm) -> None:
    ax.imshow(rgb * 0.35)
    for geometry, value in zip(bounds.geometry, values, strict=True):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        for part in getattr(geometry, "geoms", [geometry]):
            xs, ys = part.exterior.xy
            pixels = [inverse * (x, y) for x, y in zip(xs, ys, strict=True)]
            ax.add_patch(
                Polygon(pixels, closed=True, facecolor=CMAP(norm(value)), edgecolor="none")
            )
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.set_axis_off()


def main() -> None:
    grades = load_grs().set_index("cvegeo").ordinal.astype(float)
    catalogue = agebs.cities_by_size(stratify=True)
    tables = {
        (split, tag): tract_scores(split, tag) for split in ("val", "test") for tag, _ in BACKBONES
    }
    rhos = {key: city_rho(table, grades) for key, table in tables.items()}
    cities = [(c, "val") for c in validation_pick(rhos[("val", "")])] + [
        (c, "test") for c in TEST_CITIES
    ]
    norm = colors.Normalize(vmin=0, vmax=4)

    fig, axes = plt.subplots(
        len(cities), 3, figsize=(6.4, 2.1 * len(cities)), constrained_layout=True
    )
    for row, (city, split) in enumerate(cities):
        rgb, grid = rgb_of(city)
        _, layer = city_aoi(city, catalogue=catalogue)
        bounds = layer.to_crs(grid.crs)
        inverse = ~grid.transform
        name = catalogue[city].name
        fill(axes[row, 0], rgb, bounds, inverse, list(bounds.ordinal.astype(int)), norm)
        axes[row, 0].set_title(
            f"{name} ({'validation' if split == 'val' else 'test'})\ntract truth",
            loc="left",
            fontsize=7.5,
        )
        for column, (tag, label) in enumerate(BACKBONES, start=1):
            table = tables[(split, tag)]
            tract_mean = table[table.city == city].groupby("cvegeo").score.mean()
            fill(
                axes[row, column],
                rgb,
                bounds,
                inverse,
                [tract_mean.get(k, np.nan) for k in bounds.cvegeo],
                norm,
            )
            rho = rhos[(split, tag)].get(city, np.nan)
            axes[row, column].set_title(
                f"{label}\nper-tract $\\rho$ = {rho:.2f}", loc="left", fontsize=7.5
            )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    bar = fig.colorbar(
        mappable, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.015, pad=0.01
    )
    bar.set_ticks([0, 4])
    bar.set_ticklabels(["very low", "very high"])
    bar.set_label("deprivation grade", fontsize=7)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", bbox_inches="tight")
    print(f"{OUT} saved for {[c for c, _ in cities]}", flush=True)


if __name__ == "__main__":
    main()
