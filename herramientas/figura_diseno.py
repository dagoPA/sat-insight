"""Figure 1 of the manuscript: study design — where, what the model sees, what it produces.

Panel a: every municipality of the study on the national map, coloured by role. Panel b:
the true-colour composite of one held-out city with its AGEB boundaries — the imagery the
model actually consumes. Panels c and d: AGEB truth and the token-level prediction for
the same city, on one colour scale. Tapachula is the display city: median size, all five
grades present, and a within-municipality rho close to the validation mean, so the
example neither flatters nor sandbags the method.

Usage: figura_diseno.py
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
from matplotlib.patches import Rectangle  # noqa: E402

from satinsight import agebs  # noqa: E402
from satinsight.cache import load  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.tiling import TOKEN_SIZE  # noqa: E402

CITY = "tapachula"
GRADE_COLORS = ["#2166ac", "#92c5de", "#fddbc7", "#d6604d", "#b2182b"]
CMAP = colors.LinearSegmentedColormap.from_list("grs", GRADE_COLORS)

plt.rcParams.update(
    {
        "font.size": 8.5,
        "font.family": "Helvetica Neue",
        "figure.dpi": 300,
        "axes.linewidth": 0.6,
    }
)


def municipal_points() -> pd.DataFrame:
    """One centroid per study municipality with its role, cached because it takes minutes."""
    cache = DATA_ROOT / "centroides_estudio.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"municipality": str})
    partition = pd.read_csv("data/partition.csv")
    catalogue = agebs.cities_by_size(stratify=True)
    role_of_city = dict(zip(partition.ciudad, partition.split, strict=True))
    rows = []
    for key, city in catalogue.items():
        try:
            _, layer = city_aoi(key, catalogue=catalogue)
        except Exception:
            continue
        centre = layer.geometry.to_crs("EPSG:4326").union_all().centroid
        rows.append(
            {
                "municipality": city.municipality,
                "lon": centre.x,
                "lat": centre.y,
                "role": role_of_city.get(key, "train"),
            }
        )
    extra = agebs.cities_extra()
    merged = agebs.catalogue_with_extra()
    for key, city in extra.items():
        try:
            _, layer = city_aoi(key, catalogue=merged)
        except Exception:
            continue
        centre = layer.geometry.to_crs("EPSG:4326").union_all().centroid
        rows.append(
            {
                "municipality": city.municipality,
                "lon": centre.x,
                "lat": centre.y,
                "role": "expansion",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(cache, index=False)
    return table


def rgb_of(city: str) -> tuple[np.ndarray, object]:
    bands, grid, _ = load(DATA_ROOT / "composites" / f"{city}_s2.tif")
    stack = np.dstack([bands["B04"], bands["B03"], bands["B02"]])
    low, high = np.nanpercentile(stack, [2, 98])
    return np.clip((stack - low) / (high - low), 0, 1), grid


def main() -> None:
    fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])

    # a — the national map
    ax = fig.add_subplot(grid_spec[0, :])
    states = gpd.read_file("data/naturalearth/ne_10m_admin_1_states_provinces.shp")
    mexico = states[states.admin == "Mexico"]
    mexico.boundary.plot(ax=ax, color="#cccccc", linewidth=0.3)
    points = municipal_points()
    style = {
        "train": ("#9ecae1", 4, "training cities"),
        "expansion": ("#fdd0a2", 3, "expansion municipalities"),
        "val": ("#2166ac", 14, "validation cities (held out)"),
        "test": ("#b2182b", 14, "test cities (unopened)"),
    }
    for role, (colour, size, label) in style.items():
        chosen = points[points.role == role]
        ax.scatter(
            chosen.lon,
            chosen.lat,
            s=size,
            c=colour,
            label=f"{label} · {len(chosen)}",
            edgecolors="none" if size < 10 else "white",
            linewidths=0.3,
            zorder=3,
        )
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_xlim(-118, -86)
    ax.set_ylim(14, 33)
    ax.set_axis_off()
    ax.set_title(
        "a  Study municipalities and their role", loc="left", fontsize=9, fontweight="bold"
    )

    # b — the imagery with AGEB boundaries
    ax = fig.add_subplot(grid_spec[1, 0])
    rgb, grid = rgb_of(CITY)
    ax.imshow(rgb)
    catalogue = agebs.cities_by_size(stratify=True)
    _, layer = city_aoi(CITY, catalogue=catalogue)
    bounds = layer.to_crs(grid.crs)
    inverse = ~grid.transform
    for geometry in bounds.geometry:
        parts = getattr(geometry, "geoms", [geometry])
        for part in parts:
            xs, ys = part.exterior.xy
            pixels = [inverse * (x, y) for x, y in zip(xs, ys, strict=True)]
            ax.plot(
                [p[0] for p in pixels], [p[1] for p in pixels], color="white", lw=0.35, alpha=0.8
            )
    ax.add_patch(
        Rectangle(
            (40, 40), 10 * TOKEN_SIZE, 10 * TOKEN_SIZE, fill=False, edgecolor="#ffd92f", lw=1.0
        )
    )
    ax.text(44, 30, "10×10 tokens (1.6 km)", color="#ffd92f", fontsize=6.5)
    ax.set_axis_off()
    ax.set_title(
        f"b  Sentinel-2 median composite, {CITY.title()}", loc="left", fontsize=9, fontweight="bold"
    )

    # c y d — truth and prediction on one scale
    scores = pd.read_parquet("data/predicciones_val.parquet")
    tokens = (
        scores[scores.city == CITY]
        .groupby(["cvegeo", "y0", "x0"], observed=True)
        .score.mean()
        .reset_index()
    )
    grades = dict(zip(layer.cvegeo, layer.ordinal.astype(int), strict=True))

    ax = fig.add_subplot(grid_spec[1, 1])
    ax.imshow(rgb * 0.35)
    norm = colors.Normalize(vmin=0, vmax=4)
    half = TOKEN_SIZE // 2
    ax.scatter(
        tokens.x0 + half,
        tokens.y0 + half,
        c=tokens.score,
        cmap=CMAP,
        norm=norm,
        s=1.6,
        marker="s",
        linewidths=0,
    )
    ax.set_axis_off()
    ax.set_title(
        "c  Token-level prediction (weak supervision)", loc="left", fontsize=9, fontweight="bold"
    )
    colourbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, fraction=0.04, pad=0.02
    )
    colourbar.set_label("deprivation grade", fontsize=7)
    colourbar.set_ticks([0, 4])
    colourbar.set_ticklabels(["very low", "very high"])

    fig.savefig("docs/manuscript/figures/fig1_design.pdf", bbox_inches="tight")
    fig.savefig("docs/manuscript/figures/fig1_design.png", bbox_inches="tight")
    print("fig1_design saved", flush=True)

    # el gemelo de la verdad, como panel suplementario del mismo tamaño
    fig2, ax = plt.subplots(figsize=(3.6, 3.0), constrained_layout=True)
    ax.imshow(rgb * 0.35)
    truth = tokens.assign(o=tokens.cvegeo.map(grades)).dropna(subset=["o"])
    ax.scatter(
        truth.x0 + half,
        truth.y0 + half,
        c=truth.o,
        cmap=CMAP,
        norm=norm,
        s=1.6,
        marker="s",
        linewidths=0,
    )
    ax.set_axis_off()
    ax.set_title("AGEB ground truth (held out)", loc="left", fontsize=9, fontweight="bold")
    fig2.savefig("docs/manuscript/figures/fig1_truth.pdf", bbox_inches="tight")
    print("fig1_truth saved", flush=True)


if __name__ == "__main__":
    main()
