"""Figure 1 of the manuscript: study design, where, what the model sees, what it produces.

Panel a: every municipality of the study on the national map, colored by role. Panel b:
the true-color composite of one held-out city with its AGEB boundaries, the imagery the
model actually consumes. Panels c and d: the held-out tract truth and the token-level
prediction for the same city, painted as one continuous lattice of 160 m cells on one
color scale. Acámbaro is the display city: all five grades present and the highest
within-municipality rho among validation municipalities that hold every grade, so the
example shows what the map looks like where it works, with the median stated in the
caption.

Usage: fig1_design.py [city] [output stem]
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

CITY = sys.argv[1] if len(sys.argv) > 1 else "acambaro"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/manuscript/figures/fig1_design"
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
    cache = DATA_ROOT / "study_centroids.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"municipality": str})
    partition = pd.read_csv("data/partition.csv")
    catalogue = agebs.cities_by_size(stratify=True)
    role_of_city = dict(zip(partition.city, partition.split, strict=True))
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


def token_raster(tokens: pd.DataFrame, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Values painted as a continuous lattice of 160 m cells over the pixel grid.

    Each token fills its whole 16 by 16 pixel footprint, so the map reads as a surface
    with no gaps between cells; ground with no token stays transparent.
    """
    raster = np.full(shape, np.nan, dtype="float32")
    for y0, x0, value in zip(tokens.y0, tokens.x0, values, strict=True):
        raster[y0 : y0 + TOKEN_SIZE, x0 : x0 + TOKEN_SIZE] = value
    return raster


def main() -> None:
    fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])

    # a, the national map
    ax = fig.add_subplot(grid_spec[0, :])
    states = gpd.read_file("data/naturalearth/ne_10m_admin_1_states_provinces.shp")
    mexico = states[states.admin == "Mexico"]
    mexico.boundary.plot(ax=ax, color="#cccccc", linewidth=0.3)
    points = municipal_points()
    style = {
        "train": ("#9ecae1", 4, "training cities"),
        "expansion": ("#fdd0a2", 3, "expansion municipalities"),
        "val": ("#2166ac", 14, "validation cities (held out)"),
        "test": ("#b2182b", 14, "test cities (held out)"),
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

    # b, the imagery with AGEB boundaries
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
    ax.set_title(f"b  Composite, {catalogue[CITY].name}", loc="left", fontsize=9, fontweight="bold")

    # c and d, truth and prediction as one lattice on one scale
    scores = pd.read_parquet("data/predictions_val.parquet")
    tokens = (
        scores[scores.city == CITY]
        .groupby(["cvegeo", "y0", "x0"], observed=True)
        .score.mean()
        .reset_index()
    )
    grades = dict(zip(layer.cvegeo, layer.ordinal.astype(int), strict=True))
    norm = colors.Normalize(vmin=0, vmax=4)
    shape = rgb.shape[:2]
    truth = tokens.cvegeo.map(grades).to_numpy(dtype="float32")
    panels = (
        ("c  Tract truth (held out)", token_raster(tokens, truth, shape)),
        (
            "d  Prediction, weak supervision",
            token_raster(tokens, tokens.score.to_numpy(), shape),
        ),
    )
    for column, (title, raster) in enumerate(panels, start=1):
        ax = fig.add_subplot(grid_spec[1, column])
        ax.imshow(rgb * 0.35)
        image = ax.imshow(raster, cmap=CMAP, norm=norm, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    colourbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colourbar.set_label("deprivation grade", fontsize=7)
    colourbar.set_ticks([0, 4])
    colourbar.set_ticklabels(["very low", "very high"])

    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", bbox_inches="tight")
    print(f"{OUT} saved for {CITY}", flush=True)


if __name__ == "__main__":
    main()
