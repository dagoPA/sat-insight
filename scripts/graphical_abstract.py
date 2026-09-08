"""Graphical abstract: what goes in, the three ways of learning from it, what comes out.

The page is an HTML layout (`graphical_abstract.html` beside this script) filled with the
project's own artifacts: the municipal aggregates on the national map, the Sentinel-2 and
Sentinel-1 composites of the display city with the token window, and the token map beside
the tract truth it is scored against. The numbers in the text come from the canonical
results file, so the abstract cannot drift from the paper. The images are rendered here,
inlined as data URIs, and the page is rasterized with headless Chrome when one is
installed; the standalone HTML is always written, so the page can be opened or exported by
hand where Chrome is absent.

Follows the Elsevier proportions (2600 by 960 px, about 2.7:1).

Usage: graphical_abstract.py [city]
"""

import base64
import io
import logging
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

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
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from satinsight import agebs  # noqa: E402
from satinsight.cache import load  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.manuscript import canon  # noqa: E402
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.raster import stretch, to_db  # noqa: E402
from satinsight.tiling import TOKEN_SIZE  # noqa: E402

sys.path.insert(0, "scripts")
from fig1_design import CMAP, municipal_points, rgb_of, token_raster  # noqa: E402

CITY = sys.argv[1] if len(sys.argv) > 1 else "tapachula"
"""Tapachula by default: Figure 1 already shows Acámbaro, and a second city shows more."""
TEMPLATE = Path("scripts/graphical_abstract.html")
OUT = Path("docs/manuscript/figures/graphical_abstract")
CHROME = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
)
WIDTH, HEIGHT = 2600, 960


def municipal_grades() -> pd.DataFrame:
    """One point per study unit with the aggregate grade of its seat municipality."""
    points = municipal_points()
    grades = pd.concat(
        pd.read_parquet(path, columns=["municipality", "ordinal"])
        for path in sorted((DATA_ROOT / "bags").glob("*.parquet"))
    )
    labels = grades.drop_duplicates("municipality").set_index("municipality").ordinal
    points["grade"] = points.municipality.map(labels)
    return points.dropna(subset=["grade"])


def encode(figure, *, width: int, quality: int = 68, background=(250, 248, 243)) -> str:
    """Rasterizes a figure, shrinks it to `width`, and returns a JPEG data URI."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(figure)
    image = Image.open(buffer).convert("RGBA")
    flat = Image.new("RGB", image.size, background)
    flat.paste(image, mask=image.split()[-1])
    flat.thumbnail((width, width * 4))
    out = io.BytesIO()
    flat.save(out, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def national_map(norm) -> str:
    figure, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    ax.set_position([0, 0, 1, 1])
    states = gpd.read_file("data/naturalearth/ne_10m_admin_1_states_provinces.shp")
    mexico = states[states.admin == "Mexico"]
    mexico.plot(ax=ax, facecolor="#efeae1", edgecolor="#cfc8bc", linewidth=0.4)
    points = municipal_grades()
    ax.scatter(points.lon, points.lat, c=points.grade, cmap=CMAP, norm=norm, s=16, linewidths=0)
    ax.set_xlim(-118, -86)
    ax.set_ylim(14, 33)
    ax.set_axis_off()
    figure.patch.set_alpha(0)
    return encode(figure, width=900, quality=84)


def city_panels(norm) -> dict[str, str]:
    rgb, grid = rgb_of(CITY)
    catalogue = agebs.cities_by_size(stratify=True)
    _, layer = city_aoi(CITY, catalogue=catalogue)
    bounds = layer.to_crs(grid.crs)
    inverse = ~grid.transform
    height, width = rgb.shape[:2]
    scores = pd.read_parquet("data/predictions_val.parquet")
    tokens = (
        scores[scores.city == CITY]
        .groupby(["cvegeo", "y0", "x0"], observed=True)
        .score.mean()
        .reset_index()
    )
    radar_bands, _, _ = load(DATA_ROOT / "composites" / f"{CITY}_s1.tif")
    vv, vh = to_db(radar_bands["vv"]), to_db(radar_bands["vh"])
    radar = np.dstack([stretch(vv), stretch(vh), stretch(vv - vh)])

    def panel(draw) -> str:
        figure, ax = plt.subplots(figsize=(4, 4 * height / width), dpi=150)
        draw(ax)
        ax.set_axis_off()
        return encode(figure, width=470, quality=64)

    def optical(ax):
        ax.imshow(rgb)
        ax.add_patch(
            Rectangle(
                (24, 24), 10 * TOKEN_SIZE, 10 * TOKEN_SIZE, fill=False, edgecolor="#ffd92f", lw=1.6
            )
        )

    def prediction(ax):
        ax.imshow(rgb * 0.35)
        ax.imshow(
            token_raster(tokens, tokens.score.to_numpy(), (height, width)),
            cmap=CMAP,
            norm=norm,
            interpolation="nearest",
        )

    def truth(ax):
        ax.imshow(rgb * 0.35)
        for geometry, ordinal in zip(bounds.geometry, bounds.ordinal.astype(int), strict=True):
            for part in getattr(geometry, "geoms", [geometry]):
                xs, ys = part.exterior.xy
                pixels = [inverse * (x, y) for x, y in zip(xs, ys, strict=True)]
                ax.add_patch(
                    Polygon(pixels, closed=True, facecolor=CMAP(norm(ordinal)), edgecolor="none")
                )
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    return {
        "optical.jpg": panel(optical),
        "radar.jpg": panel(lambda ax: ax.imshow(radar)),
        "prediction.jpg": panel(prediction),
        "truth.jpg": panel(truth),
    }


def ramp() -> str:
    figure, ax = plt.subplots(figsize=(3, 0.35), dpi=150)
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=CMAP)
    ax.set_axis_off()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def rasterize(page: Path, destination: Path) -> bool:
    """Screenshots the page at 2x with headless Chrome; False when no Chrome is found."""
    binary = next((c for c in CHROME if Path(c).exists() or shutil.which(c)), None)
    if binary is None:
        return False
    subprocess.run(
        [
            binary,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--window-size={WIDTH},{HEIGHT}",
            "--virtual-time-budget=8000",
            f"--screenshot={destination}",
            f"file://{page.resolve()}",
        ],
        check=True,
        capture_output=True,
    )
    return True


def main() -> None:
    book = canon()
    norm = colors.Normalize(vmin=0, vmax=4)
    page = TEMPLATE.read_text()
    images = {"map.jpg": national_map(norm), "ramp.png": ramp(), **city_panels(norm)}
    for name, uri in images.items():
        assert page.count(f'src="{name}"') == 1, name
        page = page.replace(f'src="{name}"', f'src="{uri}"')
    numbers = {
        "[[RECOVERED]]": f"{book['test']['fraction']:.0%}",
        "[[BAGS]]": str(book["pool"]["bags_clean"]),
        "[[ABMIL_AUROC]]": f"{book['abmil']['map_auroc']:.2f}",
        "[[LLP_TEST]]": f"{book['test']['headline_token_within']:.2f}",
        "[[ORACLE_TEST]]": f"{book['test']['ceiling_r1']:.2f}",
        "[[RECOVERED_LARGE]]": f"{book['dofal']['test']['fraction']:.0%}",
        "[[ABMIL_AUROC_LARGE]]": f"{book['dofal']['abmil']['map_auroc']:.2f}",
        "[[LLP_TEST_LARGE]]": f"{book['dofal']['test']['headline_token_within']:.2f}",
        "[[ORACLE_TEST_LARGE]]": f"{book['dofal']['test']['ceiling_r1']:.2f}",
        "[[CITY]]": agebs.cities_by_size(stratify=True)[CITY].name,
    }
    for key, value in numbers.items():
        assert page.count(key) == 1, key
        page = page.replace(key, value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = OUT.with_suffix(".html")
    html.write_text(page)
    if rasterize(html, OUT.with_suffix(".png")):
        print(f"{OUT}.png rendered for {CITY}", flush=True)
    else:
        print(f"{html} written; no Chrome found to rasterize it", flush=True)


if __name__ == "__main__":
    main()
