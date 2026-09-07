"""Phase one figures: what the model sees and what it measures inside each AGEB.

The numbers of the proposal describe the process without showing it. These figures come from
the same composites and polygons that feed the baseline, so what shows on the page is
exactly what enters the model, with no illustration layer in between.

They all regenerate with `satinsight figures`, and their source is the composite cache.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import rasterize

from satinsight import cache
from satinsight.agebs import GRADES
from satinsight.pipeline import channels_s1, channels_s2, city_aoi
from satinsight.raster import stretch, to_db
from satinsight.texture import FIXED_RANGES_S1, MIN_PIXELS, features_of_patch, quantise

log = logging.getLogger(__name__)

GRADE_COLOUR = dict(
    zip(
        GRADES,
        [(27, 110, 114), (99, 162, 155), (216, 199, 154), (208, 138, 62), (168, 72, 12)],
        strict=True,
    )
)
"""Ordinal ramp from teal to orange, the same pair of accents the page uses."""

INK = (245, 245, 245)
BACKGROUND = (20, 24, 31)


def _rgb_from_composite(bands: dict, sensor: str) -> np.ndarray:
    """Eight-bit RGB array ready to display, according to the sensor."""
    if sensor == "s2":
        return np.dstack([stretch(bands[b]) for b in ("B04", "B03", "B02")])
    vv, vh = to_db(bands["vv"]), to_db(bands["vh"])
    return np.dstack([stretch(vv), stretch(vh), stretch(vv - vh)])


def _text(image: Image.Image, xy, text: str, size: int = 13, color=INK) -> None:
    """Writes a label with a dark halo so it reads over any background."""
    drawing = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
    except OSError:
        font = ImageFont.load_default()
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        drawing.text((x + dx, y + dy), text, font=font, fill=BACKGROUND)
    drawing.text(xy, text, font=font, fill=color)


def modality_panel(city: str, destination: Path, side: int = 520) -> Path:
    """The two sensors over the same box, side by side.

    This is the comparison the document makes in prose: optical has the urban fabric sharp
    at 10 m, radar sees it coarser but in a calibrated quantity.
    """
    root = Path("data") / "composites"
    panels = []
    for sensor, title in (("s2", "Sentinel-2 · optical"), ("s1", "Sentinel-1 · radar")):
        bands, _, labels = cache.load(cache.composite_path(city, sensor, root))
        rgb = _rgb_from_composite(bands, sensor)
        height, width = rgb.shape[:2]
        side_px = min(height, width)
        r0, c0 = (height - side_px) // 2, (width - side_px) // 2
        crop = rgb[r0 : r0 + side_px, c0 : c0 + side_px]
        image = Image.fromarray(crop).resize((side, side), Image.LANCZOS)
        _text(image, (10, 8), title)
        _text(
            image,
            (10, side - 22),
            f"{labels.get('scenes_used', '?')} scenes · annual median",
            11,
        )
        panels.append(image)

    canvas = Image.new("RGB", (side * 2 + 6, side), BACKGROUND)
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (side + 6, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    log.info("modality panel: %s", destination)
    return destination


def _thicken(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilates a boolean mask by shifting it. A one-pixel border is lost when scaling."""
    output = mask.copy()
    for axis in (0, 1):
        for sign in (1, -1):
            output |= np.roll(mask, sign * radius, axis=axis)
    return output


def ageb_panel(city: str, destination: Path, side: int = 760, window_px: int = 420) -> Path:
    """The composite with the AGEB borders on top, tinted by grade.

    Shows the unit of analysis resting on the pixels that feed it, which is what it takes to
    understand why the size of the polygon conditions what can be measured inside it.
    """
    root = Path("data") / "composites"
    _, agebs = city_aoi(city)
    bands, grid, _ = cache.load(cache.composite_path(city, "s2", root))
    rgb = _rgb_from_composite(bands, "s2")
    projected = agebs.to_crs(grid.crs)

    # one label per grade, to draw the borders in its colour
    labels = rasterize(
        [
            (g, GRADES.index(t) + 1)
            for g, t in zip(projected.geometry, projected.grade, strict=True)
        ],
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
    )
    border = np.zeros(grid.shape, dtype=bool)
    for axis in (0, 1):
        d = np.diff(labels, axis=axis) != 0
        border |= np.pad(d, [(0, 1) if i == axis else (0, 0) for i in range(2)])

    painted = rgb.copy()
    for index, grade in enumerate(GRADES, start=1):
        mask = _thicken(border & (labels == index))
        painted[mask] = GRADE_COLOUR[grade]

    # crop centred on the area with most AGEB, small so the border shows when enlarged
    rows, columns = np.where(labels > 0)
    cr, cc = int(np.median(rows)), int(np.median(columns))
    half = min(window_px, rgb.shape[0], rgb.shape[1]) // 2
    r0 = max(0, min(cr - half, rgb.shape[0] - 2 * half))
    c0 = max(0, min(cc - half, rgb.shape[1] - 2 * half))
    crop = painted[r0 : r0 + 2 * half, c0 : c0 + 2 * half]

    legend_height = 30
    image = Image.new("RGB", (side, side + legend_height), BACKGROUND)
    image.paste(Image.fromarray(crop).resize((side, side), Image.LANCZOS), (0, 0))
    _text(image, (10, 8), f"{city} · AGEB borders, tinted by deprivation grade")
    _text(image, (10, side - 22), f"{2 * half * 10 / 1000:.1f} km across · 10 m pixel", 11)

    drawing = ImageDraw.Draw(image)
    x = 10
    for grade in GRADES:
        drawing.rectangle([x, side + 11, x + 22, side + 19], fill=GRADE_COLOUR[grade])
        _text(image, (x + 28, side + 8), grade, 11)
        x += 34 + 8 * len(grade)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, optimize=True)
    log.info("AGEB panel: %s", destination)
    return destination


def _ageb_crop(rgb: np.ndarray, channel: np.ndarray, grid, geometry, margin: int = 6):
    """Cuts one AGEB out of the image and the channel, with a little air around it."""
    from satinsight.grid import polygon_window

    window = polygon_window(grid.transform, geometry, channel.shape)
    if window is None:
        return None
    rows, columns, inside = window
    r0 = max(0, rows.start - margin)
    c0 = max(0, columns.start - margin)
    r1 = min(channel.shape[0], rows.stop + margin)
    c1 = min(channel.shape[1], columns.stop + margin)
    return rgb[r0:r1, c0:c1], channel[rows, columns], inside


def contrast_panel(city: str, sensor: str, destination: Path, side: int = 190) -> Path:
    """Four AGEB of the same city, two of low deprivation and two of high.

    Under each one go its measured contrast and homogeneity. It is the way to see what the
    GLCM is capturing before believing a kappa.
    """
    root = Path("data") / "composites"
    _, agebs = city_aoi(city)
    bands, grid, _ = cache.load(cache.composite_path(city, sensor, root))
    rgb = _rgb_from_composite(bands, sensor)
    projected = agebs.to_crs(grid.crs)

    channels = channels_s2(bands) if sensor == "s2" else channels_s1(bands)
    channel_name = "s2nir" if sensor == "s2" else "s1vh"
    channel = channels[channel_name]
    span = FIXED_RANGES_S1.get(channel_name)
    if span is None:
        finite = channel[np.isfinite(channel)]
        span = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
    quantised = quantise(channel, span)

    # Choosing by polygon area is not enough: a coastal AGEB can be large and hold
    # a single usable radar pixel, because over water backscatter falls to zero and `to_db`
    # turns it null. The candidate is measured by valid pixels of the channel.
    def usable_pixels(geometry) -> int:
        parts = _ageb_crop(rgb, quantised, grid, geometry)
        if parts is None:
            return 0
        _, crop_q, inside = parts
        return int((inside & (crop_q > 0)).sum())

    area_px = projected.geometry.area / 100
    large = projected[area_px > MIN_PIXELS * 3].copy()
    selection = []
    for grades in (GRADES[:2], GRADES[3:]):
        candidates = large[large.grade.isin(grades)]
        valid = [f for f in candidates.itertuples() if usable_pixels(f.geometry) >= MIN_PIXELS]
        selection.extend(valid[:2])

    if len(selection) < 4:
        raise RuntimeError(
            f"{city}/{sensor}: only {len(selection)} AGEB with enough pixels "
            "at both ends of deprivation"
        )

    cell, cell_height = side, side + 34
    canvas = Image.new("RGB", (cell * 4 + 18, cell_height), BACKGROUND)
    for index, row in enumerate(selection[:4]):
        parts = _ageb_crop(rgb, quantised, grid, row.geometry)
        if parts is None:
            continue
        view, crop_q, inside = parts
        masked = np.where(inside & (crop_q > 0), crop_q, 0).astype(np.uint8)
        features = features_of_patch(masked)

        image = Image.fromarray(view).resize((cell, cell), Image.NEAREST)
        border = ImageDraw.Draw(image)
        border.rectangle([0, 0, cell - 1, cell - 1], outline=GRADE_COLOUR[row.grade], width=3)
        canvas.paste(image, (index * (cell + 6), 0))
        _text(
            canvas,
            (index * (cell + 6) + 4, cell + 4),
            row.grade,
            12,
            GRADE_COLOUR[row.grade],
        )
        _text(
            canvas,
            (index * (cell + 6) + 4, cell + 19),
            f"contrast {features['contrast_d1']:.2f} · homog {features['homogeneity_d1']:.2f}",
            10,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    log.info("contrast panel %s: %s", sensor, destination)
    return destination


def _dark_style(ax) -> None:
    """Leaves the axes with no frame or ticks, on the same background as the other panels."""
    ax.set_facecolor(_hex(BACKGROUND))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def national_map(destination: Path, root: Path | None = None) -> Path:
    """Places the five pilot cities inside the country.

    The size of each mark is its number of AGEB and the colour the share sitting at grade
    high or very high. Placed on the map, the bias of the sample shows at a glance: the
    two cities contributing high deprivation are in the south and on the Pacific coast.
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from satinsight.agebs import CITIES
    from satinsight.download import DATA_ROOT, ensure_naturalearth

    root = root or DATA_ROOT
    states = gpd.read_file(ensure_naturalearth(root))
    mexico = states[states["admin"] == "Mexico"]

    points = []
    for key in CITIES:
        area, agebs = city_aoi(key, root)
        lon = (area.bbox[0] + area.bbox[2]) / 2
        lat = (area.bbox[1] + area.bbox[3]) / 2
        high_share = float(agebs.grade.isin(GRADES[3:]).mean())
        points.append((key, CITIES[key].name, lon, lat, len(agebs), high_share))

    figure, ax = plt.subplots(figsize=(11, 7), dpi=150)
    figure.patch.set_facecolor(_hex(BACKGROUND))
    _dark_style(ax)

    mexico.plot(ax=ax, facecolor="#1d2530", edgecolor="#3b4653", linewidth=0.6)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "deprivation",
        [
            _hex(GRADE_COLOUR[GRADES[0]]),
            _hex(GRADE_COLOUR[GRADES[2]]),
            _hex(GRADE_COLOUR[GRADES[4]]),
        ],
    )
    # offsets picked by hand: with all of them placed to the right, the labels of
    # Acapulco and Tuxtla overlapped and Mérida's fell on the colour bar
    offsets = {
        "tuxtla": (14, -16),
        "merida": (-16, 12),
        "iztapalapa": (14, 6),
        "tapachula": (14, -14),
        "acapulco": (-16, -18),
    }
    alignment = {"merida": "right", "acapulco": "right"}

    for key, name, lon, lat, n, high_share in points:
        ax.scatter(
            lon,
            lat,
            s=40 + n * 0.55,
            c=[cmap(high_share / 0.5)],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        ax.annotate(
            f"{name}\n{n} AGEB · {100 * high_share:.0f}% high",
            (lon, lat),
            textcoords="offset points",
            xytext=offsets[key],
            ha=alignment.get(key, "left"),
            fontsize=8.5,
            color="#e6ebf1",
            family="monospace",
            zorder=4,
        )

    ax.set_xlim(-118.5, -85.5)
    ax.set_ylim(13.5, 33.5)
    ax.set_title(
        "The five pilot cities of phase one",
        color="#e6ebf1",
        fontsize=13,
        loc="left",
        pad=14,
    )
    bar = figure.colorbar(
        matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(0, 50), cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.03,
        pad=0.02,
        aspect=45,
    )
    bar.set_label("% of AGEB at high or very high deprivation", color="#a8b3c0", fontsize=9)
    bar.ax.tick_params(colors="#a8b3c0", labelsize=8)
    bar.outline.set_edgecolor("#3b4653")

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    log.info("national map: %s", destination)
    return destination


def agebs_by_city_map(destination: Path, root: Path | None = None) -> Path:
    """The complete set of AGEB of each city, tinted by grade and at a shared scale.

    The image panels show crops; this one shows the whole extent that enters the baseline.
    Sharing the scale in kilometres allows comparing the real size of the five urban
    footprints.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from satinsight.agebs import CITIES, METRIC_CRS

    root = root or Path("data")
    cities = list(CITIES)
    figure, axes = plt.subplots(1, len(cities), figsize=(16, 3.7), dpi=150)
    figure.patch.set_facecolor(_hex(BACKGROUND))

    for ax, key in zip(axes, cities, strict=True):
        _, agebs = city_aoi(key, root)
        metric = agebs.to_crs(METRIC_CRS)
        for grade, color in GRADE_COLOUR.items():
            part = metric[metric.grade == grade]
            if len(part):
                part.plot(ax=ax, facecolor=_hex(color), edgecolor="none")
        _dark_style(ax)
        ax.set_aspect("equal")

        x0, y0, x1, y1 = metric.total_bounds
        side = max(x1 - x0, y1 - y0) * 1.05
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - side / 2, cx + side / 2)
        ax.set_ylim(cy - side / 2, cy + side / 2)

        high_share = 100 * agebs.grade.isin(GRADES[3:]).mean()
        ax.set_title(
            f"{CITIES[key].name}\n{len(agebs)} AGEB · {high_share:.0f}% high"
            f" · {side / 1000:.0f} km",
            color="#e6ebf1",
            fontsize=9.5,
            family="monospace",
            pad=8,
        )

    figure.legend(
        handles=[Patch(facecolor=_hex(c), label=g) for g, c in GRADE_COLOUR.items()],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        labelcolor="#a8b3c0",
        bbox_to_anchor=(0.5, 0.02),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    log.info("AGEB map by city: %s", destination)
    return destination


COMPOSITING_STATES = {
    "complete": ("#4ade80", "composited in both modalities"),
    "partial": ("#fbbf24", "one modality ready"),
    "failed": ("#f87171", "aborted on failed reads"),
    "pending": ("#9aa7b6", "not started"),
}
"""Colour and gloss of each state a city's compositing can be in."""


def compositing_state(
    keys: list[str], root: Path | None = None, logs: Path | None = None
) -> dict[str, str]:
    """Classifies each city by what is on disk and what the logs say.

    A city that aborted leaves its name on a `FAIL` line of the log and no file at all,
    which makes it indistinguishable from one that has not started. The distinction matters
    because an aborted city is not retried on its own: the sweep skips it and finishes
    without flagging it. Only the lines after the last relaunch are read.
    """
    from satinsight.download import DATA_ROOT

    root = root or DATA_ROOT
    composites = root / "composites"
    failed: set[str] = set()
    for logfile in sorted((logs or root / "logs").glob("process_*.log")):
        text = logfile.read_text(errors="ignore")
        recent = text.rsplit("RELAUNCHED", 1)[-1]
        for line in recent.splitlines():
            if line.startswith("FAIL "):
                failed.add(line.split()[1])

    states = {}
    for key in keys:
        found = sum((composites / f"{key}_{s}.tif").exists() for s in ("s1", "s2"))
        if found == 2:
            states[key] = "complete"
        elif key in failed:
            # the failure outranks the lone file: a city that aborted on the second
            # modality leaves the first on disk and would look merely slow
            states[key] = "failed"
        elif found == 1:
            states[key] = "partial"
        else:
            states[key] = "pending"
    return states


def national_cities_map(
    destination: Path, root: Path | None = None, catalogue: dict | None = None, label: int = 18
) -> Path:
    """Places the cities of the national set, sized by AGEB and tinted by deprivation.

    Compositing state was what this map showed while the download ran, and once every city
    finished that variable stopped carrying information. What it shows now is the variable
    the sample was stratified on, which is what a reader needs to judge whether the set
    covers the country or only its comfortable half.

    Only the largest `label` cities are named. Naming all 138 pushes labels off the country
    and into the margin, which is worse than not naming them.
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from satinsight.agebs import cities_by_size
    from satinsight.download import DATA_ROOT, ensure_naturalearth

    root = root or DATA_ROOT
    catalogue = catalogue or cities_by_size(root=root, stratify=True)

    points = []
    for key, city in catalogue.items():
        try:
            area, agebs = city_aoi(key, root, catalogue=catalogue)
        except Exception:
            log.warning("no geometry for %s", key, exc_info=True)
            continue
        points.append(
            {
                "key": key,
                "name": city.name,
                "lon": (area.bbox[0] + area.bbox[2]) / 2,
                "lat": (area.bbox[1] + area.bbox[3]) / 2,
                "agebs": len(agebs),
                "high": float(agebs.grade.isin(GRADES[3:]).mean()),
            }
        )

    states = gpd.read_file(ensure_naturalearth(root))
    mexico = states[states["admin"] == "Mexico"]

    figure, ax = plt.subplots(figsize=(12.5, 8), dpi=150)
    figure.patch.set_facecolor(_hex(BACKGROUND))
    _dark_style(ax)
    mexico.plot(ax=ax, facecolor="#171d26", edgecolor="#333e4b", linewidth=0.5)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "deprivation",
        [
            _hex(GRADE_COLOUR[GRADES[0]]),
            _hex(GRADE_COLOUR[GRADES[2]]),
            _hex(GRADE_COLOUR[GRADES[4]]),
        ],
    )
    norm = matplotlib.colors.Normalize(0, 0.6)
    # the most deprived cities are drawn last so they are not covered by the large and
    # comfortable ones, which are the majority and occupy the centre of the country
    for point in sorted(points, key=lambda p: p["high"]):
        ax.scatter(
            point["lon"],
            point["lat"],
            s=18 + point["agebs"] * 0.22,
            c=[cmap(norm(point["high"]))],
            edgecolor="#0f1319",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )

    # labelled from largest to smallest, skipping any that lands on one already placed.
    # Pushing labels until they clear works with fifteen cities and with a hundred and
    # thirty-eight sends them out of the country; on a dense map, dropping a name is
    # better than placing it far from its point
    placed: list[tuple[float, float]] = []
    for point in sorted(points, key=lambda p: -p["agebs"]):
        if len(placed) >= label:
            break
        if any(
            abs(point["lon"] - lon) < 2.4 and abs(point["lat"] - lat) < 0.7 for lon, lat in placed
        ):
            continue
        placed.append((point["lon"], point["lat"]))
        ax.annotate(
            point["name"],
            (point["lon"], point["lat"]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=7.5,
            color="#c9d3de",
            family="monospace",
            zorder=8,
        )

    total = sum(p["agebs"] for p in points)
    share = np.average([p["high"] for p in points], weights=[p["agebs"] for p in points])
    ax.set_xlim(-118.5, -85.5)
    ax.set_ylim(13.5, 33.5)
    ax.set_title(
        f"The {len(points)} cities of the national set · {total:,} urban AGEB · "
        f"{100 * share:.0f}% at high or very high deprivation\n"
        "mark size is the number of AGEB, colour their deprived share",
        color="#e6ebf1",
        fontsize=12,
        loc="left",
        pad=14,
    )
    bar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm, cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.032,
        pad=0.02,
        aspect=45,
    )
    bar.set_label("share of AGEB at high or very high deprivation", color="#a8b3c0", fontsize=9)
    bar.ax.tick_params(colors="#a8b3c0", labelsize=8)
    bar.outline.set_edgecolor("#3b4653")

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    log.info("national map: %s", destination)
    return destination
