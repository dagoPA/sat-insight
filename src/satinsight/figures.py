"""Phase one figures: what the model sees and what it measures inside each AGEB.

Las cifras de la propuesta describen el proceso pero no lo muestran. Estas figuras salen de
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

GRADE_COLOUR = {
    "Muy bajo": (27, 110, 114),
    "Bajo": (99, 162, 155),
    "Medio": (216, 199, 154),
    "Alto": (208, 138, 62),
    "Muy alto": (168, 72, 12),
}
"""Ordinal ramp from teal to orange, the same pair of accents the page uses."""

TINTA = (245, 245, 245)
FONDO = (20, 24, 31)


def _rgb_from_composite(bands: dict, sensor: str) -> np.ndarray:
    """Eight-bit RGB array ready to display, according to the sensor."""
    if sensor == "s2":
        return np.dstack([stretch(bands[b]) for b in ("B04", "B03", "B02")])
    vv, vh = to_db(bands["vv"]), to_db(bands["vh"])
    return np.dstack([stretch(vv), stretch(vh), stretch(vv - vh)])


def _text(image: Image.Image, xy, texto: str, size: int = 13, color=TINTA) -> None:
    """Escribe una etiqueta con un halo oscuro para que se lea sobre cualquier fondo."""
    dibujo = ImageDraw.Draw(image)
    try:
        fuente = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
    except OSError:
        fuente = ImageFont.load_default()
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dibujo.text((x + dx, y + dy), texto, font=fuente, fill=FONDO)
    dibujo.text(xy, texto, font=fuente, fill=color)


def modality_panel(city: str, destination: Path, side: int = 520) -> Path:
    """Los dos sensores sobre el mismo recuadro, uno al side del otro.

    This is the comparison the document makes in prose: optical has the urban fabric sharp
    at 10 m, radar sees it coarser but in a calibrated quantity.
    """
    root = Path("data") / "composites"
    paneles = []
    for sensor, title in (("s2", "Sentinel-2 · optical"), ("s1", "Sentinel-1 · radar")):
        bands, _, labels = cache.load(cache.composite_path(city, sensor, root))
        rgb = _rgb_from_composite(bands, sensor)
        alto, ancho = rgb.shape[:2]
        lado_px = min(alto, ancho)
        f0, c0 = (alto - lado_px) // 2, (ancho - lado_px) // 2
        crop = rgb[f0 : f0 + lado_px, c0 : c0 + lado_px]
        image = Image.fromarray(crop).resize((side, side), Image.LANCZOS)
        _text(image, (10, 8), title)
        _text(
            image,
            (10, side - 22),
            f"{labels.get('scenes_used', '?')} scenes · mediana anual",
            11,
        )
        paneles.append(image)

    lienzo = Image.new("RGB", (side * 2 + 6, side), FONDO)
    lienzo.paste(paneles[0], (0, 0))
    lienzo.paste(paneles[1], (side + 6, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(destination, optimize=True)
    log.info("panel de brazos: %s", destination)
    return destination


def _thicken(mask: np.ndarray, radio: int = 1) -> np.ndarray:
    """Dilates a boolean mask by shifting it. A one-pixel border is lost when scaling."""
    output = mask.copy()
    for eje in (0, 1):
        for signo in (1, -1):
            output |= np.roll(mask, signo * radio, axis=eje)
    return output


def ageb_panel(city: str, destination: Path, side: int = 760, ventana_px: int = 420) -> Path:
    """The composite with the AGEB borders on top, tinted by grade.

    Shows the unit of analysis resting on the pixels that feed it, which is what it takes to
    understand why the size of the polygon conditions what can be measured inside it.
    """
    root = Path("data") / "composites"
    _, agebs = city_aoi(city)
    bands, grid, _ = cache.load(cache.composite_path(city, "s2", root))
    rgb = _rgb_from_composite(bands, "s2")
    proyectadas = agebs.to_crs(grid.crs)

    # una etiqueta por grado, para dibujar los bordes con su color
    labels = rasterize(
        [
            (g, GRADES.index(t) + 1)
            for g, t in zip(proyectadas.geometry, proyectadas.grado, strict=True)
        ],
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
    )
    border = np.zeros(grid.shape, dtype=bool)
    for eje in (0, 1):
        d = np.diff(labels, axis=eje) != 0
        border |= np.pad(d, [(0, 1) if i == eje else (0, 0) for i in range(2)])

    pintado = rgb.copy()
    for indice, grado in enumerate(GRADES, start=1):
        mask = _thicken(border & (labels == indice))
        pintado[mask] = GRADE_COLOUR[grado]

    # crop centred on the area with most AGEB, small so the border shows when enlarged
    rows, columns = np.where(labels > 0)
    cf, cc = int(np.median(rows)), int(np.median(columns))
    half = min(ventana_px, rgb.shape[0], rgb.shape[1]) // 2
    f0 = max(0, min(cf - half, rgb.shape[0] - 2 * half))
    c0 = max(0, min(cc - half, rgb.shape[1] - 2 * half))
    crop = pintado[f0 : f0 + 2 * half, c0 : c0 + 2 * half]

    alto_leyenda = 30
    image = Image.new("RGB", (side, side + alto_leyenda), FONDO)
    image.paste(Image.fromarray(crop).resize((side, side), Image.LANCZOS), (0, 0))
    _text(image, (10, 8), f"{city} · AGEB borders, tinted by deprivation grade")
    _text(image, (10, side - 22), f"{2 * half * 10 / 1000:.1f} km across · 10 m pixel", 11)

    dibujo = ImageDraw.Draw(image)
    x = 10
    for grado in GRADES:
        dibujo.rectangle([x, side + 11, x + 22, side + 19], fill=GRADE_COLOUR[grado])
        _text(image, (x + 28, side + 8), grado, 11)
        x += 34 + 8 * len(grado)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, optimize=True)
    log.info("panel de AGEB: %s", destination)
    return destination


def _ageb_crop(rgb: np.ndarray, channel: np.ndarray, grid, geometry, margen: int = 6):
    """Recorta una AGEB de la image y del channel, con un poco de aire alrededor."""
    from satinsight.grid import polygon_window

    ventana = polygon_window(grid.transform, geometry, channel.shape)
    if ventana is None:
        return None
    rows, columns, dentro = ventana
    f0 = max(0, rows.start - margen)
    c0 = max(0, columns.start - margen)
    f1 = min(channel.shape[0], rows.stop + margen)
    c1 = min(channel.shape[1], columns.stop + margen)
    return rgb[f0:f1, c0:c1], channel[rows, columns], dentro


def contrast_panel(city: str, sensor: str, destination: Path, side: int = 190) -> Path:
    """Cuatro AGEB de la misma city, dos de rezago bajo y dos de rezago alto.

    Under each one go its measured contrast and homogeneity. It is the way to see what the
    GLCM is capturing before believing a kappa.
    """
    root = Path("data") / "composites"
    _, agebs = city_aoi(city)
    bands, grid, _ = cache.load(cache.composite_path(city, sensor, root))
    rgb = _rgb_from_composite(bands, sensor)
    proyectadas = agebs.to_crs(grid.crs)

    canales = channels_s2(bands) if sensor == "s2" else channels_s1(bands)
    nombre_canal = "s2nir" if sensor == "s2" else "s1vh"
    channel = canales[nombre_canal]
    rango = FIXED_RANGES_S1.get(nombre_canal)
    if rango is None:
        finitos = channel[np.isfinite(channel)]
        rango = (float(np.percentile(finitos, 2)), float(np.percentile(finitos, 98)))
    cuantizada = quantise(channel, rango)

    # Choosing by polygon area is not enough: a coastal AGEB can be large and hold
    # a single usable radar pixel, because over water backscatter falls to zero and `to_db`
    # turns it null. The candidate is measured by valid pixels of the channel.
    def utilizables(geometry) -> int:
        partes = _ageb_crop(rgb, cuantizada, grid, geometry)
        if partes is None:
            return 0
        _, recorte_q, dentro = partes
        return int((dentro & (recorte_q > 0)).sum())

    area_px = proyectadas.geometry.area / 100
    grandes = proyectadas[area_px > MIN_PIXELS * 3].copy()
    seleccion = []
    for grados in (("Muy bajo", "Bajo"), ("Alto", "Muy alto")):
        candidates = grandes[grandes.grado.isin(grados)]
        validas = [f for f in candidates.itertuples() if utilizables(f.geometry) >= MIN_PIXELS]
        seleccion.extend(validas[:2])

    if len(seleccion) < 4:
        raise RuntimeError(
            f"{city}/{sensor}: only {len(seleccion)} AGEB with enough pixels "
            "en ambos extremos del rezago"
        )

    celda, alto_celda = side, side + 34
    lienzo = Image.new("RGB", (celda * 4 + 18, alto_celda), FONDO)
    for indice, row in enumerate(seleccion[:4]):
        partes = _ageb_crop(rgb, cuantizada, grid, row.geometry)
        if partes is None:
            continue
        vista, recorte_q, dentro = partes
        enmascarado = np.where(dentro & (recorte_q > 0), recorte_q, 0).astype(np.uint8)
        rasgos = features_of_patch(enmascarado)

        image = Image.fromarray(vista).resize((celda, celda), Image.NEAREST)
        border = ImageDraw.Draw(image)
        border.rectangle([0, 0, celda - 1, celda - 1], outline=GRADE_COLOUR[row.grado], width=3)
        lienzo.paste(image, (indice * (celda + 6), 0))
        _text(lienzo, (indice * (celda + 6) + 4, celda + 4), row.grado, 12, GRADE_COLOUR[row.grado])
        _text(
            lienzo,
            (indice * (celda + 6) + 4, celda + 19),
            f"contraste {rasgos['contrast_d1']:.2f} · homog {rasgos['homogeneity_d1']:.2f}",
            10,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(destination, optimize=True)
    log.info("panel de contraste %s: %s", sensor, destination)
    return destination


def _dark_style(ax) -> None:
    """Leaves the axes with no frame or ticks, on the same background as the other panels."""
    ax.set_facecolor(_hex(FONDO))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def national_map(destination: Path, root: Path | None = None) -> Path:
    """Places the five pilot cities inside the country.

    The size of each mark is its number of AGEB and the colour the share sitting at grade
    alto o muy alto. Puestas sobre el mapa se ve de un vistazo el sesgo de la muestra: las
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
        high_share = float(agebs.grado.isin(("Alto", "Muy alto")).mean())
        points.append((key, CITIES[key].name, lon, lat, len(agebs), high_share))

    figure, ax = plt.subplots(figsize=(11, 7), dpi=150)
    figure.patch.set_facecolor(_hex(FONDO))
    _dark_style(ax)

    mexico.plot(ax=ax, facecolor="#1d2530", edgecolor="#3b4653", linewidth=0.6)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rezago",
        [
            _hex(GRADE_COLOUR["Muy bajo"]),
            _hex(GRADE_COLOUR["Medio"]),
            _hex(GRADE_COLOUR["Muy alto"]),
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

    for key, nombre, lon, lat, n, high_share in points:
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
            f"{nombre}\n{n} AGEB · {100 * high_share:.0f}% high",
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
    log.info("mapa nacional: %s", destination)
    return destination


def agebs_by_city_map(destination: Path, root: Path | None = None) -> Path:
    """The complete set of AGEB of each city, tinted by grade and at a shared scale.

    The image panels show crops; this one shows the whole extent that enters the baseline.
    Sharing the scale in kilometres allows comparing the real size of the five
    manchas urbanas.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from satinsight.agebs import CITIES, METRIC_CRS

    root = root or Path("data")
    cities = list(CITIES)
    figure, ejes = plt.subplots(1, len(cities), figsize=(16, 3.7), dpi=150)
    figure.patch.set_facecolor(_hex(FONDO))

    for ax, key in zip(ejes, cities, strict=True):
        _, agebs = city_aoi(key, root)
        metric = agebs.to_crs(METRIC_CRS)
        for grado, color in GRADE_COLOUR.items():
            parte = metric[metric.grado == grado]
            if len(parte):
                parte.plot(ax=ax, facecolor=_hex(color), edgecolor="none")
        _dark_style(ax)
        ax.set_aspect("equal")

        x0, y0, x1, y1 = metric.total_bounds
        side = max(x1 - x0, y1 - y0) * 1.05
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - side / 2, cx + side / 2)
        ax.set_ylim(cy - side / 2, cy + side / 2)

        high_share = 100 * agebs.grado.isin(("Alto", "Muy alto")).mean()
        ax.set_title(
            f"{CITIES[key].name}\n{len(agebs)} AGEB · {high_share:.0f}% high"
            f" · {side / 1000:.0f} km",
            color="#e6ebf1",
            fontsize=9.5,
            family="monospace",
            pad=8,
        )

    figure.legend(
        handles=[Patch(facecolor=_hex(c), label=GRADE_LABELS[g]) for g, c in GRADE_COLOUR.items()],
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
    log.info("mapa de AGEB por city: %s", destination)
    return destination


GRADE_LABELS = {
    "Muy bajo": "Very low",
    "Bajo": "Low",
    "Medio": "Medium",
    "Alto": "High",
    "Muy alto": "Very high",
}
"""English name of the five grades. The figures go to the paper and the paper is English."""

STATE_LABELS = {
    "completa": "complete",
    "a medias": "partial",
    "fallida": "failed",
    "pendiente": "pending",
}
"""English name of each state, because the figures go to the paper and it is in English."""

COMPOSITING_STATES = {
    "completa": ("#4ade80", "composited in both modalities"),
    "a medias": ("#fbbf24", "one modality ready"),
    "fallida": ("#f87171", "aborted on failed reads"),
    "pendiente": ("#9aa7b6", "not started"),
}
"""Colour and gloss of each state a city's compositing can be in."""


def compositing_state(
    keys: list[str], root: Path | None = None, logs: Path | None = None
) -> dict[str, str]:
    """Classifies each city by what is on disk and what the logs say.

    A city that aborted leaves its name on a `FALLO` line of the log and no file at all,
    which makes it indistinguishable from one that has not started. The distinction matters
    porque una city abortada no se reintenta sola: el barrido la salta y termina sin
    flag it. Only the lines after the last relaunch are read.
    """
    from satinsight.download import DATA_ROOT

    root = root or DATA_ROOT
    composites = root / "composites"
    failed: set[str] = set()
    for logfile in sorted((logs or root / "logs").glob("proceso_*.log")):
        texto = logfile.read_text(errors="ignore")
        recent = texto.rsplit("RELANZADO", 1)[-1]
        for line in recent.splitlines():
            if line.startswith("FALLO "):
                failed.add(line.split()[1])

    states = {}
    for key in keys:
        found = sum((composites / f"{key}_{s}.tif").exists() for s in ("s1", "s2"))
        if found == 2:
            states[key] = "completa"
        elif key in failed:
            # the failure outranks the lone file: a city that aborted on the second
            # modality leaves the first on disk and would look merely slow
            states[key] = "fallida"
        elif found == 1:
            states[key] = "a medias"
        else:
            states[key] = "pendiente"
    return states


def national_cities_map(
    destination: Path, root: Path | None = None, catalogue: dict | None = None
) -> Path:
    """Places the cities of the national set and colours each by its compositing state.

    The size of the mark is the number of urban AGEB of the city. Laid over the country it
    shows how much of the territory the sample covers and where the work done concentrates.
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from satinsight.agebs import cities_by_size
    from satinsight.download import DATA_ROOT, ensure_naturalearth

    root = root or DATA_ROOT
    catalogue = catalogue or cities_by_size(root=root, stratify=True)
    states = compositing_state(list(catalogue), root)

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
                "nombre": city.name,
                "lon": (area.bbox[0] + area.bbox[2]) / 2,
                "lat": (area.bbox[1] + area.bbox[3]) / 2,
                "agebs": len(agebs),
                "estado": states[key],
            }
        )

    estados_lista = gpd.read_file(ensure_naturalearth(root))
    mexico = estados_lista[estados_lista["admin"] == "Mexico"]

    figure, ax = plt.subplots(figsize=(12.5, 8), dpi=150)
    figure.patch.set_facecolor(_hex(FONDO))
    _dark_style(ax)
    mexico.plot(ax=ax, facecolor="#171d26", edgecolor="#333e4b", linewidth=0.5)

    order = ["pendiente", "a medias", "fallida", "completa"]
    for estado in order:
        group = [p for p in points if p["estado"] == estado]
        if not group:
            continue
        ax.scatter(
            [p["lon"] for p in group],
            [p["lat"] for p in group],
            s=[30 + p["agebs"] * 0.2 for p in group],
            c=COMPOSITING_STATES[estado][0],
            edgecolor="#0f1319",
            linewidth=0.7,
            alpha=0.95,
            zorder=3 + order.index(estado),
        )

    # solo se rotulan las cities ya compuestas y las que fallaron: rotular las 138
    # leaves the map unreadable, and those two are the ones worth naming
    labelled = sorted(
        (p for p in points if p["estado"] != "pendiente"),
        key=lambda p: (-p["lat"], p["lon"]),
    )
    # neighbouring conurbations leave their labels on top of each other —Zapopan over
    # Guadalajara, Mexicali over Tijuana— so every label landing too close to the
    # anterior se empuja hacia abajo hasta despejarse
    placed: list[tuple[float, float]] = []
    for p in labelled:
        dx, dy = 6, 4
        while any(
            abs(p["lon"] - lon) < 1.7 and abs((p["lat"] + dy / 22) - lat) < 0.42
            for lon, lat in placed
        ):
            dy -= 11
        placed.append((p["lon"], p["lat"] + dy / 22))
        ax.annotate(
            p["nombre"],
            (p["lon"], p["lat"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.4,
            color="#c9d3de",
            family="monospace",
            zorder=8,
        )

    done = sum(1 for p in points if p["estado"] == "completa")
    total_agebs = sum(p["agebs"] for p in points)
    ax.set_xlim(-118.5, -85.5)
    ax.set_ylim(13.5, 33.5)
    ax.set_title(
        f"The {len(points)} cities of the national set · {total_agebs:,} urban AGEB\n"
        f"{done} with compositing finished",
        color="#e6ebf1",
        fontsize=13,
        loc="left",
        pad=14,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="#0f1319",
            label=f"{STATE_LABELS.get(estado, estado)} · {gloss}",
        )
        for estado, (color, gloss) in COMPOSITING_STATES.items()
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
        facecolor="#141a22",
        edgecolor="#3b4653",
    )
    for texto in legend.get_texts():
        texto.set_color("#c9d3de")

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    log.info("mapa del split nacional: %s", destination)
    return destination
