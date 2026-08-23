"""Figuras de la fase 1: qué ve el modelo y qué mide dentro de cada AGEB.

Las cifras de la propuesta describen el proceso pero no lo muestran. Estas figuras salen de
los mismos compuestos y polígonos que alimentan el baseline, de modo que lo que se ve en la
página es exactamente lo que entra al modelo, sin una capa de ilustración de por medio.

Todas se regeneran con `satinsight figuras`, y su fuente es el caché de compuestos.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import rasterize

from satinsight import cache
from satinsight.agebs import GRADOS
from satinsight.pipeline import aoi_de_ciudad, canales_s1, canales_s2
from satinsight.raster import stretch, to_db
from satinsight.textura import MINIMO_PIXELES, RANGOS_FIJOS_S1, cuantizar, rasgos_de_recorte

log = logging.getLogger(__name__)

COLOR_GRADO = {
    "Muy bajo": (27, 110, 114),
    "Bajo": (99, 162, 155),
    "Medio": (216, 199, 154),
    "Alto": (208, 138, 62),
    "Muy alto": (168, 72, 12),
}
"""Rampa ordinal de verde azulado a naranja, la misma pareja de acentos que usa la página."""

TINTA = (245, 245, 245)
FONDO = (20, 24, 31)


def _rgb_desde_compuesto(bandas: dict, sensor: str) -> np.ndarray:
    """Arreglo RGB de 8 bits listo para mostrar, según el sensor."""
    if sensor == "s2":
        return np.dstack([stretch(bandas[b]) for b in ("B04", "B03", "B02")])
    vv, vh = to_db(bandas["vv"]), to_db(bandas["vh"])
    return np.dstack([stretch(vv), stretch(vh), stretch(vv - vh)])


def _texto(imagen: Image.Image, xy, texto: str, tamano: int = 13, color=TINTA) -> None:
    """Escribe una etiqueta con un halo oscuro para que se lea sobre cualquier fondo."""
    dibujo = ImageDraw.Draw(imagen)
    try:
        fuente = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", tamano)
    except OSError:
        fuente = ImageFont.load_default()
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dibujo.text((x + dx, y + dy), texto, font=fuente, fill=FONDO)
    dibujo.text(xy, texto, font=fuente, fill=color)


def panel_brazos(ciudad: str, destino: Path, lado: int = 520) -> Path:
    """Los dos sensores sobre el mismo recuadro, uno al lado del otro.

    Es la comparación que el documento sostiene en prosa: el óptico tiene la traza urbana
    nítida a 10 m, el radar la ve más basta pero en una magnitud calibrada.
    """
    raiz = Path("data") / "compuestos"
    paneles = []
    for sensor, titulo in (("s2", "Sentinel-2 · óptico"), ("s1", "Sentinel-1 · radar")):
        bandas, _, etiquetas = cache.cargar(cache.ruta_compuesto(ciudad, sensor, raiz))
        rgb = _rgb_desde_compuesto(bandas, sensor)
        alto, ancho = rgb.shape[:2]
        lado_px = min(alto, ancho)
        f0, c0 = (alto - lado_px) // 2, (ancho - lado_px) // 2
        recorte = rgb[f0 : f0 + lado_px, c0 : c0 + lado_px]
        imagen = Image.fromarray(recorte).resize((lado, lado), Image.LANCZOS)
        _texto(imagen, (10, 8), titulo)
        _texto(
            imagen,
            (10, lado - 22),
            f"{etiquetas.get('escenas_usadas', '?')} escenas · mediana anual",
            11,
        )
        paneles.append(imagen)

    lienzo = Image.new("RGB", (lado * 2 + 6, lado), FONDO)
    lienzo.paste(paneles[0], (0, 0))
    lienzo.paste(paneles[1], (lado + 6, 0))
    destino.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(destino, optimize=True)
    log.info("panel de brazos: %s", destino)
    return destino


def _engrosar(mascara: np.ndarray, radio: int = 1) -> np.ndarray:
    """Dilata una máscara booleana desplazándola. Un borde de un píxel se pierde al escalar."""
    salida = mascara.copy()
    for eje in (0, 1):
        for signo in (1, -1):
            salida |= np.roll(mascara, signo * radio, axis=eje)
    return salida


def panel_agebs(ciudad: str, destino: Path, lado: int = 760, ventana_px: int = 420) -> Path:
    """El compuesto con los bordes de las AGEB encima, teñidos por grado.

    Muestra la unidad de análisis apoyada sobre los píxeles que la alimentan, que es lo que
    hace falta para entender por qué el tamaño del polígono condiciona lo que se puede medir
    dentro de él.
    """
    raiz = Path("data") / "compuestos"
    _, agebs = aoi_de_ciudad(ciudad)
    bandas, malla, _ = cache.cargar(cache.ruta_compuesto(ciudad, "s2", raiz))
    rgb = _rgb_desde_compuesto(bandas, "s2")
    proyectadas = agebs.to_crs(malla.crs)

    # una etiqueta por grado, para dibujar los bordes con su color
    etiquetas = rasterize(
        [
            (g, GRADOS.index(t) + 1)
            for g, t in zip(proyectadas.geometry, proyectadas.grado, strict=True)
        ],
        out_shape=malla.shape,
        transform=malla.transform,
        fill=0,
        dtype="uint8",
    )
    borde = np.zeros(malla.shape, dtype=bool)
    for eje in (0, 1):
        d = np.diff(etiquetas, axis=eje) != 0
        borde |= np.pad(d, [(0, 1) if i == eje else (0, 0) for i in range(2)])

    pintado = rgb.copy()
    for indice, grado in enumerate(GRADOS, start=1):
        mascara = _engrosar(borde & (etiquetas == indice))
        pintado[mascara] = COLOR_GRADO[grado]

    # recorte centrado en la zona con más AGEB, chico para que el borde se distinga al ampliar
    filas, columnas = np.where(etiquetas > 0)
    cf, cc = int(np.median(filas)), int(np.median(columnas))
    mitad = min(ventana_px, rgb.shape[0], rgb.shape[1]) // 2
    f0 = max(0, min(cf - mitad, rgb.shape[0] - 2 * mitad))
    c0 = max(0, min(cc - mitad, rgb.shape[1] - 2 * mitad))
    recorte = pintado[f0 : f0 + 2 * mitad, c0 : c0 + 2 * mitad]

    alto_leyenda = 30
    imagen = Image.new("RGB", (lado, lado + alto_leyenda), FONDO)
    imagen.paste(Image.fromarray(recorte).resize((lado, lado), Image.LANCZOS), (0, 0))
    _texto(imagen, (10, 8), f"{ciudad} · bordes de AGEB, teñidos por grado de rezago")
    _texto(imagen, (10, lado - 22), f"{2 * mitad * 10 / 1000:.1f} km de lado · píxel de 10 m", 11)

    dibujo = ImageDraw.Draw(imagen)
    x = 10
    for grado in GRADOS:
        dibujo.rectangle([x, lado + 11, x + 22, lado + 19], fill=COLOR_GRADO[grado])
        _texto(imagen, (x + 28, lado + 8), grado, 11)
        x += 34 + 8 * len(grado)

    destino.parent.mkdir(parents=True, exist_ok=True)
    imagen.save(destino, optimize=True)
    log.info("panel de AGEB: %s", destino)
    return destino


def _recorte_de_ageb(rgb: np.ndarray, canal: np.ndarray, malla, geometria, margen: int = 6):
    """Recorta una AGEB de la imagen y del canal, con un poco de aire alrededor."""
    from satinsight.malla import polygon_window

    ventana = polygon_window(malla.transform, geometria, canal.shape)
    if ventana is None:
        return None
    filas, columnas, dentro = ventana
    f0 = max(0, filas.start - margen)
    c0 = max(0, columnas.start - margen)
    f1 = min(canal.shape[0], filas.stop + margen)
    c1 = min(canal.shape[1], columnas.stop + margen)
    return rgb[f0:f1, c0:c1], canal[filas, columnas], dentro


def panel_contraste(ciudad: str, sensor: str, destino: Path, lado: int = 190) -> Path:
    """Cuatro AGEB de la misma ciudad, dos de rezago bajo y dos de rezago alto.

    Debajo de cada una van su contraste y su homogeneidad medidos. Es la forma de ver qué
    está capturando la GLCM antes de creerle a un kappa.
    """
    raiz = Path("data") / "compuestos"
    _, agebs = aoi_de_ciudad(ciudad)
    bandas, malla, _ = cache.cargar(cache.ruta_compuesto(ciudad, sensor, raiz))
    rgb = _rgb_desde_compuesto(bandas, sensor)
    proyectadas = agebs.to_crs(malla.crs)

    canales = canales_s2(bandas) if sensor == "s2" else canales_s1(bandas)
    nombre_canal = "s2nir" if sensor == "s2" else "s1vh"
    canal = canales[nombre_canal]
    rango = RANGOS_FIJOS_S1.get(nombre_canal)
    if rango is None:
        finitos = canal[np.isfinite(canal)]
        rango = (float(np.percentile(finitos, 2)), float(np.percentile(finitos, 98)))
    cuantizada = cuantizar(canal, rango)

    # Elegir por área del polígono no basta: una AGEB costera puede ser grande y no tener
    # un solo píxel de radar utilizable, porque sobre agua la retrodispersión cae a cero y
    # `to_db` la vuelve nula. La candidata se mide por píxeles válidos del canal.
    def utilizables(geometria) -> int:
        partes = _recorte_de_ageb(rgb, cuantizada, malla, geometria)
        if partes is None:
            return 0
        _, recorte_q, dentro = partes
        return int((dentro & (recorte_q > 0)).sum())

    area_px = proyectadas.geometry.area / 100
    grandes = proyectadas[area_px > MINIMO_PIXELES * 3].copy()
    seleccion = []
    for grados in (("Muy bajo", "Bajo"), ("Alto", "Muy alto")):
        candidatas = grandes[grandes.grado.isin(grados)]
        validas = [f for f in candidatas.itertuples() if utilizables(f.geometry) >= MINIMO_PIXELES]
        seleccion.extend(validas[:2])

    if len(seleccion) < 4:
        raise RuntimeError(
            f"{ciudad}/{sensor}: solo {len(seleccion)} AGEB con píxeles suficientes "
            "en ambos extremos del rezago"
        )

    celda, alto_celda = lado, lado + 34
    lienzo = Image.new("RGB", (celda * 4 + 18, alto_celda), FONDO)
    for indice, fila in enumerate(seleccion[:4]):
        partes = _recorte_de_ageb(rgb, cuantizada, malla, fila.geometry)
        if partes is None:
            continue
        vista, recorte_q, dentro = partes
        enmascarado = np.where(dentro & (recorte_q > 0), recorte_q, 0).astype(np.uint8)
        rasgos = rasgos_de_recorte(enmascarado)

        imagen = Image.fromarray(vista).resize((celda, celda), Image.NEAREST)
        borde = ImageDraw.Draw(imagen)
        borde.rectangle([0, 0, celda - 1, celda - 1], outline=COLOR_GRADO[fila.grado], width=3)
        lienzo.paste(imagen, (indice * (celda + 6), 0))
        _texto(
            lienzo, (indice * (celda + 6) + 4, celda + 4), fila.grado, 12, COLOR_GRADO[fila.grado]
        )
        _texto(
            lienzo,
            (indice * (celda + 6) + 4, celda + 19),
            f"contraste {rasgos['contrast_d1']:.2f} · homog {rasgos['homogeneity_d1']:.2f}",
            10,
        )

    destino.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(destino, optimize=True)
    log.info("panel de contraste %s: %s", sensor, destino)
    return destino


def _estilo_oscuro(ax) -> None:
    """Deja los ejes sin marco ni marcas, sobre el mismo fondo que los demás paneles."""
    ax.set_facecolor(_hex(FONDO))
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_visible(False)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def mapa_nacional(destino: Path, raiz: Path | None = None) -> Path:
    """Sitúa las cinco ciudades piloto dentro del país.

    El tamaño de cada marca es su número de AGEB y el color la proporción que está en grado
    alto o muy alto. Puestas sobre el mapa se ve de un vistazo el sesgo de la muestra: las
    dos ciudades que aportan rezago alto están en el sur y en la costa del Pacífico.
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from satinsight.agebs import CIUDADES
    from satinsight.ingesta import RAIZ_DATOS, asegurar_naturalearth

    raiz = raiz or RAIZ_DATOS
    estados = gpd.read_file(asegurar_naturalearth(raiz))
    mexico = estados[estados["admin"] == "Mexico"]

    puntos = []
    for clave in CIUDADES:
        area, agebs = aoi_de_ciudad(clave, raiz)
        lon = (area.bbox[0] + area.bbox[2]) / 2
        lat = (area.bbox[1] + area.bbox[3]) / 2
        altos = float(agebs.grado.isin(("Alto", "Muy alto")).mean())
        puntos.append((clave, CIUDADES[clave].nombre, lon, lat, len(agebs), altos))

    figura, ax = plt.subplots(figsize=(11, 7), dpi=150)
    figura.patch.set_facecolor(_hex(FONDO))
    _estilo_oscuro(ax)

    mexico.plot(ax=ax, facecolor="#1d2530", edgecolor="#3b4653", linewidth=0.6)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rezago",
        [_hex(COLOR_GRADO["Muy bajo"]), _hex(COLOR_GRADO["Medio"]), _hex(COLOR_GRADO["Muy alto"])],
    )
    # desplazamientos escogidos a mano: puestas todas a la derecha, las etiquetas de
    # Acapulco y Tuxtla se encimaban y la de Mérida caía sobre la barra de color
    desplazamiento = {
        "tuxtla": (14, -16),
        "merida": (-16, 12),
        "iztapalapa": (14, 6),
        "tapachula": (14, -14),
        "acapulco": (-16, -18),
    }
    alineacion = {"merida": "right", "acapulco": "right"}

    for clave, nombre, lon, lat, n, altos in puntos:
        ax.scatter(
            lon,
            lat,
            s=40 + n * 0.55,
            c=[cmap(altos / 0.5)],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        ax.annotate(
            f"{nombre}\n{n} AGEB · {100 * altos:.0f}% high",
            (lon, lat),
            textcoords="offset points",
            xytext=desplazamiento[clave],
            ha=alineacion.get(clave, "left"),
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
    barra = figura.colorbar(
        matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(0, 50), cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.03,
        pad=0.02,
        aspect=45,
    )
    barra.set_label("% of AGEB at high or very high deprivation", color="#a8b3c0", fontsize=9)
    barra.ax.tick_params(colors="#a8b3c0", labelsize=8)
    barra.outline.set_edgecolor("#3b4653")

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, facecolor=figura.get_facecolor(), bbox_inches="tight")
    plt.close(figura)
    log.info("mapa nacional: %s", destino)
    return destino


def mapa_agebs_por_ciudad(destino: Path, raiz: Path | None = None) -> Path:
    """El conjunto completo de AGEB de cada ciudad, teñido por grado y a escala común.

    Los paneles de imagen muestran recortes; este muestra la extensión entera que entra al
    baseline. Compartir la escala en kilómetros permite comparar el tamaño real de las cinco
    manchas urbanas.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from satinsight.agebs import CIUDADES, CRS_METRICO

    raiz = raiz or Path("data")
    ciudades = list(CIUDADES)
    figura, ejes = plt.subplots(1, len(ciudades), figsize=(16, 3.7), dpi=150)
    figura.patch.set_facecolor(_hex(FONDO))

    for ax, clave in zip(ejes, ciudades, strict=True):
        _, agebs = aoi_de_ciudad(clave, raiz)
        metrico = agebs.to_crs(CRS_METRICO)
        for grado, color in COLOR_GRADO.items():
            parte = metrico[metrico.grado == grado]
            if len(parte):
                parte.plot(ax=ax, facecolor=_hex(color), edgecolor="none")
        _estilo_oscuro(ax)
        ax.set_aspect("equal")

        x0, y0, x1, y1 = metrico.total_bounds
        lado = max(x1 - x0, y1 - y0) * 1.05
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - lado / 2, cx + lado / 2)
        ax.set_ylim(cy - lado / 2, cy + lado / 2)

        altos = 100 * agebs.grado.isin(("Alto", "Muy alto")).mean()
        ax.set_title(
            f"{CIUDADES[clave].nombre}\n{len(agebs)} AGEB · {altos:.0f}% high"
            f" · {lado / 1000:.0f} km",
            color="#e6ebf1",
            fontsize=9.5,
            family="monospace",
            pad=8,
        )

    figura.legend(
        handles=[Patch(facecolor=_hex(c), label=GRADOS_EN[g]) for g, c in COLOR_GRADO.items()],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        labelcolor="#a8b3c0",
        bbox_to_anchor=(0.5, 0.02),
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, facecolor=figura.get_facecolor(), bbox_inches="tight")
    plt.close(figura)
    log.info("mapa de AGEB por ciudad: %s", destino)
    return destino


GRADOS_EN = {
    "Muy bajo": "Very low",
    "Bajo": "Low",
    "Medio": "Medium",
    "Alto": "High",
    "Muy alto": "Very high",
}
"""Nombre en inglés de los cinco grados. Las figuras van al paper y el paper va en inglés."""

ESTADOS_EN = {
    "completa": "complete",
    "a medias": "partial",
    "fallida": "failed",
    "pendiente": "pending",
}
"""Nombre en inglés de cada estado, porque las figuras van al paper y el paper va en inglés."""

ESTADOS_COMPOSICION = {
    "completa": ("#4ade80", "composited in both modalities"),
    "a medias": ("#fbbf24", "one modality ready"),
    "fallida": ("#f87171", "aborted on failed reads"),
    "pendiente": ("#9aa7b6", "not started"),
}
"""Color y glosa de cada estado en que puede estar la composición de una ciudad."""


def estado_de_composicion(
    claves: list[str], raiz: Path | None = None, logs: Path | None = None
) -> dict[str, str]:
    """Clasifica cada ciudad según lo que hay en disco y lo que dicen los registros.

    Una ciudad que abortó deja su nombre en una línea `FALLO` del registro y ningún archivo,
    lo cual la vuelve indistinguible de una que todavía no empieza. La distinción importa
    porque una ciudad abortada no se reintenta sola: el barrido la salta y termina sin
    señalarla. Solo se leen las líneas posteriores al último relanzamiento.
    """
    from satinsight.ingesta import RAIZ_DATOS

    raiz = raiz or RAIZ_DATOS
    compuestos = raiz / "compuestos"
    fallidas: set[str] = set()
    for registro in sorted((logs or raiz / "logs").glob("proceso_*.log")):
        texto = registro.read_text(errors="ignore")
        reciente = texto.rsplit("RELANZADO", 1)[-1]
        for linea in reciente.splitlines():
            if linea.startswith("FALLO "):
                fallidas.add(linea.split()[1])

    estados = {}
    for clave in claves:
        hechos = sum((compuestos / f"{clave}_{s}.tif").exists() for s in ("s1", "s2"))
        if hechos == 2:
            estados[clave] = "completa"
        elif clave in fallidas:
            # el fallo manda sobre el archivo suelto: una ciudad que abortó en la segunda
            # modalidad deja la primera en disco y se vería como si solo fuera lenta
            estados[clave] = "fallida"
        elif hechos == 1:
            estados[clave] = "a medias"
        else:
            estados[clave] = "pendiente"
    return estados


def mapa_ciudades_nacionales(
    destino: Path, raiz: Path | None = None, catalogo: dict | None = None
) -> Path:
    """Sitúa las ciudades del conjunto nacional y colorea cada una según su composición.

    El tamaño de la marca es el número de AGEB urbanas de la ciudad. Puestas sobre el país
    se ve qué tanto cubre la muestra el territorio y dónde se concentra el trabajo hecho.
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from satinsight.agebs import ciudades_por_tamano
    from satinsight.ingesta import RAIZ_DATOS, asegurar_naturalearth

    raiz = raiz or RAIZ_DATOS
    catalogo = catalogo or ciudades_por_tamano(raiz=raiz, estratificar=True)
    estados = estado_de_composicion(list(catalogo), raiz)

    puntos = []
    for clave, ciudad in catalogo.items():
        try:
            area, agebs = aoi_de_ciudad(clave, raiz, catalogo=catalogo)
        except Exception:
            log.warning("sin geometría para %s", clave, exc_info=True)
            continue
        puntos.append(
            {
                "clave": clave,
                "nombre": ciudad.nombre,
                "lon": (area.bbox[0] + area.bbox[2]) / 2,
                "lat": (area.bbox[1] + area.bbox[3]) / 2,
                "agebs": len(agebs),
                "estado": estados[clave],
            }
        )

    estados_lista = gpd.read_file(asegurar_naturalearth(raiz))
    mexico = estados_lista[estados_lista["admin"] == "Mexico"]

    figura, ax = plt.subplots(figsize=(12.5, 8), dpi=150)
    figura.patch.set_facecolor(_hex(FONDO))
    _estilo_oscuro(ax)
    mexico.plot(ax=ax, facecolor="#171d26", edgecolor="#333e4b", linewidth=0.5)

    orden = ["pendiente", "a medias", "fallida", "completa"]
    for estado in orden:
        grupo = [p for p in puntos if p["estado"] == estado]
        if not grupo:
            continue
        ax.scatter(
            [p["lon"] for p in grupo],
            [p["lat"] for p in grupo],
            s=[30 + p["agebs"] * 0.2 for p in grupo],
            c=ESTADOS_COMPOSICION[estado][0],
            edgecolor="#0f1319",
            linewidth=0.7,
            alpha=0.95,
            zorder=3 + orden.index(estado),
        )

    # solo se rotulan las ciudades ya compuestas y las que fallaron: rotular las 138
    # deja el mapa ilegible, y son esas dos las que interesa poder señalar por nombre
    rotuladas = sorted(
        (p for p in puntos if p["estado"] != "pendiente"),
        key=lambda p: (-p["lat"], p["lon"]),
    )
    # las conurbaciones vecinas dejan sus rótulos uno encima de otro —Zapopan sobre
    # Guadalajara, Mexicali sobre Tijuana—, así que cada rótulo que caiga muy cerca del
    # anterior se empuja hacia abajo hasta despejarse
    puestas: list[tuple[float, float]] = []
    for p in rotuladas:
        dx, dy = 6, 4
        while any(
            abs(p["lon"] - lon) < 1.7 and abs((p["lat"] + dy / 22) - lat) < 0.42
            for lon, lat in puestas
        ):
            dy -= 11
        puestas.append((p["lon"], p["lat"] + dy / 22))
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

    hechas = sum(1 for p in puntos if p["estado"] == "completa")
    total_agebs = sum(p["agebs"] for p in puntos)
    ax.set_xlim(-118.5, -85.5)
    ax.set_ylim(13.5, 33.5)
    ax.set_title(
        f"The {len(puntos)} cities of the national set · {total_agebs:,} urban AGEB\n"
        f"{hechas} with compositing finished",
        color="#e6ebf1",
        fontsize=13,
        loc="left",
        pad=14,
    )
    marcas = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="#0f1319",
            label=f"{ESTADOS_EN.get(estado, estado)} · {glosa}",
        )
        for estado, (color, glosa) in ESTADOS_COMPOSICION.items()
    ]
    leyenda = ax.legend(
        handles=marcas,
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
        facecolor="#141a22",
        edgecolor="#3b4653",
    )
    for texto in leyenda.get_texts():
        texto.set_color("#c9d3de")

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, facecolor=figura.get_facecolor(), bbox_inches="tight")
    plt.close(figura)
    log.info("mapa del conjunto nacional: %s", destino)
    return destino
