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
from satinsight.raster import a_db, estirar
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
        return np.dstack([estirar(bandas[b]) for b in ("B04", "B03", "B02")])
    vv, vh = a_db(bandas["vv"]), a_db(bandas["vh"])
    return np.dstack([estirar(vv), estirar(vh), estirar(vv - vh)])


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
        out_shape=malla.forma,
        transform=malla.transform,
        fill=0,
        dtype="uint8",
    )
    borde = np.zeros(malla.forma, dtype=bool)
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
    from satinsight.malla import recorte_de_poligono

    ventana = recorte_de_poligono(malla.transform, geometria, canal.shape)
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
    # `a_db` la vuelve nula. La candidata se mide por píxeles válidos del canal.
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
