"""Rasgos de textura por AGEB a partir de la matriz de co-ocurrencia de niveles de gris.

La GLCM está definida sobre ventanas rectangulares y una AGEB es un polígono irregular. La
salida es recortar la AGEB a su ventana envolvente, marcar como inválido lo que cae fuera
del polígono y descartar del conteo cualquier par donde participe un píxel inválido. Eso se
consigue reservando el nivel cero para lo inválido y eliminando después su renglón y su
columna de la matriz.

La cuantización usa percentiles calculados sobre la ciudad completa, no sobre cada AGEB.
Cuantizar por AGEB normalizaría el brillo de cada polígono por separado y borraría
justamente la señal de nivel que distingue un asentamiento precario de una colonia
consolidada.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import rowcol
from skimage.feature import graycomatrix, graycoprops

log = logging.getLogger(__name__)

NIVELES = 32
"""Niveles de gris de la cuantización. Treinta y dos equilibra detalle y matrices ralas."""

DISTANCIAS = (1, 2, 4)
"""Separaciones en píxeles. A 10 m cubren de la escala de un techo a la de una manzana."""

ANGULOS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
"""Las cuatro orientaciones canónicas. Se promedian para dar invarianza a la rotación."""

PROPIEDADES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")

MINIMO_PIXELES = 50
"""Debajo de esto la matriz queda demasiado vacía para que sus estadísticos signifiquen algo."""


def rango_robusto(banda: np.ndarray, p_bajo: float = 2.0, p_alto: float = 98.0) -> tuple:
    """Percentiles de la banda completa, que fijan la escala común de cuantización."""
    validos = banda[np.isfinite(banda)]
    if validos.size == 0:
        raise ValueError("la banda no tiene un solo píxel finito")
    return float(np.percentile(validos, p_bajo)), float(np.percentile(validos, p_alto))


def cuantizar(banda: np.ndarray, rango: tuple, niveles: int = NIVELES) -> np.ndarray:
    """Lleva la banda a enteros de 1 a `niveles`, reservando el 0 para lo inválido.

    Los valores fuera del rango se recortan a los extremos en vez de descartarse: un techo
    de lámina muy brillante sigue siendo información aunque caiga sobre el percentil 98.

    Los no finitos se llevan a cero antes de convertir a entero. `np.clip` no toca los NaN,
    y convertir un NaN a entero sin signo da un resultado que la especificación no define y
    que depende de la plataforma. El `np.where` final pisa esas posiciones de todos modos,
    pero el valor intermedio no debe quedar al azar: `a_db` produce NaN en cada píxel sin
    retorno medible, así que el brazo de radar pasa por aquí constantemente.
    """
    bajo, alto = rango
    if not alto > bajo:
        raise ValueError(f"rango degenerado: {rango}")
    finitos = np.isfinite(banda)
    escalada = np.zeros(banda.shape, dtype="float64")
    np.divide(banda - bajo, alto - bajo, out=escalada, where=finitos)
    escalada = np.clip(escalada, 0.0, 1.0)
    niveles_validos = 1 + np.round(escalada * (niveles - 1)).astype(np.uint8)
    return np.where(finitos, niveles_validos, 0).astype(np.uint8)


def entropia(glcm: np.ndarray) -> float:
    """Entropía de Shannon de la matriz normalizada, promediada sobre distancias y ángulos."""
    p = glcm.astype(np.float64)
    total = p.sum(axis=(0, 1), keepdims=True)
    p = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    logaritmo = np.zeros_like(p)
    np.log2(p, out=logaritmo, where=p > 0)
    return float((-(p * logaritmo)).sum(axis=(0, 1)).mean())


def _matriz(recorte: np.ndarray, niveles: int) -> np.ndarray:
    """Calcula la GLCM del recorte y elimina el nivel reservado a los píxeles inválidos."""
    glcm = graycomatrix(
        recorte,
        distances=list(DISTANCIAS),
        angles=list(ANGULOS),
        levels=niveles + 1,
        symmetric=True,
        normed=False,
    )
    return glcm[1:, 1:, :, :]


def rasgos_de_recorte(recorte: np.ndarray, niveles: int = NIVELES) -> dict[str, float]:
    """Propiedades de Haralick de un recorte ya cuantizado, promediadas sobre los ángulos.

    Se conserva la desviación entre ángulos de una sola propiedad, el contraste, porque
    distingue una traza urbana orientada de uno sin dirección dominante.
    """
    glcm = _matriz(recorte, niveles)
    if glcm.sum() == 0:
        return dict.fromkeys([*PROPIEDADES, "entropia", "contrast_anisotropia"], np.nan)

    rasgos: dict[str, float] = {}
    for propiedad in PROPIEDADES:
        valores = graycoprops(glcm, propiedad)
        rasgos[propiedad] = float(np.nanmean(valores))
    rasgos["contrast_anisotropia"] = float(np.nanmean(np.nanstd(graycoprops(glcm, "contrast"), 1)))
    rasgos["entropia"] = entropia(glcm)
    return rasgos


def rasgos_primer_orden(valores: np.ndarray) -> dict[str, float]:
    """Estadísticos de la distribución de intensidades, sin considerar su arreglo espacial.

    Son el punto de comparación honesto de la GLCM: si la textura no agrega nada sobre la
    media y la dispersión, conviene saberlo antes de defenderla en el paper.
    """
    if valores.size == 0:
        return dict.fromkeys(["media", "desv", "p10", "p50", "p90", "rango_intercuartil"], np.nan)
    p10, p25, p50, p75, p90 = np.percentile(valores, [10, 25, 50, 75, 90])
    return {
        "media": float(np.mean(valores)),
        "desv": float(np.std(valores)),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "rango_intercuartil": float(p75 - p25),
    }


def rasgos_por_ageb(
    banda: np.ndarray,
    transform,
    geometrias: Sequence,
    claves: Sequence[str],
    *,
    prefijo: str,
    niveles: int = NIVELES,
    minimo_pixeles: int = MINIMO_PIXELES,
) -> pd.DataFrame:
    """Extrae rasgos de textura y de primer orden para cada polígono sobre una banda.

    La banda se cuantiza una sola vez con el rango de la ciudad completa, y de ahí se
    recorta cada AGEB. Devuelve un renglón por AGEB con las columnas prefijadas por el
    nombre del canal, para poder concatenar varios canales sin colisión de nombres.

    Las geometrías tienen que venir en el mismo sistema de referencia que `transform`,
    que para los compuestos es el huso UTM de las escenas y no coordenadas geográficas.
    """
    if len(geometrias) != len(claves):
        raise ValueError(f"{len(geometrias)} geometrías contra {len(claves)} claves")

    rango = rango_robusto(banda)
    cuantizada = cuantizar(banda, rango, niveles)
    alto, ancho = banda.shape
    renglones = []

    for clave, geometria in zip(claves, geometrias, strict=True):
        x_min, y_min, x_max, y_max = geometria.bounds
        fila_a, col_a = rowcol(transform, x_min, y_max)
        fila_b, col_b = rowcol(transform, x_max, y_min)
        fila_ini, fila_fin = max(0, min(fila_a, fila_b)), min(alto, max(fila_a, fila_b) + 1)
        col_ini, col_fin = max(0, min(col_a, col_b)), min(ancho, max(col_a, col_b) + 1)

        base = {"cvegeo": clave, f"{prefijo}_n_px": 0}
        if fila_fin <= fila_ini or col_fin <= col_ini:
            renglones.append(base)
            continue

        recorte = cuantizada[fila_ini:fila_fin, col_ini:col_fin]
        crudo = banda[fila_ini:fila_fin, col_ini:col_fin]
        transform_ventana = transform * transform.translation(col_ini, fila_ini)

        dentro = ~geometry_mask(
            [geometria],
            out_shape=recorte.shape,
            transform=transform_ventana,
            invert=False,
        )
        valido = dentro & (recorte > 0)
        n_px = int(valido.sum())
        base[f"{prefijo}_n_px"] = n_px
        if n_px < minimo_pixeles:
            renglones.append(base)
            continue

        enmascarado = np.where(valido, recorte, 0).astype(np.uint8)
        rasgos = rasgos_de_recorte(enmascarado, niveles)
        rasgos.update(rasgos_primer_orden(crudo[valido & np.isfinite(crudo)]))
        base.update({f"{prefijo}_{k}": v for k, v in rasgos.items()})
        renglones.append(base)

    tabla = pd.DataFrame(renglones)
    utiles = int((tabla[f"{prefijo}_n_px"] >= minimo_pixeles).sum())
    log.info("%s: %d de %d AGEB con píxeles suficientes", prefijo, utiles, len(tabla))
    return tabla
