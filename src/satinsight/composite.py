"""Compuestos mediana anuales de Sentinel-1 y Sentinel-2.

El compositing cumple aquí una única función: suprimir nubes en el óptico y speckle en
el radar. El objeto de análisis sigue siendo una imagen estática de un solo corte anual.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pystac import Item

from satinsight.aoi import Bbox
from satinsight.catalog import SCL_VALIDOS, orbita_dominante, por_nubosidad
from satinsight.raster import leer_ventana

log = logging.getLogger(__name__)

BANDAS_RGB = ("B04", "B03", "B02")
COBERTURA_MINIMA = 0.05
"""Fracción de píxeles válidos por debajo de la cual una escena se descarta."""


def compuesto_s2(
    items: list["Item"],
    bbox: Bbox,
    forma: tuple[int, int] | None = None,
    bandas: tuple[str, ...] = BANDAS_RGB,
    max_escenas: int = 36,
) -> tuple[dict[str, np.ndarray], int]:
    """Mediana por píxel de las escenas Sentinel-2 más despejadas, con máscara SCL.

    Devuelve las bandas compuestas y el número de escenas que aportaron píxeles.
    Las escenas se recorren de la más despejada a la más nublada.
    """
    if not items:
        raise ValueError("no hay escenas Sentinel-2 para componer")

    pilas: dict[str, list[np.ndarray]] = {banda: [] for banda in bandas}
    usadas = 0

    for item in por_nubosidad(items)[:max_escenas]:
        try:
            scl = leer_ventana(item.assets["SCL"].href, bbox, forma)
            mascara = np.isin(scl, list(SCL_VALIDOS))
            if mascara.mean() < COBERTURA_MINIMA:
                continue
            for banda in bandas:
                arreglo = leer_ventana(item.assets[banda].href, bbox, forma).astype("float32")
                arreglo[~mascara] = np.nan
                pilas[banda].append(arreglo)
            usadas += 1
        except Exception:
            log.warning("escena S2 omitida: %s", item.id, exc_info=True)

    if usadas == 0:
        raise RuntimeError("ninguna escena Sentinel-2 aportó píxeles válidos")

    compuesto = {banda: np.nanmedian(np.dstack(capas), axis=2) for banda, capas in pilas.items()}
    return compuesto, usadas


def compuesto_s1(
    items: list["Item"],
    bbox: Bbox,
    forma: tuple[int, int] | None = None,
    max_escenas: int = 24,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Mediana por píxel de escenas Sentinel-1 RTC de una sola geometría de órbita.

    Devuelve las polarizaciones compuestas en potencia lineal junto con los metadatos
    de la adquisición elegida.
    """
    if not items:
        raise ValueError("no hay escenas Sentinel-1 para componer")

    (estado, relativa), disponibles = orbita_dominante(items)
    seleccion = disponibles[:max_escenas]

    pilas: dict[str, list[np.ndarray]] = {"vv": [], "vh": []}
    for item in seleccion:
        try:
            for polarizacion in pilas:
                arreglo = leer_ventana(item.assets[polarizacion].href, bbox, forma)
                pilas[polarizacion].append(arreglo.astype("float32"))
        except Exception:
            log.warning("escena S1 omitida: %s", item.id, exc_info=True)

    if not pilas["vv"]:
        raise RuntimeError("ninguna escena Sentinel-1 aportó píxeles válidos")

    compuesto = {
        polarizacion: np.nanmedian(np.dstack(capas), axis=2)
        for polarizacion, capas in pilas.items()
    }
    meta = {
        "orbita": f"{estado} · relativa {relativa}",
        "escenas_usadas": len(pilas["vv"]),
        "escenas_disponibles": len(disponibles),
    }
    return compuesto, meta
