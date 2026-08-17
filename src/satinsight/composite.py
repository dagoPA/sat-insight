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

FRACCION_FALLOS = 0.3
"""Fracción de lecturas fallidas por encima de la cual el compuesto se da por roto.

Descartar la escena que no se puede leer y seguir es lo correcto frente a una escena rota,
y es la ruina frente a una avería general: si la firma de acceso caduca a media corrida
fallan casi todas, y el compuesto se devuelve armado con un puñado de escenas sin que nada
avise. Así se guardó una vez un compuesto de Acapulco con cuatro escenas de treinta.

El umbral cuenta lecturas fallidas y no escenas usadas, porque son cosas distintas. Una
escena descartada por nubosidad es un dato sobre el cielo de esa ciudad; una descartada por
error de lectura es un síntoma de avería. Mezclarlas haría abortar a Tapachula, que es de
las ciudades más nubladas del país, por una razón que no tiene nada de anómala.
"""


def _revisar_fallos(sensor: str, fallidas: int, intentadas: int, fraccion: float | None) -> None:
    """Levanta excepción cuando demasiadas lecturas fallaron para confiar en el resultado.

    `None` desactiva la comprobación. Cero es su opuesto y significa lo que aparenta: no se
    tolera ni una lectura fallida.
    """
    if fraccion is None or not intentadas:
        return
    if fallidas > fraccion * intentadas:
        raise RuntimeError(
            f"{fallidas} de {intentadas} escenas {sensor} fallaron al leerse. "
            "Un compuesto armado con las que quedan no es representativo; suele indicar "
            "que la firma de acceso caducó a media corrida o que el servicio no responde."
        )


def compuesto_s2(
    items: list["Item"],
    bbox: Bbox,
    forma: tuple[int, int] | None = None,
    bandas: tuple[str, ...] = BANDAS_RGB,
    max_escenas: int = 36,
    fraccion_fallos: float | None = FRACCION_FALLOS,
) -> tuple[dict[str, np.ndarray], int]:
    """Mediana por píxel de las escenas Sentinel-2 más despejadas, con máscara SCL.

    Devuelve las bandas compuestas y el número de escenas que aportaron píxeles.
    Las escenas se recorren de la más despejada a la más nublada.

    Aborta si demasiadas lecturas fallan. `fraccion_fallos` en `None` desactiva esa
    comprobación para quien quiera un compuesto parcial a propósito; en cero no tolera ni
    una lectura fallida.
    """
    if not items:
        raise ValueError("no hay escenas Sentinel-2 para componer")

    pilas: dict[str, list[np.ndarray]] = {banda: [] for banda in bandas}
    usadas = 0
    fallidas = 0
    seleccion = por_nubosidad(items)[:max_escenas]

    for item in seleccion:
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
            fallidas += 1
            log.warning("escena S2 omitida: %s", item.id, exc_info=True)

    _revisar_fallos("Sentinel-2", fallidas, len(seleccion), fraccion_fallos)
    if usadas == 0:
        raise RuntimeError("ninguna escena Sentinel-2 aportó píxeles válidos")

    compuesto = {banda: np.nanmedian(np.dstack(capas), axis=2) for banda, capas in pilas.items()}
    return compuesto, usadas


def compuesto_s1(
    items: list["Item"],
    bbox: Bbox,
    forma: tuple[int, int] | None = None,
    max_escenas: int = 24,
    fraccion_fallos: float | None = FRACCION_FALLOS,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Mediana por píxel de escenas Sentinel-1 RTC de una sola geometría de órbita.

    Devuelve las polarizaciones compuestas en potencia lineal junto con los metadatos
    de la adquisición elegida.

    Aborta si demasiadas lecturas fallan. `fraccion_fallos` en `None` desactiva esa
    comprobación para quien quiera un compuesto parcial a propósito; en cero no tolera ni
    una lectura fallida.
    """
    if not items:
        raise ValueError("no hay escenas Sentinel-1 para componer")

    (estado, relativa), disponibles = orbita_dominante(items)
    seleccion = disponibles[:max_escenas]

    pilas: dict[str, list[np.ndarray]] = {"vv": [], "vh": []}
    fallidas = 0
    for item in seleccion:
        # Las dos polarizaciones se leen antes de guardar ninguna: agregarlas dentro del
        # bucle dejaría VV apilado y VH no cuando la segunda lectura falla, y las medianas
        # de una y otra saldrían calculadas sobre conjuntos de escenas distintos.
        try:
            leidas = {
                polarizacion: leer_ventana(item.assets[polarizacion].href, bbox, forma).astype(
                    "float32"
                )
                for polarizacion in pilas
            }
        except Exception:
            fallidas += 1
            log.warning("escena S1 omitida: %s", item.id, exc_info=True)
            continue
        for polarizacion, arreglo in leidas.items():
            pilas[polarizacion].append(arreglo)

    _revisar_fallos("Sentinel-1", fallidas, len(seleccion), fraccion_fallos)
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
