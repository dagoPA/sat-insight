"""Fracciones de cobertura del suelo por AGEB, como baseline de densidad construida.

El riesgo abierto del proyecto es el atajo por ruralidad: que el modelo acierte leyendo
cuánto hay construido y no morfología de la privación. Para descartarlo hace falta un
baseline que responda «aquí está cuánto hay construido según otro producto, ¿el modelo
agrega algo encima?».

Los estadísticos de primer orden de los propios compuestos no responden esa pregunta. Salen
de la misma imagen que el modelo ve, así que un modelo de textura mejor les gana por
explotar mejor esa imagen, sin que eso descarte el atajo.

ESA WorldCover 2020 sí: es una clasificación publicada, a 10 m, del mismo año del censo, y
no está ajustada a las etiquetas de CONEVAL. Conviene decir con precisión qué tan
independiente es: WorldCover se deriva de Sentinel-1 y Sentinel-2 de 2020, o sea de los
mismos sensores que alimentan este trabajo. La independencia es de producto: otra cadena
de clasificación sobre la misma familia de observaciones. El paper debe declararlo con esa
precisión.

Se eligió sobre GHSL, que es lo que menciona el plan, por dos razones. GHSL no está
publicado en el STAC de Planetary Computer, y sus 100 m de resolución dejarían a la quinta
parte de las AGEB del piloto con menos de diez píxeles, que es demasiado poco para que el
baseline mismo signifique algo. WorldCover comparte la retícula de 10 m de los compuestos.
"""

import logging

import numpy as np
import pandas as pd

from satinsight.aoi import AOI
from satinsight.catalog import open_catalogue
from satinsight.malla import Grid, polygon_window
from satinsight.raster import read_window

log = logging.getLogger(__name__)

COLECCION = "esa-worldcover"
VERSION = "2020_v100"
"""Edición que coincide con el censo. El catálogo publica además 2021_v200."""

CLASES = {
    10: "arbolado",
    20: "arbustos",
    30: "pastizal",
    40: "cultivo",
    50: "construido",
    60: "desnudo",
    70: "nieve",
    80: "agua",
    90: "humedal",
    95: "manglar",
    100: "musgo",
}

SIN_DATO = 0
"""WorldCover usa el cero como ausencia de dato, y `read_window` rellena con cero fuera
de la tesela. Las dos cosas coinciden, que es lo que permite mosaicar por superposición."""


def mosaico(area: AOI, malla: Grid, catalogo=None) -> np.ndarray:
    """Cobertura del suelo sobre la retícula de la ciudad, uniendo las teselas necesarias.

    WorldCover se publica en teselas de tres grados y una ciudad puede tocar varias. Cada
    tesela se lee sobre la retícula completa —lo que cae fuera llega en cero— y se
    superpone donde la anterior no tenía dato.
    """
    catalogo = catalogo or open_catalogue()
    items = [
        item
        for item in catalogo.search(collections=[COLECCION], bbox=area.bbox).items()
        if VERSION in item.id
    ]
    if not items:
        raise RuntimeError(f"WorldCover {VERSION} no cubre el recuadro de {area.clave}")

    log.info("%s: %d teselas de WorldCover %s", area.clave, len(items), VERSION)
    salida = np.zeros(malla.shape, dtype="uint8")
    for item in items:
        tesela = read_window(item.assets["map"].href, area.bbox, malla.shape)
        faltante = salida == SIN_DATO
        salida[faltante] = tesela.astype("uint8")[faltante]

    cubierto = float((salida != SIN_DATO).mean())
    log.info("%s: %.1f%% de la retícula con cobertura", area.clave, 100 * cubierto)
    if cubierto < 0.99:
        log.warning(
            "%s: %.1f%% de la retícula quedó sin clasificar; las fracciones de esas AGEB "
            "se calculan sobre menos píxeles de los que tienen",
            area.clave,
            100 * (1 - cubierto),
        )
    return salida


def fracciones_por_ageb(
    clases: np.ndarray,
    transform,
    geometrias,
    claves,
    *,
    prefijo: str = "wc",
) -> pd.DataFrame:
    """Fracción de cada clase de cobertura dentro de cada AGEB.

    El denominador son los píxeles clasificados del polígono. Así un hueco sin dato queda
    fuera del cálculo, con lo que se evita leerlo como ausencia de construcción. El número de
    píxeles clasificados se reporta para poder auditarlo.
    """
    if len(geometrias) != len(claves):
        raise ValueError(f"{len(geometrias)} geometrías contra {len(claves)} claves")

    columnas = [f"{prefijo}_{nombre}" for nombre in CLASES.values()]
    renglones = []
    for clave, geometria in zip(claves, geometrias, strict=True):
        base = {"cvegeo": clave, f"{prefijo}_n_px": 0}
        base.update(dict.fromkeys(columnas, np.nan))

        ventana = polygon_window(transform, geometria, clases.shape)
        if ventana is not None:
            filas, cols, dentro = ventana
            valores = clases[filas, cols][dentro]
            valores = valores[valores != SIN_DATO]
            base[f"{prefijo}_n_px"] = int(valores.size)
            if valores.size:
                for codigo, nombre in CLASES.items():
                    base[f"{prefijo}_{nombre}"] = float((valores == codigo).mean())
        renglones.append(base)

    return pd.DataFrame(renglones)
