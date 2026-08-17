"""Retícula de análisis: la georreferencia que los compuestos no llevan consigo.

`leer_ventana` reproyecta el recuadro al sistema nativo de cada escena, recorta esa ventana
y la remuestrea a la forma pedida. El arreglo que sale queda entonces sobre una retícula
UTM cuya transformación afín nadie guarda, y sin ella no hay manera de decir qué píxel cae
dentro de qué AGEB.

Este módulo reconstruye esa transformación por adelantado. Al pasarle a los compuestos la
forma calculada aquí, la retícula que producen coincide con la que este módulo describe, y
los polígonos se pueden reproyectar al mismo sistema para rasterizarlos sin desfase.

La reconstrucción es válida mientras todas las escenas compartan sistema de referencia.
Sentinel-2 se entrega en teselas MGRS y una ciudad a caballo entre dos husos UTM recibiría
escenas en sistemas distintos, que al remuestrearse a la misma forma quedarían apiladas sin
estar alineadas. `crs_comun` verifica esa condición y falla temprano cuando no se cumple.
"""

import logging
from collections import Counter
from math import ceil
from typing import TYPE_CHECKING, NamedTuple

from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

if TYPE_CHECKING:
    from affine import Affine
    from pystac import Item

from satinsight.aoi import Bbox

log = logging.getLogger(__name__)

CRS_GEOGRAFICO = "EPSG:4326"
RESOLUCION_M = 10
"""Tamaño de píxel de la retícula de trabajo. Es la resolución nativa de Sentinel-2."""


class Malla(NamedTuple):
    """Retícula sobre la que viven un compuesto y las AGEB rasterizadas contra él."""

    transform: "Affine"
    forma: tuple[int, int]
    crs: str
    limites: tuple[float, float, float, float]

    @property
    def alto(self) -> int:
        return self.forma[0]

    @property
    def ancho(self) -> int:
        return self.forma[1]

    @property
    def megapixeles(self) -> float:
        return self.alto * self.ancho / 1e6


def crs_comun(items: list["Item"]) -> str:
    """Sistema de referencia compartido por las escenas, o error si hay más de uno.

    Se lee de la extensión de proyección del STAC en vez de abrir los rásteres, que
    costaría una petición de red por escena.
    """
    codigos = Counter()
    for item in items:
        codigo = item.properties.get("proj:epsg") or item.properties.get("proj:code")
        if codigo is not None:
            codigos[str(codigo)] += 1

    if not codigos:
        raise ValueError("ninguna escena declara sistema de referencia en el STAC")

    if len(codigos) > 1:
        detalle = ", ".join(f"{c}: {n} escenas" for c, n in codigos.most_common())
        raise ValueError(
            "las escenas llegan en sistemas de referencia distintos y no se pueden apilar "
            f"sobre una sola retícula ({detalle}). El recuadro cruza un huso UTM; hay que "
            "partirlo o reproyectar a una retícula común antes de componer."
        )

    codigo = next(iter(codigos))
    return codigo if codigo.upper().startswith("EPSG:") else f"EPSG:{codigo}"


def malla_de_bbox(bbox: Bbox, crs: str, resolucion_m: int = RESOLUCION_M) -> Malla:
    """Construye la retícula que `leer_ventana` produciría para este recuadro y sistema.

    El recuadro geográfico se lleva al sistema destino y se cubre con píxeles cuadrados de
    la resolución pedida, redondeando hacia arriba para no perder el borde.
    """
    limites = transform_bounds(CRS_GEOGRAFICO, crs, *bbox)
    izq, abajo, der, arriba = limites
    ancho = max(1, ceil((der - izq) / resolucion_m))
    alto = max(1, ceil((arriba - abajo) / resolucion_m))
    transform = from_bounds(izq, abajo, der, arriba, ancho, alto)
    log.info("retícula %dx%d px @%d m en %s", ancho, alto, resolucion_m, crs)
    return Malla(transform=transform, forma=(alto, ancho), crs=crs, limites=limites)


def malla_de_escenas(bbox: Bbox, items: list["Item"], resolucion_m: int = RESOLUCION_M) -> Malla:
    """Retícula derivada del sistema de referencia que declaran las escenas disponibles."""
    return malla_de_bbox(bbox, crs_comun(items), resolucion_m)
