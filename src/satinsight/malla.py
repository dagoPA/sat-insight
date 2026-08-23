"""Retícula de análisis: la georreferencia que los compuestos no llevan consigo.

`leer_ventana` reproyecta el recuadro al sistema nativo de cada escena, recorta esa ventana
y la remuestrea a la forma pedida. El arreglo que sale queda entonces sobre una retícula
UTM cuya transformación afín nadie guarda, y sin ella no hay manera de decir qué píxel cae
dentro de qué AGEB.

Este módulo reconstruye esa transformación por adelantado. Al pasarle a los compuestos la
forma calculada aquí, la retícula que producen coincide con la que este módulo describe, y
los polígonos se pueden reproyectar al mismo sistema para rasterizarlos sin desfase.

La reconstrucción es válida mientras todas las escenas compartan sistema de referencia.
Sentinel-2 se entrega en teselas MGRS, y las teselas cercanas al borde de un huso UTM se
publican en los dos husos vecinos; remuestrear ambos grupos a la misma forma y apilarlos
los desalinea sin que nada falle. `seleccionar_crs` se queda con el grupo más numeroso y
devuelve solo esas escenas, que es lo que hace la retícula reconstruible.
"""

import logging
from collections.abc import Callable
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


def _codigo_crs(item: "Item") -> str | None:
    """Lee el sistema de referencia declarado por una escena en el STAC."""
    codigo = item.properties.get("proj:epsg") or item.properties.get("proj:code")
    if codigo is None:
        return None
    codigo = str(codigo)
    return codigo if codigo.upper().startswith("EPSG:") else f"EPSG:{codigo}"


def seleccionar_crs(
    items: list["Item"], puntuar: Callable[[list["Item"]], float] | None = None
) -> tuple[str, list["Item"]]:
    """Elige un sistema de referencia y devuelve solo las escenas que lo declaran.

    Las teselas MGRS de Sentinel-2 cercanas al borde de un huso UTM se publican en los dos
    husos vecinos. Mérida está a unos cuarenta kilómetros del meridiano 90 y recibe 148
    escenas en la zona 15 contra 145 en la zona 16, cubriendo cada grupo el recuadro
    completo por su cuenta. Remuestrear ambos grupos a la misma forma y apilarlos los
    desalinea sin que nada falle.

    Quedarse con el grupo más numeroso resuelve ese caso, y falla cuando los dos husos no
    cubren lo mismo. Sobre Guasave las 87 escenas del huso 13 alcanzan la mitad del
    recuadro y las 61 del huso 12 lo cubren entero: elegir por número dejaba la ciudad sin
    compuesto de radar por falta de cobertura, teniendo dos órbitas que la cubren completa.

    Con `puntuar` la decisión pasa a la cobertura medida y el número solo desempata. La
    función recibe las escenas de un huso y devuelve cuánto del recuadro alcanzan; vive
    fuera de este módulo porque medirla exige leer píxeles.

    El código se lee de la extensión de proyección del STAC, que ahorra la petición de red
    por escena que costaría abrir cada ráster.
    """
    grupos: dict[str, list[Item]] = {}
    for item in items:
        codigo = _codigo_crs(item)
        if codigo is not None:
            grupos.setdefault(codigo, []).append(item)

    if not grupos:
        raise ValueError("ninguna escena declara sistema de referencia en el STAC")

    if puntuar is None or len(grupos) == 1:
        elegido = max(grupos, key=lambda c: len(grupos[c]))
    else:
        cobertura = {c: puntuar(v) for c, v in grupos.items()}
        elegido = max(grupos, key=lambda c: (round(cobertura[c], 2), len(grupos[c])))
        log.info(
            "cobertura por huso: %s",
            ", ".join(f"{c} {100 * v:.0f}%" for c, v in sorted(cobertura.items())),
        )
    if len(grupos) > 1:
        descartados = ", ".join(
            f"{c}: {len(v)}" for c, v in sorted(grupos.items(), key=lambda kv: -len(kv[1]))
        )
        log.warning(
            "el recuadro cruza un huso UTM y las escenas llegan en %d sistemas (%s); "
            "se compone solo con %s para no apilar retículas desalineadas",
            len(grupos),
            descartados,
            elegido,
        )
    return elegido, grupos[elegido]


def crs_comun(items: list["Item"]) -> str:
    """Sistema de referencia dominante entre las escenas."""
    return seleccionar_crs(items)[0]


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


def recorte_de_poligono(transform, geometria, forma: tuple[int, int]):
    """Rebanadas y máscara de un polígono sobre la retícula, o `None` si cae fuera.

    Es el paso común a cualquier estadístico zonal: acotar el polígono a su ventana
    envolvente y saber qué píxeles de esa ventana quedan dentro. Vive aquí para que la
    textura y las fracciones de cobertura recorten exactamente igual; si cada una lo
    resolviera por su lado, un desfase entre ambas pasaría inadvertido y las compararíamos
    creyendo que miran los mismos píxeles.

    La geometría tiene que venir en el sistema de referencia de `transform`.
    """
    from rasterio.features import geometry_mask
    from rasterio.transform import rowcol

    alto, ancho = forma
    x_min, y_min, x_max, y_max = geometria.bounds
    fila_a, col_a = rowcol(transform, x_min, y_max)
    fila_b, col_b = rowcol(transform, x_max, y_min)
    fila_ini, fila_fin = max(0, min(fila_a, fila_b)), min(alto, max(fila_a, fila_b) + 1)
    col_ini, col_fin = max(0, min(col_a, col_b)), min(ancho, max(col_a, col_b) + 1)
    if fila_fin <= fila_ini or col_fin <= col_ini:
        return None

    filas = slice(fila_ini, fila_fin)
    columnas = slice(col_ini, col_fin)
    dentro = ~geometry_mask(
        [geometria],
        out_shape=(fila_fin - fila_ini, col_fin - col_ini),
        transform=transform * transform.translation(col_ini, fila_ini),
        invert=False,
    )
    return filas, columnas, dentro


def malla_de_escenas(
    bbox: Bbox,
    items: list["Item"],
    resolucion_m: int = RESOLUCION_M,
    puntuar: Callable[[list["Item"]], float] | None = None,
) -> tuple[Malla, list["Item"]]:
    """Retícula de trabajo junto con las escenas que viven sobre ella.

    Devuelve las escenas filtradas además de la retícula: componer con las que quedaron
    fuera del sistema elegido produciría un apilado desalineado.
    """
    crs, seleccionadas = seleccionar_crs(items, puntuar)
    return malla_de_bbox(bbox, crs, resolucion_m), seleccionadas
