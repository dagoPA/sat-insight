"""Acceso al catálogo STAC de Microsoft Planetary Computer.

Las escenas se consultan y se firman aquí; la lectura de píxeles vive en `raster`.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

import planetary_computer as pc
import pystac_client

if TYPE_CHECKING:
    from pystac import Item

from satinsight.aoi import Bbox

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

COLECCION_S1 = "sentinel-1-rtc"
COLECCION_S2 = "sentinel-2-l2a"

SCL_VALIDOS = frozenset({4, 5, 6, 7})
"""Clases de la máscara SCL que cuentan como píxel utilizable: vegetación, suelo
desnudo, agua y no clasificado. El resto es nube, sombra, nieve o saturación."""


def abrir_catalogo() -> pystac_client.Client:
    """Cliente STAC con firma automática de los enlaces a los COG."""
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def buscar(
    coleccion: str,
    bbox: Bbox,
    periodo: str,
    catalogo: pystac_client.Client | None = None,
) -> list["Item"]:
    """Escenas de una colección que intersectan el recuadro en el periodo dado.

    El periodo usa la sintaxis de intervalo de STAC, por ejemplo
    ``"2020-01-01/2020-12-31"``.
    """
    catalogo = catalogo or abrir_catalogo()
    busqueda = catalogo.search(collections=[coleccion], bbox=bbox, datetime=periodo)
    return list(busqueda.items())


def resumen_nubes(items: list["Item"]) -> dict[str, float | int]:
    """Estadísticas de nubosidad de un conjunto de escenas Sentinel-2."""
    if not items:
        raise ValueError("no hay escenas que resumir")
    nubes = sorted(item.properties["eo:cloud_cover"] for item in items)
    total = len(nubes)
    return {
        "escenas": total,
        "minimo": round(nubes[0], 1),
        "maximo": round(nubes[-1], 1),
        "mediana": round(nubes[total // 2], 1),
        "pct_mayor_50": round(100 * sum(n > 50 for n in nubes) / total),
        "pct_mayor_80": round(100 * sum(n > 80 for n in nubes) / total),
    }


def por_nubosidad(items: list["Item"]) -> list["Item"]:
    """Escenas Sentinel-2 ordenadas de la más despejada a la más nublada."""
    return sorted(items, key=lambda item: item.properties["eo:cloud_cover"])


def agrupar_por_orbita(items: list["Item"]) -> dict[tuple[str, int], list["Item"]]:
    """Agrupa escenas SAR por estado de órbita y número de órbita relativa.

    Un compuesto SAR solo es coherente dentro de una misma geometría de adquisición:
    mezclar ascendente con descendente cambia el ángulo de incidencia y la dirección
    de las sombras de radar.
    """
    grupos: dict[tuple[str, int], list[Item]] = defaultdict(list)
    for item in items:
        clave = (
            item.properties.get("sat:orbit_state"),
            item.properties.get("sat:relative_orbit"),
        )
        grupos[clave].append(item)
    return dict(grupos)


def orbita_dominante(items: list["Item"]) -> tuple[tuple[str, int], list["Item"]]:
    """Geometría de adquisición con más escenas disponibles, con sus escenas."""
    grupos = agrupar_por_orbita(items)
    if not grupos:
        raise ValueError("no hay escenas SAR que agrupar")
    clave = max(grupos, key=lambda k: len(grupos[k]))
    return clave, grupos[clave]
