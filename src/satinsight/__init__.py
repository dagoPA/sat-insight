"""Desagregación del rezago social a partir de imágenes satelitales.

Herramientas reutilizables para consultar el catálogo STAC de Planetary Computer,
construir compuestos anuales de Sentinel-1 y Sentinel-2 sobre un área de interés, y
renderizar paneles de inspección visual.
"""

from satinsight.aoi import AOI, PILOTO
from satinsight.catalog import (
    COLECCION_S1,
    COLECCION_S2,
    abrir_catalogo,
    agrupar_por_orbita,
    buscar,
    orbita_dominante,
    resumen_nubes,
)
from satinsight.composite import compuesto_s1, compuesto_s2
from satinsight.raster import a_db, estirar, leer_ventana
from satinsight.render import a_data_uri, guardar_rgb

__version__ = "0.1.0"

__all__ = [
    "AOI",
    "COLECCION_S1",
    "COLECCION_S2",
    "PILOTO",
    "a_data_uri",
    "a_db",
    "abrir_catalogo",
    "agrupar_por_orbita",
    "buscar",
    "compuesto_s1",
    "compuesto_s2",
    "estirar",
    "guardar_rgb",
    "leer_ventana",
    "orbita_dominante",
    "resumen_nubes",
]
