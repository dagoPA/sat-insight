"""Desagregación del rezago social a partir de imágenes satelitales.

Herramientas reutilizables para consultar el catálogo STAC de Planetary Computer,
construir compuestos anuales de Sentinel-1 y Sentinel-2 sobre un área de interés, y
renderizar paneles de inspección visual.
"""

from satinsight.aoi import AOI, PILOT
from satinsight.catalog import (
    COLLECTION_S1,
    COLLECTION_S2,
    cloud_summary,
    dominant_orbit,
    group_by_orbit,
    open_catalogue,
    search,
)
from satinsight.composite import composite_s1, composite_s2, useful_coverage, useful_orbit
from satinsight.raster import read_window, stretch, to_db
from satinsight.render import save_rgb, to_data_uri

__version__ = "0.1.0"

__all__ = [
    "AOI",
    "COLLECTION_S1",
    "COLLECTION_S2",
    "PILOT",
    "cloud_summary",
    "composite_s1",
    "composite_s2",
    "dominant_orbit",
    "group_by_orbit",
    "open_catalogue",
    "read_window",
    "save_rgb",
    "search",
    "stretch",
    "to_data_uri",
    "to_db",
    "useful_coverage",
    "useful_orbit",
]
