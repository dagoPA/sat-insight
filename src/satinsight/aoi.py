"""Áreas de interés del proyecto.

Un AOI es un recuadro geográfico en coordenadas WGS84. Los recuadros piloto tienen todos
el mismo tamaño para que las comparaciones entre ciudades sean directas.
"""

from dataclasses import dataclass
from math import cos, radians

Bbox = tuple[float, float, float, float]

METROS_POR_GRADO = 111_320
"""Longitud de un grado de latitud. Para longitud se corrige por el coseno de la latitud."""


@dataclass(frozen=True)
class AOI:
    """Recuadro de análisis identificado por clave, con su contexto administrativo."""

    clave: str
    nombre: str
    entidad: str
    bbox: Bbox

    def __post_init__(self) -> None:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError(f"bbox degenerado en {self.clave}: {self.bbox}")
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError(f"longitud fuera de rango en {self.clave}: {self.bbox}")
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError(f"latitud fuera de rango en {self.clave}: {self.bbox}")

    @property
    def ancho_grados(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def alto_grados(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def forma_aproximada(self, resolucion_m: int = 10) -> tuple[int, int]:
        """Alto y ancho en píxeles que ocuparía el recuadro a la resolución dada."""
        centro_lat = (self.bbox[1] + self.bbox[3]) / 2
        ancho_m = self.ancho_grados * METROS_POR_GRADO * cos(radians(centro_lat))
        alto_m = self.alto_grados * METROS_POR_GRADO
        return int(alto_m / resolucion_m), int(ancho_m / resolucion_m)

    @classmethod
    def desde_poligonos(
        cls,
        clave: str,
        nombre: str,
        entidad: str,
        poligonos: object,
        margen_m: float = 0.0,
    ) -> "AOI":
        """Deriva un recuadro que envuelve un conjunto de polígonos.

        Acepta un GeoDataFrame —del que toma `total_bounds`— o directamente una tupla de
        límites en WGS84. El margen se da en metros y se convierte a grados corrigiendo la
        longitud por la latitud del centro, para que el borde sea parejo en ambos ejes.

        Sirve para que las ciudades de la fase 1 salgan de sus AGEB reales en vez de
        depender de recuadros escritos a mano.
        """
        limites = getattr(poligonos, "total_bounds", poligonos)
        lon_min, lat_min, lon_max, lat_max = (float(v) for v in limites)

        if margen_m:
            grados_lat = margen_m / METROS_POR_GRADO
            centro_lat = (lat_min + lat_max) / 2
            grados_lon = grados_lat / max(cos(radians(centro_lat)), 1e-6)
            lon_min, lon_max = lon_min - grados_lon, lon_max + grados_lon
            lat_min, lat_max = lat_min - grados_lat, lat_max + grados_lat

        return cls(
            clave=clave,
            nombre=nombre,
            entidad=entidad,
            bbox=(lon_min, lat_min, lon_max, lat_max),
        )


PILOTO: dict[str, AOI] = {
    "tuxtla": AOI(
        clave="tuxtla",
        nombre="Tuxtla Gutiérrez",
        entidad="Chiapas",
        bbox=(-93.135, 16.740, -93.095, 16.768),
    ),
    "merida": AOI(
        clave="merida",
        nombre="Mérida",
        entidad="Yucatán",
        bbox=(-89.640, 20.955, -89.600, 20.983),
    ),
    "iztapalapa": AOI(
        clave="iztapalapa",
        nombre="Iztapalapa, CDMX",
        entidad="Ciudad de México",
        bbox=(-99.100, 19.336, -99.060, 19.364),
    ),
}
"""Recuadros piloto de la fase 1, escogidos por contraste de rezago y de nubosidad."""


def obtener(clave: str) -> AOI:
    """Recupera un AOI piloto por clave, con mensaje útil si no existe."""
    try:
        return PILOTO[clave]
    except KeyError:
        disponibles = ", ".join(sorted(PILOTO))
        raise KeyError(f"AOI desconocido: {clave!r}. Disponibles: {disponibles}") from None
