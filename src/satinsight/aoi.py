"""Áreas de interés del proyecto.

Un AOI es un recuadro geográfico en coordenadas WGS84. Los recuadros piloto tienen todos
el mismo tamaño para que las comparaciones entre ciudades sean directas.
"""

from dataclasses import dataclass

Bbox = tuple[float, float, float, float]


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
        grados_por_metro = 1 / 111_320
        centro_lat = (self.bbox[1] + self.bbox[3]) / 2
        from math import cos, radians

        ancho_m = self.ancho_grados / grados_por_metro * cos(radians(centro_lat))
        alto_m = self.alto_grados / grados_por_metro
        return int(alto_m / resolucion_m), int(ancho_m / resolucion_m)


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
