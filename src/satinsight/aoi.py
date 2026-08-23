"""Areas of interest of the project.

An AOI is a geographic box in WGS84 coordinates. The pilot boxes are all the same size so
that comparisons between cities are direct.
"""

from dataclasses import dataclass
from math import cos, radians

Bbox = tuple[float, float, float, float]

METRES_PER_DEGREE = 111_320
"""Length of a degree of latitude. For longitude it is corrected by the cosine of latitude."""


@dataclass(frozen=True)
class AOI:
    """Analysis box identified by key, with its administrative context."""

    key: str
    name: str
    state: str
    bbox: Bbox

    def __post_init__(self) -> None:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError(f"degenerate bbox in {self.key}: {self.bbox}")
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError(f"longitude out of range in {self.key}: {self.bbox}")
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError(f"latitude out of range in {self.key}: {self.bbox}")

    @property
    def width_degrees(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height_degrees(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def approximate_shape(self, resolution_m: int = 10) -> tuple[int, int]:
        """Height and width in pixels the box would take at the given resolution."""
        centre_lat = (self.bbox[1] + self.bbox[3]) / 2
        width_m = self.width_degrees * METRES_PER_DEGREE * cos(radians(centre_lat))
        height_m = self.height_degrees * METRES_PER_DEGREE
        return int(height_m / resolution_m), int(width_m / resolution_m)

    @classmethod
    def from_polygons(
        cls,
        key: str,
        name: str,
        state: str,
        polygons: object,
        margin_m: float = 0.0,
    ) -> "AOI":
        """Derives a box that wraps a set of polygons.

        Accepts a GeoDataFrame —whose `total_bounds` it takes— or a tuple of bounds in
        WGS84 directly. The margin is given in metres and converted to degrees correcting
        longitude by the latitude of the centre, so the border is even on both axes.

        This is what lets the phase one boxes be derived from each city's real AGEB.
        """
        bounds = getattr(polygons, "total_bounds", polygons)
        lon_min, lat_min, lon_max, lat_max = (float(v) for v in bounds)

        if margin_m:
            degrees_lat = margin_m / METRES_PER_DEGREE
            centre_lat = (lat_min + lat_max) / 2
            degrees_lon = degrees_lat / max(cos(radians(centre_lat)), 1e-6)
            lon_min, lon_max = lon_min - degrees_lon, lon_max + degrees_lon
            lat_min, lat_max = lat_min - degrees_lat, lat_max + degrees_lat

        return cls(key=key, name=name, state=state, bbox=(lon_min, lat_min, lon_max, lat_max))


PILOT: dict[str, AOI] = {
    "tuxtla": AOI(
        key="tuxtla",
        name="Tuxtla Gutiérrez",
        state="Chiapas",
        bbox=(-93.135, 16.740, -93.095, 16.768),
    ),
    "merida": AOI(
        key="merida",
        name="Mérida",
        state="Yucatán",
        bbox=(-89.640, 20.955, -89.600, 20.983),
    ),
    "iztapalapa": AOI(
        key="iztapalapa",
        name="Iztapalapa, CDMX",
        state="Ciudad de México",
        bbox=(-99.100, 19.336, -99.060, 19.364),
    ),
}
"""Phase one pilot boxes, chosen for their contrast in deprivation and in cloud cover."""


def get(key: str) -> AOI:
    """Retrieves a pilot AOI by key, with a useful message when it does not exist."""
    try:
        return PILOT[key]
    except KeyError:
        available = ", ".join(sorted(PILOT))
        raise KeyError(f"unknown AOI: {key!r}. Available: {available}") from None
