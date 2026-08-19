"""Lectura de rásteres remotos y transformaciones de intensidad.

La lectura es siempre por ventana: se piden al servidor únicamente los píxeles del
recuadro de interés, lo cual evita descargar escenas completas de cientos de megabytes.
"""

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from satinsight.aoi import Bbox
from satinsight.catalog import firmar

CRS_GEOGRAFICO = "EPSG:4326"


def leer_ventana(
    href: str,
    bbox: Bbox,
    forma: tuple[int, int] | None = None,
) -> np.ndarray:
    """Lee del COG remoto solo la ventana que cubre el recuadro.

    El recuadro llega en coordenadas geográficas y se reproyecta al sistema nativo de
    la escena. Cuando se pasa `forma`, la ventana se remuestrea a esas dimensiones,
    lo cual sirve para alinear bandas de distinta resolución.

    La firma se renueva aquí y no al consultar el catálogo, porque un compuesto tarda más
    que la vida del token.
    """
    with rasterio.open(firmar(href)) as origen:
        limites = transform_bounds(CRS_GEOGRAFICO, origen.crs, *bbox)
        ventana = from_bounds(*limites, origen.transform)
        destino = forma or (int(ventana.height), int(ventana.width))
        # El relleno de la lectura sin límites y el centinela de la escena marcan lo mismo:
        # píxeles jamás observados. En un ráster de punto flotante los dos se vuelven NaN,
        # que es lo que las medianas del compuesto saben ignorar. Sentinel-1 RTC declara
        # -32768 y lo escribe en la sombra del radar y fuera de la franja; dejarlo pasar
        # como número hunde la mediana de toda ciudad cuyo recuadro asome del borde de la
        # escena, y un cero de relleno se leería como retrodispersión nula, que tampoco es
        # cierto. Los rásteres enteros —Sentinel-2, WorldCover— conservan el relleno en cero
        # porque no admiten NaN y su cero ya significa fuera de dato.
        flotante = np.issubdtype(np.dtype(origen.dtypes[0]), np.floating)
        datos = origen.read(
            1,
            window=ventana,
            out_shape=destino,
            boundless=True,
            fill_value=np.nan if flotante else 0,
        )
        if flotante and origen.nodata is not None:
            datos[datos == np.float32(origen.nodata)] = np.nan
        return datos


def estirar(banda: np.ndarray, inferior: float = 2, superior: float = 98) -> np.ndarray:
    """Lleva una banda a 0-255 recortando por percentiles.

    Los ceros se tratan como sin dato, que es la convención de relleno de
    `leer_ventana`. Una banda sin píxeles válidos devuelve todo a cero.
    """
    banda = np.asarray(banda, dtype="float32")
    validos = banda[np.isfinite(banda) & (banda != 0)]
    if validos.size == 0:
        return np.zeros(banda.shape, dtype="uint8")
    piso, techo = np.percentile(validos, [inferior, superior])
    if techo <= piso:
        techo = piso + 1e-6
    return np.clip((banda - piso) / (techo - piso) * 255, 0, 255).astype("uint8")


def a_db(potencia: np.ndarray) -> np.ndarray:
    """Convierte retrodispersión lineal a decibeles.

    Los valores nulos o negativos quedan como NaN, que es lo que corresponde a un
    píxel sin retorno medible.
    """
    potencia = np.asarray(potencia, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.where(potencia > 0, potencia, np.nan))


def percentiles(
    banda: np.ndarray, inferior: float = 5, superior: float = 95
) -> tuple[float, float]:
    """Par de percentiles de una banda, ignorando NaN. Útil para leyendas."""
    return (
        round(float(np.nanpercentile(banda, inferior)), 1),
        round(float(np.nanpercentile(banda, superior)), 1),
    )
