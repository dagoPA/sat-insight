"""De una ciudad a su tabla de rasgos por AGEB.

Encadena las piezas de la fase 1 en el único orden en que se pueden ejecutar: las AGEB
definen el recuadro, el recuadro define la búsqueda en el catálogo, las escenas definen la
retícula, la retícula permite componer, y el compuesto permite medir textura dentro de cada
polígono.

Cada compuesto se guarda al construirse, así que una corrida interrumpida se reanuda sin
volver a pagar la hora de descarga por ciudad y sensor.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from satinsight import cache
from satinsight.agebs import CIUDADES, agebs_de_ciudad
from satinsight.aoi import AOI
from satinsight.catalog import COLECCION_S1, COLECCION_S2, abrir_catalogo, buscar
from satinsight.composite import compuesto_s1, compuesto_s2
from satinsight.ingesta import RAIZ_DATOS
from satinsight.malla import Malla, malla_de_escenas
from satinsight.raster import a_db
from satinsight.textura import rasgos_por_ageb

log = logging.getLogger(__name__)

PERIODO_CENSO = "2020-01-01/2020-12-31"

BANDAS_S2 = ("B04", "B03", "B02", "B08")
"""Rojo, verde, azul e infrarrojo cercano. Las cuatro nativas a 10 m."""

MARGEN_M = 200.0
"""Holgura alrededor de las AGEB, para que ninguna quede cortada por el borde del recuadro."""

SENSORES = ("s2", "s1")


def aoi_de_ciudad(
    clave: str, raiz: Path = RAIZ_DATOS, *, margen_m: float = MARGEN_M
) -> tuple[AOI, gpd.GeoDataFrame]:
    """Recuadro que envuelve a las AGEB de una ciudad, junto con esas AGEB."""
    agebs = agebs_de_ciudad(clave, raiz)
    ciudad = CIUDADES[clave]
    area = AOI.desde_poligonos(clave, ciudad.nombre, ciudad.entidad, agebs, margen_m=margen_m)
    alto, ancho = area.forma_aproximada()
    log.info("%s: %d AGEB, recuadro ~%dx%d px @10 m", ciudad.nombre, len(agebs), ancho, alto)
    return area, agebs


def construir_compuesto(
    clave: str,
    sensor: str,
    area: AOI,
    *,
    periodo: str = PERIODO_CENSO,
    max_escenas: int | None = None,
    catalogo=None,
) -> tuple[dict[str, np.ndarray], Malla, dict]:
    """Compone una ciudad y un sensor desde el catálogo, sin consultar el disco."""
    if sensor not in SENSORES:
        raise ValueError(f"sensor desconocido: {sensor!r}. Válidos: {', '.join(SENSORES)}")

    catalogo = catalogo or abrir_catalogo()
    coleccion = COLECCION_S2 if sensor == "s2" else COLECCION_S1
    escenas = buscar(coleccion, area.bbox, periodo, catalogo)
    if not escenas:
        raise RuntimeError(f"el catálogo no devolvió escenas de {sensor} para {clave}")

    malla, escenas = malla_de_escenas(area.bbox, escenas)
    log.info("%s/%s: %d escenas, retícula %.1f MP", clave, sensor, len(escenas), malla.megapixeles)

    if sensor == "s2":
        bandas, usadas = compuesto_s2(escenas, area.bbox, malla.forma, BANDAS_S2, max_escenas or 36)
        etiquetas = {"escenas_disponibles": len(escenas), "escenas_usadas": usadas}
    else:
        bandas, meta = compuesto_s1(escenas, area.bbox, malla.forma, max_escenas or 24)
        etiquetas = dict(meta)

    etiquetas |= {"ciudad": clave, "sensor": sensor, "periodo": periodo, "bbox": list(area.bbox)}
    return bandas, malla, etiquetas


def _mismo_recuadro(guardado, actual, tolerancia: float = 1e-6) -> bool:
    """Compara el recuadro con el que se construyó un compuesto contra el vigente.

    El compuesto se guarda con el nombre de la ciudad, así que un cambio en la regla que
    define el recuadro dejaría en disco un archivo que cubre otra zona y se seguiría
    reutilizando sin avisar. Comparar el recuadro almacenado convierte esa corrupción
    silenciosa en una reconstrucción.
    """
    if guardado is None:
        return False
    return len(guardado) == len(actual) and all(
        abs(float(a) - float(b)) <= tolerancia for a, b in zip(guardado, actual, strict=True)
    )


def asegurar_compuesto(
    clave: str,
    sensor: str,
    *,
    area: AOI | None = None,
    raiz: Path = RAIZ_DATOS,
    periodo: str = PERIODO_CENSO,
    forzar: bool = False,
    **kwargs,
) -> tuple[dict[str, np.ndarray], Malla, dict]:
    """Devuelve el compuesto desde disco, construyéndolo la primera vez.

    Quien ya haya resuelto el recuadro puede pasarlo en `area` para ahorrarse una segunda
    lectura del shapefile de la entidad, que en Ciudad de México ronda los ochenta megas.
    """
    destino = cache.ruta_compuesto(clave, sensor, raiz / "compuestos")
    if destino.exists() and not forzar:
        guardado = cache.cargar(destino)
        if area is None or _mismo_recuadro(guardado[2].get("bbox"), area.bbox):
            log.info("compuesto en caché: %s", destino.name)
            return guardado
        log.warning(
            "%s/%s en caché cubre otro recuadro que el actual; se reconstruye", clave, sensor
        )

    if area is None:
        area, _ = aoi_de_ciudad(clave, raiz)
    bandas, malla, etiquetas = construir_compuesto(clave, sensor, area, periodo=periodo, **kwargs)
    cache.guardar(bandas, malla, destino, **etiquetas)
    return bandas, malla, etiquetas


def _division_segura(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    """Cociente que devuelve NaN donde el denominador se anula."""
    salida = np.full(numerador.shape, np.nan, dtype="float32")
    np.divide(numerador, denominador, out=salida, where=np.abs(denominador) > 1e-6)
    return salida


def canales_s2(bandas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Canales sobre los que se mide textura en el brazo óptico.

    El rojo lleva la señal de material construido, el infrarrojo cercano separa vegetación
    de suelo desnudo, y el NDVI resume ambos en un índice acotado que no depende de la
    calibración absoluta.
    """
    rojo = bandas["B04"].astype("float32")
    nir = bandas["B08"].astype("float32")
    return {
        "s2rojo": rojo,
        "s2nir": nir,
        "s2ndvi": _division_segura(nir - rojo, nir + rojo),
    }


def canales_s1(bandas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Canales del brazo radar, en decibeles.

    La conversión ocurre aquí y nunca antes de guardar: el compuesto vive en potencia
    lineal porque promediar en decibeles sesga el resultado hacia los valores bajos.
    """
    vv = a_db(bandas["vv"])
    vh = a_db(bandas["vh"])
    return {"s1vv": vv, "s1vh": vh, "s1razon": vv - vh}


CANALES = {"s2": canales_s2, "s1": canales_s1}


def rasgos_de_ciudad(
    clave: str,
    sensor: str,
    *,
    raiz: Path = RAIZ_DATOS,
    periodo: str = PERIODO_CENSO,
    forzar: bool = False,
    max_escenas: int | None = None,
) -> pd.DataFrame:
    """Tabla de rasgos por AGEB para una ciudad y un sensor, con su etiqueta ordinal."""
    area, agebs = aoi_de_ciudad(clave, raiz)
    bandas, malla, _ = asegurar_compuesto(
        clave,
        sensor,
        area=area,
        raiz=raiz,
        periodo=periodo,
        forzar=forzar,
        max_escenas=max_escenas,
    )

    proyectadas = agebs.to_crs(malla.crs)
    geometrias = list(proyectadas.geometry)
    claves = list(proyectadas["cvegeo"])

    tabla = pd.DataFrame({"cvegeo": claves})
    for nombre, canal in CANALES[sensor](bandas).items():
        parcial = rasgos_por_ageb(canal, malla.transform, geometrias, claves, prefijo=nombre)
        tabla = tabla.merge(parcial, on="cvegeo", how="left")

    etiquetas = agebs[["cvegeo", "ciudad", "grado", "ordinal", "poblacion", "viviendas"]]
    tabla = tabla.merge(etiquetas, on="cvegeo", how="left")
    tabla["area_km2"] = proyectadas.geometry.area.to_numpy() / 1e6
    return tabla


def rasgos_de_todas(
    sensor: str,
    ciudades: tuple[str, ...] = tuple(CIUDADES),
    *,
    raiz: Path = RAIZ_DATOS,
    max_escenas: int | None = None,
) -> pd.DataFrame:
    """Apila las tablas de rasgos de varias ciudades para un mismo sensor."""
    partes = [rasgos_de_ciudad(c, sensor, raiz=raiz, max_escenas=max_escenas) for c in ciudades]
    return pd.concat(partes, ignore_index=True)
