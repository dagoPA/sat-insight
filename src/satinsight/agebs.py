"""Unión de las AGEB urbanas con su Grado de Rezago Social.

CONEVAL publica la etiqueta en un libro de Excel sin geometría; INEGI publica la geometría
sin la etiqueta. La clave de trece caracteres —entidad, municipio, localidad y AGEB
concatenados— es la llave que las une.

El resultado de ese cruce es la unidad de análisis de la fase 1: un polígono con una clase
ordinal y diecisiete indicadores de privación.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from satinsight.ingesta import RAIZ_DATOS, asegurar_coneval, asegurar_inegi, capa_ageb_urbana

log = logging.getLogger(__name__)

GRADOS = ("Muy bajo", "Bajo", "Medio", "Alto", "Muy alto")
"""Las cinco clases del Grado de Rezago Social, en su orden natural."""

ORDINAL = {grado: i for i, grado in enumerate(GRADOS)}

INDICADORES = (
    "analfabeta",
    "sin_escuela_6_14",
    "sin_escuela_15_24",
    "basica_incompleta",
    "sin_salud",
    "hacinamiento",
    "sin_agua",
    "sin_excusado",
    "sin_drenaje",
    "sin_electricidad",
    "piso_tierra",
    "sin_lavadora",
    "sin_refrigerador",
    "sin_telefono",
    "sin_celular",
    "sin_computadora",
    "sin_internet",
)
"""Los diecisiete indicadores de rezago, en el orden en que vienen en el libro."""

COLUMNAS_CONEVAL = (
    "cve_ent",
    "entidad",
    "cve_mun",
    "municipio",
    "cve_loc",
    "localidad",
    "folio",
    "cvegeo",
    "poblacion",
    "viviendas",
    *INDICADORES,
    "grado",
)

FILAS_ENCABEZADO = 6
"""El libro trae título, encabezado en dos niveles y una fila en blanco antes de los datos."""

CRS_TRABAJO = "EPSG:4326"
"""Los recuadros y las consultas al STAC van en coordenadas geográficas."""


@dataclass(frozen=True)
class Ciudad:
    """Ciudad piloto, identificada por el municipio que la contiene."""

    clave: str
    nombre: str
    entidad: str
    municipio: str


CIUDADES: dict[str, Ciudad] = {
    "tuxtla": Ciudad("tuxtla", "Tuxtla Gutiérrez", "07", "07101"),
    "merida": Ciudad("merida", "Mérida", "31", "31050"),
    "iztapalapa": Ciudad("iztapalapa", "Iztapalapa", "09", "09007"),
}
"""Las tres ciudades de la fase 1, escogidas por contraste de rezago y de nubosidad."""


def cargar_grs(raiz: Path = RAIZ_DATOS, *, usar_cache: bool = True) -> pd.DataFrame:
    """Lee el Grado de Rezago Social de las AGEB urbanas del país.

    Leer el libro de Excel completo cuesta cerca de medio minuto, así que el resultado se
    guarda en parquet la primera vez.
    """
    cache = raiz / "grs_ageb_2020.parquet"
    if usar_cache and cache.exists():
        return pd.read_parquet(cache)

    libro = asegurar_coneval(raiz)
    log.info("leyendo %s", libro.name)
    tabla = pd.read_excel(
        libro, skiprows=FILAS_ENCABEZADO, header=None, names=list(COLUMNAS_CONEVAL), dtype=str
    )

    numericas = ["poblacion", "viviendas", *INDICADORES]
    tabla[numericas] = tabla[numericas].apply(pd.to_numeric, errors="coerce")
    tabla["grado"] = tabla["grado"].str.strip()

    desconocidos = set(tabla["grado"].dropna().unique()) - set(GRADOS)
    if desconocidos:
        raise ValueError(f"grados inesperados en el libro de CONEVAL: {sorted(desconocidos)}")
    tabla["ordinal"] = tabla["grado"].map(ORDINAL).astype("Int8")

    tabla = tabla.dropna(subset=["cvegeo", "grado"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(cache, index=False)
    log.info("%d AGEB urbanas con grado", len(tabla))
    return tabla


def cargar_geometria(entidad: str, raiz: Path = RAIZ_DATOS) -> gpd.GeoDataFrame:
    """Lee los polígonos de AGEB urbana de una entidad, reproyectados a WGS84."""
    capa = capa_ageb_urbana(asegurar_inegi(entidad, raiz))
    log.info("leyendo %s", capa.name)
    poligonos = gpd.read_file(capa)
    poligonos.columns = [c.lower() for c in poligonos.columns]
    if poligonos.crs is None:
        raise ValueError(f"la capa {capa} viene sin sistema de referencia")
    return poligonos.to_crs(CRS_TRABAJO)


def agebs_de_ciudad(
    clave: str, raiz: Path = RAIZ_DATOS, *, minimo_poblacion: int = 0
) -> gpd.GeoDataFrame:
    """Devuelve las AGEB de una ciudad piloto con geometría y etiqueta ordinal.

    El cruce es interno a propósito: una AGEB sin polígono o sin etiqueta queda fuera del
    conjunto de entrenamiento, y la cuenta de descartes se registra para poder auditarla.
    """
    ciudad = CIUDADES.get(clave)
    if ciudad is None:
        raise KeyError(f"ciudad desconocida: {clave!r}. Disponibles: {', '.join(sorted(CIUDADES))}")

    geometria = cargar_geometria(ciudad.entidad, raiz)
    geometria = geometria[geometria["cvegeo"].str[:5] == ciudad.municipio]

    etiquetas = cargar_grs(raiz)
    etiquetas = etiquetas[etiquetas["cve_mun"] == ciudad.municipio]

    unidas = geometria.merge(
        etiquetas.drop(columns=["cve_ent", "cve_mun", "cve_loc", "folio"]),
        on="cvegeo",
        how="inner",
    )
    log.info(
        "%s: %d polígonos, %d etiquetas, %d cruzan",
        ciudad.nombre,
        len(geometria),
        len(etiquetas),
        len(unidas),
    )

    if minimo_poblacion:
        antes = len(unidas)
        unidas = unidas[unidas["poblacion"] >= minimo_poblacion]
        log.info(
            "descartadas %d AGEB con menos de %d habitantes", antes - len(unidas), minimo_poblacion
        )

    unidas["ciudad"] = ciudad.clave
    return unidas.reset_index(drop=True)


def resumen_grados(agebs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Cuenta AGEB y población por grado, en el orden ordinal de las clases."""
    conteo = (
        agebs.groupby("grado", observed=True)
        .agg(agebs=("cvegeo", "size"), poblacion=("poblacion", "sum"))
        .reindex(GRADOS)
        .fillna(0)
        .astype(int)
    )
    conteo["pct_agebs"] = (100 * conteo["agebs"] / max(conteo["agebs"].sum(), 1)).round(1)
    return conteo
