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

CRS_METRICO = "EPSG:6372"
"""Cónica conforme de Lambert para México. Las distancias del conurbado se miden aquí."""

VECINDAD_M = 5000.0
"""Distancia al núcleo dentro de la cual un municipio vecino se considera conurbado."""

PEGADO_M = 2500.0
"""Separación máxima entre AGEB para tratarlas como parte de la misma mancha urbana."""


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
    "tapachula": Ciudad("tapachula", "Tapachula", "07", "07089"),
    "acapulco": Ciudad("acapulco", "Acapulco de Juárez", "12", "12001"),
}
"""Las cinco ciudades de la fase 1.

Las tres primeras vienen de la fase 0, escogidas por contraste de forma urbana y de
nubosidad. Entre ellas cubren mal el extremo alto del rezago: suman 1,199 AGEB de las que
69 son de grado alto, un 5.8% contra el 23% nacional, y en Iztapalapa apenas 3 de 454.

Tapachula y Acapulco entran para corregir ese sesgo. Tienen 48.6% y 36.5% de AGEB en grado
alto, y con ellas la muestra piloto se acerca a la composición del país. Sin AGEB rezagadas
en el conjunto, la puerta de decisión de la fase 1 no mide lo que pretende medir.
"""


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


def _mancha_conectada(
    agebs: gpd.GeoDataFrame, municipio_nucleo: str, pegado_m: float
) -> gpd.GeoDataFrame:
    """Se queda con la componente urbana continua que contiene al municipio núcleo.

    Un municipio mexicano suele incluir localidades urbanas separadas de su cabecera por
    decenas de kilómetros. Tapachula arrastra el puerto de la costa, y con él el recuadro
    envolvente crece cuatro veces para cubrir un espacio que está casi todo vacío.

    Dilatar cada AGEB y unir lo que se toca reconstruye las manchas urbanas; quedarse con
    la que contiene la cabecera descarta esos satélites. En Tapachula eso baja el recuadro
    de 12.4 a 1.0 megapíxeles conservando 162 de 207 AGEB.
    """
    union = agebs.geometry.buffer(pegado_m).union_all()
    partes = list(getattr(union, "geoms", [union]))
    if len(partes) <= 1:
        return agebs

    puntos = agebs.geometry.representative_point()
    mejor, mejor_n = agebs, -1
    for parte in partes:
        dentro = agebs[puntos.within(parte)]
        propias = int((dentro["cvegeo"].str[:5] == municipio_nucleo).sum())
        if propias > mejor_n:
            mejor, mejor_n = dentro, propias

    log.info(
        "mancha conectada: %d de %d AGEB, %d satélites descartados",
        len(mejor),
        len(agebs),
        len(agebs) - len(mejor),
    )
    return mejor


def agebs_de_ciudad(
    clave: str,
    raiz: Path = RAIZ_DATOS,
    *,
    minimo_poblacion: int = 0,
    conurbacion: bool = True,
    vecindad_m: float = VECINDAD_M,
    pegado_m: float = PEGADO_M,
) -> gpd.GeoDataFrame:
    """Devuelve las AGEB de una ciudad piloto con geometría y etiqueta ordinal.

    Con `conurbacion` la unidad de análisis deja de ser el municipio y pasa a ser la
    mancha urbana continua: se admiten las AGEB de cualquier municipio de la entidad que
    caiga a menos de `vecindad_m` del núcleo, y luego se recorta a la componente conectada.
    Una ciudad rara vez termina donde termina su municipio, y la periferia conurbada es
    justo donde el rezago varía.

    El cruce con las etiquetas es interno a propósito: una AGEB sin polígono o sin etiqueta
    queda fuera del conjunto, y la cuenta de descartes se registra para poder auditarla.
    """
    ciudad = CIUDADES.get(clave)
    if ciudad is None:
        raise KeyError(f"ciudad desconocida: {clave!r}. Disponibles: {', '.join(sorted(CIUDADES))}")

    geometria = cargar_geometria(ciudad.entidad, raiz)
    nucleo = geometria[geometria["cvegeo"].str[:5] == ciudad.municipio]
    if nucleo.empty:
        raise ValueError(
            f"la entidad {ciudad.entidad} no trae AGEB del municipio {ciudad.municipio}"
        )

    if conurbacion:
        metrico = geometria.to_crs(CRS_METRICO)
        envolvente = metrico[metrico["cvegeo"].isin(nucleo["cvegeo"])].geometry.union_all()
        cerca = metrico[metrico.geometry.intersects(envolvente.buffer(vecindad_m))]
        geometria = _mancha_conectada(cerca, ciudad.municipio, pegado_m).to_crs(CRS_TRABAJO)
    else:
        geometria = nucleo

    etiquetas = cargar_grs(raiz)

    unidas = geometria.merge(
        etiquetas.drop(columns=["cve_ent", "cve_mun", "cve_loc", "folio"]),
        on="cvegeo",
        how="inner",
    )
    log.info(
        "%s: %d polígonos, %d cruzan con etiqueta, %d municipios",
        ciudad.nombre,
        len(geometria),
        len(unidas),
        unidas["cve_mun"].nunique() if len(unidas) else 0,
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
