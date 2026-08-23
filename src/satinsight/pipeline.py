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
from satinsight.catalog import COLLECTION_S1, COLLECTION_S2, open_catalogue, search
from satinsight.cobertura import fracciones_por_ageb, mosaico
from satinsight.composite import compuesto_s1, compuesto_s2
from satinsight.ingesta import RAIZ_DATOS
from satinsight.malla import Grid, grid_from_scenes
from satinsight.raster import to_db
from satinsight.textura import RANGOS_FIJOS, RANGOS_FIJOS_S1, rasgos_por_ageb

log = logging.getLogger(__name__)

PERIODO_CENSO = "2020-01-01/2020-12-31"

BANDAS_S2 = ("B04", "B03", "B02", "B08", "B11")
"""Rojo, verde, azul, infrarrojo cercano e infrarrojo de onda corta.

Las cuatro primeras son nativas a 10 m. B11 llega a 20 m y se remuestrea, y entra porque
habilita el NDBI, el índice estándar de superficie construida. El riesgo central del
proyecto es que el modelo lea densidad construida en vez de morfología de la privación, así
que conviene tener el canal que mejor describe lo primero.

Azul y verde no alimentan ningún rasgo: existen para poder renderizar color natural de
cualquier ciudad.
"""

MARGEN_M = 200.0
"""Holgura alrededor de las AGEB, para que ninguna quede cortada por el borde del recuadro."""

SENSORES = ("s2", "s1")

ESCALAS = ("nativa", "fija", "percentiles")
"""Cómo se fija la escala de cuantización de la textura, y por qué hay tres opciones.

- `nativa` es el criterio de la fase 1: rango fijo en decibeles para el radar, porque gamma0
  está calibrado, y percentiles por ciudad para el óptico, porque arrastra residuos
  atmosféricos.
- `fija` cuantiza ambas modalidades con bordes fijos.
- `percentiles` estima el rango de cada ciudad en ambas.

Las dos últimas existen para poder comparar las modalidades bajo el mismo tratamiento.
Medirlas con criterios distintos confunde el sensor con el preproceso, y esa confusión es
justo lo que la comparación tiene que descartar.
"""


def _rango_de_canal(nombre: str, escala: str) -> tuple[float, float] | None:
    """Rango de cuantización de un canal bajo la escala pedida, o `None` para estimarlo."""
    if escala == "nativa":
        return RANGOS_FIJOS_S1.get(nombre)
    if escala == "fija":
        return RANGOS_FIJOS.get(nombre)
    if escala == "percentiles":
        return None
    raise ValueError(f"escala desconocida: {escala!r}. Válidas: {', '.join(ESCALAS)}")


TOPE_S2 = 20
TOPE_S1 = 16
"""Escenas que entran a cada compuesto.

El mismo tope para las cinco ciudades: la validación deja una ciudad fuera por pliegue, así
que un compuesto armado con menos escenas en una de ellas se confundiría con señal de esa
ciudad. Sentinel-2 se recorre de la escena más despejada a la más nublada, de modo que las
veinte primeras son las mejores disponibles.
"""

PROFUNDIDAD_MINIMA = 8
"""Observaciones que debe tener el píxel típico para dar el compuesto por comparable.

`composite` ya aborta cuando fallan demasiadas lecturas, que es el síntoma de una avería.
Esta segunda comprobación mira otra cosa: con cuántas observaciones se calculó la mediana
del píxel típico. Una ciudad compuesta con la mitad de observaciones que otra tiene más
ruido residual, y como la validación reparte los pliegues por ciudad, esa diferencia se
leería como señal de esa ciudad. Responde al diseño experimental, y por eso vive aquí
mientras la detección de averías vive en la librería.

Se cuenta por píxel y no por escena porque una ciudad repartida entre dos teselas MGRS
recibe escenas que solo cubren su mitad del recuadro: contarlas enteras da un número que
ninguna parte de la imagen llegó a tener.
"""


def _exigir_profundidad(clave: str, sensor: str, profundidad: int, minimo: int) -> None:
    """Avisa cuando la mediana del píxel típico se calculó con muy pocas observaciones."""
    if profundidad < minimo:
        raise RuntimeError(
            f"{clave}/{sensor}: el píxel típico se compuso con {profundidad} observaciones, "
            f"menos de las {minimo} exigidas. Comparar ciudades armadas con distinta "
            "profundidad mezcla señal con ruido de muestreo, y la validación reparte los "
            "pliegues justamente por ciudad."
        )


def aoi_de_ciudad(
    clave: str,
    raiz: Path = RAIZ_DATOS,
    *,
    margen_m: float = MARGEN_M,
    catalogo: dict | None = None,
) -> tuple[AOI, gpd.GeoDataFrame]:
    """Recuadro que envuelve a las AGEB de una ciudad, junto con esas AGEB.

    `catalogo` permite trabajar sobre el conjunto nacional que devuelve
    `agebs.ciudades_por_tamano` en vez de las cinco piloto escritas a mano.
    """
    catalogo = catalogo or CIUDADES
    agebs = agebs_de_ciudad(clave, raiz, catalogo=catalogo)
    ciudad = catalogo[clave]
    area = AOI.from_polygons(clave, ciudad.nombre, ciudad.entidad, agebs, margen_m=margen_m)
    alto, ancho = area.approximate_shape()
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
) -> tuple[dict[str, np.ndarray], Grid, dict]:
    """Compone una ciudad y un sensor desde el catálogo, sin consultar el disco."""
    if sensor not in SENSORES:
        raise ValueError(f"sensor desconocido: {sensor!r}. Válidos: {', '.join(SENSORES)}")

    catalogo = catalogo or open_catalogue()
    coleccion = COLLECTION_S2 if sensor == "s2" else COLLECTION_S1
    escenas = search(coleccion, area.bbox, periodo, catalogo)
    if not escenas:
        raise RuntimeError(f"el catálogo no devolvió escenas de {sensor} para {clave}")

    # sobre radar el huso se elige por la cobertura que alcanza cada uno y no por cuántas
    # escenas trae: los dos husos de una ciudad en el borde pueden ver mitades distintas
    puntuar = None
    if sensor == "s1":

        def puntuar(grupo):
            from satinsight.catalog import group_by_orbit
            from satinsight.composite import cobertura_util

            orbitas = group_by_orbit(grupo)
            return max((cobertura_util(v, area.bbox) for v in orbitas.values()), default=0.0)

    malla, escenas = grid_from_scenes(area.bbox, escenas, puntuar=puntuar)
    log.info("%s/%s: %d escenas, retícula %.1f MP", clave, sensor, len(escenas), malla.megapixels)

    if sensor == "s2":
        tope = max_escenas or TOPE_S2
        bandas, meta = compuesto_s2(escenas, area.bbox, malla.shape, BANDAS_S2, tope)
        etiquetas = {"escenas_disponibles": len(escenas), **meta}
        profundidad = int(meta["profundidad_mediana"])
    else:
        tope = max_escenas or TOPE_S1
        bandas, meta = compuesto_s1(escenas, area.bbox, malla.shape, tope)
        etiquetas = dict(meta)
        profundidad = int(meta["escenas_usadas"])

    _exigir_profundidad(clave, sensor, profundidad, min(PROFUNDIDAD_MINIMA, tope))
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


def _mismas_bandas(guardadas, sensor: str) -> bool:
    """Confirma que un compuesto en disco traiga las bandas que el pipeline espera hoy.

    Agregar una banda deja obsoletos los compuestos anteriores, y el archivo sigue
    cargándose sin queja hasta que un canal derivado busca la que falta. Peor todavía si el
    canal nuevo existe en unas ciudades y no en otras: esa diferencia quedaría correlacionada
    con qué ciudades se compusieron primero, que es exactamente el confundido que la
    validación por ciudad tiene que evitar.
    """
    esperadas = set(BANDAS_S2) if sensor == "s2" else {"vv", "vh"}
    return set(guardadas) == esperadas


def asegurar_compuesto(
    clave: str,
    sensor: str,
    *,
    area: AOI | None = None,
    raiz: Path = RAIZ_DATOS,
    periodo: str = PERIODO_CENSO,
    forzar: bool = False,
    **kwargs,
) -> tuple[dict[str, np.ndarray], Grid, dict]:
    """Devuelve el compuesto desde disco, construyéndolo la primera vez.

    Quien ya haya resuelto el recuadro puede pasarlo en `area` para ahorrarse una segunda
    lectura del shapefile de la entidad, que en Ciudad de México ronda los ochenta megas.
    """
    destino = cache.ruta_compuesto(clave, sensor, raiz / "compuestos")
    if destino.exists() and not forzar:
        guardado = cache.cargar(destino)
        recuadro_ok = area is None or _mismo_recuadro(guardado[2].get("bbox"), area.bbox)
        bandas_ok = _mismas_bandas(guardado[0], sensor)
        if recuadro_ok and bandas_ok:
            log.info("compuesto en caché: %s", destino.name)
            return guardado
        motivo = "cubre otro recuadro" if not recuadro_ok else "le faltan bandas"
        log.warning("%s/%s en caché %s; se reconstruye", clave, sensor, motivo)

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
    calibración absoluta. El NDBI agrega la respuesta del infrarrojo de onda corta, que es
    donde el material construido se distingue mejor del suelo desnudo.
    """
    rojo = bandas["B04"].astype("float32")
    nir = bandas["B08"].astype("float32")
    swir = bandas["B11"].astype("float32")
    return {
        "s2rojo": rojo,
        "s2nir": nir,
        "s2ndvi": _division_segura(nir - rojo, nir + rojo),
        "s2ndbi": _division_segura(swir - nir, swir + nir),
    }


def canales_s1(bandas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Canales del brazo radar, en decibeles.

    La conversión ocurre aquí y nunca antes de guardar: el compuesto vive en potencia
    lineal porque promediar en decibeles sesga el resultado hacia los valores bajos.
    """
    vv = to_db(bandas["vv"])
    vh = to_db(bandas["vh"])
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
    escala: str = "nativa",
    catalogo: dict | None = None,
) -> pd.DataFrame:
    """Tabla de rasgos por AGEB para una ciudad y un sensor, con su etiqueta ordinal.

    `escala` elige cómo se cuantiza la textura; ver `ESCALAS`.
    """
    area, agebs = aoi_de_ciudad(clave, raiz, catalogo=catalogo)
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
        parcial = rasgos_por_ageb(
            canal,
            malla.transform,
            geometrias,
            claves,
            prefijo=nombre,
            rango=_rango_de_canal(nombre, escala),
        )
        tabla = tabla.merge(parcial, on="cvegeo", how="left")

    # La cobertura del suelo no depende del sensor, pero se calcula sobre esta retícula
    # para que sus fracciones y los rasgos de textura miren exactamente los mismos píxeles.
    clases = mosaico(area, malla)
    tabla = tabla.merge(
        fracciones_por_ageb(clases, malla.transform, geometrias, claves),
        on="cvegeo",
        how="left",
    )

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
    escala: str = "nativa",
    catalogo: dict | None = None,
) -> pd.DataFrame:
    """Apila las tablas de rasgos de varias ciudades para un mismo sensor.

    Una ciudad que falle no detiene al resto: con 138 ciudades, abortar por una sola
    obliga a repetir horas de trabajo ya hecho, y qué faltó se ve en el registro.
    """
    partes = []
    for c in ciudades:
        try:
            partes.append(
                rasgos_de_ciudad(
                    c, sensor, raiz=raiz, max_escenas=max_escenas, escala=escala, catalogo=catalogo
                )
            )
        except Exception:
            log.warning("sin rasgos para %s", c, exc_info=True)
    if not partes:
        raise RuntimeError(f"ninguna ciudad dio rasgos para {sensor}")
    log.info("rasgos de %d de %d ciudades", len(partes), len(ciudades))
    return pd.concat(partes, ignore_index=True)


def fiabilidad_de_ciudades(
    sensor: str,
    ciudades: tuple[str, ...] | None = None,
    *,
    raiz: Path = RAIZ_DATOS,
    escala: str = "fija",
    catalogo: dict | None = None,
) -> pd.DataFrame:
    """Correlación entre mitades de cada rasgo, agregada sobre muchas ciudades.

    Cada ciudad aporta una correlación por rasgo, y se conserva la mediana entre ciudades
    junto con la peor. Medirlo sobre pocas ciudades deja el criterio a merced de sus
    particularidades: un rasgo puede reproducirse en cinco ciudades del sur y ser ruido
    en el norte, y el filtro lo dejaría pasar sin que nada avisara.
    """
    from satinsight.agebs import ciudades_por_tamano
    from satinsight.textura import fiabilidad_por_mitades

    catalogo = catalogo or ciudades_por_tamano(raiz=raiz, estratificar=True)
    claves = ciudades or tuple(
        p.stem.replace(f"_{sensor}", "")
        for p in sorted((raiz / "compuestos").glob(f"*_{sensor}.tif"))
    )

    partes = []
    for clave in claves:
        try:
            area, agebs = aoi_de_ciudad(clave, raiz, catalogo=catalogo)
            bandas, malla, _ = asegurar_compuesto(clave, sensor, area=area, raiz=raiz)
            agebs = agebs.to_crs(malla.crs)
            canales = canales_s2(bandas) if sensor == "s2" else canales_s1(bandas)
            for nombre, banda in canales.items():
                partes.append(
                    fiabilidad_por_mitades(
                        banda,
                        malla.transform,
                        list(agebs.geometry),
                        list(agebs.cvegeo),
                        prefijo=nombre,
                        rango=_rango_de_canal(nombre, escala),
                    ).assign(ciudad=clave)
                )
        except Exception:
            log.warning("sin fiabilidad para %s", clave, exc_info=True)

    if not partes:
        raise RuntimeError(f"ninguna ciudad dio fiabilidad para {sensor}")
    juntas = pd.concat(partes, ignore_index=True)
    resumen = (
        juntas.groupby("rasgo", observed=True)["r"]
        .agg(r_mediana="median", r_min="min", ciudades="size")
        .reset_index()
        .sort_values("r_mediana", ascending=False)
    )
    log.info("fiabilidad de %d rasgos sobre %d ciudades", len(resumen), juntas.ciudad.nunique())
    return resumen
