"""Rasgos de textura por AGEB a partir de la matriz de co-ocurrencia de niveles de gris.

La GLCM está definida sobre ventanas rectangulares y una AGEB es un polígono irregular. La
salida es recortar la AGEB a su ventana envolvente, marcar como inválido lo que cae fuera
del polígono y descartar del conteo cualquier par donde participe un píxel inválido. Eso se
consigue reservando el nivel cero para lo inválido y eliminando después su renglón y su
columna de la matriz.

La cuantización nunca es por AGEB: normalizar el brillo de cada polígono por separado
borraría la señal de nivel que distingue un asentamiento precario de una colonia
consolidada. Para el óptico la escala se estima de la ciudad completa, porque ahí hay
residuos atmosféricos y de BRDF que no son señal. Para el radar se usa un rango fijo en
decibeles, porque gamma0 está calibrado y estimarlo por ciudad tiraría la comparabilidad
entre países que justificó elegir ese sensor.

El nivel absoluto no se pierde en ningún caso: los rasgos de primer orden se calculan sobre
la banda cruda, en unidades físicas, y la GLCM describe únicamente el arreglo espacial.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import rowcol
from shapely.ops import clip_by_rect
from skimage.feature import graycomatrix, graycoprops

log = logging.getLogger(__name__)

NIVELES = 8
"""Niveles de gris de la cuantización, y la decisión más delicada del módulo.

La matriz tiene `niveles²` celdas y una AGEB aporta del orden de sus píxeles en pares. Las
2,703 AGEB del piloto tienen una mediana de 2,396 píxeles a 10 m, así que:

    32 niveles = 1024 celdas ->  2.3 pares por celda
    16 niveles =  256 celdas ->  9.4 pares por celda
     8 niveles =   64 celdas -> 37.4 pares por celda

Con dos pares por celda, la entropía y la energía miden ruido de muestreo. Lo grave es que
ese sesgo es monótono en el número de píxeles: la entropía se subestima y la energía se
sobreestima cuanto más vacía queda la matriz. El tamaño de una AGEB correlaciona con la
densidad urbana, y la densidad con el rezago, de modo que un rasgo submuestreado apunta en
la dirección del blanco sin contener información sobre él. Y como el tamaño típico cambia
entre ciudades, cada pliegue de la validación vería una estructura de ruido distinta.

Ocho niveles pierden detalle de textura y compran estadísticos que significan algo. La
elección se comprueba con `fiabilidad_por_mitades`, que mide si un rasgo se reproduce al
partir el mismo polígono en dos.
"""

DISTANCIAS = (1, 2, 4)
"""Separaciones en píxeles. A 10 m cubren de la escala de un techo a la de una manzana.

Las distancias se reportan por separado y no promediadas: son escalas distintas y llevan
información distinta. En AGEB angostas casi todos los pares a 4 píxeles cruzan el borde y
se descartan, así que promediar esa distancia con la de 1 diluye la señal buena con la
ruidosa en vez de dejar que el modelo pese cada una.
"""

ANGULOS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
"""Las cuatro orientaciones canónicas. Estas sí se promedian: la invarianza a la rotación
es deseable, porque la traza urbana no tiene una orientación privilegiada que interese."""

PROPIEDADES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")

MINIMO_PIXELES = 640
"""Píxeles mínimos para calcular textura, elegidos junto con `NIVELES`.

Con 64 celdas, 640 píxeles dan diez pares por celda, que es la regla de dedo habitual para
que los estadísticos de Haralick sean estables. Deja fuera al 11% de las AGEB del piloto.

Esas AGEB no se borran del conjunto: salen con sus rasgos de textura en nulo y conservan
los de primer orden, y la exclusión se declara al evaluar. Descartarlas en silencio sesgaría
la muestra justo hacia las AGEB grandes.
"""

RANGOS_FIJOS_S1 = {
    "s1vv": (-25.0, 5.0),
    "s1vh": (-30.0, 0.0),
    "s1razon": (0.0, 15.0),
}
"""Rangos de cuantización fijos para el radar, en decibeles.

Derivar el rango de los datos de cada ciudad tira la propiedad que justificó elegir
Sentinel-1: gamma0 es una magnitud calibrada, comparable entre países sin recalibrar. Si la
cuantización se ajusta a cada ciudad, esa comparabilidad se pierde por una decisión de
implementación, y con ella el argumento de transferencia a Brasil y Colombia.

Los bordes son fijos y físicos, no estimados. El óptico sí admite normalizar por escena
porque arrastra residuos atmosféricos y de BRDF que no son señal.
"""


def rango_robusto(banda: np.ndarray, p_bajo: float = 2.0, p_alto: float = 98.0) -> tuple:
    """Percentiles de la banda completa, que fijan la escala común de cuantización."""
    validos = banda[np.isfinite(banda)]
    if validos.size == 0:
        raise ValueError("la banda no tiene un solo píxel finito")
    return float(np.percentile(validos, p_bajo)), float(np.percentile(validos, p_alto))


def cuantizar(banda: np.ndarray, rango: tuple, niveles: int = NIVELES) -> np.ndarray:
    """Lleva la banda a enteros de 1 a `niveles`, reservando el 0 para lo inválido.

    Los valores fuera del rango se recortan a los extremos en vez de descartarse: un techo
    de lámina muy brillante sigue siendo información aunque caiga sobre el percentil 98.

    Los no finitos se llevan a cero antes de convertir a entero. `np.clip` no toca los NaN,
    y convertir un NaN a entero sin signo da un resultado que la especificación no define y
    que depende de la plataforma. El `np.where` final pisa esas posiciones de todos modos,
    pero el valor intermedio no debe quedar al azar: `a_db` produce NaN en cada píxel sin
    retorno medible, así que el brazo de radar pasa por aquí constantemente.
    """
    bajo, alto = rango
    if not alto > bajo:
        raise ValueError(f"rango degenerado: {rango}")
    finitos = np.isfinite(banda)
    escalada = np.zeros(banda.shape, dtype="float64")
    np.divide(banda - bajo, alto - bajo, out=escalada, where=finitos)
    escalada = np.clip(escalada, 0.0, 1.0)
    niveles_validos = 1 + np.round(escalada * (niveles - 1)).astype(np.uint8)
    return np.where(finitos, niveles_validos, 0).astype(np.uint8)


def entropia(glcm: np.ndarray) -> np.ndarray:
    """Entropía de Shannon por distancia, promediando los ángulos de cada una."""
    p = glcm.astype(np.float64)
    total = p.sum(axis=(0, 1), keepdims=True)
    p = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    logaritmo = np.zeros_like(p)
    np.log2(p, out=logaritmo, where=p > 0)
    return (-(p * logaritmo)).sum(axis=(0, 1)).mean(axis=1)


def _matriz(
    recorte: np.ndarray, niveles: int, distancias: Sequence[int] = DISTANCIAS
) -> np.ndarray:
    """Calcula la GLCM del recorte y elimina el nivel reservado a los píxeles inválidos."""
    glcm = graycomatrix(
        recorte,
        distances=list(distancias),
        angles=list(ANGULOS),
        levels=niveles + 1,
        symmetric=True,
        normed=False,
    )
    return glcm[1:, 1:, :, :]


def nombres_de_rasgos(distancias: Sequence[int] = DISTANCIAS) -> list[str]:
    """Nombres de las columnas de textura, una por propiedad y distancia."""
    familias = [*PROPIEDADES, "entropia", "anisotropia"]
    return [f"{familia}_d{d}" for familia in familias for d in distancias]


def rasgos_de_recorte(
    recorte: np.ndarray, niveles: int = NIVELES, distancias: Sequence[int] = DISTANCIAS
) -> dict[str, float]:
    """Propiedades de Haralick de un recorte ya cuantizado, una por distancia.

    Los ángulos se promedian, porque la traza urbana no tiene una orientación privilegiada
    que interese. Las distancias se conservan separadas, porque son escalas distintas.

    De cada propiedad se guarda además la dispersión entre ángulos bajo el nombre
    `anisotropia`, que distingue una traza orientada de una sin dirección dominante.
    """
    vacio = dict.fromkeys(nombres_de_rasgos(distancias), np.nan)
    glcm = _matriz(recorte, niveles, distancias)
    if glcm.sum() == 0:
        return vacio

    rasgos: dict[str, float] = {}
    for propiedad in PROPIEDADES:
        valores = graycoprops(glcm, propiedad)  # (distancias, ángulos)
        for indice, distancia in enumerate(distancias):
            rasgos[f"{propiedad}_d{distancia}"] = float(np.nanmean(valores[indice]))

    contraste = graycoprops(glcm, "contrast")
    for indice, distancia in enumerate(distancias):
        rasgos[f"anisotropia_d{distancia}"] = float(np.nanstd(contraste[indice]))

    for indice, distancia in enumerate(distancias):
        rasgos[f"entropia_d{distancia}"] = float(entropia(glcm)[indice])
    return rasgos


def rasgos_primer_orden(valores: np.ndarray) -> dict[str, float]:
    """Estadísticos de la distribución de intensidades, sin considerar su arreglo espacial.

    Son el punto de comparación honesto de la GLCM: si la textura no agrega nada sobre la
    media y la dispersión, conviene saberlo antes de defenderla en el paper.
    """
    if valores.size == 0:
        return dict.fromkeys(["media", "desv", "p10", "p50", "p90", "rango_intercuartil"], np.nan)
    p10, p25, p50, p75, p90 = np.percentile(valores, [10, 25, 50, 75, 90])
    return {
        "media": float(np.mean(valores)),
        "desv": float(np.std(valores)),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "rango_intercuartil": float(p75 - p25),
    }


def fiabilidad_por_mitades(
    banda: np.ndarray,
    transform,
    geometrias: Sequence,
    claves: Sequence[str],
    *,
    prefijo: str = "c",
    niveles: int = NIVELES,
    minimo_pixeles: int = MINIMO_PIXELES,
    rango: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Correlación de cada rasgo entre las dos mitades espaciales del mismo polígono.

    Un rasgo que no coincide consigo mismo al partir la AGEB en dos no está midiendo la
    AGEB: está midiendo ruido de muestreo. Como el corte es por la mediana de la coordenada
    horizontal, las dos mitades comparten morfología urbana y difieren solo en qué píxeles
    tocaron, de modo que la correlación entre ellas acota cuánta señal reproducible tiene
    el rasgo.

    Sirve para escoger qué rasgos entran al modelo con un criterio objetivo, decidido antes
    de mirar desempeño y por tanto inmune a elegir lo que conviene.
    """
    izquierdas, derechas, claves_partidas = [], [], []
    for clave, geometria in zip(claves, geometrias, strict=True):
        x_min, y_min, x_max, y_max = geometria.bounds
        medio = (x_min + x_max) / 2
        izquierda = clip_by_rect(geometria, x_min, y_min, medio, y_max)
        derecha = clip_by_rect(geometria, medio, y_min, x_max, y_max)
        if izquierda.is_empty or derecha.is_empty:
            continue
        izquierdas.append(izquierda)
        derechas.append(derecha)
        claves_partidas.append(clave)

    comunes = {
        "prefijo": prefijo,
        "niveles": niveles,
        "minimo_pixeles": minimo_pixeles,
        "rango": rango,
    }
    una = rasgos_por_ageb(banda, transform, izquierdas, claves_partidas, **comunes)
    otra = rasgos_por_ageb(banda, transform, derechas, claves_partidas, **comunes)

    filas = []
    for columna in una.columns:
        if columna == "cvegeo" or columna.endswith("_n_px"):
            continue
        a, b = una[columna], otra[columna]
        validos = a.notna() & b.notna()
        if validos.sum() < 3 or a[validos].nunique() < 2 or b[validos].nunique() < 2:
            filas.append({"rasgo": columna, "n": int(validos.sum()), "r": np.nan})
            continue
        filas.append(
            {
                "rasgo": columna,
                "n": int(validos.sum()),
                "r": float(np.corrcoef(a[validos], b[validos])[0, 1]),
            }
        )
    return pd.DataFrame(filas).sort_values("r", ascending=False).reset_index(drop=True)


def correlacion_con_tamano(tabla: pd.DataFrame, prefijo: str) -> pd.DataFrame:
    """Correlación de cada rasgo con el número de píxeles de la AGEB.

    El sesgo por submuestreo de la GLCM es monótono en el tamaño del polígono, y el tamaño
    correlaciona con la densidad urbana, que a su vez correlaciona con el rezago. Un rasgo
    muy correlacionado con el área es sospechoso: puede estar apuntando al blanco por
    construcción y no por medir morfología.
    """
    columna_px = f"{prefijo}_n_px"
    if columna_px not in tabla:
        raise KeyError(f"la tabla no trae {columna_px}")

    filas = []
    for columna in tabla.columns:
        if not columna.startswith(f"{prefijo}_") or columna == columna_px:
            continue
        validos = tabla[columna].notna() & tabla[columna_px].notna()
        if validos.sum() < 3 or tabla.loc[validos, columna].nunique() < 2:
            continue
        r = np.corrcoef(tabla.loc[validos, columna], tabla.loc[validos, columna_px])[0, 1]
        filas.append({"rasgo": columna, "n": int(validos.sum()), "r_con_n_px": float(r)})

    salida = pd.DataFrame(filas)
    if salida.empty:
        return salida
    return salida.reindex(
        salida["r_con_n_px"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def rasgos_por_ageb(
    banda: np.ndarray,
    transform,
    geometrias: Sequence,
    claves: Sequence[str],
    *,
    prefijo: str,
    niveles: int = NIVELES,
    minimo_pixeles: int = MINIMO_PIXELES,
    rango: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Extrae rasgos de textura y de primer orden para cada polígono sobre una banda.

    La banda se cuantiza una sola vez con el rango de la ciudad completa, y de ahí se
    recorta cada AGEB. Devuelve un renglón por AGEB con las columnas prefijadas por el
    nombre del canal, para poder concatenar varios canales sin colisión de nombres.

    Las geometrías tienen que venir en el mismo sistema de referencia que `transform`,
    que para los compuestos es el huso UTM de las escenas y no coordenadas geográficas.

    Con `rango` se fija la escala de cuantización en vez de estimarla de la banda. Es lo
    que el radar necesita: gamma0 está calibrado y derivar el rango de cada ciudad haría
    que la misma retrodispersión cayera en niveles distintos según dónde se midió.
    """
    if len(geometrias) != len(claves):
        raise ValueError(f"{len(geometrias)} geometrías contra {len(claves)} claves")

    rango = rango or rango_robusto(banda)
    cuantizada = cuantizar(banda, rango, niveles)
    alto, ancho = banda.shape
    renglones = []

    for clave, geometria in zip(claves, geometrias, strict=True):
        x_min, y_min, x_max, y_max = geometria.bounds
        fila_a, col_a = rowcol(transform, x_min, y_max)
        fila_b, col_b = rowcol(transform, x_max, y_min)
        fila_ini, fila_fin = max(0, min(fila_a, fila_b)), min(alto, max(fila_a, fila_b) + 1)
        col_ini, col_fin = max(0, min(col_a, col_b)), min(ancho, max(col_a, col_b) + 1)

        base = {"cvegeo": clave, f"{prefijo}_n_px": 0}
        if fila_fin <= fila_ini or col_fin <= col_ini:
            renglones.append(base)
            continue

        recorte = cuantizada[fila_ini:fila_fin, col_ini:col_fin]
        crudo = banda[fila_ini:fila_fin, col_ini:col_fin]
        transform_ventana = transform * transform.translation(col_ini, fila_ini)

        dentro = ~geometry_mask(
            [geometria],
            out_shape=recorte.shape,
            transform=transform_ventana,
            invert=False,
        )
        valido = dentro & (recorte > 0)
        n_px = int(valido.sum())
        base[f"{prefijo}_n_px"] = n_px
        if n_px < minimo_pixeles:
            renglones.append(base)
            continue

        enmascarado = np.where(valido, recorte, 0).astype(np.uint8)
        rasgos = rasgos_de_recorte(enmascarado, niveles)
        rasgos.update(rasgos_primer_orden(crudo[valido & np.isfinite(crudo)]))
        base.update({f"{prefijo}_{k}": v for k, v in rasgos.items()})
        renglones.append(base)

    tabla = pd.DataFrame(renglones)
    utiles = int((tabla[f"{prefijo}_n_px"] >= minimo_pixeles).sum())
    log.info("%s: %d de %d AGEB con píxeles suficientes", prefijo, utiles, len(tabla))
    return tabla
