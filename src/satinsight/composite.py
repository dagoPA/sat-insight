"""Compuestos mediana anuales de Sentinel-1 y Sentinel-2.

El compositing cumple aquí una única función: suprimir nubes en el óptico y speckle en
el radar. El objeto de análisis sigue siendo una imagen estática de un solo corte anual.
"""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pystac import Item

from satinsight.aoi import Bbox
from satinsight.catalog import SCL_VALIDOS, agrupar_por_orbita, por_nubosidad
from satinsight.raster import leer_ventana

log = logging.getLogger(__name__)

BANDAS_RGB = ("B04", "B03", "B02")
COBERTURA_MINIMA = 0.05
"""Fracción de píxeles válidos por debajo de la cual una escena se descarta."""

FRACCION_FALLOS = 0.3
"""Fracción de lecturas fallidas por encima de la cual el compuesto se da por roto.

Descartar la escena que no se puede leer y seguir es lo correcto frente a una escena rota,
y es la ruina frente a una avería general: si la firma de acceso caduca a media corrida
fallan casi todas, y el compuesto se devuelve armado con un puñado de escenas sin que nada
avise. Así se guardó una vez un compuesto de Acapulco con cuatro escenas de treinta.

El umbral cuenta lecturas fallidas y no escenas usadas, porque son cosas distintas. Una
escena descartada por nubosidad es un dato sobre el cielo de esa ciudad; una descartada por
error de lectura es un síntoma de avería. Mezclarlas haría abortar a Tapachula, que es de
las ciudades más nubladas del país, por una razón que no tiene nada de anómala.
"""


def _revisar_fallos(sensor: str, fallidas: int, intentadas: int, fraccion: float | None) -> None:
    """Levanta excepción cuando demasiadas lecturas fallaron para confiar en el resultado.

    `None` desactiva la comprobación. Cero es su opuesto y significa lo que aparenta: no se
    tolera ni una lectura fallida.
    """
    if fraccion is None or not intentadas:
        return
    if fallidas > fraccion * intentadas:
        raise RuntimeError(
            f"{fallidas} de {intentadas} escenas {sensor} fallaron al leerse. "
            "Un compuesto armado con las que quedan no es representativo; suele indicar "
            "que la firma de acceso caducó a media corrida o que el servicio no responde."
        )


COBERTURA_DE_TESELA = 0.02
"""Fracción del recuadro que una tesela MGRS debe alcanzar para entrar al compuesto."""


def teselas_utiles(
    items: list["Item"],
    bbox: Bbox,
    muestras: int = 2,
    minimo: float = COBERTURA_DE_TESELA,
    leer=None,
) -> list["Item"]:
    """Descarta las escenas cuya tesela MGRS no llega al recuadro.

    Sentinel-2 se entrega en teselas fijas, y el catálogo devuelve toda escena cuya tesela
    intersecte el recuadro pedido, por poco que sea. Una ciudad que cae partida entre dos
    teselas recibe entonces escenas que sobre su recuadro no traen un solo píxel: fuera de
    su huella la lectura se rellena con ceros y el SCL en cero significa sin dato, de modo
    que la máscara sale vacía y la escena se descarta ya dentro del bucle.

    El costo no es descartarlas, es haberlas elegido: la selección se queda con las más
    despejadas del año sin mirar dónde caen, y sobre San Pedro Tlaquepaque diecinueve de
    las veinte mejores resultaron ser de la tesela que no toca la ciudad. Quedaba una sola
    escena para toda la mediana.

    Se sondea entonces cada tesela una vez, a baja resolución, y se conservan las que sí
    aportan. Un recuadro repartido entre dos teselas conserva ambas, y la mediana por píxel
    las combina donde cada una tiene dato.
    """
    leer = leer or leer_ventana
    grupos: dict[str, list] = defaultdict(list)
    for item in items:
        grupos[item.properties.get("s2:mgrs_tile", "?")].append(item)

    conservadas: list = []
    for tesela, escenas in sorted(grupos.items()):
        fracciones = []
        for escena in por_nubosidad(escenas)[:muestras]:
            try:
                scl = leer(escena.assets["SCL"].href, bbox, FORMA_SONDA)
            except Exception:
                log.warning("sondeo fallido en %s", escena.id, exc_info=True)
                continue
            fracciones.append(float((scl > 0).mean()))
        cobertura = max(fracciones) if fracciones else 0.0
        if cobertura >= minimo:
            conservadas.extend(escenas)
        else:
            log.info("tesela %s descartada: cubre %.0f%% del recuadro", tesela, 100 * cobertura)
    if not conservadas:
        raise RuntimeError(f"ninguna de las {len(grupos)} teselas Sentinel-2 alcanza el recuadro")
    log.info("%d escenas en teselas útiles de %d", len(conservadas), len(items))
    return conservadas


def compuesto_s2(
    items: list["Item"],
    bbox: Bbox,
    forma: tuple[int, int] | None = None,
    bandas: tuple[str, ...] = BANDAS_RGB,
    max_escenas: int = 36,
    fraccion_fallos: float | None = FRACCION_FALLOS,
) -> tuple[dict[str, np.ndarray], int]:
    """Mediana por píxel de las escenas Sentinel-2 más despejadas, con máscara SCL.

    Devuelve las bandas compuestas y el número de escenas que aportaron píxeles.
    Las escenas se recorren de la más despejada a la más nublada.

    Aborta si demasiadas lecturas fallan. `fraccion_fallos` en `None` desactiva esa
    comprobación para quien quiera un compuesto parcial a propósito; en cero no tolera ni
    una lectura fallida.
    """
    if not items:
        raise ValueError("no hay escenas Sentinel-2 para componer")

    pilas: dict[str, list[np.ndarray]] = {banda: [] for banda in bandas}
    usadas = 0
    fallidas = 0
    seleccion = por_nubosidad(teselas_utiles(items, bbox))[:max_escenas]

    for item in seleccion:
        try:
            scl = leer_ventana(item.assets["SCL"].href, bbox, forma)
            mascara = np.isin(scl, list(SCL_VALIDOS))
            if mascara.mean() < COBERTURA_MINIMA:
                continue
            for banda in bandas:
                arreglo = leer_ventana(item.assets[banda].href, bbox, forma).astype("float32")
                arreglo[~mascara] = np.nan
                pilas[banda].append(arreglo)
            usadas += 1
        except Exception:
            fallidas += 1
            log.warning("escena S2 omitida: %s", item.id, exc_info=True)

    _revisar_fallos("Sentinel-2", fallidas, len(seleccion), fraccion_fallos)
    if usadas == 0:
        raise RuntimeError("ninguna escena Sentinel-2 aportó píxeles válidos")

    compuesto = {banda: np.nanmedian(np.dstack(capas), axis=2) for banda, capas in pilas.items()}
    return compuesto, usadas


FRACCION_VALIDA = 0.80
"""Fracción mínima de píxeles observados que se le exige a un compuesto de radar."""


def _revisar_compuesto_s1(
    compuesto: dict[str, np.ndarray], minimo: float = FRACCION_VALIDA
) -> float:
    """Rechaza un compuesto de radar que salga sin observar o con valores imposibles.

    Gamma0 en potencia lineal es estrictamente positiva. Un píxel en cero o negativo solo
    puede venir de que el sin-dato de la escena entrara a la mediana, y ese defecto no se
    delata solo: con un número par de escenas la mediana promedia el centinela con un valor
    bueno y devuelve un número intermedio que parece dato. La única forma de verlo es
    contar signos.

    Una fracción alta de NaN significa que la órbita elegida no pasa por la ciudad. Es
    preferible que la ciudad falle a que entre al conjunto con un compuesto hueco.
    """
    for polarizacion, arreglo in compuesto.items():
        finitos = np.isfinite(arreglo)
        fraccion = float(finitos.mean())
        if fraccion < minimo:
            raise RuntimeError(
                f"el compuesto Sentinel-1 solo observó el {100 * fraccion:.0f}% del recuadro "
                f"en {polarizacion}; ninguna órbita cubre la ciudad"
            )
        impropios = float((arreglo[finitos] <= 0).mean())
        if impropios > 0:
            raise RuntimeError(
                f"el {100 * impropios:.1f}% de {polarizacion} salió en cero o negativo, "
                "que gamma0 lineal no admite: el sin-dato de la escena entró a la mediana"
            )
    return min(float(np.isfinite(a).mean()) for a in compuesto.values())


MUESTRAS_DE_ORBITA = 4
"""Escenas que se sondean por órbita para estimar cuánto dato deja sobre el recuadro."""

FORMA_SONDA = (64, 64)
"""Rejilla del sondeo. Basta para medir qué fracción del recuadro cae dentro de la franja."""


def cobertura_util(
    items: list["Item"],
    bbox: Bbox,
    muestras: int = MUESTRAS_DE_ORBITA,
    leer=None,
) -> float:
    """Fracción media de píxeles observados que unas escenas dejan sobre el recuadro.

    La huella que declara el catálogo no responde esta pregunta. Sobre Mexicali, la órbita
    cuya huella cubre el 99% del recuadro entrega escenas que alternan entre 1% y 99% de
    píxeles con dato, porque la ciudad cae en el filo de la franja. Se sondean unas pocas
    escenas a baja resolución y se promedia lo que de verdad llega.

    Una lectura que falle cuenta como cero, que es como la vería el compuesto.

    `leer` se resuelve al llamar y no al definir, para que sustituir `leer_ventana` en el
    módulo baste para dejar las pruebas sin red.
    """
    leer = leer or leer_ventana
    fracciones = []
    for item in items[:muestras]:
        try:
            leida = leer(item.assets["vv"].href, bbox, FORMA_SONDA).astype("float32")
        except Exception:
            log.warning("sondeo fallido en %s", item.id, exc_info=True)
            fracciones.append(0.0)
            continue
        fracciones.append(float(np.isfinite(leida).mean()))
    return float(np.mean(fracciones)) if fracciones else 0.0


def orbita_util(
    items: list["Item"],
    bbox: Bbox,
    muestras: int = MUESTRAS_DE_ORBITA,
    leer=None,
) -> tuple[tuple[str, int], list["Item"], float]:
    """Geometría de adquisición que más píxeles observados deja sobre el recuadro.

    Manda la cobertura medida y el número de escenas solo desempata, redondeando a
    centésimas para que una diferencia de nada no tire una órbita con muchas más pasadas.
    """
    grupos = agrupar_por_orbita(items)
    if not grupos:
        raise ValueError("no hay escenas SAR que agrupar")
    cobertura = {k: cobertura_util(v, bbox, muestras, leer) for k, v in grupos.items()}
    clave = max(grupos, key=lambda k: (round(cobertura[k], 2), len(grupos[k])))
    return clave, grupos[clave], cobertura[clave]


def compuesto_s1(
    items: list["Item"],
    bbox: Bbox,
    forma: tuple[int, int] | None = None,
    max_escenas: int = 24,
    fraccion_fallos: float | None = FRACCION_FALLOS,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Mediana por píxel de escenas Sentinel-1 RTC de una sola geometría de órbita.

    Devuelve las polarizaciones compuestas en potencia lineal junto con los metadatos
    de la adquisición elegida.

    Aborta si demasiadas lecturas fallan. `fraccion_fallos` en `None` desactiva esa
    comprobación para quien quiera un compuesto parcial a propósito; en cero no tolera ni
    una lectura fallida.
    """
    if not items:
        raise ValueError("no hay escenas Sentinel-1 para componer")

    (estado, relativa), disponibles, cobertura = orbita_util(items, bbox)
    log.info("S1 órbita %s relativa %d: cubre %.0f%%", estado, relativa, 100 * cobertura)
    seleccion = disponibles[:max_escenas]

    pilas: dict[str, list[np.ndarray]] = {"vv": [], "vh": []}
    fallidas = 0
    for item in seleccion:
        # Las dos polarizaciones se leen antes de guardar ninguna: agregarlas dentro del
        # bucle dejaría VV apilado y VH no cuando la segunda lectura falla, y las medianas
        # de una y otra saldrían calculadas sobre conjuntos de escenas distintos.
        try:
            leidas = {
                polarizacion: leer_ventana(item.assets[polarizacion].href, bbox, forma).astype(
                    "float32"
                )
                for polarizacion in pilas
            }
        except Exception:
            fallidas += 1
            log.warning("escena S1 omitida: %s", item.id, exc_info=True)
            continue
        for polarizacion, arreglo in leidas.items():
            pilas[polarizacion].append(arreglo)

    _revisar_fallos("Sentinel-1", fallidas, len(seleccion), fraccion_fallos)
    if not pilas["vv"]:
        raise RuntimeError("ninguna escena Sentinel-1 aportó píxeles válidos")

    compuesto = {
        polarizacion: np.nanmedian(np.dstack(capas), axis=2)
        for polarizacion, capas in pilas.items()
    }
    valida = _revisar_compuesto_s1(compuesto)
    meta = {
        "fraccion_observada": round(valida, 3),
        "orbita": f"{estado} · relativa {relativa}",
        "escenas_usadas": len(pilas["vv"]),
        "escenas_disponibles": len(disponibles),
        "cobertura_orbita": round(cobertura, 3),
    }
    return compuesto, meta
