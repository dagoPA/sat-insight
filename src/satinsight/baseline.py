"""Baseline de la fase 1 y la comparación que decide si el proyecto sigue.

La puerta de decisión se juega contra la densidad construida. El rezago correlaciona con lo
rural y con lo poco construido, así que un modelo puede acertar leyendo cuánto hay edificado
y quedarse sin haber aprendido nada sobre privación. Ganarle al azar deja esa duda intacta.

De ahí cuatro conjuntos de rasgos, en escalones que responden preguntas distintas:

- `cobertura`: fracciones de WorldCover. Cuánto hay construido según un producto ajeno a
  estos compuestos. Es el escalón que de verdad pone a prueba el atajo por ruralidad.
- `densidad`: estadísticos de primer orden de los compuestos. Cuánto y qué tan brillante.
- `textura`: propiedades de Haralick. Cómo está arreglado, sin el nivel absoluto.
- `completo`: los tres juntos.

Si `completo` no le gana a `cobertura`, el modelo está leyendo densidad de construcción y
nada más. Si no le gana a `densidad`, la textura no aporta sobre el brillo. Conviene saber
ambas cosas antes de montar el MIL encima.

La partición es por ciudad. Entrenar y evaluar sobre AGEB vecinas de la misma mancha urbana
inflaría el resultado por autocorrelación espacial.
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from satinsight.agebs import GRADES
from satinsight.landcover import CLASSES
from satinsight.texture import feature_names

log = logging.getLogger(__name__)

SUFIJOS_DENSIDAD = ("media", "desv", "p10", "p50", "p90", "iqr")
SUFIJOS_TEXTURA = tuple(feature_names())
"""Las columnas de textura llevan propiedad y distancia, por ejemplo `contrast_d2`."""

SUFIJOS_COBERTURA = tuple(CLASSES.values())
"""Fracciones de cobertura de WorldCover, la única fuente ajena a los compuestos."""

CONJUNTOS = {
    "cobertura": SUFIJOS_COBERTURA,
    "densidad": SUFIJOS_DENSIDAD,
    "textura": SUFIJOS_TEXTURA,
    "completo": SUFIJOS_COBERTURA + SUFIJOS_DENSIDAD + SUFIJOS_TEXTURA,
}
"""Los cuatro conjuntos, que responden preguntas distintas y en ese orden.

`cobertura` pregunta si el rezago se explica por cuánto hay construido según un producto
ajeno a estos compuestos. `densidad` pregunta si el brillo de la propia imagen agrega algo
sobre eso. `textura` pregunta si el arreglo espacial agrega algo sobre el brillo. Ganarle a
`cobertura` es lo que descarta el atajo por ruralidad.
"""

SEMILLA = 0


def columnas_de_conjunto(tabla: pd.DataFrame, conjunto: str) -> list[str]:
    """Selecciona las columnas de rasgos que pertenecen a un conjunto.

    El nombre de cada columna es el canal y el sufijo del estadístico unidos por guion
    bajo, así que basta con mirar el sufijo para clasificarla.
    """
    if conjunto not in CONJUNTOS:
        raise KeyError(f"conjunto desconocido: {conjunto!r}. Válidos: {', '.join(CONJUNTOS)}")
    sufijos = CONJUNTOS[conjunto]
    return sorted(c for c in tabla.columns if any(c.endswith(f"_{s}") for s in sufijos))


def varianza_explicada(tabla: pd.DataFrame, columna: str, factor: str) -> float:
    """Fracción de la varianza de un rasgo que explica un factor categórico.

    Comparar cuánto explica la ciudad contra cuánto explica el grado dice si un rasgo sirve
    para transferir. Un rasgo cuya varianza depende sobre todo de en qué ciudad se midió
    enseña al modelo a reconocer la ciudad, y ese conocimiento no vale nada en la ciudad
    que se dejó fuera.
    """
    validos = tabla[columna].notna() & tabla[factor].notna()
    valores, grupos = tabla.loc[validos, columna], tabla.loc[validos, factor]
    if len(valores) < 2 or valores.nunique() < 2:
        return np.nan
    media = valores.mean()
    entre = sum(len(g) * (g.mean() - media) ** 2 for _, g in valores.groupby(grupos))
    total = ((valores - media) ** 2).sum()
    return float(entre / total) if total > 0 else np.nan


def diagnostico_transferencia(
    tabla: pd.DataFrame,
    conjunto: str,
    *,
    columna_grupo: str = "ciudad",
    columna_objetivo: str = "grado",
) -> pd.DataFrame:
    """Para cada rasgo, cuánta varianza explica la ciudad frente al grado.

    La razón entre ambas es la que importa. Por encima de uno, el rasgo describe mejor
    dónde se tomó la medición que qué se midió.

    Se lee junto con `split_half_reliability` y nunca sola. Un rasgo que es puro ruido sale
    con razón baja —el ruido no correlaciona con la ciudad ni con nada—, así que una razón
    buena solo significa algo en un rasgo que ya demostró reproducirse consigo mismo.
    """
    filas = []
    for columna in columnas_de_conjunto(tabla, conjunto):
        por_ciudad = varianza_explicada(tabla, columna, columna_grupo)
        por_grado = varianza_explicada(tabla, columna, columna_objetivo)
        filas.append(
            {
                "feature": columna,
                "por_ciudad": por_ciudad,
                "por_grado": por_grado,
                "razon": por_ciudad / por_grado if por_grado and por_grado > 0 else np.nan,
            }
        )
    return pd.DataFrame(filas).sort_values("razon", ascending=False).reset_index(drop=True)


def estandarizar_por_grupo(
    tabla: pd.DataFrame, columnas: list[str], columna_grupo: str = "ciudad"
) -> pd.DataFrame:
    """Lleva cada rasgo a media cero y desviación uno dentro de cada ciudad.

    Es adaptación de dominio sin supervisión: usa la distribución de los rasgos de la ciudad
    retenida, nunca sus etiquetas, así que no filtra información del objetivo. Lo que
    elimina es la deriva radiométrica y morfológica entre cities.

    El precio es real y hay que declararlo: también borra cualquier diferencia de nivel
    entre cities que sí fuera señal de rezago. Una ciudad entera más pobre que otra queda
    centrada igual que la rica. Por eso se evalúa como ablación declarada.

    Un rasgo constante dentro de una ciudad queda centrado en cero. La distinción importa: varias
    clases de cobertura valen cero en todas las AGEB —nieve,
    musgo, manglar tierra adentro— y convertirlas en columnas enteramente nulas rompe el
    binning del modelo. Los nulos que sí son ausencia de dato, como la textura de una AGEB
    demasiado pequeña, se conservan para que el modelo los trate como faltantes.
    """
    salida = tabla.copy()
    valores = salida[columnas]
    agrupado = salida.groupby(columna_grupo, observed=True)[columnas]
    media = agrupado.transform("mean")
    scale = agrupado.transform("std").where(lambda d: d > 0)

    centrada = (valores - media) / scale
    salida[columnas] = centrada.mask(scale.isna() & valores.notna(), 0.0)
    return salida


def _metricas(verdad: np.ndarray, prediccion: np.ndarray) -> dict[str, float]:
    """Métricas que respetan el orden de las cinco clases."""
    correlacion = spearmanr(verdad, prediccion).statistic if len(set(prediccion)) > 1 else np.nan
    return {
        "kappa": float(cohen_kappa_score(verdad, prediccion, weights="quadratic")),
        "exactitud": float(np.mean(verdad == prediccion)),
        "f1_macro": float(f1_score(verdad, prediccion, average="macro", zero_division=0)),
        "spearman": float(correlacion),
        "mae_ordinal": float(np.mean(np.abs(verdad - prediccion))),
    }


def _predecir(nombre: str, x_entrena, y_entrena, x_prueba) -> np.ndarray:
    """Ajusta uno de los modelos comparados y devuelve predicciones ordinales enteras."""
    if nombre == "azar":
        modelo = DummyClassifier(strategy="stratified", random_state=SEMILLA)
    elif nombre == "moda":
        modelo = DummyClassifier(strategy="most_frequent")
    elif nombre == "clasificador":
        modelo = HistGradientBoostingClassifier(random_state=SEMILLA, max_iter=300)
    elif nombre == "regresor":
        modelo = HistGradientBoostingRegressor(random_state=SEMILLA, max_iter=300)
    else:
        raise KeyError(f"modelo desconocido: {nombre!r}")

    modelo.fit(x_entrena, y_entrena)
    crudo = modelo.predict(x_prueba)
    if nombre == "regresor":
        crudo = np.clip(np.round(crudo), 0, len(GRADES) - 1)
    return crudo.astype(int)


def evaluar(
    tabla: pd.DataFrame,
    conjunto: str,
    modelo: str,
    *,
    columna_grupo: str = "ciudad",
    columna_objetivo: str = "ordinal",
    estandarizar: bool = False,
) -> pd.DataFrame:
    """Validación cruzada dejando una ciudad fuera en cada pliegue.

    Devuelve un renglón por pliegue, para poder ver si el resultado se sostiene en las
    cities o lo carga una sola.

    Con `estandarizar` cada rasgo se centra dentro de su ciudad antes de entrenar. Es una
    ablación y no el modo normal: quita la deriva radiométrica entre cities, y de paso
    cualquier diferencia de nivel entre ellas que sí fuera señal de rezago.
    """
    columnas = columnas_de_conjunto(tabla, conjunto)
    if not columnas:
        raise ValueError(f"la tabla no tiene columnas del conjunto {conjunto!r}")

    utilizable = tabla.dropna(subset=[columna_objetivo]).copy()
    if estandarizar:
        utilizable = estandarizar_por_grupo(utilizable, columnas, columna_grupo)
    renglones = []

    for ciudad in sorted(utilizable[columna_grupo].unique()):
        prueba = utilizable[utilizable[columna_grupo] == ciudad]
        entrena = utilizable[utilizable[columna_grupo] != ciudad]
        if entrena.empty or prueba.empty:
            continue

        y_entrena = entrena[columna_objetivo].astype(int).to_numpy()
        y_prueba = prueba[columna_objetivo].astype(int).to_numpy()
        prediccion = _predecir(
            modelo,
            entrena[columnas].to_numpy("float64"),
            y_entrena,
            prueba[columnas].to_numpy("float64"),
        )

        renglones.append(
            {
                "conjunto": conjunto,
                "modelo": modelo,
                "ciudad_prueba": ciudad,
                "n_entrena": len(entrena),
                "n_prueba": len(prueba),
                "n_rasgos": len(columnas),
                **_metricas(y_prueba, prediccion),
            }
        )

    return pd.DataFrame(renglones)


def comparar(
    tabla: pd.DataFrame,
    conjuntos: tuple[str, ...] = tuple(CONJUNTOS),
    modelos: tuple[str, ...] = ("azar", "moda", "clasificador", "regresor"),
    *,
    estandarizar: bool = False,
) -> pd.DataFrame:
    """Corre la rejilla completa de conjuntos de rasgos por modelos."""
    partes = []
    for conjunto in conjuntos:
        for modelo in modelos:
            if modelo in ("azar", "moda"):
                if conjunto != conjuntos[0]:
                    continue  # ignoran los rasgos; correrlos una vez alcanza
                ciego = evaluar(tabla, conjunto, modelo, estandarizar=estandarizar)
                ciego["conjunto"] = "ninguno"
                partes.append(ciego)
            else:
                partes.append(evaluar(tabla, conjunto, modelo, estandarizar=estandarizar))
    return pd.concat(partes, ignore_index=True)


def resumen(detalle: pd.DataFrame) -> pd.DataFrame:
    """Promedia los pliegues y ordena por kappa, que es la métrica que decide."""
    columnas = ["kappa", "exactitud", "f1_macro", "spearman", "mae_ordinal"]
    agregado = (
        detalle.groupby(["conjunto", "modelo"], observed=True)[columnas]
        .mean()
        .round(3)
        .sort_values("kappa", ascending=False)
    )
    return agregado.reset_index()


UMBRAL_FIABILIDAD = 0.60
"""Correlación mínima entre las dos mitades de un polígono para conservar un rasgo.

Un rasgo que no se reproduce consigo mismo al partir la AGEB en dos mide ruido de muestreo.
La prueba usa la mitad de los píxeles de cada lado, así que subestima la fiabilidad del
polígono entero: el umbral es una cota conservadora.
"""

UMBRAL_TAMANO = 0.30
"""Correlación máxima admitida entre un rasgo y el tamaño del polígono que lo produjo.

El sesgo por submuestreo de la GLCM crece con el número de píxeles, y el tamaño de una AGEB
correlaciona con la densidad urbana, que a su vez correlaciona con el rezago. Un rasgo muy
atado al área apunta al objetivo por construcción.
"""


def seleccionar_rasgos(
    tabla: pd.DataFrame,
    fiabilidad: pd.DataFrame,
    *,
    umbral_fiabilidad: float = UMBRAL_FIABILIDAD,
    umbral_tamano: float = UMBRAL_TAMANO,
) -> pd.DataFrame:
    """Decide qué rasgos de textura entran al modelo, con dos criterios independientes.

    Los dos se aplican juntos porque cada uno solo es interpretable con el otro. Un rasgo
    que es puro ruido pasa el criterio de tamaño con holgura, porque el ruido no correlaciona
    con nada; y un rasgo muy reproducible puede estar midiendo el área del polígono. Exigir
    ambos deja los que se reproducen y además no apuntan al tamaño.

    Los umbrales se fijan antes de mirar desempeño, que es lo que evita elegir el corte que
    conviene al resultado.

    `fiabilidad` viene de `textura.split_half_reliability` agregada sobre las cities, con
    columnas `rasgo` y `r_median`.
    """
    reproduce = dict(zip(fiabilidad["feature"], fiabilidad["r_median"], strict=True))
    filas = []
    for columna in columnas_de_conjunto(tabla, "textura"):
        canal = columna.rsplit("_", 2)[0]
        columna_px = f"{canal}_n_px"
        r_tamano = np.nan
        if columna_px in tabla:
            validos = tabla[columna].notna() & tabla[columna_px].notna()
            if validos.sum() > 2 and tabla.loc[validos, columna].nunique() > 1:
                r_tamano = float(
                    np.corrcoef(
                        tabla.loc[validos, columna],
                        np.log10(tabla.loc[validos, columna_px].clip(lower=1)),
                    )[0, 1]
                )

        r_mitades = reproduce.get(columna, np.nan)
        pasa_fiabilidad = bool(r_mitades >= umbral_fiabilidad) if r_mitades == r_mitades else False
        pasa_tamano = bool(abs(r_tamano) <= umbral_tamano) if r_tamano == r_tamano else False
        motivo = "kept"
        if not pasa_fiabilidad:
            motivo = "no se reproduce"
        elif not pasa_tamano:
            motivo = "atado al tamaño"
        filas.append(
            {
                "feature": columna,
                "r_mitades": r_mitades,
                "r_n_px": r_tamano,
                "kept": pasa_fiabilidad and pasa_tamano,
                "reason": motivo,
            }
        )
    return pd.DataFrame(filas)


def prueba_de_signos(diferencias: np.ndarray) -> dict[str, float]:
    """Prueba exacta de signos sobre las diferencias por pliegue.

    Con cinco cities hay cinco observaciones pareadas, y las AGEB dentro de una ciudad
    están correlacionadas espacialmente entre sí. Tratar cada AGEB como independiente
    inflaría la significancia; la unidad independiente es la ciudad.

    Cinco pliegues dan 32 asignaciones de signo posibles. Con los cinco a favor, el valor p a
    dos colas es 2/32 = 0.0625, que es el mínimo alcanzable con este tamaño de muestra: la
    prueba **nunca** puede bajar de 0.05 con cinco cities. Eso es una propiedad del diseño
    y conviene reportarla junto al resultado, porque invita a leer el tamaño del efecto y su
    intervalo antes que el valor p, y a sumar cities si se quiere evidencia concluyente.
    """
    diferencias = np.asarray(diferencias, dtype="float64")
    diferencias = diferencias[diferencias != 0]
    n = len(diferencias)
    if n == 0:
        return {"n": 0, "a_favor": 0, "p": np.nan}

    from math import comb

    favor = int((diferencias > 0).sum())
    extremo = max(favor, n - favor)
    cola = sum(comb(n, k) for k in range(extremo, n + 1))
    return {"n": n, "a_favor": favor, "p": min(1.0, 2 * cola / 2**n)}


def intervalo_por_ciudades(
    por_ciudad: pd.DataFrame,
    columna: str,
    *,
    repeticiones: int = 10000,
    semilla: int = SEMILLA,
) -> dict[str, float]:
    """Intervalo de confianza de una diferencia, remuestreando cities enteras.

    El bootstrap por conglomerados respeta que la unidad independiente es la ciudad. Con
    cinco conglomerados el intervalo sale ancho, que es la respuesta honesta al tamaño de
    muestra y no un defecto del método.
    """
    valores = por_ciudad[columna].to_numpy(dtype="float64")
    rng = np.random.default_rng(semilla)
    samples = valores[rng.integers(0, len(valores), size=(repeticiones, len(valores)))]
    medias = samples.mean(axis=1)
    return {
        "media": float(valores.mean()),
        "ic_bajo": float(np.percentile(medias, 2.5)),
        "ic_alto": float(np.percentile(medias, 97.5)),
        "fraccion_positiva": float((medias > 0).mean()),
    }


def _puntuaciones(nombre: str, x_entrena, y_entrena, x_prueba):
    """Ajusta un modelo y devuelve sus predicciones duras junto con puntuaciones continuas.

    Kappa necesita la clase predicha y el área bajo la curva necesita una puntuación que
    ordene. Un clasificador da una probabilidad por clase; un regresor da un solo número,
    que ordena igual de bien y es lo que hace falta para los umbrales acumulados.
    """
    if nombre == "clasificador":
        modelo = HistGradientBoostingClassifier(random_state=SEMILLA, max_iter=300)
        modelo.fit(x_entrena, y_entrena)
        probabilidades = modelo.predict_proba(x_prueba)
        return modelo.predict(x_prueba).astype(int), probabilidades
    if nombre == "regresor":
        modelo = HistGradientBoostingRegressor(random_state=SEMILLA, max_iter=300)
        modelo.fit(x_entrena, y_entrena)
        crudo = modelo.predict(x_prueba)
        duras = np.clip(np.round(crudo), 0, len(GRADES) - 1).astype(int)
        return duras, crudo
    duras = _predecir(nombre, x_entrena, y_entrena, x_prueba)
    return duras, duras.astype(float)


def auroc_una_contra_resto(verdad: np.ndarray, puntuaciones: np.ndarray) -> dict[str, float]:
    """Área bajo la curva de cada clase contra todas las demás, y su promedio.

    Se lee directo: 0.5 es azar y 1 es separación perfecta. Con un regresor, la puntuación
    es su salida continua, así que la curva de la clase k mide qué tan arriba quedan sus
    AGEB en ese único orden.

    Las clases de en medio salen castigadas por construcción: sus negativos incluyen a la
    vez lo que está por debajo y lo que está por encima, y separarlas exige recortar una
    banda en el centro de una scale ordenada.
    """
    salida = {}
    for k, grado in enumerate(GRADES):
        objetivo = (verdad == k).astype(int)
        if objetivo.sum() == 0 or objetivo.sum() == len(objetivo):
            continue
        # con una sola puntuación ordenada, la evidencia a favor de la clase k es la
        # cercanía a k. Usar el orden crudo invierte las clases bajas —para ellas una
        # puntuación alta significa lo contrario— y el promedio de las cinco sale en 0.5
        # por cancelación, aparentando azar donde el modelo separa bien
        marca = puntuaciones[:, k] if puntuaciones.ndim == 2 else -np.abs(puntuaciones - k)
        salida[f"auroc_{grado.lower().replace(' ', '_')}"] = float(roc_auc_score(objetivo, marca))
    if salida:
        salida["auroc_macro"] = float(np.mean(list(salida.values())))
    return salida


def auroc_acumulada(verdad: np.ndarray, puntuaciones: np.ndarray) -> dict[str, float]:
    """Área bajo la curva de cada umbral «grado mayor o igual que k».

    Preserva el orden de la scale, cosa que una contra el resto no hace, y es la misma
    descomposición que usa una pérdida ordinal.
    """
    orden = (
        puntuaciones @ np.arange(puntuaciones.shape[1]) if puntuaciones.ndim == 2 else puntuaciones
    )
    salida = {}
    for k in range(1, len(GRADES)):
        objetivo = (verdad >= k).astype(int)
        if objetivo.sum() == 0 or objetivo.sum() == len(objetivo):
            continue
        salida[f"auroc_ge_{k}"] = float(roc_auc_score(objetivo, orden))
    if salida:
        salida["auroc_acumulada_media"] = float(np.mean(list(salida.values())))
    return salida


def evaluar_particion(
    tabla: pd.DataFrame,
    conjunto: str,
    modelo: str,
    particion: pd.DataFrame,
    *,
    evaluar_en: str = "test",
    columna_grupo: str = "ciudad",
    columna_objetivo: str = "ordinal",
    remuestreos: int = 400,
) -> dict[str, float]:
    """Entrena sobre las cities de entrenamiento y mide sobre las de validación o prueba.

    Reemplaza a dejar una ciudad fuera por pliegue, que servía con cinco cities y con
    ciento treinta y ocho daría otros tantos pliegues entrenados cada uno con el 99.3% de
    los datos, donde la dispersión entre pliegues es casi toda ruido.

    La incertidumbre sale de un remuestreo por cities dentro del conjunto medido, porque
    las AGEB vecinas están correlacionadas y tratarlas como independientes estrecha los
    intervalos sin motivo.
    """
    from satinsight.splits import ciudades_de

    if evaluar_en not in ("val", "test"):
        raise KeyError(f"se mide sobre 'val' o 'test', no sobre {evaluar_en!r}")
    columnas = columnas_de_conjunto(tabla, conjunto)
    entrena = tabla[tabla[columna_grupo].isin(ciudades_de(particion, "train"))]
    mide = tabla[tabla[columna_grupo].isin(ciudades_de(particion, evaluar_en))]
    if entrena.empty or mide.empty:
        raise ValueError(
            f"la partición deja {len(entrena)} filas de entrenamiento y {len(mide)} de "
            f"{evaluar_en}; ¿coinciden las claves de ciudad?"
        )

    prediccion, puntuaciones = _puntuaciones(
        modelo, entrena[columnas], entrena[columna_objetivo], mide[columnas]
    )
    verdad = mide[columna_objetivo].to_numpy()
    metricas = {
        **_metricas(verdad, prediccion),
        **auroc_una_contra_resto(verdad, puntuaciones),
        **auroc_acumulada(verdad, puntuaciones),
    }

    # el intervalo remuestrea cities enteras y recalcula la métrica en cada réplica:
    # las AGEB vecinas están correlacionadas, y remuestrearlas sueltas daría intervalos
    # estrechos que no sobrevivirían a cambiar de ciudad, que es justo lo que se mide
    cities = mide[columna_grupo].to_numpy()
    azar = np.random.default_rng(SEMILLA)
    unicas = np.unique(cities)
    replicas: dict[str, list[float]] = defaultdict(list)
    for _ in range(remuestreos):
        elegidas = azar.choice(unicas, size=len(unicas), replace=True)
        filas = np.concatenate([np.flatnonzero(cities == c) for c in elegidas])
        v, pr, puntos = verdad[filas], prediccion[filas], puntuaciones[filas]
        if len(set(v)) < 2:
            continue
        replicas["kappa"].append(float(cohen_kappa_score(v, pr, weights="quadratic")))
        replicas["spearman"].append(float(spearmanr(v, pr).statistic))
        # todas las áreas bajo la curva reciben intervalo, incluidas las de cada umbral:
        # son las que más se citan y presentarlas desnudas invita a leer diferencias de
        # centésimas como si significaran algo
        for nombre, valor in {
            **auroc_una_contra_resto(v, puntos),
            **auroc_acumulada(v, puntos),
        }.items():
            replicas[nombre].append(valor)

    intervalos = {}
    for nombre, valores in replicas.items():
        if valores:
            intervalos[f"{nombre}_ic_bajo"] = float(np.percentile(valores, 2.5))
            intervalos[f"{nombre}_ic_alto"] = float(np.percentile(valores, 97.5))
    return {
        **metricas,
        **intervalos,
        "n_entrena": len(entrena),
        "n_mide": len(mide),
        "ciudades_mide": len(unicas),
    }


def fusionar(optico: pd.DataFrame, radar: pd.DataFrame, *, clave: str = "cvegeo") -> pd.DataFrame:
    """Une las tablas de las dos modalidades en una sola, por AGEB.

    Las columnas de contexto —cobertura del suelo, población, ciudad, grado— vienen de la
    misma fuente en ambas y se toman una vez. Las de imagen llevan el sensor en el nombre,
    así que conviven sin chocar.

    Se conservan solo las AGEB presentes en las dos. Comparar la fusión contra cada
    modalidad por separado sobre samples distintas mezclaría la diferencia de sensor con
    la de qué filas evalúa cada uno.
    """
    solo_radar = [c for c in radar.columns if c.startswith("s1")]
    faltantes = set(optico[clave]) ^ set(radar[clave])
    if faltantes:
        log.info("%d AGEB quedan fuera por faltar en una de las dos modalidades", len(faltantes))
    return optico.merge(radar[[clave, *solo_radar]], on=clave, how="inner")
