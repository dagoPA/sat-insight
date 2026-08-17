"""Baseline de la fase 1 y la comparación que decide si el proyecto sigue.

La puerta de decisión no es ganarle al azar. El rezago correlaciona con lo rural y con lo
poco construido, así que un modelo puede acertar leyendo densidad de construcción y no
haber aprendido nada sobre privación.

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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score, f1_score

from satinsight.agebs import GRADOS
from satinsight.cobertura import CLASES
from satinsight.textura import nombres_de_rasgos

log = logging.getLogger(__name__)

SUFIJOS_DENSIDAD = ("media", "desv", "p10", "p50", "p90", "rango_intercuartil")
SUFIJOS_TEXTURA = tuple(nombres_de_rasgos())
"""Las columnas de textura llevan propiedad y distancia, por ejemplo `contrast_d2`."""

SUFIJOS_COBERTURA = tuple(CLASES.values())
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
sobre eso. `textura` pregunta si el arreglo espacial agrega algo sobre el brillo. Ganarle
al azar no dice nada; ganarle a `cobertura` es lo que descarta el atajo por ruralidad.
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
        crudo = np.clip(np.round(crudo), 0, len(GRADOS) - 1)
    return crudo.astype(int)


def evaluar(
    tabla: pd.DataFrame,
    conjunto: str,
    modelo: str,
    *,
    columna_grupo: str = "ciudad",
    columna_objetivo: str = "ordinal",
) -> pd.DataFrame:
    """Validación cruzada dejando una ciudad fuera en cada pliegue.

    Devuelve un renglón por pliegue, para poder ver si el resultado se sostiene en las tres
    ciudades o lo carga una sola.
    """
    columnas = columnas_de_conjunto(tabla, conjunto)
    if not columnas:
        raise ValueError(f"la tabla no tiene columnas del conjunto {conjunto!r}")

    utilizable = tabla.dropna(subset=[columna_objetivo]).copy()
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
) -> pd.DataFrame:
    """Corre la rejilla completa de conjuntos de rasgos por modelos."""
    partes = []
    for conjunto in conjuntos:
        for modelo in modelos:
            if modelo in ("azar", "moda"):
                if conjunto != conjuntos[0]:
                    continue  # ignoran los rasgos; correrlos una vez alcanza
                ciego = evaluar(tabla, conjunto, modelo)
                ciego["conjunto"] = "ninguno"
                partes.append(ciego)
            else:
                partes.append(evaluar(tabla, conjunto, modelo))
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
