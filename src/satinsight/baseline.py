"""Phase one baseline and the comparison that decides whether the project goes on.

The decision gate is played against built density. Deprivation correlates with the rural
and with the sparsely built, so a model can get it right by reading how much is built and
come away having learned nothing about deprivation. Beating chance leaves that doubt
untouched.

Hence four feature sets, in steps that answer different questions:

- `cobertura`: WorldCover fractions. How much is built according to a product foreign to
  these composites. It is the step that really tests the rurality shortcut.
- `densidad`: first order statistics of the composites. How much and how bright.
- `textura`: Haralick properties. How it is arranged, without the absolute level.
- `completo`: the three together.

If `completo` does not beat `cobertura`, the model is reading built density and nothing
else. If it does not beat `densidad`, texture adds nothing over brightness. Both are worth
knowing before mounting the MIL on top.

The partition is by city. Training and evaluating over neighbouring AGEB of the same urban
mass would inflate the result through spatial autocorrelation.
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

DENSITY_SUFFIXES = ("mean", "std", "p10", "p50", "p90", "iqr")
TEXTURE_SUFFIXES = tuple(feature_names())
"""Texture columns carry property and distance, for example `contrast_d2`."""

COVER_SUFFIXES = tuple(CLASSES.values())
"""WorldCover cover fractions, the only source foreign to the composites."""

SETS = {
    "cobertura": COVER_SUFFIXES,
    "densidad": DENSITY_SUFFIXES,
    "textura": TEXTURE_SUFFIXES,
    "completo": COVER_SUFFIXES + DENSITY_SUFFIXES + TEXTURE_SUFFIXES,
}
"""The four sets, which answer different questions and in that order.

`cobertura` asks whether deprivation is explained by how much is built according to a
product foreign to these composites. `densidad` asks whether the brightness of the image
itself adds anything over that. `textura` asks whether the spatial arrangement adds
anything over brightness. Beating `cobertura` is what rules out the rurality shortcut.

The keys keep their Spanish because they name columns already written to disk and quoted
in the results tables; renaming them would orphan every measurement taken so far.
"""

SEED = 0


def columns_of_set(table: pd.DataFrame, split: str) -> list[str]:
    """Selecciona las columns de rasgos que pertenecen a un split.

    Each column name is the channel and the statistic suffix joined by an underscore, so
    looking at the suffix is enough to classify it.
    """
    if split not in SETS:
        raise KeyError(f"unknown set: {split!r}. Valid: {', '.join(SETS)}")
    sufijos = SETS[split]
    return sorted(c for c in table.columns if any(c.endswith(f"_{s}") for s in sufijos))


def explained_variance(table: pd.DataFrame, columna: str, factor: str) -> float:
    """Fraction of a feature's variance explained by a categorical factor.

    Comparing how much the city explains against how much the grade explains says whether a
    feature transfers. A feature whose variance depends above all on which city it was
    measured in teaches the model to recognise the city, and that knowledge is worth nothing
    in the city held out.
    """
    valid = table[columna].notna() & table[factor].notna()
    values, grupos = table.loc[valid, columna], table.loc[valid, factor]
    if len(values) < 2 or values.nunique() < 2:
        return np.nan
    mean = values.mean()
    entre = sum(len(g) * (g.mean() - mean) ** 2 for _, g in values.groupby(grupos))
    total = ((values - mean) ** 2).sum()
    return float(entre / total) if total > 0 else np.nan


def transfer_diagnostics(
    table: pd.DataFrame,
    split: str,
    *,
    group_column: str = "ciudad",
    target_column: str = "grado",
) -> pd.DataFrame:
    """For each feature, how much variance the city explains against the grade.

    The ratio between the two is what matters. Above one, the feature describes where the
    measurement was taken better than what was measured.

    Se lee junto con `split_half_reliability` y nunca sola. Un rasgo que es puro ruido sale
    with a low ratio —noise correlates with the city no more than with anything else— so a
    good ratio only means something in a feature that already proved it reproduces.
    """
    rows = []
    for columna in columns_of_set(table, split):
        per_city = explained_variance(table, columna, group_column)
        por_grado = explained_variance(table, columna, target_column)
        rows.append(
            {
                "feature": columna,
                "per_city": per_city,
                "por_grado": por_grado,
                "ratio": per_city / por_grado if por_grado and por_grado > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)


def standardise_by_group(
    table: pd.DataFrame, columns: list[str], group_column: str = "ciudad"
) -> pd.DataFrame:
    """Takes each feature to zero mean and unit deviation within each city.

    This is unsupervised domain adaptation: it uses the distribution of the held-out city's
    features, never its labels, so it leaks no information about the target. What it removes
    is the radiometric and morphological drift between cities.

    The price is real and has to be declared: it also erases any level difference between
    cities that was in fact a signal of deprivation. A whole city poorer than another ends
    up centred just like the rich one. That is why it is evaluated as a declared ablation.

    A feature constant within a city ends up centred at zero. The distinction matters: several
    clases de cobertura valen cero en todas las AGEB —nieve,
    musgo, manglar tierra adentro— y convertirlas en columns enteramente nulas rompe el
    model's binning. The nulls that really are absent data, such as the texture of an AGEB
    too small, are kept so the model treats them as missing.
    """
    output = table.copy()
    values = output[columns]
    grouped = output.groupby(group_column, observed=True)[columns]
    mean = grouped.transform("mean")
    scale = grouped.transform("std").where(lambda d: d > 0)

    centred = (values - mean) / scale
    output[columns] = centred.mask(scale.isna() & values.notna(), 0.0)
    return output


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Metrics that respect the order of the five classes."""
    correlation = spearmanr(truth, prediction).statistic if len(set(prediction)) > 1 else np.nan
    return {
        "kappa": float(cohen_kappa_score(truth, prediction, weights="quadratic")),
        "exactitud": float(np.mean(truth == prediction)),
        "f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "spearman": float(correlation),
        "mae_ordinal": float(np.mean(np.abs(truth - prediction))),
    }


def _predict(name: str, x_entrena, y_entrena, x_prueba) -> np.ndarray:
    """Ajusta uno de los modelos comparados y devuelve predicciones ordinales enteras."""
    if name == "rng":
        modelo = DummyClassifier(strategy="stratified", random_state=SEED)
    elif name == "moda":
        modelo = DummyClassifier(strategy="most_frequent")
    elif name == "clasificador":
        modelo = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
    elif name == "regresor":
        modelo = HistGradientBoostingRegressor(random_state=SEED, max_iter=300)
    else:
        raise KeyError(f"modelo desconocido: {name!r}")

    modelo.fit(x_entrena, y_entrena)
    raw = modelo.predict(x_prueba)
    if name == "regresor":
        raw = np.clip(np.round(raw), 0, len(GRADES) - 1)
    return raw.astype(int)


def evaluate(
    table: pd.DataFrame,
    split: str,
    modelo: str,
    *,
    group_column: str = "ciudad",
    target_column: str = "ordinal",
    estandarizar: bool = False,
) -> pd.DataFrame:
    """Cross-validation leaving one city out on each fold.

    Returns one row per fold, so it can be seen whether the result holds across the
    cities o lo carga una sola.

    Con `estandarizar` cada rasgo se centra dentro de su ciudad antes de entrenar. Es una
    ablation and not the normal mode: it removes the radiometric drift between cities, and
    with it any level difference between them that was in fact a signal of deprivation.
    """
    columns = columns_of_set(table, split)
    if not columns:
        raise ValueError(f"la table no tiene columns del split {split!r}")

    utilizable = table.dropna(subset=[target_column]).copy()
    if estandarizar:
        utilizable = standardise_by_group(utilizable, columns, group_column)
    rows_out = []

    for ciudad in sorted(utilizable[group_column].unique()):
        prueba = utilizable[utilizable[group_column] == ciudad]
        train = utilizable[utilizable[group_column] != ciudad]
        if train.empty or prueba.empty:
            continue

        y_entrena = train[target_column].astype(int).to_numpy()
        y_prueba = prueba[target_column].astype(int).to_numpy()
        prediction = _predict(
            modelo,
            train[columns].to_numpy("float64"),
            y_entrena,
            prueba[columns].to_numpy("float64"),
        )

        rows_out.append(
            {
                "split": split,
                "modelo": modelo,
                "ciudad_prueba": ciudad,
                "n_entrena": len(train),
                "n_prueba": len(prueba),
                "n_rasgos": len(columns),
                **_metrics(y_prueba, prediction),
            }
        )

    return pd.DataFrame(rows_out)


def compare(
    table: pd.DataFrame,
    conjuntos: tuple[str, ...] = tuple(SETS),
    modelos: tuple[str, ...] = ("rng", "moda", "clasificador", "regresor"),
    *,
    estandarizar: bool = False,
) -> pd.DataFrame:
    """Corre la rejilla completa de conjuntos de rasgos por modelos."""
    partes = []
    for split in conjuntos:
        for modelo in modelos:
            if modelo in ("rng", "moda"):
                if split != conjuntos[0]:
                    continue  # ignoran los rasgos; correrlos una vez alcanza
                ciego = evaluate(table, split, modelo, estandarizar=estandarizar)
                ciego["split"] = "ninguno"
                partes.append(ciego)
            else:
                partes.append(evaluate(table, split, modelo, estandarizar=estandarizar))
    return pd.concat(partes, ignore_index=True)


def fold_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Averages the folds and sorts by kappa, which is the deciding metric."""
    columns = ["kappa", "exactitud", "f1_macro", "spearman", "mae_ordinal"]
    agregado = (
        detail.groupby(["split", "modelo"], observed=True)[columns]
        .mean()
        .round(3)
        .sort_values("kappa", ascending=False)
    )
    return agregado.reset_index()


RELIABILITY_THRESHOLD = 0.60
"""Minimum correlation between the two halves of a polygon for a feature to be kept.

Un rasgo que no se reproduce consigo mismo al partir la AGEB en dos measured ruido de muestreo.
The test uses half the pixels on each side, so it underestimates the reliability of the
whole polygon: the threshold is a conservative bound.
"""

SIZE_THRESHOLD = 0.30
"""Maximum correlation admitted between a feature and the size of the polygon behind it.

The GLCM undersampling bias grows with the number of pixels, and the size of an AGEB
correlaciona con la densidad urbana, que a su vez correlaciona con el rezago. Un rasgo muy
tied to area points at the target by construction.
"""


def select_features(
    table: pd.DataFrame,
    reliability: pd.DataFrame,
    *,
    reliability_threshold: float = RELIABILITY_THRESHOLD,
    size_threshold: float = SIZE_THRESHOLD,
) -> pd.DataFrame:
    """Decides which texture features enter the model, on two independent criteria.

    Los dos se aplican juntos porque cada uno solo es interpretable con el otro. Un rasgo
    that is pure noise passes the size criterion with room to spare, because noise
    correlates with nothing; and a very reproducible feature may be measuring the area of
    the polygon. Demanding both leaves those that reproduce and do not point at size.

    The thresholds are set before looking at performance, which is what avoids picking the
    conviene al resultado.

    `reliability` viene de `textura.split_half_reliability` agregada sobre las cities, con
    columns `rasgo` y `r_median`.
    """
    reproduce = dict(zip(reliability["feature"], reliability["r_median"], strict=True))
    rows = []
    for columna in columns_of_set(table, "textura"):
        canal = columna.rsplit("_", 2)[0]
        px_column = f"{canal}_n_px"
        r_size = np.nan
        if px_column in table:
            valid = table[columna].notna() & table[px_column].notna()
            if valid.sum() > 2 and table.loc[valid, columna].nunique() > 1:
                r_size = float(
                    np.corrcoef(
                        table.loc[valid, columna],
                        np.log10(table.loc[valid, px_column].clip(lower=1)),
                    )[0, 1]
                )

        r_mitades = reproduce.get(columna, np.nan)
        pasa_fiabilidad = (
            bool(r_mitades >= reliability_threshold) if r_mitades == r_mitades else False
        )
        pasa_tamano = bool(abs(r_size) <= size_threshold) if r_size == r_size else False
        reason = "kept"
        if not pasa_fiabilidad:
            reason = "does not reproduce"
        elif not pasa_tamano:
            reason = "tied to size"
        rows.append(
            {
                "feature": columna,
                "r_mitades": r_mitades,
                "r_n_px": r_size,
                "kept": pasa_fiabilidad and pasa_tamano,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def sign_test(diferencias: np.ndarray) -> dict[str, float]:
    """Prueba exacta de signos sobre las diferencias por fold.

    Con cinco cities hay cinco observaciones pareadas, y las AGEB dentro de una ciudad
    are spatially correlated with each other. Treating every AGEB as independent
    would inflate significance; the independent unit is the city.

    Cinco pliegues dan 32 asignaciones de signo posibles. Con los cinco a favor, el valor p a
    two-tailed is 2/32 = 0.0625, which is the minimum attainable at this sample size: the
    test can **never** go below 0.05 with five cities. That is a property of the design and
    worth reporting beside the result, because it invites reading the effect size and its
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


def city_interval(
    per_city: pd.DataFrame,
    columna: str,
    *,
    repeticiones: int = 10000,
    semilla: int = SEED,
) -> dict[str, float]:
    """Intervalo de confianza de una diferencia, remuestreando cities enteras.

    El bootstrap por conglomerados respeta que la unidad independiente es la ciudad. Con
    five clusters the interval comes out wide, which is the honest answer to the sample
    size and not a defect of the method.
    """
    values = per_city[columna].to_numpy(dtype="float64")
    rng = np.random.default_rng(semilla)
    samples = values[rng.integers(0, len(values), size=(repeticiones, len(values)))]
    medias = samples.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ic_bajo": float(np.percentile(medias, 2.5)),
        "ic_alto": float(np.percentile(medias, 97.5)),
        "fraccion_positiva": float((medias > 0).mean()),
    }


def _scores(name: str, x_entrena, y_entrena, x_prueba):
    """Ajusta un modelo y devuelve sus predicciones hard junto con scores continuas.

    Kappa needs the predicted class and the area under the curve needs a score that orders.
    A classifier gives one probability per class; a regressor gives a single number,
    que ordena igual de bien y es lo que hace falta para los umbrales acumulados.
    """
    if name == "clasificador":
        modelo = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
        modelo.fit(x_entrena, y_entrena)
        probabilities = modelo.predict_proba(x_prueba)
        return modelo.predict(x_prueba).astype(int), probabilities
    if name == "regresor":
        modelo = HistGradientBoostingRegressor(random_state=SEED, max_iter=300)
        modelo.fit(x_entrena, y_entrena)
        raw = modelo.predict(x_prueba)
        hard = np.clip(np.round(raw), 0, len(GRADES) - 1).astype(int)
        return hard, raw
    hard = _predict(name, x_entrena, y_entrena, x_prueba)
    return hard, hard.astype(float)


def auroc_one_vs_rest(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Area under the curve of each class against all the others, and their average.

    It reads directly: 0.5 is chance and 1 is perfect separation. With a regressor the score
    is its continuous output, so the curve of class k measures how high its AGEB land in
    that single ordering.

    The middle classes are punished by construction: their negatives include at once what
    lies below and what lies above, and separating them demands carving out a
    banda en el centro de una scale ordenada.
    """
    output = {}
    for k, grado in enumerate(GRADES):
        target = (truth == k).astype(int)
        if target.sum() == 0 or target.sum() == len(target):
            continue
        # with a single ordered score, the evidence for class k is closeness to k. Using the
        # raw order inverts the low classes —for them a high score means the opposite— and
        # the average of the five comes out at 0.5 by cancellation, looking like chance
        # where the model separates well
        mark = scores[:, k] if scores.ndim == 2 else -np.abs(scores - k)
        output[f"auroc_{grado.lower().replace(' ', '_')}"] = float(roc_auc_score(target, mark))
    if output:
        output["auroc_macro"] = float(np.mean(list(output.values())))
    return output


def auroc_cumulative(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Área bajo la curva de cada threshold «grado mayor o igual que k».

        Preserva el order de la scale, cosa que una contra el resto no hace, y es la misma
    decomposition an ordinal loss uses.
    """
    order = scores @ np.arange(scores.shape[1]) if scores.ndim == 2 else scores
    output = {}
    for k in range(1, len(GRADES)):
        target = (truth >= k).astype(int)
        if target.sum() == 0 or target.sum() == len(target):
            continue
        output[f"auroc_ge_{k}"] = float(roc_auc_score(target, order))
    if output:
        output["auroc_acumulada_media"] = float(np.mean(list(output.values())))
    return output


def evaluate_partition(
    table: pd.DataFrame,
    split: str,
    modelo: str,
    partition: pd.DataFrame,
    *,
    measure_on: str = "test",
    group_column: str = "ciudad",
    target_column: str = "ordinal",
    resamples: int = 400,
) -> dict[str, float]:
    """Trains on the training cities and measures on the validation or test ones.

    Replaces leaving one city out per fold, which served with five cities and with a hundred
    and thirty-eight would give as many folds each trained on 99.3% of
    the data, where the spread between folds is almost all noise.

    La incertidumbre sale de un remuestreo por cities dentro del split medido, porque
    neighbouring AGEB are correlated and treating them as independent narrows the
    intervals sin reason.
    """
    from satinsight.splits import cities_of

    if measure_on not in ("val", "test"):
        raise KeyError(f"se measured sobre 'val' o 'test', no sobre {measure_on!r}")
    columns = columns_of_set(table, split)
    train = table[table[group_column].isin(cities_of(partition, "train"))]
    measured = table[table[group_column].isin(cities_of(partition, measure_on))]
    if train.empty or measured.empty:
        raise ValueError(
            f"the partition leaves {len(train)} training rows and {len(measured)} of "
            f"{measure_on}; ¿coinciden las claves de ciudad?"
        )

    prediction, scores = _scores(modelo, train[columns], train[target_column], measured[columns])
    truth = measured[target_column].to_numpy()
    metrics = {
        **_metrics(truth, prediction),
        **auroc_one_vs_rest(truth, scores),
        **auroc_cumulative(truth, scores),
    }

    # the interval resamples whole cities and recomputes the metric on each replicate:
    # neighbouring AGEB are correlated, and resampling them loose would give narrow intervals
    # that would not survive changing city, which is exactly what is being measured
    cities = measured[group_column].to_numpy()
    rng = np.random.default_rng(SEED)
    unique_cities = np.unique(cities)
    replicates: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        chosen = rng.choice(unique_cities, size=len(unique_cities), replace=True)
        rows = np.concatenate([np.flatnonzero(cities == c) for c in chosen])
        v, pr, scores_of = truth[rows], prediction[rows], scores[rows]
        if len(set(v)) < 2:
            continue
        replicates["kappa"].append(float(cohen_kappa_score(v, pr, weights="quadratic")))
        replicates["spearman"].append(float(spearmanr(v, pr).statistic))
        # every area under the curve gets an interval, including the per-threshold ones:
        # they are the most quoted and presenting them bare invites reading differences of
        # hundredths as if they meant something
        for name, valor in {
            **auroc_one_vs_rest(v, scores_of),
            **auroc_cumulative(v, scores_of),
        }.items():
            replicates[name].append(valor)

    intervals = {}
    for name, values in replicates.items():
        if values:
            intervals[f"{name}_ic_bajo"] = float(np.percentile(values, 2.5))
            intervals[f"{name}_ic_alto"] = float(np.percentile(values, 97.5))
    return {
        **metrics,
        **intervals,
        "n_entrena": len(train),
        "n_mide": len(measured),
        "ciudades_mide": len(unique_cities),
    }


def fuse(optical: pd.DataFrame, radar: pd.DataFrame, *, key: str = "cvegeo") -> pd.DataFrame:
    """Une las tablas de las dos modalidades en una sola, por AGEB.

        The context columns —land cover, population, city, grade— come from the
        misma fuente en ambas y se toman una vez. Las de imagen llevan el sensor en el name,
    so they coexist without clashing.

        Only the AGEB present in both are kept. Comparing the fusion against each modality
        separately over different samples would mix the difference of sensor with that of which
        rows each one evaluates.
    """
    radar_only = [c for c in radar.columns if c.startswith("s1")]
    missing = set(optical[key]) ^ set(radar[key])
    if missing:
        log.info("%d AGEB quedan fuera por faltar en una de las dos modalidades", len(missing))
    return optical.merge(radar[[key, *radar_only]], on=key, how="inner")
