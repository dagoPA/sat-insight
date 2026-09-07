"""Phase one baseline and the comparison that decides whether the project goes on.

The decision gate is played against built density. Deprivation correlates with the rural
and with the sparsely built, so a model can get it right by reading how much is built and
come away having learned nothing about deprivation. Beating chance leaves that doubt
untouched.

Hence four feature sets, in steps that answer different questions:

- `cover`: WorldCover fractions. How much is built according to a product foreign to
  these composites. It is the step that really tests the rurality shortcut.
- `intensity`: first order statistics of the composites. How much and how bright.
- `texture`: Haralick properties. How it is arranged, without the absolute level.
- `all`: the three together.

If `all` does not beat `cover`, the model is reading built density and nothing else. If
it does not beat `intensity`, texture adds nothing over brightness. Both are worth knowing
before mounting the MIL on top.

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
    "cover": COVER_SUFFIXES,
    "intensity": DENSITY_SUFFIXES,
    "texture": TEXTURE_SUFFIXES,
    "all": COVER_SUFFIXES + DENSITY_SUFFIXES + TEXTURE_SUFFIXES,
}
"""The four sets, which answer different questions and in that order.

`cover` asks whether deprivation is explained by how much is built according to a product
foreign to these composites. `intensity` asks whether the brightness of the image itself
adds anything over that. `texture` asks whether the spatial arrangement adds anything over
brightness. Beating `cover` is what rules out the rurality shortcut.
"""

MODELS = ("random", "majority", "classifier", "regressor")
"""The two blind references and the two learners compared on every feature set."""

NO_SET = "none"
"""Set label the blind models report, since they ignore the features."""

SEED = 0


def columns_of_set(table: pd.DataFrame, feature_set: str) -> list[str]:
    """Selects the feature columns that belong to one set.

    Each column name is the channel and the statistic suffix joined by an underscore, so
    looking at the suffix is enough to classify it.
    """
    if feature_set not in SETS:
        raise KeyError(f"unknown set: {feature_set!r}. Valid: {', '.join(SETS)}")
    suffixes = SETS[feature_set]
    return sorted(c for c in table.columns if any(c.endswith(f"_{s}") for s in suffixes))


def explained_variance(table: pd.DataFrame, column: str, factor: str) -> float:
    """Fraction of a feature's variance explained by a categorical factor.

    Comparing how much the city explains against how much the grade explains says whether a
    feature transfers. A feature whose variance depends above all on which city it was
    measured in teaches the model to recognise the city, and that knowledge is worth nothing
    in the city held out.
    """
    valid = table[column].notna() & table[factor].notna()
    values, groups = table.loc[valid, column], table.loc[valid, factor]
    if len(values) < 2 or values.nunique() < 2:
        return np.nan
    mean = values.mean()
    between = sum(len(g) * (g.mean() - mean) ** 2 for _, g in values.groupby(groups))
    total = ((values - mean) ** 2).sum()
    return float(between / total) if total > 0 else np.nan


def transfer_diagnostics(
    table: pd.DataFrame,
    feature_set: str,
    *,
    group_column: str = "city",
    target_column: str = "grade",
) -> pd.DataFrame:
    """For each feature, how much variance the city explains against the grade.

    The ratio between the two is what matters. Above one, the feature describes where the
    measurement was taken better than what was measured.

    It is read beside `split_half_reliability` and never alone. A feature that is pure noise
    comes out with a low ratio, noise correlates with the city no more than with anything
    else, so a good ratio only means something in a feature that already proved it
    reproduces.
    """
    rows = []
    for column in columns_of_set(table, feature_set):
        per_city = explained_variance(table, column, group_column)
        per_grade = explained_variance(table, column, target_column)
        rows.append(
            {
                "feature": column,
                "per_city": per_city,
                "per_grade": per_grade,
                "ratio": per_city / per_grade if per_grade and per_grade > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)


def standardise_by_group(
    table: pd.DataFrame, columns: list[str], group_column: str = "city"
) -> pd.DataFrame:
    """Takes each feature to zero mean and unit deviation within each city.

    This is unsupervised domain adaptation: it uses the distribution of the held-out city's
    features, never its labels, so it leaks no information about the target. What it removes
    is the radiometric and morphological drift between cities.

    The price is real and has to be declared: it also erases any level difference between
    cities that was in fact a signal of deprivation. A whole city poorer than another ends
    up centred just like the rich one. That is why it is evaluated as a declared ablation.

    A feature constant within a city ends up centred at zero. The distinction matters:
    several cover classes are zero in every AGEB, snow, moss, inland mangrove, and turning
    them into entirely null columns breaks the model's binning. The nulls that really are
    absent data, such as the texture of an AGEB too small, are kept so the model treats
    them as missing.
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
        "accuracy": float(np.mean(truth == prediction)),
        "f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "spearman": float(correlation),
        "mae_ordinal": float(np.mean(np.abs(truth - prediction))),
    }


def _predict(name: str, x_train, y_train, x_test) -> np.ndarray:
    """Fits one of the compared models and returns integer ordinal predictions."""
    if name == "random":
        model = DummyClassifier(strategy="stratified", random_state=SEED)
    elif name == "majority":
        model = DummyClassifier(strategy="most_frequent")
    elif name == "classifier":
        model = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
    elif name == "regressor":
        model = HistGradientBoostingRegressor(random_state=SEED, max_iter=300)
    else:
        raise KeyError(f"unknown model: {name!r}. Valid: {', '.join(MODELS)}")

    model.fit(x_train, y_train)
    raw = model.predict(x_test)
    if name == "regressor":
        raw = np.clip(np.round(raw), 0, len(GRADES) - 1)
    return raw.astype(int)


def evaluate(
    table: pd.DataFrame,
    feature_set: str,
    model: str,
    *,
    group_column: str = "city",
    target_column: str = "ordinal",
    standardise: bool = False,
) -> pd.DataFrame:
    """Cross-validation leaving one city out on each fold.

    Returns one row per fold, so it can be seen whether the result holds across the cities
    or a single one carries it.

    With `standardise` each feature is centred within its city before training. It is an
    ablation and not the normal mode: it removes the radiometric drift between cities, and
    with it any level difference between them that was in fact a signal of deprivation.
    """
    columns = columns_of_set(table, feature_set)
    if not columns:
        raise ValueError(f"the table has no columns of the set {feature_set!r}")

    usable = table.dropna(subset=[target_column]).copy()
    if standardise:
        usable = standardise_by_group(usable, columns, group_column)
    rows_out = []

    for city in sorted(usable[group_column].unique()):
        test = usable[usable[group_column] == city]
        train = usable[usable[group_column] != city]
        if train.empty or test.empty:
            continue

        y_train = train[target_column].astype(int).to_numpy()
        y_test = test[target_column].astype(int).to_numpy()
        prediction = _predict(
            model,
            train[columns].to_numpy("float64"),
            y_train,
            test[columns].to_numpy("float64"),
        )

        rows_out.append(
            {
                "set": feature_set,
                "model": model,
                "test_city": city,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(columns),
                **_metrics(y_test, prediction),
            }
        )

    return pd.DataFrame(rows_out)


def compare(
    table: pd.DataFrame,
    sets: tuple[str, ...] = tuple(SETS),
    models: tuple[str, ...] = MODELS,
    *,
    standardise: bool = False,
) -> pd.DataFrame:
    """Runs the full grid of feature sets by models."""
    parts = []
    for feature_set in sets:
        for model in models:
            if model in ("random", "majority"):
                if feature_set != sets[0]:
                    continue  # they ignore the features; running them once is enough
                blind = evaluate(table, feature_set, model, standardise=standardise)
                blind["set"] = NO_SET
                parts.append(blind)
            else:
                parts.append(evaluate(table, feature_set, model, standardise=standardise))
    return pd.concat(parts, ignore_index=True)


def fold_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Averages the folds and sorts by kappa, which is the deciding metric."""
    columns = ["kappa", "accuracy", "f1_macro", "spearman", "mae_ordinal"]
    aggregate = (
        detail.groupby(["set", "model"], observed=True)[columns]
        .mean()
        .round(3)
        .sort_values("kappa", ascending=False)
    )
    return aggregate.reset_index()


RELIABILITY_THRESHOLD = 0.60
"""Minimum correlation between the two halves of a polygon for a feature to be kept.

A feature that does not reproduce itself when the AGEB is split in two is measuring
sampling noise. The test uses half the pixels on each side, so it underestimates the
reliability of the whole polygon: the threshold is a conservative bound.
"""

SIZE_THRESHOLD = 0.30
"""Maximum correlation admitted between a feature and the size of the polygon behind it.

The GLCM undersampling bias grows with the number of pixels, and the size of an AGEB
correlates with urban density, which in turn correlates with deprivation. A feature tightly
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

    The two are applied together because each is only interpretable with the other. A
    feature that is pure noise passes the size criterion with room to spare, because noise
    correlates with nothing; and a very reproducible feature may be measuring the area of
    the polygon. Demanding both leaves those that reproduce and do not point at size.

    The thresholds are set before looking at performance, which is what avoids picking the
    ones that suit the result.

    `reliability` comes from `texture.split_half_reliability` aggregated over the cities,
    with columns `feature` and `r_median`.
    """
    reproduce = dict(zip(reliability["feature"], reliability["r_median"], strict=True))
    rows = []
    for column in columns_of_set(table, "texture"):
        channel = column.rsplit("_", 2)[0]
        px_column = f"{channel}_n_px"
        r_size = np.nan
        if px_column in table:
            valid = table[column].notna() & table[px_column].notna()
            if valid.sum() > 2 and table.loc[valid, column].nunique() > 1:
                r_size = float(
                    np.corrcoef(
                        table.loc[valid, column],
                        np.log10(table.loc[valid, px_column].clip(lower=1)),
                    )[0, 1]
                )

        r_halves = reproduce.get(column, np.nan)
        passes_reliability = (
            bool(r_halves >= reliability_threshold) if r_halves == r_halves else False
        )
        passes_size = bool(abs(r_size) <= size_threshold) if r_size == r_size else False
        reason = "kept"
        if not passes_reliability:
            reason = "does not reproduce"
        elif not passes_size:
            reason = "tied to size"
        rows.append(
            {
                "feature": column,
                "r_halves": r_halves,
                "r_n_px": r_size,
                "kept": passes_reliability and passes_size,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def sign_test(differences: np.ndarray) -> dict[str, float]:
    """Exact sign test over the per-fold differences.

    With five cities there are five paired observations, and the AGEB within a city are
    spatially correlated with each other. Treating every AGEB as independent would inflate
    significance; the independent unit is the city.

    Five folds give 32 possible sign assignments. With all five in favour, the two-tailed p
    value is 2/32 = 0.0625, which is the minimum attainable at this sample size: the test
    can **never** go below 0.05 with five cities. That is a property of the design and worth
    reporting beside the result, because it invites reading the effect size and its interval
    before the p value, and adding cities if conclusive evidence is wanted.
    """
    differences = np.asarray(differences, dtype="float64")
    differences = differences[differences != 0]
    n = len(differences)
    if n == 0:
        return {"n": 0, "in_favour": 0, "p": np.nan}

    from math import comb

    favour = int((differences > 0).sum())
    extreme = max(favour, n - favour)
    tail = sum(comb(n, k) for k in range(extreme, n + 1))
    return {"n": n, "in_favour": favour, "p": min(1.0, 2 * tail / 2**n)}


def city_interval(
    per_city: pd.DataFrame,
    column: str,
    *,
    resamples: int = 10000,
    seed: int = SEED,
) -> dict[str, float]:
    """Confidence interval of a difference, resampling whole cities.

    The clustered bootstrap respects that the independent unit is the city. With five
    clusters the interval comes out wide, which is the honest answer to the sample size and
    not a defect of the method.
    """
    values = per_city[column].to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(resamples, len(values)))]
    means = samples.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "positive_fraction": float((means > 0).mean()),
    }


def _scores(name: str, x_train, y_train, x_test):
    """Fits a model and returns its hard predictions beside continuous scores.

    Kappa needs the predicted class and the area under the curve needs a score that orders.
    A classifier gives one probability per class; a regressor gives a single number, which
    orders just as well and is what the cumulative thresholds need.
    """
    if name == "classifier":
        model = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        return model.predict(x_test).astype(int), probabilities
    if name == "regressor":
        model = HistGradientBoostingRegressor(random_state=SEED, max_iter=300)
        model.fit(x_train, y_train)
        raw = model.predict(x_test)
        hard = np.clip(np.round(raw), 0, len(GRADES) - 1).astype(int)
        return hard, raw
    hard = _predict(name, x_train, y_train, x_test)
    return hard, hard.astype(float)


def auroc_one_vs_rest(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Area under the curve of each class against all the others, and their average.

    It reads directly: 0.5 is chance and 1 is perfect separation. With a regressor the score
    is its continuous output, so the curve of class k measures how high its AGEB land in
    that single ordering.

    The middle classes are punished by construction: their negatives include at once what
    lies below and what lies above, and separating them demands carving out a band in the
    centre of an ordered scale.
    """
    output = {}
    for k, grade in enumerate(GRADES):
        target = (truth == k).astype(int)
        if target.sum() == 0 or target.sum() == len(target):
            continue
        # with a single ordered score, the evidence for class k is closeness to k. Using the
        # raw order inverts the low classes, for them a high score means the opposite, and
        # the average of the five comes out at 0.5 by cancellation, looking like chance
        # where the model separates well
        mark = scores[:, k] if scores.ndim == 2 else -np.abs(scores - k)
        output[f"auroc_{grade.lower().replace(' ', '_')}"] = float(roc_auc_score(target, mark))
    if output:
        output["auroc_macro"] = float(np.mean(list(output.values())))
    return output


def auroc_cumulative(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Area under the curve of each threshold "grade at least k".

    It preserves the order of the scale, which one against the rest does not, and it is the
    same decomposition an ordinal loss uses.
    """
    order = scores @ np.arange(scores.shape[1]) if scores.ndim == 2 else scores
    output = {}
    for k in range(1, len(GRADES)):
        target = (truth >= k).astype(int)
        if target.sum() == 0 or target.sum() == len(target):
            continue
        output[f"auroc_ge_{k}"] = float(roc_auc_score(target, order))
    if output:
        output["auroc_cumulative_mean"] = float(np.mean(list(output.values())))
    return output


def evaluate_partition(
    table: pd.DataFrame,
    feature_set: str,
    model: str,
    partition: pd.DataFrame,
    *,
    measure_on: str = "test",
    group_column: str = "city",
    target_column: str = "ordinal",
    resamples: int = 400,
) -> dict[str, float]:
    """Trains on the training cities and measures on the validation or test ones.

    Replaces leaving one city out per fold, which served with five cities and with a hundred
    and thirty-eight would give as many folds each trained on 99.3% of the data, where the
    spread between folds is almost all noise.

    The uncertainty comes from resampling cities within the measured split, because
    neighbouring AGEB are correlated and treating them as independent narrows the intervals
    for no reason.
    """
    from satinsight.splits import cities_of

    if measure_on not in ("val", "test"):
        raise KeyError(f"measure_on must be 'val' or 'test', not {measure_on!r}")
    columns = columns_of_set(table, feature_set)
    train = table[table[group_column].isin(cities_of(partition, "train"))]
    measured = table[table[group_column].isin(cities_of(partition, measure_on))]
    if train.empty or measured.empty:
        raise ValueError(
            f"the partition leaves {len(train)} training rows and {len(measured)} of "
            f"{measure_on}; do the city keys match?"
        )

    prediction, scores = _scores(model, train[columns], train[target_column], measured[columns])
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
        for name, value in {
            **auroc_one_vs_rest(v, scores_of),
            **auroc_cumulative(v, scores_of),
        }.items():
            replicates[name].append(value)

    intervals = {}
    for name, values in replicates.items():
        if values:
            intervals[f"{name}_ci_low"] = float(np.percentile(values, 2.5))
            intervals[f"{name}_ci_high"] = float(np.percentile(values, 97.5))
    return {
        **metrics,
        **intervals,
        "n_train": len(train),
        "n_measured": len(measured),
        "cities_measured": len(unique_cities),
    }


def fuse(optical: pd.DataFrame, radar: pd.DataFrame, *, key: str = "cvegeo") -> pd.DataFrame:
    """Joins the tables of the two modalities into one, by AGEB.

    The context columns, land cover, population, city, grade, come from the same source in
    both and are taken once. The image columns carry the sensor in their name, so they
    coexist without clashing.

    Only the AGEB present in both are kept. Comparing the fusion against each modality
    separately over different samples would mix the difference of sensor with that of which
    rows each one evaluates.
    """
    radar_only = [c for c in radar.columns if c.startswith("s1")]
    missing = set(optical[key]) ^ set(radar[key])
    if missing:
        log.info("%d AGEB left out for missing in one of the two modalities", len(missing))
    return optical.merge(radar[[key, *radar_only]], on=key, how="inner")
