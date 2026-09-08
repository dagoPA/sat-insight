"""Uncertainty on the map, and the audit of where it fails by name.

A map published without a statement of confidence is not usable for targeting, and it is
the objection a remote sensing referee raises first. Three questions are answered here,
all from the persisted per-seed scores, training nothing:

1. Does seed disagreement know where the map is wrong? Three seeds give a spread per
   tract; if that spread ranks the errors, it is an uncertainty layer, and if it does not,
   the honest thing is to say the map carries none.
2. What does it buy operationally? Discarding the least confident tracts inside each
   municipality should raise the correlation over what survives. That selective curve is
   the form a targeting office would actually use: leave the doubtful tracts to a survey.
3. Can an interval be attached? Split conformal calibrated on the validation cities and
   evaluated on the test cities gives a grade interval with a stated coverage. Because
   tracts within a city are anything but exchangeable with tracts in another, the
   clustered variant takes the quantile over per-city quantiles, and per-city coverage is
   reported so that a city where the interval fails cannot hide inside the average.

The audit closes with the test cities ranked by map quality, so the worst one is named in
the paper rather than buried in a mean.

Usage: uncertainty.py
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from satinsight import backbone  # noqa: E402

RESAMPLES = 2000
MIN_AGEB = 20
"""A municipality needs this many tracts for a within correlation to mean anything."""

DISCARD_STEPS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
COVERAGE = 0.90


def per_ageb(path: str) -> pd.DataFrame:
    """Ensemble score and seed disagreement per tract.

    The map value of a tract is the mean of its token scores, averaged over seeds. The
    disagreement is the spread of the three per-seed tract values, which is the quantity a
    user could compute without ever seeing a label.
    """
    scores = pd.read_parquet(path)
    by_seed = (
        scores.groupby(["seed", "city", "municipality", "cvegeo"], observed=True)
        .score.mean()
        .reset_index()
    )
    grouped = by_seed.groupby(["city", "municipality", "cvegeo"], observed=True).score
    table = grouped.agg(["mean", "std", "count"]).reset_index()
    table = table.rename(columns={"mean": "score", "std": "spread", "count": "seeds"})
    logging.info("%s: %d tracts, %d seeds", path, len(table), int(table.seeds.max()))
    return table


def with_truth(table: pd.DataFrame) -> pd.DataFrame:
    """Joins the held-out tract grade and drops what has none."""
    truth = pd.read_parquet("data/grs_ageb_2020.parquet")[["cvegeo", "ordinal", "population"]]
    joined = table.merge(truth, on="cvegeo", how="inner")
    joined["ordinal"] = joined.ordinal.astype(int)
    logging.info("%d of %d tracts carry a grade", len(joined), len(table))
    return joined


def clustered_interval(values: dict, city_of: dict, seed: int = 0) -> tuple[float, float]:
    """City-clustered bootstrap of a mean over municipalities."""
    by_city: dict = {}
    for municipality, value in values.items():
        by_city.setdefault(city_of[municipality], []).append(value)
    groups = [np.array(v) for v in by_city.values()]
    rng = np.random.default_rng(seed)
    means = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        chosen = rng.integers(0, len(groups), len(groups))
        means[k] = float(np.concatenate([groups[j] for j in chosen]).mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def usable(frame: pd.DataFrame):
    """Municipalities with enough tracts and more than one grade among them."""
    for municipality, group in frame.groupby("municipality", observed=True):
        if len(group) >= MIN_AGEB and group.ordinal.nunique() > 1:
            yield municipality, group


def spread_knows_error(frame: pd.DataFrame, isotonic) -> dict:
    """Whether seed disagreement ranks the tract errors, measured within municipality."""
    frame = frame.copy()
    frame["predicted"] = isotonic.predict(frame.score)
    frame["error"] = (frame.predicted - frame.ordinal).abs()
    rhos, city_of = {}, {}
    for municipality, group in usable(frame):
        if group.spread.nunique() < 2:
            continue
        rhos[municipality] = float(spearmanr(group.spread, group.error).statistic)
        city_of[municipality] = group.city.iloc[0]
    low, high = clustered_interval(rhos, city_of)
    return {
        "municipalities": len(rhos),
        "rho_spread_error": float(np.mean(list(rhos.values()))),
        "ci_low": low,
        "ci_high": high,
    }


def _survives_every_step(group: pd.DataFrame) -> bool:
    """Whether a municipality stays measurable at every point of the discard sweep."""
    for fraction in DISCARD_STEPS:
        keep = group.sort_values("spread").head(round(len(group) * (1 - fraction)))
        if len(keep) < MIN_AGEB or keep.ordinal.nunique() < 2:
            return False
    return True


def selective_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Map quality against how many of the least confident tracts are set aside.

    The discard happens inside each municipality, because that is the unit a targeting
    office works in and the unit the correlation is computed over.

    The set of municipalities is fixed once, to those that stay above the minimum at every
    point of the sweep. Letting municipalities drop out as the cut deepens would confound
    the curve with attrition: the deepest cut would be measured on the largest, easiest
    municipalities and the rise would be an artifact of which ones remain. The cost is a
    smaller panel, reported next to the curve.
    """
    eligible = {
        municipality for municipality, group in usable(frame) if _survives_every_step(group)
    }
    logging.info("%d municipalities hold at every discard level", len(eligible))

    rows = []
    for fraction in DISCARD_STEPS:
        rhos, city_of = {}, {}
        for municipality, group in usable(frame):
            if municipality not in eligible:
                continue
            keep = group.sort_values("spread").head(round(len(group) * (1 - fraction)))
            rhos[municipality] = float(spearmanr(keep.score, keep.ordinal).statistic)
            city_of[municipality] = keep.city.iloc[0]
        low, high = clustered_interval(rhos, city_of)
        rows.append(
            {
                "discarded": fraction,
                "municipalities": len(rhos),
                "rho_within": float(np.mean(list(rhos.values()))),
                "ci_low": low,
                "ci_high": high,
            }
        )
        logging.info(
            "discard %.0f%%: rho %+.3f [%+.3f, %+.3f] over %d municipalities",
            100 * fraction,
            rows[-1]["rho_within"],
            low,
            high,
            len(rhos),
        )
    return pd.DataFrame(rows)


def conformal(calibration: pd.DataFrame, evaluation: pd.DataFrame, isotonic) -> pd.DataFrame:
    """Split-conformal grade intervals, marginal and clustered by city.

    The nonconformity score is the absolute distance between the grade the isotonic map
    predicts and the grade the census publishes. The marginal quantile treats every
    calibration tract as exchangeable with every evaluation tract, which is the assumption
    the clustered variant exists to doubt: it takes each calibration city's own quantile
    and then the quantile across cities, so a city unlike the calibration set is priced in.
    """
    calibration = calibration.copy()
    evaluation = evaluation.copy()
    calibration["residual"] = (isotonic.predict(calibration.score) - calibration.ordinal).abs()
    evaluation["residual"] = (isotonic.predict(evaluation.score) - evaluation.ordinal).abs()

    n = len(calibration)
    level = min(1.0, np.ceil((n + 1) * COVERAGE) / n)
    marginal = float(np.quantile(calibration.residual, level))

    per_city = calibration.groupby("city", observed=True).residual.quantile(COVERAGE)
    clustered = float(np.quantile(per_city, COVERAGE))

    rows = []
    city_rows = []
    for name, width in (("marginal", marginal), ("clustered", clustered)):
        covered = evaluation.residual <= width
        by_city = evaluation.assign(covered=covered).groupby("city", observed=True).covered.mean()
        city_rows.append(
            by_city.rename("coverage").reset_index().assign(rule=name, half_width_grades=width)
        )
        rows.append(
            {
                "rule": name,
                "nominal": COVERAGE,
                "half_width_grades": width,
                "coverage": float(covered.mean()),
                "worst_city_coverage": float(by_city.min()),
                "worst_city": str(by_city.idxmin()),
                "best_city_coverage": float(by_city.max()),
                "cities_below_nominal": int((by_city < COVERAGE).sum()),
                "cities": int(by_city.size),
            }
        )
        logging.info(
            "%s: half width %.2f grades, coverage %.3f, worst city %s at %.3f, "
            "%d of %d cities below nominal",
            name,
            width,
            rows[-1]["coverage"],
            rows[-1]["worst_city"],
            rows[-1]["worst_city_coverage"],
            rows[-1]["cities_below_nominal"],
            rows[-1]["cities"],
        )
    pd.concat(city_rows, ignore_index=True).to_csv(
        backbone.suffixed("data/uncertainty_coverage_city.csv"), index=False
    )
    return pd.DataFrame(rows)


def audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Every evaluated city ranked by the map quality it gets, worst first."""
    rows = []
    for city, group in frame.groupby("city", observed=True):
        rhos = [rho for _, rho in _within_of(group)]
        if not rhos:
            continue
        rows.append(
            {
                "city": city,
                "municipalities": len(rhos),
                "agebs": len(group),
                "rho_within": float(np.mean(rhos)),
                "grades_present": int(group.ordinal.nunique()),
                "share_high": float((group.ordinal >= 3).mean()),
                "median_spread": float(group.spread.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("rho_within").reset_index(drop=True)


def _within_of(frame: pd.DataFrame):
    for municipality, group in usable(frame):
        yield municipality, float(spearmanr(group.score, group.ordinal).statistic)


def main() -> None:
    validation = with_truth(per_ageb(backbone.suffixed("data/predictions_val.parquet")))
    test = with_truth(per_ageb(backbone.suffixed("data/predictions_test.parquet")))

    isotonic = IsotonicRegression(out_of_bounds="clip").fit(validation.score, validation.ordinal)

    rows = []
    for split, frame in (("val", validation), ("test", test)):
        result = spread_knows_error(frame, isotonic)
        result["split"] = split
        rows.append(result)
        logging.info(
            "%s: spread vs error rho %+.3f [%+.3f, %+.3f] over %d municipalities",
            split,
            result["rho_spread_error"],
            result["ci_low"],
            result["ci_high"],
            result["municipalities"],
        )
    pd.DataFrame(rows).to_csv(backbone.suffixed("data/uncertainty_spread.csv"), index=False)

    curves = []
    for split, frame in (("val", validation), ("test", test)):
        logging.info("selective curve on %s", split)
        curve = selective_curve(frame)
        curve.insert(0, "split", split)
        curves.append(curve)
    pd.concat(curves, ignore_index=True).to_csv(
        backbone.suffixed("data/uncertainty_selective.csv"), index=False
    )

    conformal(validation, test, isotonic).to_csv(
        backbone.suffixed("data/uncertainty_conformal.csv"), index=False
    )

    ranking = audit(test)
    ranking.to_csv(backbone.suffixed("data/city_audit_test.csv"), index=False)
    print(ranking.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
