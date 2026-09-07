"""Scores the downloadable global proxies a practitioner would use instead of the map.

The RWI comparison covers the leading gridded wealth product and leaves the two cheaper
objections open: that built-up density alone orders neighborhoods as well as the map, and
that nighttime lights, the oldest proxy in this literature, already do the job. Both are
free downloads, so both are the first thing a referee will ask about.

The protocol is the one the RWI comparison already uses, kept identical so the numbers sit
in the same table: every product is scored per AGEB on the same universe as the map, the
correlation with the tract grade is computed within each municipality, and the interval is
a city-clustered bootstrap of the paired difference. Orientation is fixed in
`incumbents.DEPRIVATION_SIGN` before anything is measured.

Consumes the persisted scores; trains nothing and opens nothing.

Usage: global_products.py [data/predictions_val.parquet]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from satinsight.agebs import catalogue_with_extra  # noqa: E402
from satinsight.incumbents import (  # noqa: E402
    DEPRIVATION_SIGN,
    GHSL_PRODUCTS,
    LABELS,
    NIGHTLIGHTS_KEY,
    ensure_ghsl,
    ensure_nightlights,
    tiles_for,
    zonal_mean,
)
from satinsight.pipeline import city_aoi  # noqa: E402
from satinsight.splits import cities_of  # noqa: E402

RESAMPLES = 2000
MIN_AGEB = 20
"""Municipalities with fewer tracts are dropped, as in the RWI comparison."""

PRODUCTS = [*GHSL_PRODUCTS, NIGHTLIGHTS_KEY]


def values_per_ageb(cities, catalogue, cache: Path) -> pd.DataFrame:
    """Tract-level value of every product, cached because the rasters are large."""
    if cache.exists():
        table = pd.read_parquet(cache)
        logging.info("reusing %s (%d AGEB)", cache.name, len(table))
        return table

    areas = {}
    for city in cities:
        try:
            area, agebs = city_aoi(city, catalogue=catalogue)
        except Exception as error:  # a city without geometry is skipped, as elsewhere
            logging.warning("%s: skipped (%s)", city, error)
            continue
        areas[city] = (area, agebs)

    tiles = tiles_for([area.bbox for area, _ in areas.values()])
    logging.info("GHSL tiles required: %s", tiles)
    sources = {key: ensure_ghsl(key, tiles) for key in GHSL_PRODUCTS}
    sources[NIGHTLIGHTS_KEY] = ensure_nightlights()

    frames = []
    for city, (_, agebs) in areas.items():
        logging.info("%s: %d AGEB", city, len(agebs))
        joined = pd.DataFrame(
            {
                "city": city,
                "cvegeo": agebs.cvegeo.to_numpy(),
                "ordinal": agebs.ordinal.astype(int).to_numpy(),
            }
        )
        for key, paths in sources.items():
            measured = zonal_mean(paths, agebs, agebs.cvegeo, column=key)
            joined = joined.merge(measured, on="cvegeo", how="left")
        frames.append(joined)

    table = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache, index=False)
    logging.info("wrote %s (%d AGEB)", cache.name, len(table))
    return table


def paired_interval(diff: dict, city_of: dict, seed: int = 0) -> tuple[float, float]:
    """City-clustered bootstrap of the mean paired difference."""
    by_city: dict = {}
    for municipality, value in diff.items():
        by_city.setdefault(city_of[municipality], []).append(value)
    groups = [np.array(v) for v in by_city.values()]
    rng = np.random.default_rng(seed)
    means = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        chosen = rng.integers(0, len(groups), len(groups))
        means[k] = float(np.concatenate([groups[j] for j in chosen]).mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/predictions_val.parquet"
    split = "test" if "test" in source else "val"
    suffix = "_test" if split == "test" else ""

    scores = pd.read_parquet(source)
    per_ageb = (
        scores.groupby(["city", "municipality", "cvegeo"], observed=True).score.mean().reset_index()
    )

    partition = pd.read_csv("data/partition.csv")
    catalogue = catalogue_with_extra()
    cities = sorted(cities_of(partition, split))
    products = values_per_ageb(
        cities, catalogue, Path(f"data/global_products_agebs{suffix}.parquet")
    )

    both = per_ageb.merge(products.drop(columns=["city"]), on="cvegeo", how="inner")
    logging.info("shared universe: %d AGEB over %d cities", len(both), both.city.nunique())

    rows = []
    for product in PRODUCTS:
        sign = DEPRIVATION_SIGN[product]
        frame = both[both[product].notna()].copy()
        frame["proxy"] = sign * frame[product]

        ours, theirs, city_of = {}, {}, {}
        for municipality, group in frame.groupby("municipality", observed=True):
            if len(group) < MIN_AGEB or group.ordinal.nunique() < 2:
                continue
            if group.proxy.nunique() < 2:
                continue
            ours[municipality] = float(spearmanr(group.score, group.ordinal).statistic)
            theirs[municipality] = float(spearmanr(group.proxy, group.ordinal).statistic)
            city_of[municipality] = group.city.iloc[0]

        if not ours:
            logging.warning("%s: no municipality survived the filters", product)
            continue

        diff = {m: ours[m] - theirs[m] for m in ours}
        low, high = paired_interval(diff, city_of)
        high_grade = (frame.ordinal >= 4).to_numpy()
        pooled_auc = (
            float(roc_auc_score(high_grade, frame.proxy))
            if high_grade.any() and not high_grade.all()
            else float("nan")
        )
        ours_auc = (
            float(roc_auc_score(high_grade, frame.score))
            if high_grade.any() and not high_grade.all()
            else float("nan")
        )
        rows.append(
            {
                "product": product,
                "label": LABELS[product],
                "agebs": len(frame),
                "municipalities": len(ours),
                "centroid_fallback": int((frame[f"{product}_n_px"] == 0).sum()),
                "ours_within": float(np.mean(list(ours.values()))),
                "proxy_within": float(np.mean(list(theirs.values()))),
                "difference": float(np.mean(list(diff.values()))),
                "ci_low": low,
                "ci_high": high,
                "wins": int(sum(d > 0 for d in diff.values())),
                "ours_auc_high": ours_auc,
                "proxy_auc_high": pooled_auc,
            }
        )
        print(
            f"{product:>14}  ours {rows[-1]['ours_within']:+.3f} · proxy "
            f"{rows[-1]['proxy_within']:+.3f} · paired {rows[-1]['difference']:+.3f} "
            f"[{low:+.3f}, {high:+.3f}] · wins {rows[-1]['wins']}/{len(ours)} · "
            f"AUROC {ours_auc:.3f} vs {pooled_auc:.3f}",
            flush=True,
        )

    output = Path(f"data/global_products{suffix}.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    logging.info("wrote %s", output)


if __name__ == "__main__":
    main()
