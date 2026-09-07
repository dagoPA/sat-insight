"""Partitions cities into training and test, with no city split between the two.

The unit of partition is the city, never the AGEB. Neighbouring AGEB share streets,
building stock and the same acquisition geometry, so dealing them at random puts halves
of the same neighbourhood on both sides and the score comes out inflated. Holding out
whole cities asks the question the project actually cares about: does this transfer to a
city the model has never seen.

Cities are dealt into training, validation and test in an 80/10/10 split. Validation
selects, test is held out for reporting. Leaving a single city out at a time made sense
with five of them; with 138 it would mean 138 folds, each training on 99.3% of the data,
and the spread between folds would be mostly noise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROPORTIONS = (0.8, 0.1, 0.1)
"""How cities are dealt between training, validation and test."""

SETS = ("train", "val", "test")

SEED = 20200101
"""Fixed so the partition can be rebuilt from scratch; it is the census reference date."""

STRATA = 3
"""Bins per stratifying variable. Three by three leaves nine strata over 138 cities."""


def _bins(values: pd.Series, n: int) -> pd.Series:
    """Rank-based bins that survive ties and skew.

    Quantile cuts on a heavily tied column collapse into fewer bins than asked for, and
    city size is skewed enough that fixed-width cuts would leave one bin holding almost
    everything.
    """
    ranks = values.rank(method="first")
    return pd.cut(ranks, bins=n, labels=False).astype(int)


def assign(
    cities: pd.DataFrame,
    *,
    proportions: tuple[float, float, float] = PROPORTIONS,
    seed: int = SEED,
    city_column: str = "key",
    size_column: str = "agebs",
    stratum_column: str = "high_share",
) -> pd.DataFrame:
    """Deals whole cities into training, validation and test, balanced on two variables.

    Both are stratified because both bias the result on their own. Sorting only by size
    leaves the test set with the wrong mix of deprived AGEB, and the high grades are rare
    enough that an unlucky draw could leave a split with almost none.

    The deal walks an order that alternates strata and hands out places on a repeating
    pattern, so the proportions come out exact and every stratum is spread across the
    three sets rather than sampled into them.
    """
    missing = {city_column, size_column, stratum_column} - set(cities.columns)
    if missing:
        raise KeyError(f"the city table is missing {sorted(missing)}")
    if abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError(f"proportions must add up to one, got {proportions}")
    if len(cities) < len(SETS):
        raise ValueError(f"{len(cities)} cities cannot fill {len(SETS)} sets")

    table = cities[[city_column, size_column, stratum_column]].copy()
    table.columns = ["city", "n_agebs", "stratum_value"]
    table["stratum"] = (
        _bins(table.n_agebs, STRATA).astype(str)
        + "-"
        + _bins(table.stratum_value, STRATA).astype(str)
    )

    rng = np.random.default_rng(seed)
    pattern = _pattern(proportions)

    # the deal runs inside each stratum and not over a global order: walking a
    # single list, a whole stratum can land on positions the cycle always sends to the same
    # set, and validation ends up with half the deprivation of training even though the
    # global split adds up. Each stratum now hands over its own share.
    assigned: dict[str, str] = {}
    offset = 0
    for _stratum, group in table.groupby("stratum", observed=True, sort=True):
        keys = group.city.to_numpy().copy()
        rng.shuffle(keys)
        for i, city in enumerate(keys):
            assigned[city] = pattern[(i + offset) % len(pattern)]
        # the offset stops every stratum from handing its first city to the same set,
        # which with small strata would bias the whole deal
        offset = (offset + len(keys)) % len(pattern)
    table["split"] = table.city.map(assigned)

    summary = table.groupby("split", observed=True).agg(
        cities=("city", "size"),
        agebs=("n_agebs", "sum"),
        mean_deprivation=("stratum_value", "mean"),
    )
    log.info("partition:\n%s", summary)
    return table[["city", "split", "n_agebs", "stratum_value", "stratum"]]


def _pattern(proportions: tuple[float, float, float], steps: int = 10) -> list[str]:
    """Cycle of destinations that reproduces the requested proportions.

    Dealing by cycle rather than by sampling makes the proportions come out exact and
    stops any set from taking a run of the same stratum.
    """
    quotas = [round(p * steps) for p in proportions]
    quotas[0] += steps - sum(quotas)
    # test and validation sit at the start of the cycle, apart from each other, so they
    # spread along the order instead of piling up at one end
    pattern = ["test", "val"] * min(quotas[2], quotas[1])
    pattern += ["test"] * (quotas[2] - min(quotas[2], quotas[1]))
    pattern += ["val"] * (quotas[1] - min(quotas[2], quotas[1]))
    pattern += ["train"] * quotas[0]
    return pattern


def cities_of(partition: pd.DataFrame, split: str) -> list[str]:
    """City keys of one of the three sets."""
    if split not in SETS:
        raise KeyError(f"unknown set {split!r}, expected one of {SETS}")
    return partition.loc[partition.split == split, "city"].tolist()


def check(partition: pd.DataFrame, instances: pd.DataFrame | None = None) -> None:
    """Fails loudly if a city, an AGEB or a bag ended up on both sides of the partition.

    Worth running even though `assign` cannot produce a leak by construction: the tables
    get rebuilt, filtered and merged by hand along the way, and a leak found by the
    reviewer instead of by us costs the paper.
    """
    repeated = partition.city[partition.city.duplicated()].unique()
    if len(repeated):
        raise ValueError(f"cities assigned more than once: {sorted(repeated)}")

    if instances is None:
        return
    per_city = partition.set_index("city").split
    marked = instances.assign(split=instances.city.map(per_city))
    unassigned = marked.split.isna().sum()
    if unassigned:
        raise ValueError(f"{unassigned} instances belong to a city outside the partition")
    for column in ("cvegeo", "municipality"):
        if column not in marked.columns:
            continue
        crossing = marked.groupby(column, observed=True).split.nunique()
        if (crossing > 1).any():
            offenders = crossing[crossing > 1].index.tolist()[:5]
            raise ValueError(f"{column} spanning both sides of the partition: {offenders}")
    log.info("partition clean over %d instances", len(instances))


def municipality_owner(table: pd.DataFrame, catalogue: dict) -> dict[str, str]:
    """Assigns each municipality key to a single city, so no AGEB belongs to two.

    A city box wraps its conurbated urban footprint, and the footprints of two
    neighbouring cities overlap: Guadalajara and Zapopan are separate catalogue entries and
    share 302 AGEB. Extracted under both, those AGEB enter the table twice, and if the two
    cities fall on different sides of the partition the model sees in training rows that
    are later measured on it. Over the national partition there were 1,880 such AGEB.

    The city whose municipality key is the AGEB's own decides, which is the one the
    catalogue names. A municipality no city claims, it enters by proximity, not by being
    the core, goes to whichever holds most of its AGEB, and ties break by name so the
    assignment does not depend on the order the files were read in.
    """
    owner = {c.municipality: key for key, c in catalogue.items()}
    output: dict[str, str] = {}
    for municipality_key, group in table.groupby(table.cvegeo.str[:5], observed=True):
        if municipality_key in owner and owner[municipality_key] in set(group.city):
            output[municipality_key] = owner[municipality_key]
            continue
        counts = group.city.value_counts()
        output[municipality_key] = sorted(counts[counts == counts.max()].index)[0]
    return output


def deduplicate(table: pd.DataFrame, catalogue: dict) -> pd.DataFrame:
    """Keeps a single row per AGEB, under the city that owns its municipality key."""
    owner = municipality_owner(table, catalogue)
    municipality_key = table.cvegeo.str[:5]
    output = table[table.city == municipality_key.map(owner)].copy()
    surplus = len(table) - len(output)
    if surplus:
        log.info("%d rows duplicated between conurbated cities dropped", surplus)
    repeated = output.cvegeo.duplicated().sum()
    if repeated:
        raise ValueError(f"{repeated} AGEB still repeated after deduplicate")
    return output.reset_index(drop=True)
