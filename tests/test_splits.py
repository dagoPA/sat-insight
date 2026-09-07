"""Tests for the spatial partition. The whole point is that nothing leaks."""

import numpy as np
import pandas as pd
import pytest

from satinsight.splits import SETS, assign, check, cities_of


def catalogue(n=138, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "key": [f"city{i:03d}" for i in range(n)],
            "agebs": rng.integers(150, 1200, n),
            "high_share": rng.random(n) * 0.5,
        }
    )


def test_every_city_lands_in_exactly_one_place():
    p = assign(catalogue())
    assert len(p) == 138
    assert p.city.is_unique
    assert set(p.split) == set(SETS)


def test_the_split_is_eighty_ten_ten():
    p = assign(catalogue(n=138))
    shares = p.split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.8, abs=0.02)
    assert shares["val"] == pytest.approx(0.1, abs=0.02)
    assert shares["test"] == pytest.approx(0.1, abs=0.02)


def test_other_proportions_are_honoured():
    p = assign(catalogue(n=100), proportions=(0.6, 0.2, 0.2))
    shares = p.split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.6, abs=0.02)


def test_proportions_must_add_up():
    with pytest.raises(ValueError, match="add up to one"):
        assign(catalogue(), proportions=(0.8, 0.2, 0.2))


def test_the_partition_is_reproducible():
    a, b = assign(catalogue()), assign(catalogue())
    pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_moves_cities_around():
    a = assign(catalogue(), seed=1).set_index("city").split
    b = assign(catalogue(), seed=2).set_index("city").split
    assert (a != b).any()


def test_the_three_sets_never_share_a_city():
    p = assign(catalogue())
    lists = {c: set(cities_of(p, c)) for c in SETS}
    assert not lists["train"] & lists["val"]
    assert not lists["train"] & lists["test"]
    assert not lists["val"] & lists["test"]


def test_deprivation_is_balanced_across_the_three():
    p = assign(catalogue(n=138))
    means = p.groupby("split").stratum_value.mean()
    assert means.max() - means.min() < 0.08


def test_size_is_balanced_across_the_three():
    p = assign(catalogue(n=138))
    means = p.groupby("split").n_agebs.mean()
    assert means.max() / means.min() < 1.5


def test_assign_refuses_a_catalogue_too_small():
    with pytest.raises(ValueError, match="cannot fill"):
        assign(catalogue(n=2))


def test_assign_demands_its_columns():
    with pytest.raises(KeyError, match="high_share"):
        assign(catalogue().drop(columns=["high_share"]))


def test_cities_of_rejects_an_unknown_set():
    with pytest.raises(KeyError, match="unknown set"):
        cities_of(assign(catalogue(n=20)), "training")


def test_check_catches_an_ageb_on_both_sides():
    p = assign(catalogue(n=20))
    two = list(p.city[:2])
    instances = pd.DataFrame(
        {
            "city": two,
            "cvegeo": ["0710100010001", "0710100010001"],
            "municipality": ["07101", "07101"],
        }
    )
    p.loc[p.city == two[0], "split"] = "train"
    p.loc[p.city == two[1], "split"] = "test"
    with pytest.raises(ValueError, match="spanning both sides"):
        check(p, instances)


def test_check_catches_an_instance_from_an_unknown_city():
    p = assign(catalogue(n=20))
    with pytest.raises(ValueError, match="outside the partition"):
        check(p, pd.DataFrame({"city": ["ghost"], "cvegeo": ["x"], "municipality": ["y"]}))


def test_check_passes_on_a_clean_partition():
    p = assign(catalogue(n=20))
    instances = pd.DataFrame(
        {
            "city": p.city,
            # each city with its own municipality: a conurbation belongs to a single city,
            # and sharing it between two would be exactly the leak being hunted
            "municipality": [f"07{i:03d}" for i in range(len(p))],
            "cvegeo": [f"071010001{i:04d}" for i in range(len(p))],
        }
    )
    check(p, instances)


def _conurbated_table():
    """Two neighbouring cities that share AGEB, like Guadalajara and Zapopan."""
    return pd.DataFrame(
        {
            "cvegeo": ["1403900010001", "1403900010002", "1412000010001", "1403900010001"],
            "city": ["guadalajara", "guadalajara", "zapopan", "zapopan"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


class _City:
    def __init__(self, municipality):
        self.municipality = municipality


def test_a_shared_ageb_ends_under_a_single_city():
    from satinsight.splits import deduplicate

    catalogue = {"guadalajara": _City("14039"), "zapopan": _City("14120")}
    d = deduplicate(_conurbated_table(), catalogue)
    assert len(d) == 3
    assert d.cvegeo.is_unique
    assert d.loc[d.cvegeo == "1403900010001", "city"].item() == "guadalajara"


def test_a_municipality_no_city_claims_goes_to_the_one_holding_most():
    from satinsight.splits import municipality_owner

    table = pd.DataFrame(
        {
            "cvegeo": ["1409800010001", "1409800010002", "1409800010003"],
            "city": ["guadalajara", "guadalajara", "zapopan"],
        }
    )
    catalogue = {"guadalajara": _City("14039"), "zapopan": _City("14120")}
    assert municipality_owner(table, catalogue)["14098"] == "guadalajara"


def test_deduplicate_leaves_the_partition_without_a_leak():
    from satinsight.splits import deduplicate

    catalogue = {"guadalajara": _City("14039"), "zapopan": _City("14120")}
    d = deduplicate(_conurbated_table(), catalogue)
    partition = pd.DataFrame({"city": ["guadalajara", "zapopan"], "split": ["train", "val"]})
    d["municipality"] = d.cvegeo.str[:5]
    check(partition, d[["city", "cvegeo", "municipality"]])
