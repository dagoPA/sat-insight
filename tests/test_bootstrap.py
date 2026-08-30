"""The clustered bootstrap must widen when the signal clusters by city."""

import numpy as np

from satinsight.llp import bootstrap_within


def test_mean_matches_the_plain_mean():
    per_bag = [("m1", 0.1), ("m2", 0.3), ("m3", 0.5)]
    city_of = {"m1": "a", "m2": "a", "m3": "b"}
    mean, _ = bootstrap_within(per_bag, city_of, n_resamples=200)
    assert abs(mean - 0.3) < 1e-9


def test_clustered_signal_widens_the_interval():
    rng = np.random.default_rng(0)
    city_of = {}
    # same bag values in both designs; what differs is how they group into cities
    values = rng.normal(0.2, 0.15, 40)
    clustered = [(f"m{i}", float(0.4 if i < 20 else 0.0) + values[i] * 0) for i in range(40)]
    spread = [(f"m{i}", float(values[i])) for i in range(40)]
    for i in range(40):
        city_of[f"m{i}"] = f"c{i // 10}"
    _, wide = bootstrap_within(clustered, city_of, n_resamples=500)
    _, narrow = bootstrap_within(spread, {f"m{i}": f"c{i}" for i in range(40)}, n_resamples=500)
    assert wide > narrow
