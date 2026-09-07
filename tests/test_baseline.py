"""Tests of the baseline over synthetic tables with known signal."""

import numpy as np
import pandas as pd
import pytest

from satinsight.agebs import GRADES
from satinsight.baseline import (
    NO_SET,
    columns_of_set,
    compare,
    evaluate,
    explained_variance,
    fold_summary,
    select_features,
    standardise_by_group,
    transfer_diagnostics,
)
from satinsight.landcover import CLASSES
from satinsight.texture import feature_names

CITIES = ("tuxtla", "merida", "iztapalapa")


def synthetic_table(per_city=120, strength=1.0, seed=0):
    """Table whose features carry signal of the ordinal, graded by `strength`.

    With a high strength the model must recover the order; with strength zero the features
    are pure noise and no model should beat the random reference consistently.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for city in CITIES:
        ordinal = rng.integers(0, 5, per_city)
        noise = rng.normal(0, 1, per_city)
        columns = {
            "cvegeo": [f"{city}{i:04d}" for i in range(per_city)],
            "city": city,
            "ordinal": ordinal,
            "grade": [GRADES[i] for i in ordinal],
            "c_mean": strength * ordinal + noise,
            "c_std": strength * ordinal * 0.5 + noise,
            "c_p10": rng.normal(0, 1, per_city),
            "c_p50": rng.normal(0, 1, per_city),
            "c_p90": rng.normal(0, 1, per_city),
            "c_iqr": rng.normal(0, 1, per_city),
        }
        # the texture names come from the module itself, so renaming a feature breaks
        # the test instead of leaving it measuring an empty set
        for cover_class in CLASSES.values():
            columns[f"wc_{cover_class}"] = rng.random(per_city)
        for suffix in feature_names():
            carries_signal = suffix.startswith(("contrast_", "homogeneity_"))
            sign = -1 if suffix.startswith("homogeneity_") else 1
            signal = sign * strength * ordinal if carries_signal else 0.0
            columns[f"c_{suffix}"] = signal + rng.normal(0, 1, per_city)
        parts.append(pd.DataFrame(columns))
    return pd.concat(parts, ignore_index=True)


def test_sets_separate_intensity_from_texture():
    table = synthetic_table(10)
    intensity = columns_of_set(table, "intensity")
    texture = columns_of_set(table, "texture")

    assert "c_mean" in intensity
    assert "c_contrast_d1" not in intensity
    assert "c_contrast_d1" in texture
    assert "c_mean" not in texture
    assert not set(intensity) & set(texture)


def test_the_full_set_is_the_union_of_the_three_steps():
    table = synthetic_table(10)
    full = set(columns_of_set(table, "all"))
    expected = set()
    for step in ("cover", "intensity", "texture"):
        expected |= set(columns_of_set(table, step))
    assert full == expected


def test_cover_does_not_mix_with_the_other_steps():
    table = synthetic_table(10)
    cover = set(columns_of_set(table, "cover"))
    assert "wc_built" in cover
    assert not cover & set(columns_of_set(table, "intensity"))
    assert not cover & set(columns_of_set(table, "texture"))


def test_unknown_set_fails():
    with pytest.raises(KeyError, match="unknown set"):
        columns_of_set(synthetic_table(10), "invented")


def test_validation_leaves_one_city_out_per_fold():
    detail = evaluate(synthetic_table(60), "all", "classifier")
    assert len(detail) == len(CITIES)
    assert set(detail["test_city"]) == set(CITIES)
    for _, row in detail.iterrows():
        assert row["n_train"] == 120
        assert row["n_test"] == 60


def test_with_signal_the_model_beats_chance():
    table = synthetic_table(200, strength=1.5)
    model = evaluate(table, "all", "classifier")["kappa"].mean()
    random = evaluate(table, "all", "random")["kappa"].mean()
    assert model > random + 0.1


def test_without_signal_the_model_does_not_beat_chance():
    table = synthetic_table(200, strength=0.0)
    model = evaluate(table, "all", "classifier")["kappa"].mean()
    assert model < 0.15


def test_the_majority_has_null_kappa():
    detail = evaluate(synthetic_table(100), "intensity", "majority")
    assert detail["kappa"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_the_regressor_predicts_inside_the_ordinal_range():
    table = synthetic_table(100, strength=2.0)
    detail = evaluate(table, "all", "regressor")
    assert (detail["mae_ordinal"] >= 0).all()
    assert (detail["accuracy"] <= 1).all()


def test_compare_runs_the_blind_models_once():
    detail = compare(synthetic_table(50))
    blind = detail[detail["model"].isin(["random", "majority"])]
    assert set(blind["set"]) == {NO_SET}
    assert len(blind) == 2 * len(CITIES)


def test_the_summary_sorts_by_kappa():
    aggregate = fold_summary(compare(synthetic_table(80, strength=1.5)))
    assert list(aggregate["kappa"]) == sorted(aggregate["kappa"], reverse=True)
    assert {"set", "model", "kappa"} <= set(aggregate.columns)


def test_explained_variance_recognises_a_perfect_factor():
    table = pd.DataFrame({"v": [1.0, 1.0, 5.0, 5.0], "g": ["a", "a", "b", "b"]})
    assert explained_variance(table, "v", "g") == pytest.approx(1.0)


def test_an_unrelated_factor_explains_little():
    table = pd.DataFrame({"v": [1.0, 5.0, 1.0, 5.0], "g": ["a", "a", "b", "b"]})
    assert explained_variance(table, "v", "g") == pytest.approx(0.0, abs=1e-9)


def test_diagnostics_expose_the_feature_that_only_knows_the_city():
    """A feature that separates cities without separating grades must show a high ratio."""
    table = synthetic_table(80, strength=1.5)
    table["c_mean"] = table["city"].map({c: i * 10.0 for i, c in enumerate(CITIES)})
    d = transfer_diagnostics(table, "intensity").set_index("feature")
    assert d.loc["c_mean", "ratio"] > 10
    assert d.loc["c_std", "ratio"] < d.loc["c_mean", "ratio"]


def test_standardise_centres_within_each_city():
    table = synthetic_table(60, strength=1.0)
    columns = columns_of_set(table, "intensity")
    e = standardise_by_group(table, columns)
    for _, group in e.groupby("city"):
        assert group["c_mean"].mean() == pytest.approx(0.0, abs=1e-9)
        assert group["c_mean"].std() == pytest.approx(1.0, abs=1e-9)


def test_a_constant_feature_ends_at_zero_and_not_null():
    """Several cover classes are zero in every AGEB.

    Turning them into entirely null columns breaks the model's binning, so the degenerate
    case has to end up centred at zero.
    """
    table = synthetic_table(40)
    table["wc_snow"] = 0.0
    e = standardise_by_group(table, ["wc_snow"])
    assert e["wc_snow"].notna().all()
    assert (e["wc_snow"] == 0.0).all()


def test_standardise_keeps_the_nulls_that_are_absent_data():
    table = synthetic_table(40)
    table.loc[:5, "c_mean"] = np.nan
    e = standardise_by_group(table, ["c_mean"])
    assert e["c_mean"].isna().sum() == 6


def test_the_standardised_ablation_runs_through():
    table = synthetic_table(60, strength=1.5)
    detail = compare(table, standardise=True)
    assert not detail.empty
    assert detail["kappa"].notna().all()


def test_a_table_without_features_of_the_set_fails():
    table = pd.DataFrame({"city": ["a"], "ordinal": [1], "something_else": [3.0]})
    with pytest.raises(ValueError, match="set"):
        evaluate(table, "texture", "classifier")


def reliability_of(table, value=0.9):
    """Synthetic reliability table with the same value for every feature."""
    features = columns_of_set(table, "texture")
    return pd.DataFrame({"feature": features, "r_median": [value] * len(features)})


def test_a_feature_that_does_not_reproduce_is_left_out():
    table = synthetic_table(40)
    table["c_n_px"] = 1000
    reliability = reliability_of(table)
    reliability.loc[reliability.feature == "c_contrast_d1", "r_median"] = 0.2

    sel = select_features(table, reliability).set_index("feature")
    assert not sel.loc["c_contrast_d1", "kept"]
    assert sel.loc["c_contrast_d1", "reason"] == "does not reproduce"


def test_a_feature_tied_to_size_is_left_out():
    """A feature that is a function of polygon area points at the target by construction."""
    table = synthetic_table(60)
    rng = np.random.default_rng(3)
    table["c_n_px"] = rng.integers(700, 20000, len(table))
    table["c_contrast_d1"] = np.log10(table["c_n_px"]) * 5

    sel = select_features(table, reliability_of(table)).set_index("feature")
    assert not sel.loc["c_contrast_d1", "kept"]
    assert sel.loc["c_contrast_d1", "reason"] == "tied to size"


def test_a_reliable_and_independent_feature_is_kept():
    table = synthetic_table(60)
    rng = np.random.default_rng(4)
    table["c_n_px"] = rng.integers(700, 20000, len(table))
    table["c_homogeneity_d1"] = rng.normal(0, 1, len(table))

    sel = select_features(table, reliability_of(table)).set_index("feature")
    assert sel.loc["c_homogeneity_d1", "kept"]
    assert sel.loc["c_homogeneity_d1", "reason"] == "kept"


def test_pure_noise_is_caught_by_reliability_and_not_by_size():
    """The two criteria need each other.

    A feature that is noise passes the size criterion with room to spare, because noise
    correlates with nothing. Only reliability detects it.
    """
    table = synthetic_table(60)
    rng = np.random.default_rng(5)
    table["c_n_px"] = rng.integers(700, 20000, len(table))
    table["c_energy_d4"] = rng.normal(0, 1, len(table))

    reliability = reliability_of(table)
    reliability.loc[reliability.feature == "c_energy_d4", "r_median"] = 0.05
    sel = select_features(table, reliability).set_index("feature")

    assert abs(sel.loc["c_energy_d4", "r_n_px"]) < 0.30
    assert not sel.loc["c_energy_d4", "kept"]


def test_a_feature_without_reliability_measurement_is_left_out():
    table = synthetic_table(40)
    table["c_n_px"] = 1000
    reliability = reliability_of(table)
    sel = select_features(table, reliability[reliability.feature != "c_contrast_d2"]).set_index(
        "feature"
    )
    assert not sel.loc["c_contrast_d2", "kept"]


def test_the_auroc_of_the_low_classes_is_not_inverted():
    """With a single ordered score, the evidence for k is closeness to k.

    Using the raw order inverts the low classes and the average lands at 0.5 by
    cancellation, looking like chance where the model separates almost perfectly.
    """
    from satinsight.baseline import auroc_one_vs_rest

    truth = np.array([0, 1, 2, 3, 4] * 20)
    almost_perfect = truth + np.random.default_rng(0).normal(0, 0.2, len(truth))
    r = auroc_one_vs_rest(truth, almost_perfect)
    assert r[f"auroc_{GRADES[0].lower().replace(' ', '_')}"] > 0.9
    assert r["auroc_macro"] > 0.85


def test_the_cumulative_auroc_respects_the_order():
    from satinsight.baseline import auroc_cumulative

    truth = np.array([0, 1, 2, 3, 4] * 20)
    r = auroc_cumulative(truth, truth.astype(float))
    assert all(v == 1.0 for k, v in r.items() if k.startswith("auroc_ge_"))


def test_fuse_joins_the_columns_of_both_modalities():
    from satinsight.baseline import fuse

    optical = pd.DataFrame(
        {"cvegeo": ["a", "b"], "city": ["x", "x"], "ordinal": [1, 2], "s2red_mean": [1.0, 2.0]}
    )
    radar = pd.DataFrame(
        {"cvegeo": ["a", "b"], "city": ["x", "x"], "ordinal": [1, 2], "s1vv_mean": [3.0, 4.0]}
    )
    joined = fuse(optical, radar)
    assert list(joined.columns) == ["cvegeo", "city", "ordinal", "s2red_mean", "s1vv_mean"]
    assert len(joined) == 2


def test_fuse_keeps_only_the_agebs_present_in_both():
    from satinsight.baseline import fuse

    optical = pd.DataFrame({"cvegeo": ["a", "b", "c"], "s2red_mean": [1.0, 2.0, 3.0]})
    radar = pd.DataFrame({"cvegeo": ["b", "c", "d"], "s1vv_mean": [4.0, 5.0, 6.0]})
    joined = fuse(optical, radar)
    assert sorted(joined.cvegeo) == ["b", "c"]
