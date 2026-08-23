"""Tests of the MIL pieces that carry logic. The network itself needs no deep learning here."""

import numpy as np
import pytest

from satinsight.bagdata import Bag
from satinsight.mil import attention_per_ageb
from satinsight.train_mil import _bag_weights, score_heatmap


def test_attention_adds_up_within_each_ageb():
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    keys = np.array(["a", "a", "b", "b"])
    assert attention_per_ageb(weights, keys) == pytest.approx({"a": 0.3, "b": 0.7})


def test_attention_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="against"):
        attention_per_ageb(np.array([0.5, 0.5]), np.array(["a"]))


def test_the_rare_grade_gets_the_heaviest_weight():
    """Cuatro bolsas de grado alto contra ciento setenta de grado bajo.

    Sin reponderar, predecir la clase mayoritaria en todas partes es un óptimo local que el
    modelo alcanza en dos épocas y del que no sale.
    """
    bags = [Bag("c", "m", np.zeros((1, 2)), np.array(["a"]), 1)] * 90
    bags += [Bag("c", "m", np.zeros((1, 2)), np.array(["a"]), 4)] * 10
    w = _bag_weights(bags, 5)
    assert w[4] > w[1]
    assert w[0] == 0.0  # una clase ausente no recibe peso


def _bag(city, keys, n_por_clave=1):
    claves = np.repeat(keys, n_por_clave)
    return Bag(city, "m", np.zeros((len(claves), 2)), claves, 2)


def test_the_heatmap_scores_perfectly_when_attention_follows_the_grade():
    bag = _bag("x", ["a", "b", "c", "d"])
    grades = {"a": 0, "b": 1, "c": 3, "d": 4}
    r = score_heatmap([bag], [np.array([0.1, 0.2, 0.3, 0.4])], grades)
    assert r["spearman_mean"] == pytest.approx(1.0)
    assert r["spearman_pooled"] == pytest.approx(1.0)


def test_the_heatmap_scores_negative_when_attention_runs_backwards():
    bag = _bag("x", ["a", "b", "c", "d"])
    grades = {"a": 0, "b": 1, "c": 3, "d": 4}
    r = score_heatmap([bag], [np.array([0.4, 0.3, 0.2, 0.1])], grades)
    assert r["spearman_mean"] == pytest.approx(-1.0)


def test_a_bag_of_one_grade_contributes_nothing():
    """No hay orden que recuperar dentro de una bolsa cuyas AGEB comparten grado."""
    bag = _bag("x", ["a", "b", "c"])
    r = score_heatmap([bag], [np.array([0.5, 0.3, 0.2])], {"a": 1, "b": 1, "c": 1})
    assert r["bags_scored"] == 0
    assert np.isnan(r["spearman_mean"])
    assert r["agebs_scored"] == 3


def test_bags_with_too_few_agebs_are_skipped():
    bag = _bag("x", ["a", "b"])
    r = score_heatmap([bag], [np.array([0.7, 0.3])], {"a": 0, "b": 4})
    assert r["bags_scored"] == 0
    assert r["agebs_scored"] == 0


def test_agebs_without_a_grade_are_left_out():
    bag = _bag("x", ["a", "b", "c", "sin_grado"])
    grades = {"a": 0, "b": 2, "c": 4}
    r = score_heatmap([bag], [np.array([0.1, 0.2, 0.3, 0.4])], grades)
    assert r["agebs_scored"] == 3
    assert r["spearman_mean"] == pytest.approx(1.0)
