"""Tests for the spatial partition. The whole point is that nothing leaks."""

import numpy as np
import pandas as pd
import pytest

from satinsight.splits import assign, check, folds


def catalogo(n=138, seed=0):
    azar = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "clave": [f"ciudad{i:03d}" for i in range(n)],
            "agebs": azar.integers(150, 1200, n),
            "altos": azar.random(n) * 0.5,
        }
    )


def test_every_city_lands_in_exactly_one_place():
    p = assign(catalogo())
    assert len(p) == 138
    assert p.ciudad.is_unique
    assert set(p.conjunto) == {"train", "test"}


def test_the_test_set_has_the_size_asked_for():
    p = assign(catalogo(), n_test=20, n_folds=5)
    assert (p.conjunto == "test").sum() == 20
    assert p[p.conjunto == "train"].pliegue.notna().all()
    assert sorted(p.pliegue.dropna().unique()) == [0, 1, 2, 3, 4]


def test_the_partition_is_reproducible():
    a, b = assign(catalogo()), assign(catalogo())
    pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_moves_cities_around():
    a = assign(catalogo(), seed=1).set_index("ciudad").conjunto
    b = assign(catalogo(), seed=2).set_index("ciudad").conjunto
    assert (a != b).any()


def test_folds_never_validate_on_a_city_they_trained_with():
    p = assign(catalogo())
    for entrena, valida in folds(p):
        assert not set(entrena) & set(valida)
    assert len(folds(p)) == 5


def test_the_test_cities_appear_in_no_fold():
    p = assign(catalogo())
    prueba = set(p[p.conjunto == "test"].ciudad)
    for entrena, valida in folds(p):
        assert not prueba & set(entrena) and not prueba & set(valida)


def test_deprivation_is_balanced_between_train_and_test():
    p = assign(catalogo(n=138))
    medias = p.groupby("conjunto").estrato_valor.mean()
    assert abs(medias["train"] - medias["test"]) < 0.08


def test_assign_refuses_a_catalogue_too_small():
    with pytest.raises(ValueError, match="cannot fill"):
        assign(catalogo(n=10), n_test=20, n_folds=5)


def test_assign_demands_its_columns():
    with pytest.raises(KeyError, match="altos"):
        assign(catalogo().drop(columns=["altos"]))


def test_check_catches_an_ageb_on_both_sides():
    p = assign(catalogo(n=20), n_test=4, n_folds=2)
    dos = list(p.ciudad[:2])
    instancias = pd.DataFrame(
        {
            "ciudad": dos,
            "cvegeo": ["0710100010001", "0710100010001"],
            "municipio": ["07101", "07101"],
        }
    )
    p.loc[p.ciudad == dos[0], "conjunto"] = "train"
    p.loc[p.ciudad == dos[1], "conjunto"] = "test"
    with pytest.raises(ValueError, match="spanning both sides"):
        check(p, instancias)


def test_check_catches_an_instance_from_an_unknown_city():
    p = assign(catalogo(n=20), n_test=4, n_folds=2)
    with pytest.raises(ValueError, match="outside the partition"):
        check(p, pd.DataFrame({"ciudad": ["fantasma"], "cvegeo": ["x"], "municipio": ["y"]}))


def test_check_passes_on_a_clean_partition():
    p = assign(catalogo(n=20), n_test=4, n_folds=2)
    instancias = pd.DataFrame(
        {
            "ciudad": p.ciudad,
            "cvegeo": [f"071010001{i:04d}" for i in range(len(p))],
            # cada ciudad con municipio propio: una conurbación pertenece a una sola
            # ciudad, y compartirlo entre dos sería precisamente la fuga que se busca
            "municipio": [f"07{i:03d}" for i in range(len(p))],
        }
    )
    check(p, instancias)
