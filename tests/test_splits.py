"""Tests for the spatial partition. The whole point is that nothing leaks."""

import numpy as np
import pandas as pd
import pytest

from satinsight.splits import CONJUNTOS, assign, check, ciudades_de


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
    assert set(p.conjunto) == set(CONJUNTOS)


def test_the_split_is_eighty_ten_ten():
    p = assign(catalogo(n=138))
    reparto = p.conjunto.value_counts(normalize=True)
    assert reparto["train"] == pytest.approx(0.8, abs=0.02)
    assert reparto["val"] == pytest.approx(0.1, abs=0.02)
    assert reparto["test"] == pytest.approx(0.1, abs=0.02)


def test_other_proportions_are_honoured():
    p = assign(catalogo(n=100), proporciones=(0.6, 0.2, 0.2))
    reparto = p.conjunto.value_counts(normalize=True)
    assert reparto["train"] == pytest.approx(0.6, abs=0.02)


def test_proportions_must_add_up():
    with pytest.raises(ValueError, match="add up to one"):
        assign(catalogo(), proporciones=(0.8, 0.2, 0.2))


def test_the_partition_is_reproducible():
    a, b = assign(catalogo()), assign(catalogo())
    pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_moves_cities_around():
    a = assign(catalogo(), seed=1).set_index("ciudad").conjunto
    b = assign(catalogo(), seed=2).set_index("ciudad").conjunto
    assert (a != b).any()


def test_the_three_sets_never_share_a_city():
    p = assign(catalogo())
    listas = {c: set(ciudades_de(p, c)) for c in CONJUNTOS}
    assert not listas["train"] & listas["val"]
    assert not listas["train"] & listas["test"]
    assert not listas["val"] & listas["test"]


def test_deprivation_is_balanced_across_the_three():
    p = assign(catalogo(n=138))
    medias = p.groupby("conjunto").estrato_valor.mean()
    assert medias.max() - medias.min() < 0.08


def test_size_is_balanced_across_the_three():
    p = assign(catalogo(n=138))
    medias = p.groupby("conjunto").tamano.mean()
    assert medias.max() / medias.min() < 1.5


def test_assign_refuses_a_catalogue_too_small():
    with pytest.raises(ValueError, match="cannot fill"):
        assign(catalogo(n=2))


def test_assign_demands_its_columns():
    with pytest.raises(KeyError, match="altos"):
        assign(catalogo().drop(columns=["altos"]))


def test_ciudades_de_rejects_an_unknown_set():
    with pytest.raises(KeyError, match="unknown set"):
        ciudades_de(assign(catalogo(n=20)), "entrenamiento")


def test_check_catches_an_ageb_on_both_sides():
    p = assign(catalogo(n=20))
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
    p = assign(catalogo(n=20))
    with pytest.raises(ValueError, match="outside the partition"):
        check(p, pd.DataFrame({"ciudad": ["fantasma"], "cvegeo": ["x"], "municipio": ["y"]}))


def test_check_passes_on_a_clean_partition():
    p = assign(catalogo(n=20))
    instancias = pd.DataFrame(
        {
            "ciudad": p.ciudad,
            # cada ciudad con municipio propio: una conurbación pertenece a una sola
            # ciudad, y compartirlo entre dos sería precisamente la fuga que se busca
            "municipio": [f"07{i:03d}" for i in range(len(p))],
            "cvegeo": [f"071010001{i:04d}" for i in range(len(p))],
        }
    )
    check(p, instancias)
