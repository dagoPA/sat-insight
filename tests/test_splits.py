"""Tests for the spatial partition. The whole point is that nothing leaks."""

import numpy as np
import pandas as pd
import pytest

from satinsight.splits import SETS, assign, check, cities_of


def catalogue(n=138, seed=0):
    azar = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "clave": [f"ciudad{i:03d}" for i in range(n)],
            "agebs": azar.integers(150, 1200, n),
            "altos": azar.random(n) * 0.5,
        }
    )


def test_every_city_lands_in_exactly_one_place():
    p = assign(catalogue())
    assert len(p) == 138
    assert p.ciudad.is_unique
    assert set(p.split) == set(SETS)


def test_the_split_is_eighty_ten_ten():
    p = assign(catalogue(n=138))
    reparto = p.split.value_counts(normalize=True)
    assert reparto["train"] == pytest.approx(0.8, abs=0.02)
    assert reparto["val"] == pytest.approx(0.1, abs=0.02)
    assert reparto["test"] == pytest.approx(0.1, abs=0.02)


def test_other_proportions_are_honoured():
    p = assign(catalogue(n=100), proportions=(0.6, 0.2, 0.2))
    reparto = p.split.value_counts(normalize=True)
    assert reparto["train"] == pytest.approx(0.6, abs=0.02)


def test_proportions_must_add_up():
    with pytest.raises(ValueError, match="add up to one"):
        assign(catalogue(), proportions=(0.8, 0.2, 0.2))


def test_the_partition_is_reproducible():
    a, b = assign(catalogue()), assign(catalogue())
    pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_moves_cities_around():
    a = assign(catalogue(), seed=1).set_index("ciudad").split
    b = assign(catalogue(), seed=2).set_index("ciudad").split
    assert (a != b).any()


def test_the_three_sets_never_share_a_city():
    p = assign(catalogue())
    listas = {c: set(cities_of(p, c)) for c in SETS}
    assert not listas["train"] & listas["val"]
    assert not listas["train"] & listas["test"]
    assert not listas["val"] & listas["test"]


def test_deprivation_is_balanced_across_the_three():
    p = assign(catalogue(n=138))
    medias = p.groupby("split").stratum_value.mean()
    assert medias.max() - medias.min() < 0.08


def test_size_is_balanced_across_the_three():
    p = assign(catalogue(n=138))
    medias = p.groupby("split").n_agebs.mean()
    assert medias.max() / medias.min() < 1.5


def test_assign_refuses_a_catalogue_too_small():
    with pytest.raises(ValueError, match="cannot fill"):
        assign(catalogue(n=2))


def test_assign_demands_its_columns():
    with pytest.raises(KeyError, match="altos"):
        assign(catalogue().drop(columns=["altos"]))


def test_ciudades_de_rejects_an_unknown_set():
    with pytest.raises(KeyError, match="unknown set"):
        cities_of(assign(catalogue(n=20)), "entrenamiento")


def test_check_catches_an_ageb_on_both_sides():
    p = assign(catalogue(n=20))
    dos = list(p.ciudad[:2])
    instancias = pd.DataFrame(
        {
            "ciudad": dos,
            "cvegeo": ["0710100010001", "0710100010001"],
            "municipio": ["07101", "07101"],
        }
    )
    p.loc[p.ciudad == dos[0], "split"] = "train"
    p.loc[p.ciudad == dos[1], "split"] = "test"
    with pytest.raises(ValueError, match="spanning both sides"):
        check(p, instancias)


def test_check_catches_an_instance_from_an_unknown_city():
    p = assign(catalogue(n=20))
    with pytest.raises(ValueError, match="outside the partition"):
        check(p, pd.DataFrame({"ciudad": ["fantasma"], "cvegeo": ["x"], "municipio": ["y"]}))


def test_check_passes_on_a_clean_partition():
    p = assign(catalogue(n=20))
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


def _tabla_conurbada():
    """Dos cities vecinas que comparten AGEB, como Guadalajara y Zapopan."""
    return pd.DataFrame(
        {
            "cvegeo": ["1403900010001", "1403900010002", "1412000010001", "1403900010001"],
            "ciudad": ["guadalajara", "guadalajara", "zapopan", "zapopan"],
            "valor": [1.0, 2.0, 3.0, 4.0],
        }
    )


class _City:
    def __init__(self, municipality):
        self.municipality = municipality


def test_una_ageb_compartida_queda_bajo_una_sola_ciudad():
    from satinsight.splits import deduplicate

    catalogue = {"guadalajara": _City("14039"), "zapopan": _City("14120")}
    d = deduplicate(_tabla_conurbada(), catalogue)
    assert len(d) == 3
    assert d.cvegeo.is_unique
    assert d.loc[d.cvegeo == "1403900010001", "ciudad"].item() == "guadalajara"


def test_un_municipio_que_ninguna_ciudad_reclama_va_a_la_que_mas_tiene():
    from satinsight.splits import municipality_owner

    tabla = pd.DataFrame(
        {
            "cvegeo": ["1409800010001", "1409800010002", "1409800010003"],
            "ciudad": ["guadalajara", "guadalajara", "zapopan"],
        }
    )
    catalogue = {"guadalajara": _City("14039"), "zapopan": _City("14120")}
    assert municipality_owner(tabla, catalogue)["14098"] == "guadalajara"


def test_desduplicar_deja_la_particion_sin_fuga():
    from satinsight.splits import deduplicate

    catalogue = {"guadalajara": _City("14039"), "zapopan": _City("14120")}
    d = deduplicate(_tabla_conurbada(), catalogue)
    partition = pd.DataFrame({"ciudad": ["guadalajara", "zapopan"], "split": ["train", "val"]})
    d["municipio"] = d.cvegeo.str[:5]
    check(partition, d[["ciudad", "cvegeo", "municipio"]])
