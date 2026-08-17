"""Pruebas de la capa de AGEB que no tocan disco ni red."""

import pandas as pd
import pytest

from satinsight.agebs import CIUDADES, GRADOS, INDICADORES, ORDINAL, resumen_grados


def tabla(grados, poblaciones=None):
    poblaciones = poblaciones or [100] * len(grados)
    return pd.DataFrame(
        {
            "cvegeo": [f"{i:013d}" for i in range(len(grados))],
            "grado": grados,
            "poblacion": poblaciones,
        }
    )


def test_el_orden_ordinal_es_de_menor_a_mayor_rezago():
    assert ORDINAL["Muy bajo"] == 0
    assert ORDINAL["Muy alto"] == len(GRADOS) - 1
    assert list(ORDINAL.values()) == sorted(ORDINAL.values())


def test_hay_diecisiete_indicadores_sin_repetir():
    assert len(INDICADORES) == 17
    assert len(set(INDICADORES)) == 17


def test_cada_ciudad_declara_un_municipio_de_su_entidad():
    for clave, ciudad in CIUDADES.items():
        assert ciudad.clave == clave
        assert len(ciudad.entidad) == 2
        assert len(ciudad.municipio) == 5
        assert ciudad.municipio.startswith(ciudad.entidad)


def test_el_resumen_cuenta_todas_las_clases_aunque_falten():
    resumen = resumen_grados(tabla(["Bajo", "Bajo", "Alto"]))
    assert list(resumen.index) == list(GRADOS)
    assert resumen.loc["Bajo", "agebs"] == 2
    assert resumen.loc["Alto", "agebs"] == 1
    assert resumen.loc["Muy alto", "agebs"] == 0


def test_el_resumen_suma_poblacion_por_clase():
    resumen = resumen_grados(tabla(["Bajo", "Bajo", "Alto"], [10, 20, 5]))
    assert resumen.loc["Bajo", "poblacion"] == 30
    assert resumen.loc["Alto", "poblacion"] == 5


def test_los_porcentajes_suman_cien():
    resumen = resumen_grados(tabla(["Muy bajo", "Bajo", "Medio", "Alto"]))
    assert resumen["pct_agebs"].sum() == pytest.approx(100.0, abs=0.2)


def test_una_tabla_vacia_no_divide_entre_cero():
    resumen = resumen_grados(tabla([]))
    assert resumen["agebs"].sum() == 0
    assert resumen["pct_agebs"].sum() == 0
