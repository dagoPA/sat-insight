"""Pruebas de la capa de AGEB que no tocan disco ni red."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from satinsight.agebs import (
    CIUDADES,
    CRS_METRICO,
    GRADOS,
    INDICADORES,
    ORDINAL,
    _mancha_conectada,
    resumen_grados,
)


def cuadros(especificaciones):
    """GeoDataFrame métrico de AGEB cuadradas de 1 km, dadas como (clave, x_km, y_km)."""
    return gpd.GeoDataFrame(
        {"cvegeo": [c for c, _, _ in especificaciones]},
        geometry=[
            box(x * 1000, y * 1000, x * 1000 + 1000, y * 1000 + 1000)
            for _, x, y in especificaciones
        ],
        crs=CRS_METRICO,
    )


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


def test_la_mancha_descarta_el_satelite_lejano():
    agebs = cuadros(
        [
            ("0710100010001", 0, 0),
            ("0710100010002", 1, 0),
            ("0799900010001", 40, 0),  # localidad desprendida a cuarenta kilómetros
        ]
    )
    mancha = _mancha_conectada(agebs, "07101", pegado_m=2500)
    assert set(mancha["cvegeo"]) == {"0710100010001", "0710100010002"}


def test_la_mancha_conserva_al_vecino_pegado():
    agebs = cuadros(
        [
            ("0710100010001", 0, 0),
            ("0710200010001", 2, 0),  # otro municipio, pero conurbado
        ]
    )
    mancha = _mancha_conectada(agebs, "07101", pegado_m=2500)
    assert len(mancha) == 2


def test_una_mancha_unica_pasa_intacta():
    agebs = cuadros([("0710100010001", 0, 0), ("0710100010002", 1, 0)])
    assert len(_mancha_conectada(agebs, "07101", pegado_m=2500)) == 2


def test_se_elige_la_mancha_con_mas_agebs_del_nucleo():
    agebs = cuadros(
        [
            ("0799900010001", 0, 0),
            ("0799900010002", 1, 0),
            ("0799900010003", 2, 0),
            ("0710100010001", 40, 0),
            ("0710100010002", 41, 0),
        ]
    )
    mancha = _mancha_conectada(agebs, "07101", pegado_m=2500)
    assert set(mancha["cvegeo"]) == {"0710100010001", "0710100010002"}


def test_un_pegado_generoso_une_lo_que_uno_estricto_separa():
    agebs = cuadros([("0710100010001", 0, 0), ("0710100010002", 10, 0)])
    assert len(_mancha_conectada(agebs, "07101", pegado_m=1000)) == 1
    assert len(_mancha_conectada(agebs, "07101", pegado_m=6000)) == 2
