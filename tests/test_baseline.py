"""Pruebas del baseline sobre tablas sintéticas con señal conocida."""

import numpy as np
import pandas as pd
import pytest

from satinsight.baseline import columnas_de_conjunto, comparar, evaluar, resumen
from satinsight.cobertura import CLASES
from satinsight.textura import nombres_de_rasgos

CIUDADES = ("tuxtla", "merida", "iztapalapa")


def tabla_sintetica(n_por_ciudad=120, fuerza=1.0, semilla=0):
    """Tabla donde los rasgos llevan señal del ordinal, graduable con `fuerza`.

    Con fuerza alta el modelo debe recuperar el orden; con fuerza cero los rasgos son ruido
    puro y ningún modelo debería superar al azar de manera consistente.
    """
    rng = np.random.default_rng(semilla)
    partes = []
    for ciudad in CIUDADES:
        ordinal = rng.integers(0, 5, n_por_ciudad)
        ruido = rng.normal(0, 1, n_por_ciudad)
        columnas = {
            "cvegeo": [f"{ciudad}{i:04d}" for i in range(n_por_ciudad)],
            "ciudad": ciudad,
            "ordinal": ordinal,
            "c_media": fuerza * ordinal + ruido,
            "c_desv": fuerza * ordinal * 0.5 + ruido,
            "c_p10": rng.normal(0, 1, n_por_ciudad),
            "c_p50": rng.normal(0, 1, n_por_ciudad),
            "c_p90": rng.normal(0, 1, n_por_ciudad),
            "c_rango_intercuartil": rng.normal(0, 1, n_por_ciudad),
        }
        # Los nombres de textura salen del propio módulo, para que renombrar un rasgo
        # rompa la prueba en vez de dejarla midiendo un conjunto vacío en silencio.
        for clase in CLASES.values():
            columnas[f"wc_{clase}"] = rng.random(n_por_ciudad)
        for sufijo in nombres_de_rasgos():
            lleva_senal = sufijo.startswith(("contrast_", "homogeneity_"))
            signo = -1 if sufijo.startswith("homogeneity_") else 1
            señal = signo * fuerza * ordinal if lleva_senal else 0.0
            columnas[f"c_{sufijo}"] = señal + rng.normal(0, 1, n_por_ciudad)
        partes.append(pd.DataFrame(columnas))
    return pd.concat(partes, ignore_index=True)


def test_los_conjuntos_separan_densidad_de_textura():
    tabla = tabla_sintetica(10)
    densidad = columnas_de_conjunto(tabla, "densidad")
    textura = columnas_de_conjunto(tabla, "textura")

    assert "c_media" in densidad
    assert "c_contrast_d1" not in densidad
    assert "c_contrast_d1" in textura
    assert "c_media" not in textura
    assert not set(densidad) & set(textura)


def test_el_conjunto_completo_es_la_union_de_los_tres_escalones():
    tabla = tabla_sintetica(10)
    completo = set(columnas_de_conjunto(tabla, "completo"))
    esperado = set()
    for escalon in ("cobertura", "densidad", "textura"):
        esperado |= set(columnas_de_conjunto(tabla, escalon))
    assert completo == esperado


def test_la_cobertura_no_se_mezcla_con_los_otros_escalones():
    tabla = tabla_sintetica(10)
    cobertura = set(columnas_de_conjunto(tabla, "cobertura"))
    assert "wc_construido" in cobertura
    assert not cobertura & set(columnas_de_conjunto(tabla, "densidad"))
    assert not cobertura & set(columnas_de_conjunto(tabla, "textura"))


def test_conjunto_desconocido_falla():
    with pytest.raises(KeyError, match="conjunto desconocido"):
        columnas_de_conjunto(tabla_sintetica(10), "inventado")


def test_la_validacion_deja_una_ciudad_fuera_por_pliegue():
    detalle = evaluar(tabla_sintetica(60), "completo", "clasificador")
    assert len(detalle) == len(CIUDADES)
    assert set(detalle["ciudad_prueba"]) == set(CIUDADES)
    for _, fila in detalle.iterrows():
        assert fila["n_entrena"] == 120
        assert fila["n_prueba"] == 60


def test_con_senal_el_modelo_le_gana_al_azar():
    tabla = tabla_sintetica(200, fuerza=1.5)
    modelo = evaluar(tabla, "completo", "clasificador")["kappa"].mean()
    azar = evaluar(tabla, "completo", "azar")["kappa"].mean()
    assert modelo > azar + 0.1


def test_sin_senal_el_modelo_no_le_gana_al_azar():
    tabla = tabla_sintetica(200, fuerza=0.0)
    modelo = evaluar(tabla, "completo", "clasificador")["kappa"].mean()
    assert modelo < 0.15


def test_la_moda_tiene_kappa_nulo():
    detalle = evaluar(tabla_sintetica(100), "densidad", "moda")
    assert detalle["kappa"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_el_regresor_predice_dentro_del_rango_ordinal():
    tabla = tabla_sintetica(100, fuerza=2.0)
    detalle = evaluar(tabla, "completo", "regresor")
    assert (detalle["mae_ordinal"] >= 0).all()
    assert (detalle["exactitud"] <= 1).all()


def test_comparar_corre_los_modelos_ciegos_una_sola_vez():
    detalle = comparar(tabla_sintetica(50))
    ciegos = detalle[detalle["modelo"].isin(["azar", "moda"])]
    assert set(ciegos["conjunto"]) == {"ninguno"}
    assert len(ciegos) == 2 * len(CIUDADES)


def test_el_resumen_ordena_por_kappa():
    agregado = resumen(comparar(tabla_sintetica(80, fuerza=1.5)))
    assert list(agregado["kappa"]) == sorted(agregado["kappa"], reverse=True)
    assert {"conjunto", "modelo", "kappa"} <= set(agregado.columns)


def test_una_tabla_sin_rasgos_del_conjunto_falla():
    tabla = pd.DataFrame({"ciudad": ["a"], "ordinal": [1], "otra_cosa": [3.0]})
    with pytest.raises(ValueError, match="conjunto"):
        evaluar(tabla, "textura", "clasificador")
