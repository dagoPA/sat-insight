"""Pruebas de los rasgos de textura, sobre imágenes sintéticas de propiedades conocidas."""

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from satinsight.textura import (
    DISTANCIAS,
    NIVELES,
    cuantizar,
    rango_robusto,
    rasgos_de_recorte,
    rasgos_por_ageb,
    rasgos_primer_orden,
)

TRANSFORM = Affine.translation(0.0, 2000.0) * Affine.scale(10.0, -10.0)
"""Retícula métrica de 10 m con origen arriba a la izquierda, como la de un compuesto."""


def imagen_con_parche(semilla=0):
    """Fondo liso con un parche central de mucha varianza."""
    rng = np.random.default_rng(semilla)
    banda = rng.normal(0.2, 0.01, (200, 200))
    banda[50:150, 50:150] += rng.normal(0, 0.15, (100, 100))
    return banda


def caja_de_pixeles(fila_ini, fila_fin, col_ini, col_fin):
    """Traduce un rango de filas y columnas al polígono que lo cubre."""
    x_min, y_max = TRANSFORM * (col_ini, fila_ini)
    x_max, y_min = TRANSFORM * (col_fin, fila_fin)
    return box(x_min, y_min, x_max, y_max)


def test_cuantizar_respeta_el_rango_reservando_el_cero():
    banda = np.linspace(0.0, 1.0, 100).reshape(10, 10)
    q = cuantizar(banda, (0.0, 1.0), NIVELES)
    assert q.min() == 1
    assert q.max() == NIVELES


def test_los_no_finitos_quedan_en_cero():
    banda = np.full((4, 4), 0.5)
    banda[0, 0] = np.nan
    q = cuantizar(banda, (0.0, 1.0))
    assert q[0, 0] == 0
    assert (q[1:] > 0).all()


def test_los_valores_fuera_de_rango_se_recortan_sin_perderse():
    banda = np.array([[-5.0, 0.5, 9.0]])
    q = cuantizar(banda, (0.0, 1.0))
    assert q[0, 0] == 1
    assert q[0, 2] == NIVELES


def test_rango_degenerado_falla():
    with pytest.raises(ValueError, match="degenerado"):
        cuantizar(np.ones((3, 3)), (1.0, 1.0))


def test_rango_robusto_ignora_los_no_finitos():
    banda = np.array([[np.nan, 1.0, 2.0, 3.0, np.inf]])
    bajo, alto = rango_robusto(banda, 0, 100)
    assert bajo == pytest.approx(1.0)
    assert alto == pytest.approx(3.0)


def test_banda_sin_pixeles_finitos_falla():
    with pytest.raises(ValueError, match="finito"):
        rango_robusto(np.full((3, 3), np.nan))


def test_una_region_texturada_contrasta_mas_que_una_lisa():
    banda = imagen_con_parche()
    lisa = caja_de_pixeles(5, 45, 5, 45)  # 1600 px, fuera del parche
    aspera = caja_de_pixeles(60, 100, 60, 100)  # 1600 px, dentro del parche
    tabla = rasgos_por_ageb(banda, TRANSFORM, [lisa, aspera], ["lisa", "aspera"], prefijo="c")
    fila = tabla.set_index("cvegeo")

    assert fila.loc["aspera", "c_contrast_d1"] > 5 * fila.loc["lisa", "c_contrast_d1"]
    assert fila.loc["aspera", "c_homogeneity_d1"] < fila.loc["lisa", "c_homogeneity_d1"]
    assert fila.loc["aspera", "c_entropia_d1"] > fila.loc["lisa", "c_entropia_d1"]


def test_cada_distancia_sale_como_columna_propia():
    banda = imagen_con_parche()
    aspera = caja_de_pixeles(60, 100, 60, 100)
    tabla = rasgos_por_ageb(banda, TRANSFORM, [aspera], ["x"], prefijo="c")
    for distancia in DISTANCIAS:
        assert f"c_contrast_d{distancia}" in tabla.columns
    # a mayor separación, mayor contraste en una superficie sin estructura periódica
    assert tabla.loc[0, "c_contrast_d4"] > tabla.loc[0, "c_contrast_d1"]


def test_un_rango_fijo_no_depende_de_la_banda():
    """Dos bandas con distinto nivel deben cuantizarse igual si el rango se fija.

    Es lo que sostiene la comparabilidad del radar entre ciudades y entre países.
    """
    rng = np.random.default_rng(1)
    patron = rng.normal(0, 1, (60, 60))
    caja = caja_de_pixeles(0, 60, 0, 60)
    fijo = (-10.0, 10.0)

    a = rasgos_por_ageb(patron, TRANSFORM, [caja], ["x"], prefijo="c", rango=fijo)
    b = rasgos_por_ageb(patron + 3.0, TRANSFORM, [caja], ["x"], prefijo="c", rango=fijo)
    libre_a = rasgos_por_ageb(patron, TRANSFORM, [caja], ["x"], prefijo="c")
    libre_b = rasgos_por_ageb(patron + 3.0, TRANSFORM, [caja], ["x"], prefijo="c")

    assert a.loc[0, "c_contrast_d1"] != pytest.approx(b.loc[0, "c_contrast_d1"])
    assert libre_a.loc[0, "c_contrast_d1"] == pytest.approx(libre_b.loc[0, "c_contrast_d1"])


def test_un_poligono_diminuto_se_reporta_sin_rasgos():
    banda = imagen_con_parche()
    minusculo = caja_de_pixeles(10, 12, 10, 12)
    tabla = rasgos_por_ageb(banda, TRANSFORM, [minusculo], ["x"], prefijo="c", minimo_pixeles=50)
    assert tabla.loc[0, "c_n_px"] < 50
    assert "c_contrast_d1" not in tabla.columns


def test_un_poligono_fuera_del_raster_no_rompe():
    banda = imagen_con_parche()
    lejos = box(500_000, 500_000, 500_100, 500_100)
    tabla = rasgos_por_ageb(banda, TRANSFORM, [lejos], ["fuera"], prefijo="c")
    assert tabla.loc[0, "c_n_px"] == 0


def test_geometrias_y_claves_desparejas_fallan():
    banda = imagen_con_parche()
    with pytest.raises(ValueError, match="geometrías"):
        rasgos_por_ageb(banda, TRANSFORM, [caja_de_pixeles(0, 10, 0, 10)], ["a", "b"], prefijo="c")


def test_un_recorte_constante_no_tiene_entropia():
    constante = np.full((30, 30), 5, dtype=np.uint8)
    rasgos = rasgos_de_recorte(constante)
    assert rasgos["entropia_d1"] == pytest.approx(0.0, abs=1e-9)
    assert rasgos["contrast_d1"] == pytest.approx(0.0, abs=1e-9)


def test_un_recorte_vacio_devuelve_nan():
    rasgos = rasgos_de_recorte(np.zeros((20, 20), dtype=np.uint8))
    assert np.isnan(rasgos["contrast_d1"])


def test_primer_orden_reproduce_la_media_y_la_dispersion():
    valores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rasgos = rasgos_primer_orden(valores)
    assert rasgos["media"] == pytest.approx(3.0)
    assert rasgos["p50"] == pytest.approx(3.0)
    assert rasgos["desv"] == pytest.approx(np.std(valores))


def test_primer_orden_sin_datos_devuelve_nan():
    rasgos = rasgos_primer_orden(np.array([]))
    assert all(np.isnan(v) for v in rasgos.values())
