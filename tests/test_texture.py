"""Pruebas de los rasgos de textura, sobre imágenes sintéticas de propiedades conocidas."""

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from satinsight.texture import (
    DISTANCES,
    LEVELS,
    features_of_patch,
    features_per_ageb,
    first_order_features,
    quantise,
    robust_range,
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
    q = quantise(banda, (0.0, 1.0), LEVELS)
    assert q.min() == 1
    assert q.max() == LEVELS


def test_los_no_finitos_quedan_en_cero():
    banda = np.full((4, 4), 0.5)
    banda[0, 0] = np.nan
    q = quantise(banda, (0.0, 1.0))
    assert q[0, 0] == 0
    assert (q[1:] > 0).all()


def test_los_valores_fuera_de_rango_se_recortan_sin_perderse():
    banda = np.array([[-5.0, 0.5, 9.0]])
    q = quantise(banda, (0.0, 1.0))
    assert q[0, 0] == 1
    assert q[0, 2] == LEVELS


def test_rango_degenerado_falla():
    with pytest.raises(ValueError, match="degenerate range"):
        quantise(np.ones((3, 3)), (1.0, 1.0))


def test_rango_robusto_ignora_los_no_finitos():
    banda = np.array([[np.nan, 1.0, 2.0, 3.0, np.inf]])
    bajo, alto = robust_range(banda, 0, 100)
    assert bajo == pytest.approx(1.0)
    assert alto == pytest.approx(3.0)


def test_banda_sin_pixeles_finitos_falla():
    with pytest.raises(ValueError, match="not one finite pixel"):
        robust_range(np.full((3, 3), np.nan))


def test_una_region_texturada_contrasta_mas_que_una_lisa():
    banda = imagen_con_parche()
    lisa = caja_de_pixeles(5, 45, 5, 45)  # 1600 px, fuera del parche
    aspera = caja_de_pixeles(60, 100, 60, 100)  # 1600 px, dentro del parche
    tabla = features_per_ageb(banda, TRANSFORM, [lisa, aspera], ["lisa", "aspera"], prefix="c")
    fila = tabla.set_index("cvegeo")

    assert fila.loc["aspera", "c_contrast_d1"] > 5 * fila.loc["lisa", "c_contrast_d1"]
    assert fila.loc["aspera", "c_homogeneity_d1"] < fila.loc["lisa", "c_homogeneity_d1"]
    assert fila.loc["aspera", "c_entropy_d1"] > fila.loc["lisa", "c_entropy_d1"]


def test_cada_distancia_sale_como_columna_propia():
    banda = imagen_con_parche()
    aspera = caja_de_pixeles(60, 100, 60, 100)
    tabla = features_per_ageb(banda, TRANSFORM, [aspera], ["x"], prefix="c")
    for distancia in DISTANCES:
        assert f"c_contrast_d{distancia}" in tabla.columns
    # a mayor separación, mayor contraste en una superficie sin estructura periódica
    assert tabla.loc[0, "c_contrast_d4"] > tabla.loc[0, "c_contrast_d1"]


def test_un_rango_fijo_no_depende_de_la_banda():
    """Dos bandas con distinto nivel deben cuantizarse igual si el rango se fija.

    Es lo que sostiene la comparabilidad del radar entre cities y entre países.
    """
    rng = np.random.default_rng(1)
    patron = rng.normal(0, 1, (60, 60))
    caja = caja_de_pixeles(0, 60, 0, 60)
    fijo = (-10.0, 10.0)

    a = features_per_ageb(patron, TRANSFORM, [caja], ["x"], prefix="c", value_range=fijo)
    b = features_per_ageb(patron + 3.0, TRANSFORM, [caja], ["x"], prefix="c", value_range=fijo)
    libre_a = features_per_ageb(patron, TRANSFORM, [caja], ["x"], prefix="c")
    libre_b = features_per_ageb(patron + 3.0, TRANSFORM, [caja], ["x"], prefix="c")

    assert a.loc[0, "c_contrast_d1"] != pytest.approx(b.loc[0, "c_contrast_d1"])
    assert libre_a.loc[0, "c_contrast_d1"] == pytest.approx(libre_b.loc[0, "c_contrast_d1"])


def test_un_poligono_diminuto_se_reporta_sin_rasgos():
    banda = imagen_con_parche()
    minusculo = caja_de_pixeles(10, 12, 10, 12)
    tabla = features_per_ageb(banda, TRANSFORM, [minusculo], ["x"], prefix="c", min_pixels=50)
    assert tabla.loc[0, "c_n_px"] < 50
    assert "c_contrast_d1" not in tabla.columns


def test_un_poligono_fuera_del_raster_no_rompe():
    banda = imagen_con_parche()
    lejos = box(500_000, 500_000, 500_100, 500_100)
    tabla = features_per_ageb(banda, TRANSFORM, [lejos], ["fuera"], prefix="c")
    assert tabla.loc[0, "c_n_px"] == 0


def test_geometrias_y_claves_desparejas_fallan():
    banda = imagen_con_parche()
    with pytest.raises(ValueError, match="geometries"):
        features_per_ageb(banda, TRANSFORM, [caja_de_pixeles(0, 10, 0, 10)], ["a", "b"], prefix="c")


def test_un_recorte_constante_no_tiene_entropia():
    constante = np.full((30, 30), 5, dtype=np.uint8)
    rasgos = features_of_patch(constante)
    assert rasgos["entropy_d1"] == pytest.approx(0.0, abs=1e-9)
    assert rasgos["contrast_d1"] == pytest.approx(0.0, abs=1e-9)


def test_un_recorte_vacio_devuelve_nan():
    rasgos = features_of_patch(np.zeros((20, 20), dtype=np.uint8))
    assert np.isnan(rasgos["contrast_d1"])


def test_primer_orden_reproduce_la_media_y_la_dispersion():
    valores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rasgos = first_order_features(valores)
    assert rasgos["mean"] == pytest.approx(3.0)
    assert rasgos["p50"] == pytest.approx(3.0)
    assert rasgos["std"] == pytest.approx(np.std(valores))


def test_primer_orden_sin_datos_devuelve_nan():
    rasgos = first_order_features(np.array([]))
    assert all(np.isnan(v) for v in rasgos.values())
