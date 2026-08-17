import numpy as np

from satinsight.raster import a_db, estirar, percentiles


def test_estirar_devuelve_rango_completo():
    banda = np.arange(1, 101, dtype="float32").reshape(10, 10)
    salida = estirar(banda, inferior=0, superior=100)
    assert salida.dtype == np.uint8
    assert salida.min() == 0
    assert salida.max() == 255


def test_estirar_sin_pixeles_validos_devuelve_ceros():
    salida = estirar(np.zeros((4, 4), dtype="float32"))
    assert salida.shape == (4, 4)
    assert not salida.any()


def test_estirar_trata_el_cero_como_sin_dato():
    banda = np.array([[0, 0], [10, 20]], dtype="float32")
    salida = estirar(banda, inferior=0, superior=100)
    # el mínimo del estiramiento sale de 10, no del relleno
    assert salida[1, 0] == 0
    assert salida[1, 1] == 255


def test_estirar_con_banda_constante_no_divide_entre_cero():
    salida = estirar(np.full((3, 3), 7.0, dtype="float32"))
    assert np.isfinite(salida).all()


def test_a_db_convierte_potencia_conocida():
    potencia = np.array([[1.0, 0.1, 10.0]], dtype="float32")
    resultado = a_db(potencia)
    np.testing.assert_allclose(resultado, [[0.0, -10.0, 10.0]], atol=1e-4)


def test_a_db_marca_los_no_positivos_como_nan():
    resultado = a_db(np.array([[0.0, -1.0, 1.0]], dtype="float32"))
    assert np.isnan(resultado[0, 0])
    assert np.isnan(resultado[0, 1])
    assert resultado[0, 2] == 0.0


def test_percentiles_ignora_nan():
    banda = np.array([[np.nan, 1.0, 2.0, 3.0, 4.0]], dtype="float32")
    inferior, superior = percentiles(banda, 0, 100)
    assert inferior == 1.0
    assert superior == 4.0
