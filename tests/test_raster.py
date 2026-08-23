import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from satinsight.raster import percentiles, read_window, stretch, to_db


def test_estirar_devuelve_rango_completo():
    banda = np.arange(1, 101, dtype="float32").reshape(10, 10)
    salida = stretch(banda, lower=0, upper=100)
    assert salida.dtype == np.uint8
    assert salida.min() == 0
    assert salida.max() == 255


def test_estirar_sin_pixeles_validos_devuelve_ceros():
    salida = stretch(np.zeros((4, 4), dtype="float32"))
    assert salida.shape == (4, 4)
    assert not salida.any()


def test_estirar_trata_el_cero_como_sin_dato():
    banda = np.array([[0, 0], [10, 20]], dtype="float32")
    salida = stretch(banda, lower=0, upper=100)
    # el mínimo del estiramiento sale de 10, no del relleno
    assert salida[1, 0] == 0
    assert salida[1, 1] == 255


def test_estirar_con_banda_constante_no_divide_entre_cero():
    salida = stretch(np.full((3, 3), 7.0, dtype="float32"))
    assert np.isfinite(salida).all()


def test_a_db_convierte_potencia_conocida():
    potencia = np.array([[1.0, 0.1, 10.0]], dtype="float32")
    resultado = to_db(potencia)
    np.testing.assert_allclose(resultado, [[0.0, -10.0, 10.0]], atol=1e-4)


def test_a_db_marca_los_no_positivos_como_nan():
    resultado = to_db(np.array([[0.0, -1.0, 1.0]], dtype="float32"))
    assert np.isnan(resultado[0, 0])
    assert np.isnan(resultado[0, 1])
    assert resultado[0, 2] == 0.0


def test_percentiles_ignora_nan():
    banda = np.array([[np.nan, 1.0, 2.0, 3.0, 4.0]], dtype="float32")
    inferior, superior = percentiles(banda, 0, 100)
    assert inferior == 1.0
    assert superior == 4.0


def test_el_centinela_de_la_escena_se_vuelve_nan(tmp_path):
    """El valor de «sin dato» de un ráster de punto flotante no puede entrar a la mediana.

    Sentinel-1 RTC declara -32768 y lo escribe fuera de la franja y en la sombra del radar.
    Leído como número hunde el compuesto de las cities que asoman del borde de la escena.
    """
    ruta = tmp_path / "sar.tif"
    datos = np.full((8, 8), 0.25, dtype="float32")
    datos[:4] = -32768.0
    perfil = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(4326),
        "transform": from_origin(-93.14, 16.77, 0.005, 0.005),
        "nodata": -32768.0,
    }
    with rasterio.open(ruta, "w", **perfil) as destino:
        destino.write(datos, 1)

    leida = read_window(str(ruta), (-93.14, 16.73, -93.10, 16.77), (8, 8))
    assert np.isnan(leida[:4]).all()
    assert np.allclose(leida[4:], 0.25)


def test_un_raster_entero_conserva_el_relleno_en_cero(tmp_path):
    """Sentinel-2 y WorldCover llegan en enteros, que no admiten NaN."""
    ruta = tmp_path / "optico.tif"
    perfil = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "uint16",
        "crs": CRS.from_epsg(4326),
        "transform": from_origin(-93.14, 16.77, 0.005, 0.005),
    }
    with rasterio.open(ruta, "w", **perfil) as destino:
        destino.write(np.full((8, 8), 1200, dtype="uint16"), 1)

    leida = read_window(str(ruta), (-93.20, 16.73, -93.10, 16.77), (8, 16))
    assert leida.dtype == np.uint16
    assert (leida == 0).any()
