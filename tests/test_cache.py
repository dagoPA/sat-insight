"""Pruebas del cacheo de compuestos. Escriben en tmp_path, nunca en la red."""

import numpy as np
import pytest

from satinsight.cache import composite_path, exists, load, save
from satinsight.grid import grid_from_bbox

BBOX = (-93.135, 16.740, -93.095, 16.768)


@pytest.fixture
def malla():
    return grid_from_bbox(BBOX, "EPSG:32615", resolution_m=100)


def bandas_de(malla, nombres=("vv", "vh")):
    rng = np.random.default_rng(0)
    return {n: rng.random(malla.shape).astype("float32") for n in nombres}


def test_ida_y_vuelta_conserva_los_valores(tmp_path, malla):
    original = bandas_de(malla)
    destino = save(original, malla, tmp_path / "x.tif")
    recuperadas, _, _ = load(destino)

    assert set(recuperadas) == set(original)
    for nombre, arreglo in original.items():
        np.testing.assert_allclose(recuperadas[nombre], arreglo, rtol=1e-6)


def test_ida_y_vuelta_conserva_la_georreferencia(tmp_path, malla):
    save(bandas_de(malla), malla, tmp_path / "x.tif")
    _, recuperada, _ = load(tmp_path / "x.tif")

    assert recuperada.shape == malla.shape
    assert recuperada.crs == malla.crs
    assert recuperada.transform == pytest.approx(malla.transform, abs=1e-6)


def test_el_orden_de_las_bandas_se_conserva(tmp_path, malla):
    original = bandas_de(malla, ("B04", "B03", "B02", "B08"))
    save(original, malla, tmp_path / "x.tif")
    recuperadas, _, _ = load(tmp_path / "x.tif")
    assert list(recuperadas) == list(original)


def test_las_etiquetas_sobreviven(tmp_path, malla):
    save(bandas_de(malla), malla, tmp_path / "x.tif", scenes_used=17, orbit="ascendente · 99")
    _, _, etiquetas = load(tmp_path / "x.tif")
    assert etiquetas["scenes_used"] == 17
    assert etiquetas["orbit"] == "ascendente · 99"


def test_los_nan_sobreviven(tmp_path, malla):
    bandas = bandas_de(malla, ("vv",))
    bandas["vv"][0, 0] = np.nan
    save(bandas, malla, tmp_path / "x.tif")
    recuperadas, _, _ = load(tmp_path / "x.tif")
    assert np.isnan(recuperadas["vv"][0, 0])


def test_guardar_sin_bandas_falla(tmp_path, malla):
    with pytest.raises(ValueError, match="no bands to save"):
        save({}, malla, tmp_path / "x.tif")


def test_bandas_de_formas_distintas_fallan(tmp_path, malla):
    bandas = bandas_de(malla, ("vv",))
    bandas["vh"] = np.zeros((3, 3), dtype="float32")
    with pytest.raises(ValueError, match="do not share a shape"):
        save(bandas, malla, tmp_path / "x.tif")


def test_forma_que_no_coincide_con_la_malla_falla(tmp_path, malla):
    bandas = {"vv": np.zeros((5, 5), dtype="float32")}
    with pytest.raises(ValueError, match="does not match the grid"):
        save(bandas, malla, tmp_path / "x.tif")


def test_la_ruta_distingue_ciudad_y_sensor(tmp_path):
    a = composite_path("tuxtla", "s1", tmp_path)
    b = composite_path("tuxtla", "s2", tmp_path)
    assert a != b
    assert a.suffix == ".tif"


def test_existe_reporta_ausencia(tmp_path):
    assert not exists("merida", "s1", tmp_path)
