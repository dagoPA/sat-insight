import pytest

from satinsight.aoi import AOI, PILOT, get


def test_todos_los_pilotos_son_validos():
    for clave, area in PILOT.items():
        assert area.key == clave
        assert area.width_degrees > 0
        assert area.height_degrees > 0


def test_los_pilotos_comparten_tamano():
    anchos = {round(a.width_degrees, 6) for a in PILOT.values()}
    altos = {round(a.height_degrees, 6) for a in PILOT.values()}
    assert len(anchos) == 1, "los recuadros piloto deben ser comparables entre sí"
    assert len(altos) == 1


def test_approximate_shape_es_plausible():
    alto, ancho = PILOT["tuxtla"].approximate_shape(resolution_m=10)
    assert 250 < alto < 350
    assert 350 < ancho < 480


@pytest.mark.parametrize(
    "bbox",
    [
        (-93.0, 16.7, -93.1, 16.8),  # longitudes invertidas
        (-93.1, 16.8, -93.0, 16.7),  # latitudes invertidas
        (-200.0, 16.7, -93.0, 16.8),  # longitud fuera de rango
        (-93.1, -95.0, -93.0, 16.8),  # latitud fuera de rango
    ],
)
def test_bbox_invalido_falla(bbox):
    with pytest.raises(ValueError):
        AOI(key="x", name="x", state="x", bbox=bbox)


def test_obtener_desconocido_sugiere_disponibles():
    with pytest.raises(KeyError, match="tuxtla"):
        get("saltillo")
