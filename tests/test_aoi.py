import pytest

from satinsight.aoi import AOI, PILOTO, obtener


def test_todos_los_pilotos_son_validos():
    for clave, area in PILOTO.items():
        assert area.clave == clave
        assert area.ancho_grados > 0
        assert area.alto_grados > 0


def test_los_pilotos_comparten_tamano():
    anchos = {round(a.ancho_grados, 6) for a in PILOTO.values()}
    altos = {round(a.alto_grados, 6) for a in PILOTO.values()}
    assert len(anchos) == 1, "los recuadros piloto deben ser comparables entre sí"
    assert len(altos) == 1


def test_forma_aproximada_es_plausible():
    alto, ancho = PILOTO["tuxtla"].forma_aproximada(resolucion_m=10)
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
        AOI(clave="x", nombre="x", entidad="x", bbox=bbox)


def test_obtener_desconocido_sugiere_disponibles():
    with pytest.raises(KeyError, match="tuxtla"):
        obtener("saltillo")
