"""Pruebas de la retícula de análisis. Sin red: los items del catálogo son dobles."""

from dataclasses import dataclass, field

import pytest

from satinsight.grid import grid_from_bbox, grid_from_scenes, select_crs


@dataclass
class ItemFalso:
    """Doble de un item del STAC, del que solo interesan sus propiedades."""

    properties: dict = field(default_factory=dict)


def item(epsg=32615, clave="proj:epsg"):
    return ItemFalso(properties={clave: epsg})


BBOX_TUXTLA = (-93.135, 16.740, -93.095, 16.768)


def test_la_malla_cubre_el_recuadro_completo():
    malla = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615", resolution_m=10)
    izq, abajo, der, arriba = malla.bounds
    assert malla.width * 10 >= der - izq
    assert malla.height * 10 >= arriba - abajo


def test_el_tamano_de_pixel_es_el_pedido():
    malla = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615", resolution_m=10)
    assert malla.transform.a == pytest.approx(10, abs=0.5)
    assert malla.transform.e == pytest.approx(-10, abs=0.5)


def test_bajar_la_resolucion_reduce_los_pixeles():
    fina = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615", resolution_m=10)
    burda = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615", resolution_m=20)
    assert burda.width < fina.width
    assert burda.megapixels < fina.megapixels


def test_la_esquina_superior_izquierda_coincide_con_los_limites():
    malla = grid_from_bbox(BBOX_TUXTLA, "EPSG:32615")
    assert malla.transform.c == pytest.approx(malla.bounds[0])
    assert malla.transform.f == pytest.approx(malla.bounds[3])


def test_crs_unico_se_acepta():
    malla, escenas = grid_from_scenes(BBOX_TUXTLA, [item(), item(), item()])
    assert malla.crs == "EPSG:32615"
    assert len(escenas) == 3


def test_crs_ya_prefijado_no_se_duplica():
    malla, _ = grid_from_scenes(BBOX_TUXTLA, [item(epsg="EPSG:32615")])
    assert malla.crs == "EPSG:32615"


def test_se_lee_tambien_la_clave_moderna():
    malla, _ = grid_from_scenes(BBOX_TUXTLA, [item(epsg="EPSG:32616", clave="proj:code")])
    assert malla.crs == "EPSG:32616"


def test_con_husos_mezclados_gana_la_mayoria():
    escenas = [item(32615), item(32615), item(32616)]
    crs, seleccionadas = select_crs(escenas)
    assert crs == "EPSG:32615"
    assert len(seleccionadas) == 2


def test_las_escenas_del_huso_descartado_no_se_devuelven():
    escenas = [item(32616), item(32615), item(32615), item(32615)]
    malla, seleccionadas = grid_from_scenes(BBOX_TUXTLA, escenas)
    assert malla.crs == "EPSG:32615"
    assert len(seleccionadas) == 3
    assert all(s.properties["proj:epsg"] == 32615 for s in seleccionadas)


def test_las_escenas_sin_proyeccion_se_ignoran_si_hay_otras():
    escenas = [ItemFalso(properties={}), item(32615)]
    crs, seleccionadas = select_crs(escenas)
    assert crs == "EPSG:32615"
    assert len(seleccionadas) == 1


def test_sin_proyeccion_declarada_falla():
    with pytest.raises(ValueError, match="reference system"):
        grid_from_scenes(BBOX_TUXTLA, [ItemFalso(properties={})])


def test_el_huso_se_elige_por_cobertura_y_no_por_numero():
    """El caso de Guasave: el huso más numeroso ve la mitad del recuadro.

    Ochenta y siete escenas del huso 13 alcanzan la mitad de la ciudad y sesenta y una
    del huso 12 la cubren entera. Elegir por número dejaba la ciudad sin compuesto.
    """
    muchas = [item("EPSG:32613", clave="proj:code") for _ in range(87)]
    pocas = [item("EPSG:32612", clave="proj:code") for _ in range(61)]
    cobertura = {"EPSG:32613": 0.53, "EPSG:32612": 1.0}

    def puntuar(grupo):
        return cobertura[grupo[0].properties["proj:code"]]

    elegido, seleccionadas = select_crs(muchas + pocas, puntuar)
    assert elegido == "EPSG:32612"
    assert len(seleccionadas) == 61


def test_sin_puntuacion_sigue_mandando_el_numero():
    muchas = [item("EPSG:32613", clave="proj:code") for _ in range(87)]
    pocas = [item("EPSG:32612", clave="proj:code") for _ in range(61)]
    assert select_crs(muchas + pocas)[0] == "EPSG:32613"


def test_a_igual_cobertura_desempata_el_numero():
    muchas = [item("EPSG:32613", clave="proj:code") for _ in range(87)]
    pocas = [item("EPSG:32612", clave="proj:code") for _ in range(61)]
    elegido, _ = select_crs(muchas + pocas, lambda grupo: 1.0)
    assert elegido == "EPSG:32613"
