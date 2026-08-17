"""Pruebas de la retícula de análisis. Sin red: los items del catálogo son dobles."""

from dataclasses import dataclass, field

import pytest

from satinsight.malla import malla_de_bbox, malla_de_escenas


@dataclass
class ItemFalso:
    """Doble de un item del STAC, del que solo interesan sus propiedades."""

    properties: dict = field(default_factory=dict)


def item(epsg=32615, clave="proj:epsg"):
    return ItemFalso(properties={clave: epsg})


BBOX_TUXTLA = (-93.135, 16.740, -93.095, 16.768)


def test_la_malla_cubre_el_recuadro_completo():
    malla = malla_de_bbox(BBOX_TUXTLA, "EPSG:32615", resolucion_m=10)
    izq, abajo, der, arriba = malla.limites
    assert malla.ancho * 10 >= der - izq
    assert malla.alto * 10 >= arriba - abajo


def test_el_tamano_de_pixel_es_el_pedido():
    malla = malla_de_bbox(BBOX_TUXTLA, "EPSG:32615", resolucion_m=10)
    assert malla.transform.a == pytest.approx(10, abs=0.5)
    assert malla.transform.e == pytest.approx(-10, abs=0.5)


def test_bajar_la_resolucion_reduce_los_pixeles():
    fina = malla_de_bbox(BBOX_TUXTLA, "EPSG:32615", resolucion_m=10)
    burda = malla_de_bbox(BBOX_TUXTLA, "EPSG:32615", resolucion_m=20)
    assert burda.ancho < fina.ancho
    assert burda.megapixeles < fina.megapixeles


def test_la_esquina_superior_izquierda_coincide_con_los_limites():
    malla = malla_de_bbox(BBOX_TUXTLA, "EPSG:32615")
    assert malla.transform.c == pytest.approx(malla.limites[0])
    assert malla.transform.f == pytest.approx(malla.limites[3])


def test_crs_unico_se_acepta():
    malla = malla_de_escenas(BBOX_TUXTLA, [item(), item(), item()])
    assert malla.crs == "EPSG:32615"


def test_crs_ya_prefijado_no_se_duplica():
    malla = malla_de_escenas(BBOX_TUXTLA, [item(epsg="EPSG:32615")])
    assert malla.crs == "EPSG:32615"


def test_se_lee_tambien_la_clave_moderna():
    malla = malla_de_escenas(BBOX_TUXTLA, [item(epsg="EPSG:32616", clave="proj:code")])
    assert malla.crs == "EPSG:32616"


def test_husos_mezclados_fallan_con_mensaje_util():
    escenas = [item(32615), item(32615), item(32616)]
    with pytest.raises(ValueError, match="huso UTM"):
        malla_de_escenas(BBOX_TUXTLA, escenas)


def test_sin_proyeccion_declarada_falla():
    with pytest.raises(ValueError, match="sistema de referencia"):
        malla_de_escenas(BBOX_TUXTLA, [ItemFalso(properties={})])
