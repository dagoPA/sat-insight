"""Pruebas del compositing. La lectura remota se sustituye por un doble."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from satinsight import composite
from satinsight.composite import _revisar_fallos, compuesto_s1

BBOX = (-93.135, 16.740, -93.095, 16.768)
FORMA = (8, 8)


@dataclass
class ActivoFalso:
    href: str


@dataclass
class ItemFalso:
    """Doble de una escena SAR, con lo mínimo que mira `orbita_dominante`."""

    id: str
    assets: dict = field(default_factory=dict)
    properties: dict = field(default_factory=dict)


def escena_sar(nombre: str, orbita: int = 99) -> ItemFalso:
    return ItemFalso(
        id=nombre,
        assets={"vv": ActivoFalso(f"{nombre}/vv"), "vh": ActivoFalso(f"{nombre}/vh")},
        properties={"sat:orbit_state": "ascending", "sat:relative_orbit": orbita},
    )


def test_pocos_fallos_no_abortan():
    _revisar_fallos("Sentinel-2", fallidas=2, intentadas=20, fraccion=0.3)


def test_demasiados_fallos_abortan():
    with pytest.raises(RuntimeError, match="fallaron al leerse"):
        _revisar_fallos("Sentinel-2", fallidas=15, intentadas=20, fraccion=0.3)


def test_none_desactiva_la_comprobacion():
    _revisar_fallos("Sentinel-2", fallidas=20, intentadas=20, fraccion=None)


def test_cero_es_el_extremo_estricto_y_no_el_desactivado():
    _revisar_fallos("Sentinel-2", fallidas=0, intentadas=20, fraccion=0.0)
    with pytest.raises(RuntimeError, match="fallaron al leerse"):
        _revisar_fallos("Sentinel-2", fallidas=1, intentadas=20, fraccion=0.0)


def test_sin_escenas_intentadas_no_divide_entre_cero():
    _revisar_fallos("Sentinel-1", fallidas=0, intentadas=0, fraccion=0.3)


def test_una_lectura_rota_no_desincroniza_las_polarizaciones(monkeypatch):
    """Si VH falla, VV no debe quedar apilado por su cuenta.

    Las medianas de una y otra polarización saldrían calculadas sobre conjuntos de escenas
    distintos, y la razón entre ambas dejaría de significar lo que dice significar.
    """

    def leer(href, bbox, forma=None):
        if href == "b/vh":
            raise OSError("lectura rota")
        return np.ones(FORMA, dtype="float32")

    monkeypatch.setattr(composite, "leer_ventana", leer)
    escenas = [escena_sar(nombre) for nombre in "abcde"]  # una sola falla, bajo el umbral
    bandas, meta = compuesto_s1(escenas, BBOX, FORMA)

    assert bandas["vv"].shape == FORMA
    assert meta["escenas_usadas"] == 4


def test_si_fallan_casi_todas_aborta(monkeypatch):
    def leer(href, bbox, forma=None):
        raise OSError("HTTP response code: 403")

    monkeypatch.setattr(composite, "leer_ventana", leer)
    escenas = [escena_sar(nombre) for nombre in "abcde"]
    with pytest.raises(RuntimeError, match="fallaron al leerse"):
        compuesto_s1(escenas, BBOX, FORMA)


def test_sin_escenas_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match="no hay escenas"):
        compuesto_s1([], BBOX, FORMA)


def test_la_orbita_se_elige_por_cobertura_medida(monkeypatch):
    """Una órbita con menos pasadas gana si es la que de verdad ve la ciudad.

    Es el caso de Mexicali: la órbita más repetida del catálogo roza el recuadro por el
    filo de la franja y deja casi todo sin observar.
    """
    escenas = [escena_sar(f"filo{i}", orbita=166) for i in range(5)]
    escenas += [escena_sar(f"plena{i}", orbita=173) for i in range(2)]

    def leer(href, bbox, forma):
        arreglo = np.full(forma, np.nan, dtype="float32")
        if href.startswith("plena"):
            arreglo[:] = 1.0
        else:
            arreglo[0, 0] = 1.0
        return arreglo

    monkeypatch.setattr(composite, "leer_ventana", leer)
    clave, seleccion, cobertura = composite.orbita_util(escenas, BBOX)
    assert clave == ("ascending", 173)
    assert len(seleccion) == 2
    assert cobertura == pytest.approx(1.0)


def test_a_igual_cobertura_gana_la_orbita_con_mas_escenas(monkeypatch):
    escenas = [escena_sar(f"a{i}", orbita=10) for i in range(2)]
    escenas += [escena_sar(f"b{i}", orbita=20) for i in range(6)]
    monkeypatch.setattr(composite, "leer_ventana", lambda h, b, f: np.ones(f, dtype="float32"))
    clave, seleccion, _ = composite.orbita_util(escenas, BBOX)
    assert clave == ("ascending", 20)
    assert len(seleccion) == 6


def test_una_orbita_ilegible_cuenta_como_sin_cobertura(monkeypatch):
    def leer(href, bbox, forma):
        raise OSError("403")

    monkeypatch.setattr(composite, "leer_ventana", leer)
    assert composite.cobertura_util([escena_sar("x")], BBOX) == 0.0


def test_un_compuesto_de_radar_con_ceros_se_rechaza():
    """Gamma0 lineal es positiva: un cero delata al sin-dato colado en la mediana."""
    arreglo = np.full((8, 8), 0.2, dtype="float32")
    arreglo[0, 0] = 0.0
    with pytest.raises(RuntimeError, match="cero o negativo"):
        composite._revisar_compuesto_s1({"vv": arreglo})


def test_un_compuesto_de_radar_con_valor_intermedio_se_rechaza():
    """El caso que no se ve: la mediana promedia -32768 con un valor bueno."""
    arreglo = np.full((8, 8), -16384.0, dtype="float32")
    with pytest.raises(RuntimeError, match="cero o negativo"):
        composite._revisar_compuesto_s1({"vv": arreglo})


def test_un_compuesto_de_radar_mayormente_sin_observar_se_rechaza():
    arreglo = np.full((10, 10), np.nan, dtype="float32")
    arreglo[:5] = 0.2
    with pytest.raises(RuntimeError, match="ninguna órbita cubre"):
        composite._revisar_compuesto_s1({"vv": arreglo})


def test_un_compuesto_de_radar_sano_pasa():
    arreglo = np.full((10, 10), 0.2, dtype="float32")
    arreglo[0] = np.nan
    assert composite._revisar_compuesto_s1({"vv": arreglo, "vh": arreglo}) == pytest.approx(0.9)


def escena_optica(nombre: str, tesela: str, nubes: float = 10.0) -> ItemFalso:
    return ItemFalso(
        id=nombre,
        assets={b: ActivoFalso(f"{nombre}/{b}") for b in ("SCL", "B04", "B03", "B02")},
        properties={"s2:mgrs_tile": tesela, "eo:cloud_cover": nubes},
    )


def test_se_descarta_la_tesela_que_no_toca_el_recuadro(monkeypatch):
    """El caso de San Pedro Tlaquepaque: las más despejadas son de la tesela equivocada.

    Fuera de su huella la lectura llega en ceros y el SCL en cero significa sin dato, de
    modo que la escena no aporta un solo píxel por muy despejada que venga.
    """
    escenas = [escena_optica(f"lejos{i}", "13QFD", nubes=1.0) for i in range(19)]
    escenas += [escena_optica("encima", "13QFC", nubes=40.0)]

    def leer(href, bbox, forma):
        return np.full(forma, 0 if href.startswith("lejos") else 4, dtype="uint8")

    monkeypatch.setattr(composite, "leer_ventana", leer)
    utiles = composite.teselas_utiles(escenas, BBOX)
    assert [e.id for e in utiles] == ["encima"]


def test_un_recuadro_partido_conserva_las_dos_teselas(monkeypatch):
    escenas = [escena_optica(f"a{i}", "14QKH") for i in range(3)]
    escenas += [escena_optica(f"b{i}", "14QLH") for i in range(3)]

    def leer(href, bbox, forma):
        arreglo = np.zeros(forma, dtype="uint8")
        if href.startswith("a"):
            arreglo[:, : forma[1] // 2] = 4
        else:
            arreglo[:, forma[1] // 2 :] = 5
        return arreglo

    monkeypatch.setattr(composite, "leer_ventana", leer)
    assert len(composite.teselas_utiles(escenas, BBOX)) == 6


def test_sin_ninguna_tesela_util_falla(monkeypatch):
    monkeypatch.setattr(composite, "leer_ventana", lambda h, b, f: np.zeros(f, dtype="uint8"))
    with pytest.raises(RuntimeError, match="ninguna de las"):
        composite.teselas_utiles([escena_optica("x", "14QKH")], BBOX)
