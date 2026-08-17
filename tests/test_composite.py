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
