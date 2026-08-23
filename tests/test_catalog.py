import pytest

from satinsight.catalog import by_cloud_cover, cloud_summary, dominant_orbit, group_by_orbit


class EscenaFalsa:
    """Sustituto mínimo de pystac.Item para probar la lógica pura del módulo."""

    def __init__(self, identificador: str, **propiedades):
        self.id = identificador
        self.properties = propiedades


def optica(identificador, nubes):
    return EscenaFalsa(identificador, **{"eo:cloud_cover": nubes})


def sar(identificador, estado, relativa):
    return EscenaFalsa(
        identificador,
        **{"sat:orbit_state": estado, "sat:relative_orbit": relativa},
    )


def test_resumen_nubes_calcula_proporciones():
    escenas = [optica(f"e{i}", n) for i, n in enumerate([0, 10, 55, 60, 85, 90, 95, 99])]
    resumen = cloud_summary(escenas)
    assert resumen["scenes"] == 8
    assert resumen["minimum"] == 0
    assert resumen["maximum"] == 99
    assert resumen["pct_over_50"] == 75
    assert resumen["pct_over_80"] == 50


def test_resumen_nubes_sin_escenas_falla():
    with pytest.raises(ValueError, match="no scenes to summarise"):
        cloud_summary([])


def test_por_nubosidad_ordena_ascendente():
    escenas = [optica("a", 80), optica("b", 5), optica("c", 40)]
    assert [e.id for e in by_cloud_cover(escenas)] == ["b", "c", "a"]


def test_agrupar_por_orbita_separa_geometrias():
    escenas = [
        sar("a", "ascending", 99),
        sar("b", "descending", 99),
        sar("c", "ascending", 99),
        sar("d", "ascending", 143),
    ]
    grupos = group_by_orbit(escenas)
    assert len(grupos) == 3
    assert len(grupos[("ascending", 99)]) == 2


def test_orbita_dominante_elige_la_mas_poblada():
    escenas = [
        sar("a", "ascending", 99),
        sar("b", "descending", 41),
        sar("c", "descending", 41),
        sar("d", "descending", 41),
    ]
    clave, seleccion = dominant_orbit(escenas)
    assert clave == ("descending", 41)
    assert len(seleccion) == 3


def test_orbita_dominante_sin_escenas_falla():
    with pytest.raises(ValueError, match="no SAR scenes to group"):
        dominant_orbit([])
