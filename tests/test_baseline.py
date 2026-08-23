"""Pruebas del baseline sobre tablas sintéticas con señal conocida."""

import numpy as np
import pandas as pd
import pytest

from satinsight.agebs import GRADES
from satinsight.baseline import (
    columnas_de_conjunto,
    comparar,
    diagnostico_transferencia,
    estandarizar_por_grupo,
    evaluar,
    resumen,
    seleccionar_rasgos,
    varianza_explicada,
)
from satinsight.landcover import CLASSES
from satinsight.texture import feature_names

CITIES = ("tuxtla", "merida", "iztapalapa")


def tabla_sintetica(n_por_ciudad=120, fuerza=1.0, semilla=0):
    """Tabla donde los rasgos llevan señal del ordinal, graduable con `fuerza`.

    Con fuerza alta el modelo debe recuperar el orden; con fuerza cero los rasgos son ruido
    puro y ningún modelo debería superar al azar de manera consistente.
    """
    rng = np.random.default_rng(semilla)
    partes = []
    for ciudad in CITIES:
        ordinal = rng.integers(0, 5, n_por_ciudad)
        ruido = rng.normal(0, 1, n_por_ciudad)
        columnas = {
            "cvegeo": [f"{ciudad}{i:04d}" for i in range(n_por_ciudad)],
            "ciudad": ciudad,
            "ordinal": ordinal,
            "grado": [GRADES[i] for i in ordinal],
            "c_media": fuerza * ordinal + ruido,
            "c_desv": fuerza * ordinal * 0.5 + ruido,
            "c_p10": rng.normal(0, 1, n_por_ciudad),
            "c_p50": rng.normal(0, 1, n_por_ciudad),
            "c_p90": rng.normal(0, 1, n_por_ciudad),
            "c_rango_intercuartil": rng.normal(0, 1, n_por_ciudad),
        }
        # Los nombres de textura salen del propio módulo, para que renombrar un rasgo
        # rompa la prueba, con lo que se evita que quede midiendo un conjunto vacío.
        for clase in CLASSES.values():
            columnas[f"wc_{clase}"] = rng.random(n_por_ciudad)
        for sufijo in feature_names():
            lleva_senal = sufijo.startswith(("contrast_", "homogeneity_"))
            signo = -1 if sufijo.startswith("homogeneity_") else 1
            señal = signo * fuerza * ordinal if lleva_senal else 0.0
            columnas[f"c_{sufijo}"] = señal + rng.normal(0, 1, n_por_ciudad)
        partes.append(pd.DataFrame(columnas))
    return pd.concat(partes, ignore_index=True)


def test_los_conjuntos_separan_densidad_de_textura():
    tabla = tabla_sintetica(10)
    densidad = columnas_de_conjunto(tabla, "densidad")
    textura = columnas_de_conjunto(tabla, "textura")

    assert "c_media" in densidad
    assert "c_contrast_d1" not in densidad
    assert "c_contrast_d1" in textura
    assert "c_media" not in textura
    assert not set(densidad) & set(textura)


def test_el_conjunto_completo_es_la_union_de_los_tres_escalones():
    tabla = tabla_sintetica(10)
    completo = set(columnas_de_conjunto(tabla, "completo"))
    esperado = set()
    for escalon in ("cobertura", "densidad", "textura"):
        esperado |= set(columnas_de_conjunto(tabla, escalon))
    assert completo == esperado


def test_la_cobertura_no_se_mezcla_con_los_otros_escalones():
    tabla = tabla_sintetica(10)
    cobertura = set(columnas_de_conjunto(tabla, "cobertura"))
    assert "wc_built" in cobertura
    assert not cobertura & set(columnas_de_conjunto(tabla, "densidad"))
    assert not cobertura & set(columnas_de_conjunto(tabla, "textura"))


def test_conjunto_desconocido_falla():
    with pytest.raises(KeyError, match="conjunto desconocido"):
        columnas_de_conjunto(tabla_sintetica(10), "inventado")


def test_la_validacion_deja_una_ciudad_fuera_por_pliegue():
    detalle = evaluar(tabla_sintetica(60), "completo", "clasificador")
    assert len(detalle) == len(CITIES)
    assert set(detalle["ciudad_prueba"]) == set(CITIES)
    for _, fila in detalle.iterrows():
        assert fila["n_entrena"] == 120
        assert fila["n_prueba"] == 60


def test_con_senal_el_modelo_le_gana_al_azar():
    tabla = tabla_sintetica(200, fuerza=1.5)
    modelo = evaluar(tabla, "completo", "clasificador")["kappa"].mean()
    azar = evaluar(tabla, "completo", "azar")["kappa"].mean()
    assert modelo > azar + 0.1


def test_sin_senal_el_modelo_no_le_gana_al_azar():
    tabla = tabla_sintetica(200, fuerza=0.0)
    modelo = evaluar(tabla, "completo", "clasificador")["kappa"].mean()
    assert modelo < 0.15


def test_la_moda_tiene_kappa_nulo():
    detalle = evaluar(tabla_sintetica(100), "densidad", "moda")
    assert detalle["kappa"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_el_regresor_predice_dentro_del_rango_ordinal():
    tabla = tabla_sintetica(100, fuerza=2.0)
    detalle = evaluar(tabla, "completo", "regresor")
    assert (detalle["mae_ordinal"] >= 0).all()
    assert (detalle["exactitud"] <= 1).all()


def test_comparar_corre_los_modelos_ciegos_una_sola_vez():
    detalle = comparar(tabla_sintetica(50))
    ciegos = detalle[detalle["modelo"].isin(["azar", "moda"])]
    assert set(ciegos["conjunto"]) == {"ninguno"}
    assert len(ciegos) == 2 * len(CITIES)


def test_el_resumen_ordena_por_kappa():
    agregado = resumen(comparar(tabla_sintetica(80, fuerza=1.5)))
    assert list(agregado["kappa"]) == sorted(agregado["kappa"], reverse=True)
    assert {"conjunto", "modelo", "kappa"} <= set(agregado.columns)


def test_la_varianza_explicada_reconoce_un_factor_perfecto():
    tabla = pd.DataFrame({"v": [1.0, 1.0, 5.0, 5.0], "g": ["a", "a", "b", "b"]})
    assert varianza_explicada(tabla, "v", "g") == pytest.approx(1.0)


def test_un_factor_sin_relacion_explica_poco():
    tabla = pd.DataFrame({"v": [1.0, 5.0, 1.0, 5.0], "g": ["a", "a", "b", "b"]})
    assert varianza_explicada(tabla, "v", "g") == pytest.approx(0.0, abs=1e-9)


def test_el_diagnostico_delata_el_rasgo_que_solo_conoce_la_ciudad():
    """Un rasgo que separa cities sin separar grados debe salir con razón alta."""
    tabla = tabla_sintetica(80, fuerza=1.5)
    tabla["c_media"] = tabla["ciudad"].map({c: i * 10.0 for i, c in enumerate(CITIES)})
    d = diagnostico_transferencia(tabla, "densidad").set_index("feature")
    assert d.loc["c_media", "razon"] > 10
    assert d.loc["c_desv", "razon"] < d.loc["c_media", "razon"]


def test_estandarizar_centra_dentro_de_cada_ciudad():
    tabla = tabla_sintetica(60, fuerza=1.0)
    columnas = columnas_de_conjunto(tabla, "densidad")
    e = estandarizar_por_grupo(tabla, columnas)
    for _, grupo in e.groupby("ciudad"):
        assert grupo["c_media"].mean() == pytest.approx(0.0, abs=1e-9)
        assert grupo["c_media"].std() == pytest.approx(1.0, abs=1e-9)


def test_un_rasgo_constante_queda_en_cero_y_no_en_nulo():
    """Varias clases de cobertura valen cero en todas las AGEB.

    Convertirlas en columnas enteramente nulas rompe el binning del modelo, así que el
    caso degenerado tiene que quedar centrado en cero.
    """
    tabla = tabla_sintetica(40)
    tabla["wc_snow"] = 0.0
    e = estandarizar_por_grupo(tabla, ["wc_snow"])
    assert e["wc_snow"].notna().all()
    assert (e["wc_snow"] == 0.0).all()


def test_estandarizar_conserva_los_nulos_que_son_ausencia_de_dato():
    tabla = tabla_sintetica(40)
    tabla.loc[:5, "c_media"] = np.nan
    e = estandarizar_por_grupo(tabla, ["c_media"])
    assert e["c_media"].isna().sum() == 6


def test_la_ablacion_estandarizada_corre_completa():
    tabla = tabla_sintetica(60, fuerza=1.5)
    detalle = comparar(tabla, estandarizar=True)
    assert not detalle.empty
    assert detalle["kappa"].notna().all()


def test_una_tabla_sin_rasgos_del_conjunto_falla():
    tabla = pd.DataFrame({"ciudad": ["a"], "ordinal": [1], "otra_cosa": [3.0]})
    with pytest.raises(ValueError, match="conjunto"):
        evaluar(tabla, "textura", "clasificador")


def fiabilidad_de(tabla, valor=0.9):
    """Tabla de fiabilidad sintética con el mismo valor para todos los rasgos."""
    rasgos = columnas_de_conjunto(tabla, "textura")
    return pd.DataFrame({"feature": rasgos, "r_median": [valor] * len(rasgos)})


def test_un_rasgo_que_no_se_reproduce_queda_fuera():
    tabla = tabla_sintetica(40)
    tabla["c_n_px"] = 1000
    fiab = fiabilidad_de(tabla)
    fiab.loc[fiab.feature == "c_contrast_d1", "r_median"] = 0.2

    sel = seleccionar_rasgos(tabla, fiab).set_index("feature")
    assert not sel.loc["c_contrast_d1", "kept"]
    assert sel.loc["c_contrast_d1", "reason"] == "no se reproduce"


def test_un_rasgo_atado_al_tamano_queda_fuera():
    """Un rasgo que es una función del área del polígono apunta al blanco por construcción."""
    tabla = tabla_sintetica(60)
    rng = np.random.default_rng(3)
    tabla["c_n_px"] = rng.integers(700, 20000, len(tabla))
    tabla["c_contrast_d1"] = np.log10(tabla["c_n_px"]) * 5

    sel = seleccionar_rasgos(tabla, fiabilidad_de(tabla)).set_index("feature")
    assert not sel.loc["c_contrast_d1", "kept"]
    assert sel.loc["c_contrast_d1", "reason"] == "atado al tamaño"


def test_un_rasgo_fiable_e_independiente_se_conserva():
    tabla = tabla_sintetica(60)
    rng = np.random.default_rng(4)
    tabla["c_n_px"] = rng.integers(700, 20000, len(tabla))
    tabla["c_homogeneity_d1"] = rng.normal(0, 1, len(tabla))

    sel = seleccionar_rasgos(tabla, fiabilidad_de(tabla)).set_index("feature")
    assert sel.loc["c_homogeneity_d1", "kept"]
    assert sel.loc["c_homogeneity_d1", "reason"] == "kept"


def test_el_ruido_puro_lo_atrapa_la_fiabilidad_y_no_el_tamano():
    """Los dos criterios se necesitan mutuamente.

    Un rasgo que es ruido pasa el criterio de tamaño con holgura, porque el ruido no
    correlaciona con nada. Solo la fiabilidad lo detecta.
    """
    tabla = tabla_sintetica(60)
    rng = np.random.default_rng(5)
    tabla["c_n_px"] = rng.integers(700, 20000, len(tabla))
    tabla["c_energy_d4"] = rng.normal(0, 1, len(tabla))

    fiab = fiabilidad_de(tabla)
    fiab.loc[fiab.feature == "c_energy_d4", "r_median"] = 0.05
    sel = seleccionar_rasgos(tabla, fiab).set_index("feature")

    assert abs(sel.loc["c_energy_d4", "r_n_px"]) < 0.30
    assert not sel.loc["c_energy_d4", "kept"]


def test_un_rasgo_sin_medicion_de_fiabilidad_queda_fuera():
    tabla = tabla_sintetica(40)
    tabla["c_n_px"] = 1000
    fiab = fiabilidad_de(tabla)
    sel = seleccionar_rasgos(tabla, fiab[fiab.feature != "c_contrast_d2"]).set_index("feature")
    assert not sel.loc["c_contrast_d2", "kept"]


def test_el_auroc_de_las_clases_bajas_no_sale_invertido():
    """Con una sola puntuación ordenada, la evidencia a favor de k es la cercanía a k.

    Usar el orden crudo invierte las clases bajas y el promedio sale en 0.5 por
    cancelación, aparentando azar donde el modelo separa casi perfecto.
    """
    import numpy as np

    from satinsight.baseline import auroc_una_contra_resto

    verdad = np.array([0, 1, 2, 3, 4] * 20)
    casi_perfecto = verdad + np.random.default_rng(0).normal(0, 0.2, len(verdad))
    r = auroc_una_contra_resto(verdad, casi_perfecto)
    assert r["auroc_muy_bajo"] > 0.9
    assert r["auroc_macro"] > 0.85


def test_el_auroc_acumulado_respeta_el_orden():
    import numpy as np

    from satinsight.baseline import auroc_acumulada

    verdad = np.array([0, 1, 2, 3, 4] * 20)
    r = auroc_acumulada(verdad, verdad.astype(float))
    assert all(v == 1.0 for k, v in r.items() if k.startswith("auroc_ge_"))


def test_la_fusion_junta_las_columnas_de_las_dos_modalidades():
    import pandas as pd

    from satinsight.baseline import fusionar

    optico = pd.DataFrame(
        {"cvegeo": ["a", "b"], "ciudad": ["x", "x"], "ordinal": [1, 2], "s2rojo_media": [1.0, 2.0]}
    )
    radar = pd.DataFrame(
        {"cvegeo": ["a", "b"], "ciudad": ["x", "x"], "ordinal": [1, 2], "s1vv_media": [3.0, 4.0]}
    )
    juntas = fusionar(optico, radar)
    assert list(juntas.columns) == ["cvegeo", "ciudad", "ordinal", "s2rojo_media", "s1vv_media"]
    assert len(juntas) == 2


def test_la_fusion_conserva_solo_las_ageb_de_las_dos():
    import pandas as pd

    from satinsight.baseline import fusionar

    optico = pd.DataFrame({"cvegeo": ["a", "b", "c"], "s2rojo_media": [1.0, 2.0, 3.0]})
    radar = pd.DataFrame({"cvegeo": ["b", "c", "d"], "s1vv_media": [4.0, 5.0, 6.0]})
    juntas = fusionar(optico, radar)
    assert sorted(juntas.cvegeo) == ["b", "c"]
