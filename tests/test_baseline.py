"""Pruebas del baseline sobre tablas sintéticas con señal conocida."""

import numpy as np
import pandas as pd
import pytest

from satinsight.agebs import GRADES
from satinsight.baseline import (
    columns_of_set,
    compare,
    evaluate,
    explained_variance,
    fold_summary,
    select_features,
    standardise_by_group,
    transfer_diagnostics,
)
from satinsight.landcover import CLASSES
from satinsight.texture import feature_names

CITIES = ("tuxtla", "merida", "iztapalapa")


def synthetic_table(n_por_ciudad=120, strength=1.0, semilla=0):
    """Tabla donde los features llevan señal del ordinal, graduable con `strength`.

    Con strength alta el modelo debe recuperar el orden; con strength cero los features son ruido
    puro y ningún modelo debería superar al rng de manera consistente.
    """
    rng = np.random.default_rng(semilla)
    partes = []
    for ciudad in CITIES:
        ordinal = rng.integers(0, 5, n_por_ciudad)
        ruido = rng.normal(0, 1, n_por_ciudad)
        columns = {
            "cvegeo": [f"{ciudad}{i:04d}" for i in range(n_por_ciudad)],
            "ciudad": ciudad,
            "ordinal": ordinal,
            "grado": [GRADES[i] for i in ordinal],
            "c_mean": strength * ordinal + ruido,
            "c_std": strength * ordinal * 0.5 + ruido,
            "c_p10": rng.normal(0, 1, n_por_ciudad),
            "c_p50": rng.normal(0, 1, n_por_ciudad),
            "c_p90": rng.normal(0, 1, n_por_ciudad),
            "c_rango_intercuartil": rng.normal(0, 1, n_por_ciudad),
        }
        # Los nombres de textura salen del propio módulo, para que renombrar un rasgo
        # rompa la prueba, con lo que se evita que quede midiendo un split vacío.
        for clase in CLASSES.values():
            columns[f"wc_{clase}"] = rng.random(n_por_ciudad)
        for sufijo in feature_names():
            lleva_senal = sufijo.startswith(("contrast_", "homogeneity_"))
            signo = -1 if sufijo.startswith("homogeneity_") else 1
            señal = signo * strength * ordinal if lleva_senal else 0.0
            columns[f"c_{sufijo}"] = señal + rng.normal(0, 1, n_por_ciudad)
        partes.append(pd.DataFrame(columns))
    return pd.concat(partes, ignore_index=True)


def test_los_conjuntos_separan_densidad_de_textura():
    table = synthetic_table(10)
    densidad = columns_of_set(table, "densidad")
    textura = columns_of_set(table, "textura")

    assert "c_mean" in densidad
    assert "c_contrast_d1" not in densidad
    assert "c_contrast_d1" in textura
    assert "c_mean" not in textura
    assert not set(densidad) & set(textura)


def test_el_conjunto_completo_es_la_union_de_los_tres_escalones():
    table = synthetic_table(10)
    completo = set(columns_of_set(table, "completo"))
    esperado = set()
    for escalon in ("cobertura", "densidad", "textura"):
        esperado |= set(columns_of_set(table, escalon))
    assert completo == esperado


def test_la_cobertura_no_se_mezcla_con_los_otros_escalones():
    table = synthetic_table(10)
    cobertura = set(columns_of_set(table, "cobertura"))
    assert "wc_built" in cobertura
    assert not cobertura & set(columns_of_set(table, "densidad"))
    assert not cobertura & set(columns_of_set(table, "textura"))


def test_conjunto_desconocido_falla():
    with pytest.raises(KeyError, match="unknown set"):
        columns_of_set(synthetic_table(10), "inventado")


def test_la_validacion_deja_una_ciudad_fuera_por_pliegue():
    detail = evaluate(synthetic_table(60), "completo", "clasificador")
    assert len(detail) == len(CITIES)
    assert set(detail["ciudad_prueba"]) == set(CITIES)
    for _, fila in detail.iterrows():
        assert fila["n_entrena"] == 120
        assert fila["n_prueba"] == 60


def test_con_senal_el_modelo_le_gana_al_azar():
    table = synthetic_table(200, strength=1.5)
    modelo = evaluate(table, "completo", "clasificador")["kappa"].mean()
    rng = evaluate(table, "completo", "rng")["kappa"].mean()
    assert modelo > rng + 0.1


def test_sin_senal_el_modelo_no_le_gana_al_azar():
    table = synthetic_table(200, strength=0.0)
    modelo = evaluate(table, "completo", "clasificador")["kappa"].mean()
    assert modelo < 0.15


def test_la_moda_tiene_kappa_nulo():
    detail = evaluate(synthetic_table(100), "densidad", "moda")
    assert detail["kappa"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_el_regresor_predice_dentro_del_rango_ordinal():
    table = synthetic_table(100, strength=2.0)
    detail = evaluate(table, "completo", "regresor")
    assert (detail["mae_ordinal"] >= 0).all()
    assert (detail["exactitud"] <= 1).all()


def test_comparar_corre_los_modelos_ciegos_una_sola_vez():
    detail = compare(synthetic_table(50))
    ciegos = detail[detail["modelo"].isin(["rng", "moda"])]
    assert set(ciegos["split"]) == {"ninguno"}
    assert len(ciegos) == 2 * len(CITIES)


def test_el_resumen_ordena_por_kappa():
    aggregate = fold_summary(compare(synthetic_table(80, strength=1.5)))
    assert list(aggregate["kappa"]) == sorted(aggregate["kappa"], reverse=True)
    assert {"split", "modelo", "kappa"} <= set(aggregate.columns)


def test_la_varianza_explicada_reconoce_un_factor_perfecto():
    table = pd.DataFrame({"v": [1.0, 1.0, 5.0, 5.0], "g": ["a", "a", "b", "b"]})
    assert explained_variance(table, "v", "g") == pytest.approx(1.0)


def test_un_factor_sin_relacion_explica_poco():
    table = pd.DataFrame({"v": [1.0, 5.0, 1.0, 5.0], "g": ["a", "a", "b", "b"]})
    assert explained_variance(table, "v", "g") == pytest.approx(0.0, abs=1e-9)


def test_el_diagnostico_delata_el_rasgo_que_solo_conoce_la_ciudad():
    """Un rasgo que separa cities sin separar grados debe salir con razón alta."""
    table = synthetic_table(80, strength=1.5)
    table["c_mean"] = table["ciudad"].map({c: i * 10.0 for i, c in enumerate(CITIES)})
    d = transfer_diagnostics(table, "densidad").set_index("feature")
    assert d.loc["c_mean", "ratio"] > 10
    assert d.loc["c_std", "ratio"] < d.loc["c_mean", "ratio"]


def test_estandarizar_centra_dentro_de_cada_ciudad():
    table = synthetic_table(60, strength=1.0)
    columns = columns_of_set(table, "densidad")
    e = standardise_by_group(table, columns)
    for _, grupo in e.groupby("ciudad"):
        assert grupo["c_mean"].mean() == pytest.approx(0.0, abs=1e-9)
        assert grupo["c_mean"].std() == pytest.approx(1.0, abs=1e-9)


def test_un_rasgo_constante_queda_en_cero_y_no_en_nulo():
    """Varias clases de cobertura valen cero en todas las AGEB.

    Convertirlas en columns enteramente nulas rompe el binning del modelo, así que el
    caso degenerado tiene que quedar centrado en cero.
    """
    table = synthetic_table(40)
    table["wc_snow"] = 0.0
    e = standardise_by_group(table, ["wc_snow"])
    assert e["wc_snow"].notna().all()
    assert (e["wc_snow"] == 0.0).all()


def test_estandarizar_conserva_los_nulos_que_son_ausencia_de_dato():
    table = synthetic_table(40)
    table.loc[:5, "c_mean"] = np.nan
    e = standardise_by_group(table, ["c_mean"])
    assert e["c_mean"].isna().sum() == 6


def test_la_ablacion_estandarizada_corre_completa():
    table = synthetic_table(60, strength=1.5)
    detail = compare(table, estandarizar=True)
    assert not detail.empty
    assert detail["kappa"].notna().all()


def test_una_tabla_sin_rasgos_del_conjunto_falla():
    table = pd.DataFrame({"ciudad": ["a"], "ordinal": [1], "otra_cosa": [3.0]})
    with pytest.raises(ValueError, match="split"):
        evaluate(table, "textura", "clasificador")


def fiabilidad_de(table, valor=0.9):
    """Tabla de reliability sintética con el mismo valor para todos los features."""
    features = columns_of_set(table, "textura")
    return pd.DataFrame({"feature": features, "r_median": [valor] * len(features)})


def test_un_rasgo_que_no_se_reproduce_queda_fuera():
    table = synthetic_table(40)
    table["c_n_px"] = 1000
    fiab = fiabilidad_de(table)
    fiab.loc[fiab.feature == "c_contrast_d1", "r_median"] = 0.2

    sel = select_features(table, fiab).set_index("feature")
    assert not sel.loc["c_contrast_d1", "kept"]
    assert sel.loc["c_contrast_d1", "reason"] == "does not reproduce"


def test_un_rasgo_atado_al_tamano_queda_fuera():
    """Un rasgo que es una función del área del polígono apunta al blanco por construcción."""
    table = synthetic_table(60)
    rng = np.random.default_rng(3)
    table["c_n_px"] = rng.integers(700, 20000, len(table))
    table["c_contrast_d1"] = np.log10(table["c_n_px"]) * 5

    sel = select_features(table, fiabilidad_de(table)).set_index("feature")
    assert not sel.loc["c_contrast_d1", "kept"]
    assert sel.loc["c_contrast_d1", "reason"] == "tied to size"


def test_un_rasgo_fiable_e_independiente_se_conserva():
    table = synthetic_table(60)
    rng = np.random.default_rng(4)
    table["c_n_px"] = rng.integers(700, 20000, len(table))
    table["c_homogeneity_d1"] = rng.normal(0, 1, len(table))

    sel = select_features(table, fiabilidad_de(table)).set_index("feature")
    assert sel.loc["c_homogeneity_d1", "kept"]
    assert sel.loc["c_homogeneity_d1", "reason"] == "kept"


def test_el_ruido_puro_lo_atrapa_la_fiabilidad_y_no_el_tamano():
    """Los dos criterios se necesitan mutuamente.

    Un rasgo que es ruido pasa el criterio de tamaño con holgura, porque el ruido no
    correlaciona con nada. Solo la reliability lo detecta.
    """
    table = synthetic_table(60)
    rng = np.random.default_rng(5)
    table["c_n_px"] = rng.integers(700, 20000, len(table))
    table["c_energy_d4"] = rng.normal(0, 1, len(table))

    fiab = fiabilidad_de(table)
    fiab.loc[fiab.feature == "c_energy_d4", "r_median"] = 0.05
    sel = select_features(table, fiab).set_index("feature")

    assert abs(sel.loc["c_energy_d4", "r_n_px"]) < 0.30
    assert not sel.loc["c_energy_d4", "kept"]


def test_un_rasgo_sin_medicion_de_fiabilidad_queda_fuera():
    table = synthetic_table(40)
    table["c_n_px"] = 1000
    fiab = fiabilidad_de(table)
    sel = select_features(table, fiab[fiab.feature != "c_contrast_d2"]).set_index("feature")
    assert not sel.loc["c_contrast_d2", "kept"]


def test_el_auroc_de_las_clases_bajas_no_sale_invertido():
    """Con una sola puntuación ordenada, la evidencia a favor de k es la cercanía a k.

    Usar el orden crudo invierte las clases bajas y el promedio sale en 0.5 por
    cancelación, aparentando rng donde el modelo separa casi perfecto.
    """
    import numpy as np

    from satinsight.baseline import auroc_one_vs_rest

    verdad = np.array([0, 1, 2, 3, 4] * 20)
    casi_perfecto = verdad + np.random.default_rng(0).normal(0, 0.2, len(verdad))
    r = auroc_one_vs_rest(verdad, casi_perfecto)
    assert r["auroc_muy_bajo"] > 0.9
    assert r["auroc_macro"] > 0.85


def test_el_auroc_acumulado_respeta_el_orden():
    import numpy as np

    from satinsight.baseline import auroc_cumulative

    verdad = np.array([0, 1, 2, 3, 4] * 20)
    r = auroc_cumulative(verdad, verdad.astype(float))
    assert all(v == 1.0 for k, v in r.items() if k.startswith("auroc_ge_"))


def test_la_fusion_junta_las_columnas_de_las_dos_modalidades():
    import pandas as pd

    from satinsight.baseline import fuse

    optico = pd.DataFrame(
        {"cvegeo": ["a", "b"], "ciudad": ["x", "x"], "ordinal": [1, 2], "s2rojo_media": [1.0, 2.0]}
    )
    radar = pd.DataFrame(
        {"cvegeo": ["a", "b"], "ciudad": ["x", "x"], "ordinal": [1, 2], "s1vv_media": [3.0, 4.0]}
    )
    juntas = fuse(optico, radar)
    assert list(juntas.columns) == ["cvegeo", "ciudad", "ordinal", "s2rojo_media", "s1vv_media"]
    assert len(juntas) == 2


def test_la_fusion_conserva_solo_las_ageb_de_las_dos():
    import pandas as pd

    from satinsight.baseline import fuse

    optico = pd.DataFrame({"cvegeo": ["a", "b", "c"], "s2rojo_media": [1.0, 2.0, 3.0]})
    radar = pd.DataFrame({"cvegeo": ["b", "c", "d"], "s1vv_media": [4.0, 5.0, 6.0]})
    juntas = fuse(optico, radar)
    assert sorted(juntas.cvegeo) == ["b", "c"]
