"""Tests for bag assembly. Geometry is built by hand, nothing is read from disk."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from satinsight.bags import build, locate, municipal_labels
from satinsight.grid import grid_from_bbox
from satinsight.tiling import grid

BBOX = (-93.135, 16.740, -93.095, 16.768)
CRS = "EPSG:32615"


def malla_y_tiles(size=64):
    malla = grid_from_bbox(BBOX, CRS)
    return malla, grid(malla.shape, size=size)


def agebs_falsas(malla, cortes=2, municipios=("07101", "07102")):
    """Parte el recuadro en franjas verticales, una AGEB por franja."""
    izq, abajo, der, arriba = malla.bounds
    ancho = (der - izq) / cortes
    filas = []
    for i in range(cortes):
        mun = municipios[i % len(municipios)]
        filas.append(
            {
                "cvegeo": f"{mun}0001{i:03d}",
                "grado": ["Muy bajo", "Alto"][i % 2],
                "ordinal": [0, 3][i % 2],
                "poblacion": 1000 * (i + 1),
                "geometry": box(izq + i * ancho, abajo, izq + (i + 1) * ancho, arriba),
            }
        )
    return gpd.GeoDataFrame(filas, crs=CRS)


def test_every_patch_lands_in_exactly_one_ageb():
    malla, tiles = malla_y_tiles()
    tabla = locate(tiles, malla, agebs_falsas(malla))
    assert len(tabla) == len(tiles)
    assert tabla.tile.is_unique
    assert set(tabla.municipio) == {"07101", "07102"}


def test_patches_outside_every_ageb_are_dropped():
    malla, tiles = malla_y_tiles()
    izq, abajo, der, arriba = malla.bounds
    # una sola AGEB que cubre la mitad izquierda
    solo_izquierda = gpd.GeoDataFrame(
        [{"cvegeo": "0710100010001", "geometry": box(izq, abajo, (izq + der) / 2, arriba)}],
        crs=CRS,
    )
    tabla = locate(tiles, malla, solo_izquierda)
    assert 0 < len(tabla) < len(tiles)


def test_locate_on_no_tiles_returns_an_empty_frame():
    malla, _ = malla_y_tiles()
    vacia = locate([], malla, agebs_falsas(malla))
    assert vacia.empty and "municipio" in vacia.columns


def test_the_bag_label_is_weighted_by_population():
    agebs = gpd.GeoDataFrame(
        {
            "cvegeo": ["0710100010001", "0710100010002"],
            "ordinal": [0, 4],
            "poblacion": [1, 999],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        crs=CRS,
    )
    etiquetas = municipal_labels(agebs)
    assert len(etiquetas) == 1
    # la AGEB poblada manda: la media ponderada queda muy cerca de 4
    assert etiquetas.ordinal.iloc[0] == 4
    assert etiquetas.ordinal_continuo.iloc[0] > 3.9
    assert etiquetas.grado.iloc[0] == "Muy alto"


def test_a_municipality_with_no_population_still_gets_a_label():
    agebs = gpd.GeoDataFrame(
        {
            "cvegeo": ["0710100010001", "0710100010002"],
            "ordinal": [0, 2],
            "poblacion": [0, 0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        crs=CRS,
    )
    assert municipal_labels(agebs).ordinal.iloc[0] == 1


def test_municipal_labels_demands_its_columns():
    agebs = gpd.GeoDataFrame({"cvegeo": ["0710100010001"], "geometry": [box(0, 0, 1, 1)]}, crs=CRS)
    with pytest.raises(KeyError, match="ordinal"):
        municipal_labels(agebs)


def test_build_returns_matching_instance_and_bag_tables():
    malla, tiles = malla_y_tiles()
    instancias, bolsas = build(tiles, malla, agebs_falsas(malla), "prueba")
    assert set(instancias.municipio) == set(bolsas.municipio)
    assert bolsas.instances.sum() == len(instancias)
    assert (instancias.ciudad == "prueba").all()


def test_bags_below_the_minimum_are_dropped_with_their_instances():
    malla, tiles = malla_y_tiles()
    agebs = agebs_falsas(malla, cortes=8, municipios=tuple(f"0710{i}" for i in range(8)))
    instancias, bolsas = build(tiles, malla, agebs, "prueba", min_instances=1000)
    assert bolsas.empty and instancias.empty


def test_build_fails_when_nothing_lands_inside():
    malla, tiles = malla_y_tiles()
    lejos = gpd.GeoDataFrame(
        [{"cvegeo": "0710100010001", "ordinal": 1, "poblacion": 10, "geometry": box(0, 0, 1, 1)}],
        crs=CRS,
    )
    with pytest.raises(ValueError, match="no patch landed"):
        build(tiles, malla, lejos, "prueba")


def test_the_cumulative_shares_are_population_weighted():
    """La proporción de población que vive en AGEB de grado k o más.

    Es el agregado que un dato municipal de verdad conoce, y el único de los tres que
    obliga a localizar: predecir que un décimo de la población vive en AGEB rezagada exige
    identificar cuál décimo.
    """
    agebs = gpd.GeoDataFrame(
        {
            "cvegeo": ["0710100010001", "0710100010002", "0710100010003"],
            "ordinal": [0, 2, 4],
            "poblacion": [800, 100, 100],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        crs=CRS,
    )
    fila = municipal_labels(agebs).iloc[0]
    assert fila.p1 == pytest.approx(0.2)
    assert fila.p3 == pytest.approx(0.1)
    assert fila.p4 == pytest.approx(0.1)
    # el redondeo aplasta esa estructura a una sola clase y pierde que un décimo de la
    # población vive en el grado más alto
    assert fila.ordinal == 1


def test_the_shares_fall_as_the_threshold_rises():
    agebs = gpd.GeoDataFrame(
        {
            "cvegeo": [f"071010001000{i}" for i in range(5)],
            "ordinal": [0, 1, 2, 3, 4],
            "poblacion": [100] * 5,
            "geometry": [box(i, 0, i + 1, 1) for i in range(5)],
        },
        crs=CRS,
    )
    fila = municipal_labels(agebs).iloc[0]
    assert fila.p1 > fila.p2 > fila.p3 > fila.p4
    assert fila.p1 == pytest.approx(0.8)
