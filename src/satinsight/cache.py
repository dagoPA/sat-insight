"""Persistencia de los compuestos anuales como GeoTIFF.

Componer una ciudad cuesta cerca de una hora por sensor, y el costo lo domina el sobrecosto
de abrir cada COG remoto, muy por encima de los bytes transferidos. Guardar el resultado en disco
convierte ese gasto en algo que se paga una sola vez.

El compuesto de Sentinel-1 se guarda en potencia lineal, que es como lo entrega
`compuesto_s1`. La conversión a decibeles ocurre al leer, nunca antes de escribir: promediar
en decibeles y promediar en potencia dan resultados distintos, y el promedio en decibeles
está sesgado hacia los valores bajos.

El archivo lleva la retícula consigo —transformación afín, sistema de referencia y nombre de
cada banda— de modo que al recargarlo se recupera todo lo necesario para cruzarlo con
polígonos sin volver a consultar el catálogo.
"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from satinsight.malla import Grid

log = logging.getLogger(__name__)

RAIZ_COMPUESTOS = Path("data") / "compuestos"


def ruta_compuesto(ciudad: str, sensor: str, raiz: Path = RAIZ_COMPUESTOS) -> Path:
    """Ubicación canónica del compuesto de una ciudad y un sensor."""
    return raiz / f"{ciudad}_{sensor}.tif"


def guardar(bandas: dict[str, np.ndarray], malla: Grid, destino: Path, **etiquetas) -> Path:
    """Escribe las bandas de un compuesto en un GeoTIFF con su georreferencia.

    Los nombres de banda se conservan en la descripción de cada una, y cualquier metadato
    extra —escenas usadas, órbita elegida— viaja como etiqueta del archivo para poder
    auditar después con qué se construyó.
    """
    if not bandas:
        raise ValueError("no hay bandas que guardar")

    nombres = list(bandas)
    formas = {b.shape for b in bandas.values()}
    if len(formas) > 1:
        raise ValueError(f"las bandas no comparten forma: {formas}")
    forma = formas.pop()
    if forma != malla.shape:
        raise ValueError(
            f"la forma de las bandas {forma} no coincide con la retícula {malla.shape}"
        )

    destino.parent.mkdir(parents=True, exist_ok=True)
    perfil = {
        "driver": "GTiff",
        "height": malla.height,
        "width": malla.width,
        "count": len(nombres),
        "dtype": "float32",
        "crs": malla.crs,
        "transform": malla.transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(destino, "w", **perfil) as salida:
        for indice, nombre in enumerate(nombres, start=1):
            salida.write(bandas[nombre].astype("float32"), indice)
            salida.set_band_description(indice, nombre)
        if etiquetas:
            salida.update_tags(
                **{k: json.dumps(v, ensure_ascii=False) for k, v in etiquetas.items()}
            )

    log.info("guardado %s (%s, %.1f MP)", destino.name, ", ".join(nombres), malla.megapixels)
    return destino


def cargar(origen: Path) -> tuple[dict[str, np.ndarray], Grid, dict]:
    """Recupera un compuesto y la retícula sobre la que fue escrito."""
    with rasterio.open(origen) as fuente:
        nombres = [
            descripcion or f"banda_{i}"
            for i, descripcion in enumerate(fuente.descriptions, start=1)
        ]
        bandas = {nombre: fuente.read(i) for i, nombre in enumerate(nombres, start=1)}
        malla = Grid(
            transform=fuente.transform,
            shape=(fuente.height, fuente.width),
            crs=str(fuente.crs),
            bounds=tuple(fuente.bounds),
        )
        etiquetas = {}
        for clave, valor in fuente.tags().items():
            try:
                etiquetas[clave] = json.loads(valor)
            except (json.JSONDecodeError, TypeError):
                etiquetas[clave] = valor
    return bandas, malla, etiquetas


def existe(ciudad: str, sensor: str, raiz: Path = RAIZ_COMPUESTOS) -> bool:
    """Indica si el compuesto ya está en disco."""
    return ruta_compuesto(ciudad, sensor, raiz).exists()
