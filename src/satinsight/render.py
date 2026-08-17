"""Renderizado de paneles de inspección visual."""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

TAMANO_PANEL = (440, 308)


def guardar_rgb(
    rojo: np.ndarray,
    verde: np.ndarray,
    azul: np.ndarray,
    destino: Path,
    tamano: tuple[int, int] = TAMANO_PANEL,
) -> Path:
    """Compone tres bandas ya estiradas a 0-255 y las escribe como PNG."""
    imagen = Image.fromarray(np.dstack([rojo, verde, azul]))
    destino.parent.mkdir(parents=True, exist_ok=True)
    imagen.resize(tamano, Image.LANCZOS).save(destino, optimize=True)
    return destino


def a_data_uri(ruta: Path, calidad: int = 82) -> str:
    """Codifica una imagen como data URI JPEG, apto para incrustar en HTML.

    Los artefactos publicados bloquean peticiones a servidores externos, así que las
    imágenes tienen que viajar dentro del propio documento.
    """
    imagen = Image.open(ruta).convert("RGB")
    memoria = io.BytesIO()
    imagen.save(memoria, format="JPEG", quality=calidad, optimize=True, progressive=True)
    codificado = base64.b64encode(memoria.getvalue()).decode()
    return f"data:image/jpeg;base64,{codificado}"
