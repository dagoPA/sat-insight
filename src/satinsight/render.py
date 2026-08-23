"""Rendering of visual inspection panels."""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

PANEL_SIZE = (440, 308)


def save_rgb(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    destination: Path,
    size: tuple[int, int] = PANEL_SIZE,
) -> Path:
    """Stacks three bands already stretched to 0-255 and writes them as a PNG."""
    image = Image.fromarray(np.dstack([red, green, blue]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.resize(size, Image.LANCZOS).save(destination, optimize=True)
    return destination


def to_data_uri(path: Path, quality: int = 82) -> str:
    """Encodes an image as a JPEG data URI, ready to embed in HTML.

    Published artefacts block requests to outside servers, so images have to travel
    inside the document itself.
    """
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"
