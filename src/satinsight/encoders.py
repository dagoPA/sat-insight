"""Turns patches into feature vectors with a frozen foundation model.

The model is never fine-tuned. Vectors are extracted once, written to disk, and every
later experiment trains on those. That keeps the expensive half of the pipeline out of
the training loop and makes an ablation over MIL variants cost minutes.

Torch is imported inside the encoder rather than at module level, so the tiling, the bag
assembly and the extraction loop can all be exercised with a stand-in encoder and no
deep learning stack installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from satinsight.textura import RANGOS_FIJOS

log = logging.getLogger(__name__)

WAVELENGTHS_UM = {
    "B02": 0.490,
    "B03": 0.560,
    "B04": 0.665,
    "B08": 0.842,
    "B11": 1.610,
    "vv": 55500.0,
    "vh": 55500.0,
}
"""Central wavelength of each channel in micrometres.

The optical figures are the Sentinel-2 band centres. Sentinel-1 rides C band at 5.405
GHz, which is 5.55 cm, and the value is written in the same unit so one dictionary
covers both sensors. A wavelength-conditioned model generates its input projection from
these numbers, so pairing a channel with the wrong one silently produces a valid-looking
vector that means nothing; the units its checkpoint expects have to be confirmed against
the reference implementation before the first extraction is trusted.
"""

BATCH = 64
"""Patches per forward pass."""


def normalize(patch: np.ndarray, names: list[str]) -> np.ndarray:
    """Maps each channel onto [0, 1] with the fixed ranges the baseline settled on.

    Stretching each patch by its own percentiles would make identical ground look
    different depending on what else shares the patch, which is the failure the phase one
    ablation measured: fixed ranges beat per-image percentiles by 0.25 of kappa on radar
    and 0.21 on optical. Unobserved pixels are filled with the middle of the range, since
    a foundation model has no way to represent a hole.
    """
    if len(names) != patch.shape[0]:
        raise ValueError(f"{len(names)} channel names for {patch.shape[0]} channels")
    salida = np.empty_like(patch, dtype="float32")
    for i, nombre in enumerate(names):
        clave = {"B04": "s2rojo", "B08": "s2nir"}.get(nombre, nombre)
        rango = RANGOS_FIJOS.get(clave)
        if rango is None:
            canal = patch[i][np.isfinite(patch[i])]
            rango = (float(canal.min()), float(canal.max())) if canal.size else (0.0, 1.0)
        bajo, alto = rango
        escalado = (patch[i] - bajo) / max(alto - bajo, 1e-9)
        salida[i] = np.nan_to_num(np.clip(escalado, 0.0, 1.0), nan=0.5)
    return salida


@runtime_checkable
class PatchEncoder(Protocol):
    """What the extraction loop needs from any feature extractor."""

    dim: int

    def embed(self, batch: np.ndarray, wavelengths: list[float]) -> np.ndarray:
        """Takes (n, channel, row, column) and returns (n, dim)."""
        ...


class DofaEncoder:
    """DOFA, a wavelength-conditioned foundation model, held frozen.

    It accepts any number of channels because it generates its input projection from the
    wavelength of each one, which is what lets radar and optical share a single extractor
    and what makes the transfer to other sensors cheap.
    """

    def __init__(self, checkpoint: str = "dofa_base_patch16_224", device: str | None = None):
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._cargar(checkpoint)
        self.model.eval().to(self.device)
        for parametro in self.model.parameters():
            parametro.requires_grad_(False)
        self.dim = int(getattr(self.model, "embed_dim", 768))
        log.info("%s loaded on %s, %d dimensions", checkpoint, self.device, self.dim)

    def _cargar(self, checkpoint: str):
        """Fetches the pretrained weights, preferring torchgeo over torch.hub."""
        try:
            import torchgeo.models as modelos

            return modelos.dofa_base_patch16_224(weights=modelos.DOFABase16_Weights.DOFA_MAE)
        except Exception:
            log.warning("torchgeo unavailable, falling back to torch.hub", exc_info=True)
            return self._torch.hub.load("zhu-xlab/DOFA", checkpoint, pretrained=True)

    def embed(self, batch: np.ndarray, wavelengths: list[float]) -> np.ndarray:
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).float().to(self.device)
        with torch.inference_mode():
            salida = self.model.forward_features(tensor, wave_list=wavelengths)
            if salida.ndim == 3:
                # (n, token, dim): the leading token is the summary and the rest are the
                # patch tokens; the summary is what represents the whole instance
                salida = salida[:, 0]
        return salida.float().cpu().numpy()


def extract(
    bands: dict[str, np.ndarray],
    tiles: list,
    encoder: PatchEncoder,
    *,
    order: list[str] | None = None,
    batch: int = BATCH,
) -> np.ndarray:
    """Runs every patch of a city through the encoder and returns (n_tiles, dim)."""
    from satinsight.tiling import stack

    order = order or sorted(bands)
    faltantes = [n for n in order if n not in WAVELENGTHS_UM]
    if faltantes:
        raise KeyError(f"no wavelength registered for {faltantes}")
    longitudes = [WAVELENGTHS_UM[n] for n in order]

    vectores = []
    for inicio in range(0, len(tiles), batch):
        lote = np.stack(
            [normalize(stack(bands, t, order), order) for t in tiles[inicio : inicio + batch]]
        )
        vectores.append(encoder.embed(lote, longitudes))
    if not vectores:
        return np.empty((0, encoder.dim), dtype="float32")
    salida = np.concatenate(vectores)
    log.info("%d patches encoded into %d dimensions", *salida.shape)
    return salida


def save(embeddings: np.ndarray, destino: Path, **etiquetas) -> Path:
    """Writes the vectors as half precision, which halves the disk for no measurable loss."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destino,
        embeddings=embeddings.astype("float16"),
        **{k: np.asarray(v) for k, v in etiquetas.items()},
    )
    log.info("%s (%.1f MB)", destino.name, destino.stat().st_size / 1e6)
    return destino


def load(origen: Path) -> tuple[np.ndarray, dict]:
    """Reads back what `save` wrote, restoring the vectors to single precision."""
    with np.load(origen, allow_pickle=False) as datos:
        embeddings = datos["embeddings"].astype("float32")
        etiquetas = {k: datos[k] for k in datos.files if k != "embeddings"}
    return embeddings, etiquetas
