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

from satinsight.texture import FIXED_RANGES
from satinsight.tiling import MIN_VALID_FRACTION, TOKEN_SIZE

log = logging.getLogger(__name__)

WAVELENGTHS_UM = {
    "B02": 0.490,
    "B03": 0.560,
    "B04": 0.665,
    "B08": 0.842,
    "B11": 1.610,
    "vv": 3.75,
    "vh": 3.75,
}
"""What each channel is called in the units the model conditions its input projection on.

The optical figures are the Sentinel-2 band centres in micrometres. The radar pair is
the surprise: Sentinel-1 rides C band at 5.405 GHz, which is 5.55 cm, and the physical
figure is the wrong one to pass. The DOFA v1 weights were trained with 3.75 standing in
as a modality marker for VV and VH, so that is what the checkpoint recognises.

Getting this wrong costs nothing visible. The model would accept 55500, generate a
projection for it, and return vectors of the right shape carrying no useful signal. The
numbers are pinned here after reading them out of the reference implementation, and any
new checkpoint has to be checked the same way before its first extraction is believed.
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
        clave = {"B04": "s2red", "B08": "s2nir"}.get(nombre, nombre)
        rango = FIXED_RANGES.get(clave)
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
        """Takes (n, channel, row, column) and returns one summary vector per window."""
        ...

    def embed_tokens(self, batch: np.ndarray, wavelengths: list[float]) -> np.ndarray:
        """Takes (n, channel, row, column) and returns (n, token, dim).

        One vector per token is what the MIL bag needs: the summary of a whole 224 px
        window spans twenty AGEB and cannot be scored against any of them.
        """
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
        self.device = device or self._mejor_dispositivo()
        self.model = self._cargar(checkpoint)
        self.model.eval().to(self.device)
        for parametro in self.model.parameters():
            parametro.requires_grad_(False)
        self.dim = int(getattr(self.model, "embed_dim", 768))
        log.info("%s loaded on %s, %d dimensions", checkpoint, self.device, self.dim)

    def _mejor_dispositivo(self) -> str:
        """Picks the fastest accelerator present, Apple silicon included."""
        torch = self._torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _cargar(self, checkpoint: str):
        """Fetches the pretrained weights through torchgeo, which hosts the checkpoints."""
        import torchgeo.models as modelos

        constructor = getattr(modelos, checkpoint.replace("-", "_"), None)
        if constructor is None:
            disponibles = sorted(n for n in dir(modelos) if n.startswith("dofa_"))
            raise KeyError(f"unknown checkpoint {checkpoint!r}. Available: {disponibles}")
        pesos = modelos.DOFABase16_Weights.DOFA_MAE if "base" in checkpoint else None
        return constructor(weights=pesos)

    def _tensor(self, batch: np.ndarray):
        return self._torch.from_numpy(np.ascontiguousarray(batch)).float().to(self.device)

    def embed(self, batch: np.ndarray, wavelengths: list[float]) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            salida = self.model.forward_features(self._tensor(batch), wavelengths)
            if salida.ndim == 3:
                salida = salida[:, 0]
        return salida.float().cpu().numpy()

    def embed_tokens(self, batch: np.ndarray, wavelengths: list[float]) -> np.ndarray:
        """Runs the transformer and hands back every token instead of their average.

        `forward_features` folds the tokens into one vector before returning, so the run
        is reproduced here up to that last step. It touches only the pieces the published
        architecture is built from —the input projection, the positional embedding, the
        blocks and the final norm— and repeats them in the order the reference
        implementation does, so a checkpoint that loads at all will run through this.

        Each token has already attended to the rest of its window, so it carries the 2.24
        km around it while still describing its own 160 m.
        """
        torch = self._torch
        tensor = self._tensor(batch)
        longitudes = torch.tensor(wavelengths, device=tensor.device, dtype=tensor.dtype)
        with torch.inference_mode():
            tokens, _ = self.model.patch_embed(tensor, longitudes)
            tokens = tokens + self.model.pos_embed[:, 1:, :]
            resumen = (self.model.cls_token + self.model.pos_embed[:, :1, :]).expand(
                tokens.shape[0], -1, -1
            )
            estado = torch.cat((resumen, tokens), dim=1)
            for bloque in self.model.blocks:
                estado = bloque(estado)
            estado = self.model.fc_norm(estado)
        return estado[:, 1:].float().cpu().numpy()


def extract(
    bands: dict[str, np.ndarray],
    windows: list,
    encoder: PatchEncoder,
    *,
    order: list[str] | None = None,
    batch: int = BATCH,
    token_size: int = TOKEN_SIZE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> tuple[np.ndarray, list]:
    """Encodes every window of a city and returns one vector per surviving token.

    Gives back the vectors and the instances they belong to, in the same order, so the
    two never have to be lined up again by hand later.
    """
    from satinsight.tiling import instances, stack

    order = order or sorted(bands)
    faltantes = [n for n in order if n not in WAVELENGTHS_UM]
    if faltantes:
        raise KeyError(f"no wavelength registered for {faltantes}")
    longitudes = [WAVELENGTHS_UM[n] for n in order]

    tokens, indices = instances(windows, bands, token_size, min_valid_fraction)
    if not windows:
        return np.empty((0, encoder.dim), dtype="float32"), []

    vectores = []
    for inicio in range(0, len(windows), batch):
        lote = np.stack(
            [normalize(stack(bands, w, order), order) for w in windows[inicio : inicio + batch]]
        )
        salida = encoder.embed_tokens(lote, longitudes)
        vectores.append(salida.reshape(-1, salida.shape[-1]))
    matriz = np.concatenate(vectores)[indices]
    log.info("%d instances encoded into %d dimensions", *matriz.shape)
    return matriz, tokens


def save(embeddings: np.ndarray, destino: Path, **etiquetas) -> Path:
    """Writes the vectors as half precision, which halves the disk for no measurable loss.

    Label columns of strings arrive from pandas as arrays of objects, and numpy can only
    store those by pickling. Reading a pickle back means trusting whatever the file
    contains, so they are narrowed to fixed-width text and the archive stays loadable
    with pickling switched off.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    limpias = {}
    for clave, valor in etiquetas.items():
        arreglo = np.asarray(valor)
        limpias[clave] = arreglo.astype("U") if arreglo.dtype == object else arreglo
    np.savez_compressed(destino, embeddings=embeddings.astype("float16"), **limpias)
    log.info("%s (%.1f MB)", destino.name, destino.stat().st_size / 1e6)
    return destino


def load(origen: Path) -> tuple[np.ndarray, dict]:
    """Reads back what `save` wrote, restoring the vectors to single precision."""
    with np.load(origen, allow_pickle=False) as datos:
        embeddings = datos["embeddings"].astype("float32")
        etiquetas = {k: datos[k] for k in datos.files if k != "embeddings"}
    return embeddings, etiquetas
