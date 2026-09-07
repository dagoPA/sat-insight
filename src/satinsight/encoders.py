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
    output = np.empty_like(patch, dtype="float32")
    for i, name in enumerate(names):
        key = {"B04": "s2red", "B08": "s2nir"}.get(name, name)
        span = FIXED_RANGES.get(key)
        if span is None:
            channel = patch[i][np.isfinite(patch[i])]
            span = (float(channel.min()), float(channel.max())) if channel.size else (0.0, 1.0)
        low, high = span
        scaled = (patch[i] - low) / max(high - low, 1e-9)
        output[i] = np.nan_to_num(np.clip(scaled, 0.0, 1.0), nan=0.5)
    return output


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
        self.device = device or self._best_device()
        self.model = self._load(checkpoint)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dim = int(getattr(self.model, "embed_dim", 768))
        log.info("%s loaded on %s, %d dimensions", checkpoint, self.device, self.dim)

    def _best_device(self) -> str:
        """Picks the fastest accelerator present, Apple silicon included."""
        torch = self._torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self, checkpoint: str):
        """Fetches the pretrained weights through torchgeo, which hosts the checkpoints."""
        import torchgeo.models as models

        constructor = getattr(models, checkpoint.replace("-", "_"), None)
        if constructor is None:
            available = sorted(n for n in dir(models) if n.startswith("dofa_"))
            raise KeyError(f"unknown checkpoint {checkpoint!r}. Available: {available}")
        if "base" in checkpoint:
            weights = models.DOFABase16_Weights.DOFA_MAE
        elif "large" in checkpoint:
            weights = models.DOFALarge16_Weights.DOFA_MAE
        else:
            weights = None
        return constructor(weights=weights)

    def _tensor(self, batch: np.ndarray):
        return self._torch.from_numpy(np.ascontiguousarray(batch)).float().to(self.device)

    def embed(self, batch: np.ndarray, wavelengths: list[float]) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            output = self.model.forward_features(self._tensor(batch), wavelengths)
            if output.ndim == 3:
                output = output[:, 0]
        return output.float().cpu().numpy()

    def embed_tokens(self, batch: np.ndarray, wavelengths: list[float]) -> np.ndarray:
        """Runs the transformer and hands back every token instead of their average.

        `forward_features` folds the tokens into one vector before returning, so the run
        is reproduced here up to that last step. It touches only the pieces the published
        architecture is built from, the input projection, the positional embedding, the
        blocks and the final norm, and repeats them in the order the reference
        implementation does, so a checkpoint that loads at all will run through this.

        Each token has already attended to the rest of its window, so it carries the 2.24
        km around it while still describing its own 160 m.
        """
        torch = self._torch
        tensor = self._tensor(batch)
        wavelengths_list = torch.tensor(wavelengths, device=tensor.device, dtype=tensor.dtype)
        with torch.inference_mode():
            tokens, _ = self.model.patch_embed(tensor, wavelengths_list)
            tokens = tokens + self.model.pos_embed[:, 1:, :]
            summary = (self.model.cls_token + self.model.pos_embed[:, :1, :]).expand(
                tokens.shape[0], -1, -1
            )
            state = torch.cat((summary, tokens), dim=1)
            for block in self.model.blocks:
                state = block(state)
            state = self.model.fc_norm(state)
        return state[:, 1:].float().cpu().numpy()


CFM_WAVELENGTHS_NM = {
    "B02": (492.4, 66.0),
    "B03": (559.8, 36.0),
    "B04": (664.6, 31.0),
    "B08": (832.8, 106.0),
    "B11": (1613.7, 91.0),
    "vv": (5.55e7, 1e9),
    "vh": (5.55e7, 1e9),
}
"""Central wavelength and bandwidth in nanometres, the units Copernicus-FM conditions on.

Sentinel-2A band centres and widths from the ESA mission handbook. The radar pair is
given the C-band wavelength (5.405 GHz, 5.55 cm) with a nominal bandwidth, the way the
Copernicus-FM release treats Sentinel-1 GRD; VV and VH share one kernel, as in DOFA.
"""

CFM_REFERENCE_DAY = 18444.0
"""Days since 1970-01-01 for 2020-07-01, the middle of the annual composites."""


class CopernicusFmEncoder(DofaEncoder):
    """Copernicus-FM, a spectral-hypernetwork foundation model with metadata conditioning.

    Like DOFA it generates its patch projection from the wavelength of each channel, and
    on top it embeds the location, the date and the footprint of every window, so the
    extraction hands it those from the grid. The per-token path replicates
    `forward_features` up to the final norm, as the DOFA path does.
    """

    needs_metadata = True

    def __init__(self, checkpoint: str = "copernicusfm_base", device: str | None = None):
        super().__init__(checkpoint, device)

    def _load(self, checkpoint: str):
        import torchgeo.models as models

        if checkpoint != "copernicusfm_base":
            raise KeyError(f"unknown Copernicus-FM checkpoint {checkpoint!r}")
        return models.copernicusfm_base(weights=models.CopernicusFM_Base_Weights.CopernicusFM_ViT)

    def embed(self, batch, wavelengths, metadata=None):
        return self.embed_tokens(batch, wavelengths, metadata).mean(axis=1)

    def embed_tokens(self, batch, wavelengths, metadata=None):
        """Every token of every window, conditioned on band physics and window metadata.

        `wavelengths` are channel names here, resolved through `CFM_WAVELENGTHS_NM`;
        `metadata` is (n, 4) of longitude, latitude, day and area in km2, NaN where
        unknown, in which case the model falls back to its learned placeholder tokens.
        """
        torch = self._torch
        model = self.model
        tensor = self._tensor(batch)
        names = list(wavelengths)
        wvs = torch.tensor([CFM_WAVELENGTHS_NM[n][0] for n in names], device=tensor.device)
        bws = torch.tensor([CFM_WAVELENGTHS_NM[n][1] for n in names], device=tensor.device)
        if metadata is None:
            metadata = np.full((len(batch), 4), np.nan, dtype="float32")
        meta = torch.from_numpy(np.asarray(metadata, dtype="float32")).to(tensor.device)
        with torch.inference_mode():
            x = model.patch_embed_spectral(tensor, wavelengths=wvs.float(), bandwidths=bws.float())
            pos_embed = model.pos_embed
            embed_dim = pos_embed.shape[-1]
            lons, lats, times, areas = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3]
            if torch.isnan(lons).any() or torch.isnan(lats).any():
                coord = model.coord_token
            else:
                coord = model.get_coord_pos_embed(lons, lats, embed_dim)
            coord = model.coord_fc(coord)
            area = (
                model.scale_token
                if torch.isnan(areas).any()
                else model.get_area_pos_embed(areas, embed_dim)
            )
            area = model.scale_fc(area)
            time = (
                model.time_token
                if torch.isnan(times).any()
                else model.get_time_pos_embed(times, embed_dim)
            )
            time = model.time_fc(time)
            pos_embed = pos_embed + coord + area + time
            x = x + pos_embed[:, 1:, :]
            cls = (model.cls_token + pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1)
            for block in model.blocks:
                x = block(x)
            x = model.fc_norm(x)
        return x[:, 1:].float().cpu().numpy()


def window_metadata(windows: list, grid, day: float = CFM_REFERENCE_DAY) -> np.ndarray:
    """Longitude, latitude, day and footprint area of each window, for Copernicus-FM."""
    import rasterio.warp

    rows = np.array([w.center_px[0] for w in windows])
    cols = np.array([w.center_px[1] for w in windows])
    x, y = grid.transform * (cols, rows)
    lon, lat = rasterio.warp.transform(grid.crs, "EPSG:4326", np.atleast_1d(x), np.atleast_1d(y))
    side_km = windows[0].size * abs(grid.transform.a) / 1000.0 if windows else 0.0
    area = np.full(len(windows), side_km * side_km)
    return np.column_stack([lon, lat, np.full(len(windows), day), area]).astype("float32")


def extract(
    bands: dict[str, np.ndarray],
    windows: list,
    encoder: PatchEncoder,
    *,
    order: list[str] | None = None,
    batch: int = BATCH,
    token_size: int = TOKEN_SIZE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    metadata: np.ndarray | None = None,
) -> tuple[np.ndarray, list]:
    """Encodes every window of a city and returns one vector per surviving token.

    Gives back the vectors and the instances they belong to, in the same order, so the
    two never have to be lined up again by hand later. An encoder that declares
    `needs_metadata` receives the channel names in place of DOFA's wavelengths, plus the
    per-window `metadata` rows, which it resolves through its own tables.
    """
    from satinsight.tiling import instances, stack

    order = order or sorted(bands)
    missing = [n for n in order if n not in WAVELENGTHS_UM]
    if missing:
        raise KeyError(f"no wavelength registered for {missing}")
    wavelengths_list = (
        order if getattr(encoder, "needs_metadata", False) else [WAVELENGTHS_UM[n] for n in order]
    )

    tokens, indices = instances(windows, bands, token_size, min_valid_fraction)
    if not windows:
        return np.empty((0, encoder.dim), dtype="float32"), []

    vectors = []
    for start in range(0, len(windows), batch):
        batch_in = np.stack(
            [normalize(stack(bands, w, order), order) for w in windows[start : start + batch]]
        )
        if getattr(encoder, "needs_metadata", False):
            chunk = None if metadata is None else metadata[start : start + batch]
            output = encoder.embed_tokens(batch_in, wavelengths_list, chunk)
        else:
            output = encoder.embed_tokens(batch_in, wavelengths_list)
        vectors.append(output.reshape(-1, output.shape[-1]))
    matrix = np.concatenate(vectors)[indices]
    log.info("%d instances encoded into %d dimensions", *matrix.shape)
    return matrix, tokens


def save(embeddings: np.ndarray, destination: Path, **labels) -> Path:
    """Writes the vectors as half precision, which halves the disk for no measurable loss.

    Label columns of strings arrive from pandas as arrays of objects, and numpy can only
    store those by pickling. Reading a pickle back means trusting whatever the file
    contains, so they are narrowed to fixed-width text and the archive stays loadable
    with pickling switched off.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {}
    for key, value in labels.items():
        array = np.asarray(value)
        cleaned[key] = array.astype("U") if array.dtype == object else array
    np.savez_compressed(destination, embeddings=embeddings.astype("float16"), **cleaned)
    log.info("%s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
    return destination


def load(source: Path) -> tuple[np.ndarray, dict]:
    """Reads back what `save` wrote, restoring the vectors to single precision."""
    with np.load(source, allow_pickle=False) as data:
        embeddings = data["embeddings"].astype("float32")
        labels = {k: data[k] for k in data.files if k != "embeddings"}
    return embeddings, labels
