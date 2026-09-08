"""Partial fine-tuning of the frozen backbone under the label-proportion loss.

The rest of the paper never moves the encoder. This module lets the last transformer
blocks move, which is the ablation a reviewer asks for when the frozen bound is low: does
the map improve when the representation may adapt to the aggregates? Running every
window through the whole encoder at every step would cost an extraction pass per epoch,
so the encoder is split once: the frozen prefix is applied to every window of every city
a single time and its output cached to disk, and training runs only the trainable suffix
(the last blocks and the final norm) on those cached states, shared by both sensors as
in the pretrained model. Starting from the pretrained weights the suffix reproduces the
frozen vectors exactly, so the head's standardization statistics stay those of the
frozen features and every comparison with the frozen protocol is like for like.

Tokens are addressed by their position on the city grid: a token at pixel (y0, x0) sits
in window (y0 // 224, x0 // 224) at index ((y0 % 224) // 16) * 14 + (x0 % 224) // 16 of
that window's 196 patch tokens. A bag gathers the windows its tokens fall in, runs them
through the suffix as one batch, and picks its tokens out; the fused vector is the
concatenation of the two sensors' tokens at the same position, as everywhere else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from satinsight.dataset import CHANNELS
from satinsight.download import DATA_ROOT
from satinsight.encoders import WAVELENGTHS_UM, normalize
from satinsight.tiling import MIN_VALID_FRACTION, TOKEN_SIZE, WINDOW_SIZE

log = logging.getLogger(__name__)

ACTIVATIONS = DATA_ROOT / "activations"
TOKENS_PER_SIDE = WINDOW_SIZE // TOKEN_SIZE
TOKENS_PER_WINDOW = TOKENS_PER_SIDE * TOKENS_PER_SIDE


def cache_paths(city: str, sensor: str, split_at: int) -> tuple[Path, Path]:
    """Where the frozen-prefix states and the window positions of one city live."""
    stem = ACTIVATIONS / f"{city}_{sensor}_b{split_at}"
    return stem.with_suffix(".npy"), stem.with_name(stem.name + "_windows.npy")


def prefix_states(encoder, batch: np.ndarray, wavelengths: list[float], split_at: int):
    """Runs the input projection, positional embedding and the first `split_at` blocks."""
    torch = encoder._torch
    model = encoder.model
    tensor = encoder._tensor(batch)
    wavelengths_list = torch.tensor(wavelengths, device=tensor.device, dtype=tensor.dtype)
    with torch.inference_mode():
        tokens, _ = model.patch_embed(tensor, wavelengths_list)
        tokens = tokens + model.pos_embed[:, 1:, :]
        summary = (model.cls_token + model.pos_embed[:, :1, :]).expand(tokens.shape[0], -1, -1)
        state = torch.cat((summary, tokens), dim=1)
        for block in model.blocks[:split_at]:
            state = block(state)
    return state.half().cpu().numpy()


def cache_city(city: str, sensor: str, encoder, split_at: int, *, batch: int = 32) -> str:
    """Caches the frozen-prefix state of every window of one city and sensor."""
    from satinsight import tiling
    from satinsight.cache import load

    states_path, windows_path = cache_paths(city, sensor, split_at)
    if states_path.exists() and windows_path.exists():
        return "SKIP"
    bands, _, _ = load(DATA_ROOT / "composites" / f"{city}_{sensor}.tif")
    order = CHANNELS[sensor]
    bands = {c: bands[c] for c in order}
    windows = tiling.select(bands, min_valid_fraction=MIN_VALID_FRACTION)
    if not windows:
        return "EMPTY"
    wavelengths = [WAVELENGTHS_UM[n] for n in order]
    chunks = []
    for start in range(0, len(windows), batch):
        stack = np.stack(
            [
                normalize(tiling.stack(bands, w, order), order)
                for w in windows[start : start + batch]
            ]
        )
        chunks.append(prefix_states(encoder, stack, wavelengths, split_at))
    states = np.concatenate(chunks)
    ACTIVATIONS.mkdir(parents=True, exist_ok=True)
    np.save(states_path, states)
    np.save(windows_path, np.array([(w.y0, w.x0) for w in windows], dtype=np.int64))
    return f"OK {len(windows)} windows"


@dataclass
class CityStates:
    """The cached windows of one city and sensor, memory-mapped, addressable by pixel."""

    states: np.ndarray
    windows: np.ndarray
    lookup: dict = field(default_factory=dict)

    @classmethod
    def open(cls, city: str, sensor: str, split_at: int) -> CityStates:
        states_path, windows_path = cache_paths(city, sensor, split_at)
        windows = np.load(windows_path)
        lookup = {(int(y), int(x)): i for i, (y, x) in enumerate(windows)}
        return cls(np.load(states_path, mmap_mode="r"), windows, lookup)

    def address(self, y0: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Window index and in-window token index (offset by one for the class token)."""
        wy = (np.asarray(y0) // WINDOW_SIZE) * WINDOW_SIZE
        wx = (np.asarray(x0) // WINDOW_SIZE) * WINDOW_SIZE
        window = np.array([self.lookup[(int(a), int(b))] for a, b in zip(wy, wx, strict=True)])
        inside = ((np.asarray(y0) % WINDOW_SIZE) // TOKEN_SIZE) * TOKENS_PER_SIDE + (
            np.asarray(x0) % WINDOW_SIZE
        ) // TOKEN_SIZE
        return window, inside.astype(np.int64) + 1


@dataclass
class BagAddress:
    """Everything the suffix needs to rebuild one bag's fused vectors."""

    windows: dict[str, np.ndarray]
    """Per sensor, the unique window indices the bag touches, in the order they are batched."""

    rows: dict[str, np.ndarray]
    """Per sensor, for every token of the bag, its row among the batched windows."""

    inside: dict[str, np.ndarray]
    """Per sensor, for every token of the bag, its index inside its window."""


def address_bag(bag, states: dict[str, CityStates]) -> BagAddress:
    windows, rows, inside = {}, {}, {}
    for sensor, city_states in states.items():
        window, token = city_states.address(bag.y0, bag.x0)
        unique, position = np.unique(window, return_inverse=True)
        windows[sensor] = unique
        rows[sensor] = position
        inside[sensor] = token
    return BagAddress(windows, rows, inside)


def build_suffix(encoder, split_at: int):
    """The trainable tail of the encoder: the blocks from `split_at` on and the final norm.

    Returned as a module that maps cached states to token vectors, initialized from the
    pretrained weights so that its output equals the frozen extraction at step zero.
    """
    import copy

    from torch import nn

    class Suffix(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(copy.deepcopy(b) for b in encoder.model.blocks[split_at:])
            self.norm = copy.deepcopy(encoder.model.fc_norm)
            for parameter in self.parameters():
                parameter.requires_grad_(True)

        def forward(self, state):
            for block in self.blocks:
                state = block(state)
            return self.norm(state)

    return Suffix()


def fused_vectors(suffix, bag, address: BagAddress, states: dict[str, CityStates], torch, device):
    """Runs the bag's windows through the suffix and gathers its fused token vectors."""
    parts = []
    for sensor in ("s2", "s1"):
        batch = torch.from_numpy(
            np.ascontiguousarray(states[sensor].states[address.windows[sensor]])
        ).float()
        out = suffix(batch.to(device))
        rows = torch.from_numpy(address.rows[sensor]).to(device)
        inside = torch.from_numpy(address.inside[sensor]).to(device)
        parts.append(out[rows, inside])
    return torch.cat(parts, dim=1)
