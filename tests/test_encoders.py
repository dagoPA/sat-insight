"""Tests for the extraction loop, with a stand-in encoder and no deep learning stack."""

import numpy as np
import pytest

from satinsight.encoders import WAVELENGTHS_UM, extract, load, normalize, save
from satinsight.tiling import grid


class FakeEncoder:
    """Returns the mean of each channel, so what it received can be checked."""

    dim = 2
    tokens = 196

    def __init__(self):
        self.wavelengths = None
        self.batches = 0

    def embed(self, batch, wavelengths):
        self.wavelengths = wavelengths
        self.batches += 1
        return batch.mean(axis=(2, 3)).astype("float32")

    def embed_tokens(self, batch, wavelengths):
        """One vector per token, with the token index in the first component."""
        self.wavelengths = wavelengths
        self.batches += 1
        n = batch.shape[0]
        output = np.zeros((n, self.tokens, self.dim), dtype="float32")
        output[..., 0] = np.arange(self.tokens)[None, :]
        output[..., 1] = batch.mean(axis=(2, 3))[:, :1]
        return output


def bands(height=224, width=448):
    return {
        "vv": np.full((height, width), 0.1, dtype="float32"),
        "vh": np.full((height, width), 0.05, dtype="float32"),
    }


def test_normalize_lands_inside_the_unit_range():
    patch = np.stack([np.full((8, 8), 0.1), np.full((8, 8), 0.05)]).astype("float32")
    output = normalize(patch, ["vv", "vh"])
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_normalize_fills_holes_with_the_middle_of_the_range():
    patch = np.full((1, 4, 4), np.nan, dtype="float32")
    assert (normalize(patch, ["vv"]) == 0.5).all()


def test_normalize_checks_the_channel_names_match():
    with pytest.raises(ValueError, match="channel names"):
        normalize(np.zeros((2, 4, 4), "float32"), ["vv"])


def test_extract_returns_one_vector_per_token():
    b = bands()
    windows = grid((224, 448), size=224)
    encoder = FakeEncoder()
    matrix, tokens = extract(b, windows, encoder, order=["vv", "vh"], batch=1)
    assert len(windows) == 2
    assert matrix.shape == (len(tokens), 2)
    assert len(tokens) == 2 * 196
    assert all(t.size == 16 for t in tokens)


def test_each_vector_keeps_the_token_it_came_from():
    """Row i has to correspond to token i, not to another of the same window."""
    b = bands()
    windows = grid((224, 448), size=224)
    matrix, tokens = extract(b, windows, FakeEncoder(), order=["vv", "vh"])
    expected = [i % 196 for i in range(len(tokens))]
    assert matrix[:, 0].astype(int).tolist() == expected


def test_extract_hands_over_the_wavelength_of_each_channel_in_order():
    encoder = FakeEncoder()
    extract(bands(224, 224), grid((224, 224), 224), encoder, order=["vh", "vv"])
    assert encoder.wavelengths == [WAVELENGTHS_UM["vh"], WAVELENGTHS_UM["vv"]]


def test_extract_refuses_a_channel_with_no_wavelength():
    b = bands()
    b["invented"] = np.zeros((224, 448), "float32")
    with pytest.raises(KeyError, match="invented"):
        extract(b, grid((224, 448), 224), FakeEncoder(), order=["invented"])


def test_extract_on_no_windows_returns_an_empty_matrix():
    matrix, tokens = extract(bands(), [], FakeEncoder(), order=["vv", "vh"])
    assert matrix.shape == (0, 2) and tokens == []


def test_save_and_load_round_trip(tmp_path):
    vectors = np.random.default_rng(0).random((7, 5)).astype("float32")
    path = save(vectors, tmp_path / "x.npz", tile=np.arange(7))
    read_back, labels = load(path)
    assert read_back.shape == vectors.shape
    assert np.allclose(read_back, vectors, atol=1e-3)
    assert (labels["tile"] == np.arange(7)).all()


def test_string_labels_survive_the_round_trip(tmp_path):
    """AGEB keys arrive as text and must not force reading with pickle."""
    import pandas as pd

    keys = pd.Series(["0710100010001", "0710100010002"]).to_numpy()
    path = save(np.zeros((2, 3), "float32"), tmp_path / "y.npz", cvegeo=keys)
    _, labels = load(path)
    assert list(labels["cvegeo"]) == ["0710100010001", "0710100010002"]
