"""Tests for the extraction loop, with a stand-in encoder and no deep learning stack."""

import numpy as np
import pytest

from satinsight.encoders import WAVELENGTHS_UM, extract, load, normalize, save
from satinsight.tiling import grid


class EncoderFalso:
    """Devuelve la media de cada canal, para poder comprobar qué recibió."""

    dim = 2

    def __init__(self):
        self.longitudes = None
        self.lotes = 0

    def embed(self, batch, wavelengths):
        self.longitudes = wavelengths
        self.lotes += 1
        return batch.mean(axis=(2, 3)).astype("float32")


def bandas(alto=128, ancho=128):
    return {
        "vv": np.full((alto, ancho), 0.1, dtype="float32"),
        "vh": np.full((alto, ancho), 0.05, dtype="float32"),
    }


def test_normalize_lands_inside_the_unit_range():
    patch = np.stack([np.full((8, 8), 0.1), np.full((8, 8), 0.05)]).astype("float32")
    salida = normalize(patch, ["vv", "vh"])
    assert salida.min() >= 0.0 and salida.max() <= 1.0


def test_normalize_fills_holes_with_the_middle_of_the_range():
    patch = np.full((1, 4, 4), np.nan, dtype="float32")
    assert (normalize(patch, ["vv"]) == 0.5).all()


def test_normalize_checks_the_channel_names_match():
    with pytest.raises(ValueError, match="channel names"):
        normalize(np.zeros((2, 4, 4), "float32"), ["vv"])


def test_extract_returns_one_vector_per_patch():
    b = bandas()
    tiles = grid((128, 128), size=64)
    encoder = EncoderFalso()
    salida = extract(b, tiles, encoder, order=["vv", "vh"], batch=2)
    assert salida.shape == (len(tiles), 2)
    assert encoder.lotes == 2


def test_extract_hands_over_the_wavelength_of_each_channel_in_order():
    encoder = EncoderFalso()
    extract(bandas(), grid((64, 64), 64), encoder, order=["vh", "vv"])
    assert encoder.longitudes == [WAVELENGTHS_UM["vh"], WAVELENGTHS_UM["vv"]]


def test_extract_refuses_a_channel_with_no_wavelength():
    b = bandas()
    b["inventado"] = np.zeros((128, 128), "float32")
    with pytest.raises(KeyError, match="inventado"):
        extract(b, grid((128, 128), 64), EncoderFalso(), order=["inventado"])


def test_extract_on_no_patches_returns_an_empty_matrix():
    salida = extract(bandas(), [], EncoderFalso(), order=["vv", "vh"])
    assert salida.shape == (0, 2)


def test_save_and_load_round_trip(tmp_path):
    vectores = np.random.default_rng(0).random((7, 5)).astype("float32")
    ruta = save(vectores, tmp_path / "x.npz", tile=np.arange(7))
    leidos, etiquetas = load(ruta)
    assert leidos.shape == vectores.shape
    assert np.allclose(leidos, vectores, atol=1e-3)
    assert (etiquetas["tile"] == np.arange(7)).all()
