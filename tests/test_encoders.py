"""Tests for the extraction loop, with a stand-in encoder and no deep learning stack."""

import numpy as np
import pytest

from satinsight.encoders import WAVELENGTHS_UM, extract, load, normalize, save
from satinsight.tiling import grid


class EncoderFalso:
    """Devuelve la media de cada canal, para poder comprobar qué recibió."""

    dim = 2
    tokens = 196

    def __init__(self):
        self.longitudes = None
        self.lotes = 0

    def embed(self, batch, wavelengths):
        self.longitudes = wavelengths
        self.lotes += 1
        return batch.mean(axis=(2, 3)).astype("float32")

    def embed_tokens(self, batch, wavelengths):
        """Un vector por token, con el índice del token en la primera componente."""
        self.longitudes = wavelengths
        self.lotes += 1
        n = batch.shape[0]
        salida = np.zeros((n, self.tokens, self.dim), dtype="float32")
        salida[..., 0] = np.arange(self.tokens)[None, :]
        salida[..., 1] = batch.mean(axis=(2, 3))[:, :1]
        return salida


def bandas(alto=224, ancho=448):
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


def test_extract_returns_one_vector_per_token():
    b = bandas()
    ventanas = grid((224, 448), size=224)
    encoder = EncoderFalso()
    matriz, tokens = extract(b, ventanas, encoder, order=["vv", "vh"], batch=1)
    assert len(ventanas) == 2
    assert matriz.shape == (len(tokens), 2)
    assert len(tokens) == 2 * 196
    assert all(t.size == 16 for t in tokens)


def test_each_vector_keeps_the_token_it_came_from():
    """La fila i tiene que corresponder al token i, no a otro de la misma ventana."""
    b = bandas()
    ventanas = grid((224, 448), size=224)
    matriz, tokens = extract(b, ventanas, EncoderFalso(), order=["vv", "vh"])
    esperado = [i % 196 for i in range(len(tokens))]
    assert matriz[:, 0].astype(int).tolist() == esperado


def test_extract_hands_over_the_wavelength_of_each_channel_in_order():
    encoder = EncoderFalso()
    extract(bandas(224, 224), grid((224, 224), 224), encoder, order=["vh", "vv"])
    assert encoder.longitudes == [WAVELENGTHS_UM["vh"], WAVELENGTHS_UM["vv"]]


def test_extract_refuses_a_channel_with_no_wavelength():
    b = bandas()
    b["inventado"] = np.zeros((224, 448), "float32")
    with pytest.raises(KeyError, match="inventado"):
        extract(b, grid((224, 448), 224), EncoderFalso(), order=["inventado"])


def test_extract_on_no_windows_returns_an_empty_matrix():
    matriz, tokens = extract(bandas(), [], EncoderFalso(), order=["vv", "vh"])
    assert matriz.shape == (0, 2) and tokens == []


def test_save_and_load_round_trip(tmp_path):
    vectores = np.random.default_rng(0).random((7, 5)).astype("float32")
    ruta = save(vectores, tmp_path / "x.npz", tile=np.arange(7))
    leidos, etiquetas = load(ruta)
    assert leidos.shape == vectores.shape
    assert np.allclose(leidos, vectores, atol=1e-3)
    assert (etiquetas["tile"] == np.arange(7)).all()


def test_string_labels_survive_the_round_trip(tmp_path):
    """Las claves de AGEB llegan como texto y no deben obligar a leer con pickle."""
    import pandas as pd

    claves = pd.Series(["0710100010001", "0710100010002"]).to_numpy()
    ruta = save(np.zeros((2, 3), "float32"), tmp_path / "y.npz", cvegeo=claves)
    _, etiquetas = load(ruta)
    assert list(etiquetas["cvegeo"]) == ["0710100010001", "0710100010002"]
