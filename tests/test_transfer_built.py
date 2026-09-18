"""The built fraction under a token, and the refusal of an unclassified mosaic."""

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "scripts")


def test_built_share_ignores_pixels_without_cover():
    from transfer_bags import BUILT_CODE, TOKEN_SIZE, built_share

    from satinsight import landcover

    classes = np.full((TOKEN_SIZE, 2 * TOKEN_SIZE), 10, dtype="uint8")
    classes[:, :TOKEN_SIZE] = BUILT_CODE
    classes[: TOKEN_SIZE // 2, TOKEN_SIZE:] = landcover.NO_DATA
    tokens = pd.DataFrame({"y0": [0, 0], "x0": [0, TOKEN_SIZE]})
    share = built_share(classes, tokens)
    assert share.tolist() == pytest.approx([1.0, 0.0])


def test_a_mosaic_left_unclassified_is_refused(monkeypatch):
    import transfer_bags

    from satinsight import landcover

    empty = np.full((32, 32), landcover.NO_DATA, dtype="uint8")
    monkeypatch.setattr(transfer_bags, "load", lambda path: (None, None, None))
    monkeypatch.setattr(transfer_bags, "transfer_aoi", lambda key: None)
    monkeypatch.setattr(transfer_bags.landcover, "mosaic", lambda aoi, grid: empty)
    with pytest.raises(RuntimeError, match="read failed"):
        transfer_bags.built_fraction("anywhere", pd.DataFrame({"y0": [0], "x0": [0]}))
