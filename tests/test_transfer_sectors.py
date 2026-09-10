"""Brazilian tract files resolve for every state, in either of the two places they live."""

from pathlib import Path

from satinsight.transfer import IBGE_SECTORS, sector_path


def test_hand_box_states_keep_their_original_files():
    root = Path("/data")
    for state, name in IBGE_SECTORS.items():
        assert sector_path(state, root) == root / "transfer" / name


def test_catalogued_states_resolve_under_setores():
    assert sector_path("AC", Path("/data")) == Path("/data/transfer/setores/AC_setores_CD2022.gpkg")
    assert sector_path("AL", Path("/data")) == Path("/data/transfer/setores/AL_setores_CD2022.gpkg")
