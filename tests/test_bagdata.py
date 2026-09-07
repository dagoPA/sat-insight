"""Tests of the bag loader logic that needs no files on disk."""

import pytest

from satinsight.bagdata import fusion_partner


def test_the_fusion_partner_swaps_the_modality_and_keeps_the_backbone_suffix():
    assert fusion_partner("s2") == "s1"
    assert fusion_partner("s1") == "s2"
    assert fusion_partner("s2_dofal") == "s1_dofal"
    assert fusion_partner("s1_cfm") == "s2_cfm"


def test_a_sensor_without_a_modality_prefix_has_no_partner():
    with pytest.raises(KeyError, match="partner"):
        fusion_partner("wc")
