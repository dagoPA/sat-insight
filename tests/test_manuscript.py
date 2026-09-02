"""Tests of the guard the manuscript figures depend on.

`agrees` is the whole reason the drivers can claim that a figure and the text cannot
disagree, so its three behaviours are pinned: it passes a recomputation that matches the
canon, it raises on one that does not, and it refuses a key the canon never carried rather
than treating an absent number as a match.
"""

import json

import pytest

from satinsight.manuscript import agrees, canon


def test_agrees_returns_a_matching_value():
    assert agrees(0.22703, 0.227, name="curve at 771 bags") == pytest.approx(0.22703)


def test_agrees_accepts_the_rounding_of_the_canon():
    """The canon stores four decimals, so a fifth-decimal difference is not a drift."""
    assert agrees(0.35383, 0.3538, name="map within") == pytest.approx(0.35383)


def test_agrees_raises_when_the_artifact_has_drifted():
    with pytest.raises(ValueError, match="diverged"):
        agrees(0.31, 0.227, name="curve at 771 bags")


def test_agrees_names_the_quantity_it_rejected():
    with pytest.raises(ValueError, match="curve at 771 bags"):
        agrees(0.31, 0.227, name="curve at 771 bags")


def test_agrees_refuses_a_number_the_canon_never_carried():
    with pytest.raises(KeyError, match="absent from the canon"):
        agrees(0.5, None, name="a quantity nobody recorded")


def test_canon_reads_a_results_file(tmp_path):
    path = tmp_path / "canon.json"
    path.write_text(json.dumps({"curve": {"771": {"spearman_within": 0.227}}}))
    assert canon(path)["curve"]["771"]["spearman_within"] == 0.227
