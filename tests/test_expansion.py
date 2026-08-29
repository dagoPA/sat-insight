"""The expansion must never move a key that already names a composite on disk."""

import pytest

from satinsight import agebs


@pytest.fixture(scope="module")
def catalogues():
    base = agebs.cities_by_size(stratify=True)
    extra = agebs.cities_extra()
    return base, extra


def test_no_base_key_is_reused_by_the_expansion(catalogues):
    base, extra = catalogues
    assert not set(base) & set(extra)


def test_no_municipality_appears_in_both(catalogues):
    base, extra = catalogues
    assert not {c.municipality for c in base.values()} & {c.municipality for c in extra.values()}


def test_the_merged_catalogue_keeps_every_base_key(catalogues):
    base, _ = catalogues
    merged = agebs.catalogue_with_extra()
    for key, city in base.items():
        assert merged[key].municipality == city.municipality


def test_the_five_renamed_cities_keep_their_key(catalogues):
    """Cancún, La Paz, Matamoros, Tonalá and Cuauhtémoc gain homonyms when the floor drops."""
    base, _ = catalogues
    merged = agebs.catalogue_with_extra()
    for key in ("benitojuarez", "lapaz", "matamoros", "tonala", "cuauhtemoc"):
        assert key in merged, f"{key} lost its key to the expansion"
        assert merged[key].municipality == base[key].municipality
