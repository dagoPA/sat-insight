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


def test_no_bag_municipality_of_any_base_city_enters_the_expansion():
    """A city's box covers neighbouring municipalities too; none of them may re-enter.

    Nine validation municipalities were also expansion bags before this was enforced, and
    the whole expanded-pool result set had to be re-measured. The exclusion must cover
    every municipality any base city turned into a bag, whatever split that city is in.
    """
    import pandas as pd

    from satinsight.dataset import paths
    from satinsight.download import DATA_ROOT

    bags_dir = paths(DATA_ROOT)["bags"]
    if not any(bags_dir.glob("*.parquet")):
        pytest.skip("no bags on disk")
    base = agebs.cities_by_size(stratify=True)
    extra_muns = {c.municipality for c in agebs.cities_extra().values()}
    for key in base:
        f = bags_dir / f"{key}.parquet"
        if f.exists():
            overlap = set(pd.read_parquet(f, columns=["municipio"]).municipio) & extra_muns
            assert not overlap, f"{key}: {sorted(overlap)} train and evaluate the same ground"
