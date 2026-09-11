"""Newcomer municipalities never take a key the catalogue already uses."""

from satinsight.agebs import City, beyond_keys


def test_collisions_carry_the_municipality_key():
    taken = {"guadalupe", "zaragoza"}
    newcomers = {
        "guadalupe": City("guadalupe", "Guadalupe", "32", "32017"),
        "abasolo": City("abasolo", "Abasolo", "11", "11001"),
    }
    out = beyond_keys(taken, newcomers)
    assert set(out) == {"guadalupe32017", "abasolo"}
    assert out["guadalupe32017"].key == "guadalupe32017"
    assert out["guadalupe32017"].municipality == "32017"


def test_newcomers_do_not_collide_with_each_other():
    newcomers = {"union": City("union", "Unión", "01", "01001")}
    first = beyond_keys(set(), newcomers)
    second = beyond_keys(set(first), {"union": City("union", "Unión", "02", "02002")})
    assert set(second) == {"union02002"}
