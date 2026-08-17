"""Sondeo del catálogo STAC de Planetary Computer sobre el AOI piloto."""

import planetary_computer as pc
import pystac_client

# Tuxtla Gutiérrez, Chiapas — zona de nubosidad persistente
BBOX = [-93.135, 16.740, -93.095, 16.768]
YEAR = "2020-01-01/2020-12-31"

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

s2 = list(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=YEAR).items())
print(f"S2 L2A escenas 2020: {len(s2)}")
clouds = sorted((i.properties["eo:cloud_cover"], i.id) for i in s2)
print(f"  min nubes: {clouds[0][0]:.1f}%  -> {clouds[0][1]}")
print(f"  max nubes: {clouds[-1][0]:.1f}% -> {clouds[-1][1]}")
over50 = sum(1 for c, _ in clouds if c > 50)
over80 = sum(1 for c, _ in clouds if c > 80)
print(f"  escenas >50% nubes: {over50}/{len(s2)} ({100*over50/len(s2):.0f}%)")
print(f"  escenas >80% nubes: {over80}/{len(s2)} ({100*over80/len(s2):.0f}%)")

for coll in ["sentinel-1-rtc", "sentinel-1-grd"]:
    try:
        items = list(cat.search(collections=[coll], bbox=BBOX, datetime=YEAR).items())
        print(f"{coll} escenas 2020: {len(items)}")
        if items:
            it = items[0]
            print(f"  assets: {list(it.assets)}")
            print(f"  props: orbit={it.properties.get('sat:orbit_state')} "
                  f"mode={it.properties.get('sar:instrument_mode')}")
    except Exception as e:
        print(f"{coll}: ERROR {type(e).__name__}: {e}")
